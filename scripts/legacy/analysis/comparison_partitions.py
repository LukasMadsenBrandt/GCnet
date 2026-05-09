"""Benchmark and compare repeated Louvain partition/coassociation strategies."""

import math
import os
import sys
import time
import concurrent.futures
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from sklearn.cluster import AgglomerativeClustering
from community import community_louvain

from gene_analysis_common.network import create_network
from gene_analysis.io.paths import results_path
from gene_analysis_benito.granger_causality import filter_gene_pairs as filter_gene_pairs_benito
from gene_analysis_benito.granger_causality import collect_significant_edges as collect_significant_edges_benito

from gene_analysis_kutsche.granger_causality import filter_gene_pairs as filter_gene_pairs_kutsche
from gene_analysis_kutsche.granger_causality import collect_significant_edges as collect_significant_edges_kutsche

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------

debugging = True
plots_dir = os.path.join(os.getcwd(), 'plots')


# ---------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------

def debug_print(*args):
    """Print debug output when global debugging is enabled."""
    if debugging:
        print(" ".join(map(str, args)))


def debug_log(*args, log_file=results_path("logs", "debug_log.txt")):
    """Print and append debug output when global debugging is enabled."""
    if debugging:
        message = " ".join(map(str, args))
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")


# ---------------------------------------------------------------------
# Louvain utilities
# ---------------------------------------------------------------------

def run_louvain_once(G, seed):
    """Run one seeded Louvain partition on an undirected view of the graph."""
    partition = community_louvain.best_partition(G.to_undirected(), random_state=seed)
    return partition


def run_multiple_louvain_parallel(G, n_runs=100, existing_partitions=None):
    """Run multiple Louvain partitions in parallel and append to existing results."""
    if existing_partitions is None:
        existing_partitions = []
    partitions = existing_partitions.copy()
    start_seed = len(existing_partitions)

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_louvain_once, G, seed)
            for seed in range(start_seed, n_runs)
        ]
        for future in concurrent.futures.as_completed(futures):
            partitions.append(future.result())

    debug_print(f"Consensus detection complete: {len(partitions)} partitions run (expected {n_runs}).")
    return partitions


# ---------------------------------------------------------------------
# OLD serial coassociation (for comparison & benchmark)
# ---------------------------------------------------------------------

def build_coassociation_matrix_serial(G, partitions):
    """
    Original serial implementation – kept for comparison and for
    correctness checks in the benchmark.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    coassoc = np.zeros((n, n), dtype=float)

    for part in partitions:
        for i in range(n):
            for j in range(i, n):
                if part.get(nodes[i]) == part.get(nodes[j]):
                    coassoc[i, j] += 1.0
                    if i != j:
                        coassoc[j, i] += 1.0

    if len(partitions) > 0:
        coassoc /= float(len(partitions))
    return nodes, coassoc


# ---------------------------------------------------------------------
# NEW: parallel + incremental coassociation
# ---------------------------------------------------------------------

def coassoc_for_chunk(partitions_chunk, nodes):
    """
    Build an unnormalized coassociation count matrix for a subset of
    partitions. This runs in a worker process.
    Returns:
        coassoc_counts: (n x n) int matrix with counts
        n_partitions:   number of partitions summarized
    """
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    coassoc_counts = np.zeros((n, n), dtype=np.int32)

    for part in partitions_chunk:
        # labels[i] = community label for node i in this partition
        labels = np.full(n, -1, dtype=np.int32)
        for node, comm in part.items():
            idx = node_index.get(node, None)
            if idx is not None:
                labels[idx] = comm

        # For each community label, increment all (i, j) pairs in that community
        unique_comms = np.unique(labels)
        for c in unique_comms:
            if c == -1:
                continue
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                continue
            coassoc_counts[np.ix_(idx, idx)] += 1

    return coassoc_counts, len(partitions_chunk)


def build_coassociation_matrix_parallel(
    G,
    partitions,
    n_threads=None,
    prev_state=None,   # (coassoc_counts, n_prev) or None
):
    """
    Parallel + (optionally) incremental coassociation builder.

    Parameters
    ----------
    G : nx.Graph
        Graph used for node ordering.
    partitions : list[dict]
        Partitions: dict[node] -> community label.
        Assumption: new partitions are appended.
    n_threads : int or None
        Number of worker processes.
    prev_state : tuple or None
        (coassoc_counts, n_prev) from previous call, to reuse work.

    Returns
    -------
    nodes : list
    coassoc : np.ndarray (float)
    new_state : (coassoc_counts, n_total)
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n_threads is None:
        n_threads = os.cpu_count() or 1

    if prev_state is None:
        coassoc_counts = np.zeros((n, n), dtype=np.int32)
        n_prev = 0
    else:
        coassoc_counts, n_prev = prev_state

    # Only process NEW partitions since last call
    new_partitions = partitions[n_prev:]

    if not new_partitions:
        # Nothing new: just normalize what we have
        if n_prev == 0:
            coassoc = np.zeros((n, n), dtype=float)
        else:
            coassoc = coassoc_counts.astype(float) / float(n_prev)
        return nodes, coassoc, (coassoc_counts, n_prev)

    chunk_size = max(1, math.ceil(len(new_partitions) / n_threads))
    chunks = [
        new_partitions[i:i + chunk_size]
        for i in range(0, len(new_partitions), chunk_size)
    ]

    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(coassoc_for_chunk, chunk, nodes) for chunk in chunks]
        for fut in futures:
            chunk_counts, n_chunk = fut.result()
            coassoc_counts += chunk_counts
            n_prev += n_chunk

    coassoc = coassoc_counts.astype(float) / float(n_prev)
    return nodes, coassoc, (coassoc_counts, n_prev)


# ---------------------------------------------------------------------
# Consensus partition (now can use serial or parallel coassoc)
# ---------------------------------------------------------------------

def consensus_partition(
    G,
    n_runs=20,
    n_clusters=None,
    gene_of_interest=None,
    plot=None,
    existing_partitions=None,
    use_parallel=True,
    coassoc_state=None,
    n_threads=None,
):
    """
    Compute consensus partition using Louvain + coassociation matrix,
    with an option to use the new parallel/incremental summarization.

    Returns:
      consensus, coassoc, partitions, num_genes_same_comm, nodes, coassoc_state
    """
    if existing_partitions is None:
        existing_partitions = []

    partitions = run_multiple_louvain_parallel(G, n_runs, existing_partitions)

    # Build coassociation matrix (serial or parallel)
    if use_parallel:
        nodes, coassoc, coassoc_state = build_coassociation_matrix_parallel(
            G,
            partitions,
            n_threads=n_threads,
            prev_state=coassoc_state,
        )
    else:
        nodes, coassoc = build_coassociation_matrix_serial(G, partitions)
        # Build a compatible state so you *could* switch to incremental later
        coassoc_counts = (coassoc * len(partitions)).astype(np.int32)
        coassoc_state = (coassoc_counts, len(partitions))

    plot = True  # as in your original code

    # If n_clusters is not provided, compute it as the average number of
    # communities over all partitions.
    if n_clusters is None:
        total_communities = sum(len(set(partition.values())) for partition in partitions)
        avg_communities = total_communities / len(partitions)
        n_clusters = int(round(avg_communities)) if avg_communities > 1 else 1

    # Convert coassociation to distance matrix
    distance_matrix = 1 - coassoc
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    consensus = {node: labels[i] for i, node in enumerate(nodes)}

    # Compute union of genes ever in the same community as gene_of_interest
    union_genes = set()
    if gene_of_interest is not None:
        for partition in partitions:
            if gene_of_interest in partition:
                current_comm = partition[gene_of_interest]
                union_genes.update(
                    [node for node, comm in partition.items() if comm == current_comm]
                )

        if plot:
            plot_coassociation_for_gene(coassoc, nodes, gene_of_interest, n_runs, plots_dir)
        num_genes_same_comm = len(union_genes)
    else:
        num_genes_same_comm = None

    return consensus, coassoc, partitions, num_genes_same_comm, nodes, coassoc_state


# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------

def plot_coassociation_for_gene(coassoc, nodes, gene_of_interest, n_runs, save_dir):
    """
    Plot an interactive scatter plot showing the coassociation frequency
    for each gene with respect to the gene_of_interest.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        idx = nodes.index(gene_of_interest)
    except ValueError:
        raise ValueError(f"Gene {gene_of_interest} not found in the nodes list.")

    frequencies = coassoc[idx, :]

    df = pd.DataFrame({
        'Gene': nodes,
        'Coassociation Frequency': frequencies,
        'Index': list(range(len(nodes)))
    })

    # Remove the gene of interest itself
    df = df[df['Gene'] != gene_of_interest]

    fig = px.scatter(
        df,
        x='Index',
        y='Coassociation Frequency',
        hover_data={'Gene': True, 'Coassociation Frequency': ':.4f'},
        title=f'Coassociation Frequencies for Genes with {gene_of_interest}',
    )

    fig.update_layout(
        xaxis=dict(
            showticklabels=False,
            title="",
        ),
        yaxis_title="Coassociation Frequency",
    )

    html_filename = f"coassoc_{gene_of_interest}_n_runs_{n_runs}.html"
    html_path = os.path.join(save_dir, html_filename)
    pio.write_html(fig, file=html_path, auto_open=False)
    debug_print(f"Interactive coassociation plot saved to {html_path}")

    png_filename = f"coassoc_{gene_of_interest}_n_runs_{n_runs}.png"
    png_path = os.path.join(save_dir, png_filename)
    try:
        fig.write_image(png_path)
        debug_print(f"Static coassociation plot saved to {png_path}")
    except Exception as e:
        debug_print(f"Error saving static image: {e}")

    return fig


# ---------------------------------------------------------------------
# CSV + quantiles utilities
# ---------------------------------------------------------------------

def save_gene_frequencies_to_csv(nodes, coassoc, gene_of_interest, run_count, save_dir="gene_frequencies"):
    """
    Saves coassociation frequencies of all genes w.r.t. gene_of_interest to CSV.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        gene_index = nodes.index(gene_of_interest)
        gene_frequencies = [
            {"Gene": gene, "Coassociation Frequency": coassoc[gene_index, i]}
            for i, gene in enumerate(nodes)
        ]

        df = pd.DataFrame(gene_frequencies)
        df = df.sort_values(by="Coassociation Frequency", ascending=False)

        csv_filename = f"{gene_of_interest}_coassoc_{run_count}_runs.csv"
        csv_path = os.path.join(save_dir, csv_filename)

        df.to_csv(csv_path, index=False)
        print(f"Gene frequencies saved to {csv_path}")

    except ValueError:
        print(f"Gene {gene_of_interest} not found in the network.")


def compute_quantile_thresholds(coassoc, nodes, quantiles=[0.90, 0.95]):
    """
    Compute coassociation frequency thresholds for given quantiles.
    """
    freq_values = coassoc.flatten()
    freq_values = freq_values[freq_values > 0]  # Remove zeros
    if freq_values.size == 0:
        return {q: 0.0 for q in quantiles}

    thresholds = {q: np.quantile(freq_values, q) for q in quantiles}
    return thresholds


# ---------------------------------------------------------------------
# BENCHMARK UTILITIES
# ---------------------------------------------------------------------

def generate_random_partitions(nodes, n_partitions, min_comms=2, max_comms=10, seed=42):
    """
    Generate random partitions as dict[node] -> community_label
    for a fixed list of nodes (for benchmarking).
    """
    rng = np.random.default_rng(seed)
    n = len(nodes)
    partitions = []

    for r in range(n_partitions):
        k = int(rng.integers(min_comms, max_comms + 1))
        labels = rng.integers(0, k, size=n)
        part = {node: int(labels[i]) for i, node in enumerate(nodes)}
        partitions.append(part)

    return partitions


def benchmark_coassociation_builders(
    n_nodes=200,
    partition_counts=(50, 100, 200),
    n_threads=None,
):
    """
    Compare serial vs parallel coassociation building for different
    numbers of partitions.
    """
    if n_threads is None:
        n_threads = os.cpu_count() or 1

    nodes = [f"gene_{i}" for i in range(n_nodes)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    print(f"Using {n_threads} worker(s) for parallel version.")
    print(f"Benchmark on {n_nodes} nodes.\n")

    for n_partitions in partition_counts:
        print(f"=== n_partitions = {n_partitions} ===")

        partitions = generate_random_partitions(nodes, n_partitions, seed=123)

        # Serial
        t0 = time.perf_counter()
        nodes_serial, coassoc_serial = build_coassociation_matrix_serial(G, partitions)
        t1 = time.perf_counter()

        # Parallel (full build, no prev_state)
        t2 = time.perf_counter()
        nodes_parallel, coassoc_parallel, _ = build_coassociation_matrix_parallel(
            G,
            partitions,
            n_threads=n_threads,
            prev_state=None,
        )
        t3 = time.perf_counter()

        time_serial = t1 - t0
        time_parallel = t3 - t2

        if nodes_serial != nodes_parallel:
            print("  [WARNING] Node order differs between implementations!")
        else:
            max_diff = float(np.max(np.abs(coassoc_serial - coassoc_parallel)))
            print(f"  max |serial - parallel| = {max_diff:.3e}")

        print(f"  serial   time: {time_serial:.4f} s")
        print(f"  parallel time: {time_parallel:.4f} s")
        if time_parallel > 0:
            print(f"  speedup (serial/parallel): {time_serial / time_parallel:.2f}x")
        print()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == '__main__':
    # Change working directory to the directory of the script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Parameters
    p_threshold = 0.0015
    genelist_global = ["MECP2"]
    gene_of_interest = "MECP2"
    # output_dir = f"1st_{gene_of_interest}_freq_{p_threshold}"
    output_dir = results_path("coassociation", f"2nd_{gene_of_interest}_freq_{p_threshold}")

    debug_log(f"Starting stability check for {genelist_global}")

    # Load gene pairs and build the network
    filtered_pairs = filter_gene_pairs_kutsche(
        filepath="granger_results_p00015_MECP2_433406of202677932.csv",
        p_threshold=p_threshold,
        starting_genes=genelist_global,
        higher_threshold_for_starting_genes=p_threshold
    )

    significant_edges = collect_significant_edges_kutsche(
        filtered_pairs,
        p_value_threshold=p_threshold,
        file=True,
        filepath=filtered_pairs,
        starting_genes=genelist_global,
        higher_threshold_for_starting_genes=p_threshold
    )
    G = create_network(significant_edges)

    # Initialize parameters
    current_runs = 5000   # Start at 5000
    increment = 0.2       # Increase by 20% each step
    tolerance = 0.05      # 5% threshold for stability based on 90% quantile

    previous_coassoc = None
    previous_thresholds = None
    previous_top_genes = set()
    previous_runs = set()

    partitions = []        # store all partitions
    coassoc_state = None   # (coassoc_counts, n_prev) for incremental build


    # -----------------------------------------------------------------
    # Optional: run the benchmark to compare serial vs parallel
    # -----------------------------------------------------------------
    print("\n========== COASSOCIATION BENCHMARK low partition count high number of nodes ==========")
    benchmark_coassociation_builders(
        n_nodes=5000,
        partition_counts=(50, 60, 72),
        n_threads=os.cpu_count()- 1
        #n_threads=1
    )

    benchmark_coassociation_builders(
        n_nodes=5000,
        partition_counts=(50, 60, 72),
        #n_threads=os.cpu_count()- 1
        n_threads=1
    )
    print("\n========== COASSOCIATION BENCHMARK high partition count low number of nodes ==========")
    benchmark_coassociation_builders(
        n_nodes=1000,
        partition_counts=(5000, 6000, 7200),
        n_threads=os.cpu_count()- 1
        #n_threads=1
    )
    benchmark_coassociation_builders(
        n_nodes=1000,
        partition_counts=(5000, 6000, 7200),
        #n_threads=os.cpu_count()- 1
        n_threads=1
    )

    print("\n========== COASSOCIATION BENCHMARK high partition count high number of nodes ==========")
    benchmark_coassociation_builders(
        n_nodes=5000,
        partition_counts=(5000, 6000, 7200),
        n_threads=os.cpu_count()- 1
        #n_threads=1
    )
    benchmark_coassociation_builders(
        n_nodes=5000,
        partition_counts=(5000, 6000, 7200),
        #n_threads=os.cpu_count()- 1
        n_threads=1
    )
    # Stability search
    while False:
        if current_runs in previous_runs:
            # Already computed this run count
            current_runs += math.floor(int(current_runs * increment))
            continue

        previous_runs.add(current_runs)
        debug_log(f"\n🔄 Running Louvain clustering for {current_runs} runs...")

        # Run consensus with parallel coassociation
        consensus, coassoc, partitions, n_of_genes_in_interest_gene_community, nodes, coassoc_state = consensus_partition(
            G,
            n_runs=current_runs,
            gene_of_interest=gene_of_interest,
            plot=True,
            existing_partitions=partitions,
            use_parallel=True,
            coassoc_state=coassoc_state,
            n_threads=os.cpu_count()
        )

        # Compute quantile thresholds
        current_thresholds = compute_quantile_thresholds(coassoc, nodes)

        # Save gene frequencies to CSV
        save_gene_frequencies_to_csv(nodes, coassoc, gene_of_interest, current_runs, save_dir=output_dir)

        # Identify top 5% genes
        num_top_genes = max(1, int(0.05 * len(nodes)))
        try:
            gene_index = nodes.index(gene_of_interest)
            current_top_genes = sorted(
                [(gene, coassoc[gene_index, i]) for i, gene in enumerate(nodes)],
                key=lambda x: x[1],
                reverse=True
            )[:num_top_genes]
        except ValueError:
            raise ValueError(f"Gene {gene_of_interest} not found in the network.")

        current_top_genes_names = {gene for gene, _ in current_top_genes}

        if previous_coassoc is not None:
            diff_matrix = np.abs(previous_coassoc - coassoc)
            relative_diff = diff_matrix / np.clip(previous_coassoc, a_min=1e-9, a_max=None)
            quantile_90_rel_diff = np.quantile(relative_diff, 0.90)

            overlapping_genes = previous_top_genes.intersection(current_top_genes_names)
            overlap_percentage = len(overlapping_genes) / num_top_genes * 100.0

            debug_log(f"📊 Run: {current_runs}")
            debug_log(f"   - 90% Quantile of Relative Differences: {quantile_90_rel_diff:.6f} (Threshold: {tolerance})")
            debug_log(f"   - Top 5% Genes Overlap: {overlap_percentage:.2f}% (Threshold: 95%)")
            debug_log(f"   - Overlapping Genes ({len(overlapping_genes)}/{num_top_genes})")

            if quantile_90_rel_diff <= tolerance and overlap_percentage >= 95:
                debug_log(f"✅ Stability detected at {current_runs} runs.")
                break

        previous_coassoc = coassoc
        previous_thresholds = current_thresholds
        previous_top_genes = current_top_genes_names

        # Increase runs by 20%
        current_runs += math.floor(int(current_runs * increment))
        debug_log(f"🔼 Increasing runs to {current_runs}")
