"""Optimized consensus/coassociation utilities used by the canonical pipeline."""

import math
import os
import time
import logging
import concurrent.futures

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from sklearn.cluster import AgglomerativeClustering
from community import community_louvain

from gene_analysis_common.network import create_network
from gene_analysis_benito.granger_causality import filter_gene_pairs as filter_gene_pairs_benito
from gene_analysis_benito.granger_causality import collect_significant_edges as collect_significant_edges_benito

from gene_analysis_kutsche.granger_causality import filter_gene_pairs as filter_gene_pairs_kutsche
from gene_analysis_kutsche.granger_causality import collect_significant_edges as collect_significant_edges_kutsche

from concurrent.futures import ProcessPoolExecutor

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from gene_analysis.analysis.stability import GeneStabilityConfig, compute_csv_stability_metrics
from gene_analysis.io.paths import results_path


@dataclass(frozen=True)
class GeneFreqDiffConfig:
    """Settings for comparing two coassociation-frequency CSV files."""

    gene_col: str = "Gene"
    freq_col: str = "Coassociation Frequency"
    case_insensitive: bool = True

    # "union" includes all genes, missing treated as 0.0
    # "inter" includes only genes present in both
    rowset_mode: str = "inter"

    # "previous" | "max" | "mean"
    denominator: str = "previous"
    eps: float = 1e-9

    # Quantile to report (you had 0.80 in your snippet)
    quantile_p: float = 0.80

    # Top overlap settings
    top_fraction: float = 0.05
    top_k: Optional[int] = None

    # Optional thresholds for “OK/NOT OK” reporting
    rel_diff_tolerance: Optional[float] = None
    overlap_threshold_percent: float = 95.0


def _norm_gene(g: str, case_insensitive: bool) -> str:
    return g.lower() if case_insensitive else g


def read_gene_freq_csv(path: str, gene_col: str, freq_col: str, *, case_insensitive: bool) -> Dict[str, float]:
    """Read gene frequencies from a CSV into a normalized mapping."""
    data: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path}: Missing CSV header.")

        headers = {h.lower(): h for h in reader.fieldnames}
        gcol = headers.get(gene_col.lower(), gene_col)
        fcol = headers.get(freq_col.lower(), freq_col)

        if gcol not in reader.fieldnames:
            raise ValueError(f"{path}: Gene column '{gene_col}' not found. Columns: {reader.fieldnames}")
        if fcol not in reader.fieldnames:
            raise ValueError(f"{path}: Frequency column '{freq_col}' not found. Columns: {reader.fieldnames}")

        for row in reader:
            gene = (row.get(gcol) or "").strip()
            freq_raw = (row.get(fcol) or "").strip()
            if not gene:
                continue
            try:
                freq = float(freq_raw)
            except ValueError:
                continue
            data[_norm_gene(gene, case_insensitive)] = freq

    return data


def _rel_diff(prev: float, curr: float, *, denominator: str, eps: float) -> float:
    num = abs(prev - curr)
    if denominator == "previous":
        den = max(prev, eps)
    elif denominator == "max":
        den = max(prev, curr, eps)
    elif denominator == "mean":
        den = max((prev + curr) / 2.0, eps)
    else:
        raise ValueError(f"Invalid denominator: {denominator}")
    return num / den


def _quantile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 1:
        return max(values)

    xs = sorted(values)
    pos = p * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _top_k_from_freqs(freqs: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    return sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:k]


def compare_gene_frequency_csvs(
    previous_file: str,
    current_file: str,
    *,
    out_dir: str,
    cfg: GeneFreqDiffConfig = GeneFreqDiffConfig(),
    sort_by: str = "RelDiff",      # "RelDiff" | "AbsDiff" | "Gene"
    sort_desc: bool = True,
) -> Tuple[str, str, float, float, int]:
    """
    Compares two gene-frequency CSVs and writes:
      - diff csv
      - summary txt

    Returns:
      (diff_csv_path, summary_txt_path, q_val, overlap_pct, k)
    """
    prev = read_gene_freq_csv(previous_file, cfg.gene_col, cfg.freq_col, case_insensitive=cfg.case_insensitive)
    curr = read_gene_freq_csv(current_file,  cfg.gene_col, cfg.freq_col, case_insensitive=cfg.case_insensitive)

    prev_genes = set(prev.keys())
    curr_genes = set(curr.keys())
    union_genes = prev_genes | curr_genes
    inter_genes = prev_genes & curr_genes

    if cfg.rowset_mode.lower() == "union":
        genes_for_rows = union_genes
    elif cfg.rowset_mode.lower() == "inter":
        genes_for_rows = inter_genes
    else:
        raise ValueError("rowset_mode must be 'union' or 'inter'")

    rows = []
    rel_diffs_for_stats: List[float] = []

    for g in genes_for_rows:
        p = prev.get(g, 0.0)
        c = curr.get(g, 0.0)
        abs_diff = abs(p - c)
        rel_diff = _rel_diff(p, c, denominator=cfg.denominator, eps=cfg.eps)

        if g in inter_genes:
            rel_diffs_for_stats.append(rel_diff)

        rows.append({
            "Gene": g,  # normalized if case_insensitive
            "PrevFreq": p,
            "CurrFreq": c,
            "AbsDiff": abs_diff,
            "RelDiff": rel_diff,
            "InPrev": 1 if g in prev_genes else 0,
            "InCurr": 1 if g in curr_genes else 0,
        })

    if sort_by not in {"RelDiff", "AbsDiff", "Gene"}:
        raise ValueError("sort_by must be 'RelDiff', 'AbsDiff', or 'Gene'")
    rows.sort(key=lambda r: r[sort_by], reverse=sort_desc)

    q_val = _quantile(rel_diffs_for_stats, cfg.quantile_p)

    n_prev = len(prev)
    n_curr = len(curr)
    if cfg.top_k is not None:
        k = int(cfg.top_k)
    else:
        k = max(1, int(round(cfg.top_fraction * min(n_prev, n_curr))))

    top_prev = set(g for g, _ in _top_k_from_freqs(prev, k))
    top_curr = set(g for g, _ in _top_k_from_freqs(curr, k))
    overlap = top_prev & top_curr
    overlap_pct = (len(overlap) / k) * 100.0 if k > 0 else float("nan")

    os.makedirs(out_dir, exist_ok=True)
    prev_tag = os.path.splitext(os.path.basename(previous_file))[0]
    curr_tag = os.path.splitext(os.path.basename(current_file))[0]

    diff_csv_path = os.path.join(out_dir, f"gene_freq_diff__{prev_tag}__to__{curr_tag}.csv")
    summary_txt_path = os.path.join(out_dir, f"gene_freq_diff_summary__{prev_tag}__to__{curr_tag}.txt")

    # Write diff CSV
    with open(diff_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Gene", "PrevFreq", "CurrFreq", "AbsDiff", "RelDiff", "InPrev", "InCurr"])
        for r in rows:
            w.writerow([
                r["Gene"],
                f"{r['PrevFreq']:.12g}",
                f"{r['CurrFreq']:.12g}",
                f"{r['AbsDiff']:.12g}",
                f"{r['RelDiff']:.12g}",
                r["InPrev"],
                r["InCurr"],
            ])

    # Summary + optional checks
    stability_lines = []
    if cfg.rel_diff_tolerance is not None:
        ok = (q_val <= cfg.rel_diff_tolerance)
        stability_lines.append(
            f" - Quantile({cfg.quantile_p:.2f}) of Relative Differences: {q_val:.6f} "
            f"(Tolerance: {cfg.rel_diff_tolerance}) -> {'OK' if ok else 'NOT OK'}"
        )
    else:
        stability_lines.append(f" - Quantile({cfg.quantile_p:.2f}) of Relative Differences: {q_val:.6f}")

    ok_overlap = (overlap_pct >= cfg.overlap_threshold_percent)
    stability_lines.append(
        f" - Top-{k} Overlap: {overlap_pct:.2f}% "
        f"(Threshold: {cfg.overlap_threshold_percent:.2f}%) -> {'OK' if ok_overlap else 'NOT OK'}"
    )

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("Gene Frequency Comparison Summary\n")
        f.write("================================\n")
        f.write(f"Previous file : {previous_file}\n")
        f.write(f"Current file  : {current_file}\n\n")
        f.write(f"Prev genes: {n_prev}\n")
        f.write(f"Curr genes: {n_curr}\n")
        f.write(f"Intersection used for rel-diff stats: {len(inter_genes)} genes\n\n")
        f.write("\n".join(stability_lines) + "\n\n")
        f.write(f"Diff table written to: {diff_csv_path}\n")

    return diff_csv_path, summary_txt_path, q_val, overlap_pct, k

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------

def setup_logging(log_file: str = "run.log", level=logging.INFO):
    """
    Configure root logger with both console and file handlers.
    """
    if os.path.dirname(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handlers = []

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    handlers.append(ch)

    # File
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(formatter)
    handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers)


logger = logging.getLogger(__name__)

plots_dir = os.path.join(os.getcwd(), 'plots')


# ---------------------------------------------------------------------
# Louvain utilities (fast, process-based)
# ---------------------------------------------------------------------

_GLOBAL_GRAPH = None  # will hold G_undirected in worker processes


def _init_louvain_worker(G_undirected):
    """
    Initializer for ProcessPoolExecutor workers.
    Sets the global graph so it doesn't need to be pickled per task.
    """
    global _GLOBAL_GRAPH
    _GLOBAL_GRAPH = G_undirected


def _run_louvain_seeds(seeds):
    """
    Run Louvain for a batch of seeds on the global graph in a worker.
    Returns a list of partitions, one per seed (same order).
    """
    parts = []
    for seed in seeds:
        part = community_louvain.best_partition(_GLOBAL_GRAPH, random_state=seed)
        parts.append(part)
    return parts


def run_multiple_louvain_parallel(
    G_undirected,
    n_runs=100,
    existing_partitions=None,
    n_workers=None,
    batch_per_worker=True,
):
    """
    Run Louvain multiple times in parallel using processes.

    Parameters
    ----------
    G_undirected : nx.Graph
        Precomputed undirected version of your graph.
    n_runs : int
        Total number of runs desired (including existing_partitions).
    existing_partitions : list[dict] or None
        Partitions already computed. New partitions will be appended.
    n_workers : int or None
        Number of worker processes; defaults to os.cpu_count().
    batch_per_worker : bool
        If True, each worker will handle a chunk of seeds in one task
        (reduces IPC overhead). If False, one seed per task.

    Returns
    -------
    partitions : list[dict]
        All partitions (old + new), length == n_runs.
    """
    if existing_partitions is None:
        existing_partitions = []

    partitions = list(existing_partitions)
    start_seed = len(existing_partitions)
    n_new = n_runs - start_seed

    if n_new <= 0:
        logger.info(
            "No new Louvain runs needed (already have %d >= %d).",
            len(existing_partitions), n_runs
        )
        return partitions

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    seeds = list(range(start_seed, n_runs))

    logger.info(
        "Running Louvain %d→%d (adding %d runs) with %d worker(s)...",
        start_seed, n_runs, n_new, n_workers
    )
    t0 = time.perf_counter()

    # Start process pool after G_undirected is created
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_louvain_worker,
        initargs=(G_undirected,)
    ) as executor:

        if batch_per_worker:
            # Chunk seeds into roughly equal batches per worker
            chunk_size = max(1, math.ceil(n_new / n_workers))
            seed_chunks = [seeds[i:i + chunk_size] for i in range(0, n_new, chunk_size)]

            futures = [executor.submit(_run_louvain_seeds, chunk) for chunk in seed_chunks]
            total_done = 0
            for fut in concurrent.futures.as_completed(futures):
                parts_chunk = fut.result()
                partitions.extend(parts_chunk)
                total_done += len(parts_chunk)
                logger.debug("  Received %d partitions (total now %d)", len(parts_chunk), total_done + start_seed)
        else:
            # One seed per task, preserves strict ordering but more overhead
            for i, part in enumerate(executor.map(_run_louvain_seeds, [[s] for s in seeds]), start=1):
                # part is a list of length 1
                partitions.extend(part)
                if i % 50 == 0 or i == n_new:
                    logger.debug("  Completed %d/%d Louvain runs", i, n_new)

    elapsed = time.perf_counter() - t0
    logger.info("Louvain runs complete: %d partitions total (%.2fs)", len(partitions), elapsed)

    return partitions


# ---------------------------------------------------------------------
# Parallel + incremental coassociation
# ---------------------------------------------------------------------

def coassoc_for_chunk(partitions_chunk, nodes):
    """
    Build an unnormalized coassociation count matrix for a subset of
    partitions. This runs in a worker process.
    Returns:
        coassoc_counts: (n x n) int matrix with counts
        n_partitions:   number of partitions summarized
    """
    import numpy as _np  # local import for multiprocessing safety

    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    coassoc_counts = _np.zeros((n, n), dtype=_np.int32)

    for part in partitions_chunk:
        labels = _np.full(n, -1, dtype=_np.int32)
        for node, comm in part.items():
            idx = node_index.get(node, None)
            if idx is not None:
                labels[idx] = comm

        unique_comms = _np.unique(labels)
        for c in unique_comms:
            if c == -1:
                continue
            idx = _np.where(labels == c)[0]
            if idx.size == 0:
                continue
            coassoc_counts[_np.ix_(idx, idx)] += 1

    return coassoc_counts, len(partitions_chunk)


def build_coassociation_matrix_parallel(
    G,
    partitions,
    n_threads=None,
    prev_state=None,   # (coassoc_counts, n_prev) or None
):
    """
    Parallel + (optionally) incremental coassociation builder.

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

    new_partitions = partitions[n_prev:]
    if not new_partitions:
        if n_prev == 0:
            coassoc = np.zeros((n, n), dtype=float)
        else:
            coassoc = coassoc_counts.astype(float) / float(n_prev)
        logger.debug("No new partitions for coassociation; returning existing matrix built from %d partitions.", n_prev)
        return nodes, coassoc, (coassoc_counts, n_prev)

    logger.info("Building coassociation from %d NEW partitions (total after update: %d)...",
                len(new_partitions), len(partitions))
    t0 = time.perf_counter()

    chunk_size = max(1, math.ceil(len(new_partitions) / n_threads))
    chunks = [
        new_partitions[i:i + chunk_size]
        for i in range(0, len(new_partitions), chunk_size)
    ]
    logger.debug("Using %d worker(s), chunk size %d, %d chunk(s) total.",
                 n_threads, chunk_size, len(chunks))

    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(coassoc_for_chunk, chunk, nodes) for chunk in chunks]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            chunk_counts, n_chunk = fut.result()
            coassoc_counts += chunk_counts
            n_prev += n_chunk
            logger.debug("  Processed chunk %d/%d (%d partitions)", i, len(chunks), n_chunk)

    coassoc = coassoc_counts.astype(float) / float(n_prev)
    elapsed = time.perf_counter() - t0
    logger.info("Coassociation build complete (%d partitions total) in %.2fs", n_prev, elapsed)

    return nodes, coassoc, (coassoc_counts, n_prev)


# ---------------------------------------------------------------------
# Consensus partition
# ---------------------------------------------------------------------

def consensus_partition(
    G,
    G_undirected,
    n_runs=20,
    n_clusters=None,
    gene_of_interest=None,
    plot=True,
    existing_partitions=None,
    coassoc_state=None,
    n_threads=None,
    n_louvain_workers=None,
):
    """
    Compute consensus partition using Louvain + coassociation matrix.

    Returns:
      consensus, coassoc, partitions, num_genes_same_comm, nodes, coassoc_state
    """
    if existing_partitions is None:
        existing_partitions = []

    logger.info("Starting consensus partition for %d runs.", n_runs)
    t0_total = time.perf_counter()

    # Run Louvain multiple times on the undirected graph
    partitions = run_multiple_louvain_parallel(
        G_undirected,
        n_runs=n_runs,
        existing_partitions=existing_partitions,
        n_workers=n_louvain_workers,
        batch_per_worker=True,
    )

    # Build/update coassociation matrix
    nodes, coassoc, coassoc_state = build_coassociation_matrix_parallel(
        G,
        partitions,
        n_threads=n_threads,
        prev_state=coassoc_state,
    )

    # Number of clusters
    if n_clusters is None:
        total_communities = sum(len(set(partition.values())) for partition in partitions)
        avg_communities = total_communities / len(partitions)
        n_clusters = int(round(avg_communities)) if avg_communities > 1 else 1
        logger.info("Estimated n_clusters = %d (avg communities across partitions: %.2f).",
                    n_clusters, avg_communities)

    distance_matrix = 1 - coassoc
    logger.info("Running AgglomerativeClustering with %d clusters on distance matrix of shape %s.",
                n_clusters, distance_matrix.shape)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    consensus = {node: labels[i] for i, node in enumerate(nodes)}

    # Genes in same community as gene_of_interest across all partitions
    union_genes = set()
    if gene_of_interest is not None:
        for partition in partitions:
            if gene_of_interest in partition:
                current_comm = partition[gene_of_interest]
                union_genes.update(
                    [node for node, comm in partition.items() if comm == current_comm]
                )

        num_genes_same_comm = len(union_genes)
        logger.info("Gene '%s' co-clustered with %d unique genes across all runs.",
                    gene_of_interest, num_genes_same_comm)

        if plot:
            plot_coassociation_for_gene(coassoc, nodes, gene_of_interest, n_runs, plots_dir)
    else:
        num_genes_same_comm = None

    logger.info("Consensus partition completed in %.2fs.", time.perf_counter() - t0_total)
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
        logger.warning("Gene %s not found in nodes list; skipping plot.", gene_of_interest)
        return None

    frequencies = coassoc[idx, :]

    df = pd.DataFrame({
        'Gene': nodes,
        'Coassociation Frequency': frequencies,
        'Index': list(range(len(nodes)))
    })

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
    logger.info("Interactive coassociation plot saved to %s", html_path)

    png_filename = f"coassoc_{gene_of_interest}_n_runs_{n_runs}.png"
    png_path = os.path.join(save_dir, png_filename)
    try:
        fig.write_image(png_path)
        logger.info("Static coassociation plot saved to %s", png_path)
    except Exception as e:
        logger.warning("Error saving static image: %s", e)

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
    except ValueError:
        logger.warning("Gene %s not found in the network; skipping CSV export.", gene_of_interest)
        return

    gene_frequencies = [
        {"Gene": gene, "Coassociation Frequency": coassoc[gene_index, i]}
        for i, gene in enumerate(nodes)
    ]

    df = pd.DataFrame(gene_frequencies)
    df = df.sort_values(by="Coassociation Frequency", ascending=False)

    csv_filename = f"{gene_of_interest}_coassoc_{run_count}_runs.csv"
    csv_path = os.path.join(save_dir, csv_filename)

    df.to_csv(csv_path, index=False)
    logger.info("Gene frequencies saved to %s", csv_path)


def compute_quantile_thresholds(coassoc, nodes, quantiles=[0.90, 0.95]):
    """
    Compute coassociation frequency thresholds for given quantiles.
    """
    freq_values = coassoc.flatten()
    freq_values = freq_values[freq_values > 0]  # Remove zeros
    if freq_values.size == 0:
        thresholds = {q: 0.0 for q in quantiles}
        logger.debug("Quantile thresholds (no positive values): %s", thresholds)
        return thresholds

    thresholds = {q: float(np.quantile(freq_values, q)) for q in quantiles}
    logger.debug("Quantile thresholds: %s", thresholds)
    return thresholds


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == '__main__':
    # ---------------------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------------------
    current_runs = 5000       # you can start lower now, since each step is cheaper
    increment = 0.2        # Increase by 20%
    tolerance = 0.05       # 5% threshold for stability based on 90% quantile
    strict_threshold = 0.0015
    p_threshold = 0.0402
    genelist_global = ["ZEB2"]
    gene_of_interest = "ZEB2"
    output_dir = results_path("coassociation", f"BASE_BENITO_GORILLA2_1st_{gene_of_interest}_freq_{p_threshold}")
    gc_filepath = "granger_causality_results_truncated_benito_gorilla.csv"

    

    # Set up logging first
    setup_logging(log_file=f"{output_dir}/run_{current_runs}_{gene_of_interest}_{p_threshold}.log", level=logging.INFO)
    logger.info("=== Starting stability check pipeline ===")

    # Change working directory to the directory of the script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)



    logger.info("Starting stability check for %s (p_threshold=%.6f)", genelist_global, p_threshold)
    
    significant_edges = collect_significant_edges_kutsche(
        gc_filepath,
        p_value_threshold=strict_threshold,
        file=True,
        filepath=gc_filepath,
        starting_genes=genelist_global,
        higher_threshold_for_starting_genes=p_threshold
    )
    logger.info("Significant edges: %d", len(significant_edges))
    G = create_network(significant_edges)
    # Precompute undirected graph once for Louvain
    logger.info("Creating undirected view for Louvain...")
    G_undirected = G.to_undirected(as_view=False)
    logger.info("Undirected graph has %d nodes and %d edges", G_undirected.number_of_nodes(), G_undirected.number_of_edges())



    previous_coassoc = None
    previous_thresholds = None
    previous_top_genes = set()
    previous_runs = set()

    partitions = []
    coassoc_state = None
    previous_freq_csv_path = None
    diff_out_dir = os.path.join(output_dir, "freq_diffs")

    previous_freq_csv_path = None
    workers = max(1, (os.cpu_count() or 1) - 1)

    csv_cfg = GeneStabilityConfig(
        gene_col="Gene",
        freq_col="Coassociation Frequency",
        case_insensitive=True,
        rowset_mode="inter",
        denominator="previous",
        quantile_p=0.90,
        top_fraction=0.05,
        top_k=None,
    )

    while True:
        logger.info("=== Stability iteration with %d runs ===", current_runs)

        consensus, coassoc, partitions, n_genes_same_comm, nodes, coassoc_state = consensus_partition(
            G,
            G_undirected,
            n_runs=current_runs,
            gene_of_interest=gene_of_interest,
            plot=False,
            existing_partitions=partitions,
            coassoc_state=coassoc_state,
            n_threads=workers,
            n_louvain_workers=workers,
        )

        save_gene_frequencies_to_csv(
            nodes,
            coassoc,
            gene_of_interest,
            current_runs,
            save_dir=output_dir,
        )

        current_freq_csv_path = os.path.join(
            output_dir,
            f"{gene_of_interest}_coassoc_{current_runs}_runs.csv",
        )

        if previous_freq_csv_path is not None:
            q_rel, overlap_pct, k = compute_csv_stability_metrics(
                previous_freq_csv_path,
                current_freq_csv_path,
                cfg=csv_cfg,
            )

            logger.info(
                "CSV stability metrics → Q(%.2f)=%.6f | Top-%d overlap=%.2f%%",
                csv_cfg.quantile_p,
                q_rel,
                k,
                overlap_pct,
            )

            if q_rel <= tolerance and overlap_pct >= 95.0:
                logger.info("Stability detected at %d runs. Stopping.", current_runs)
                break

        previous_freq_csv_path = current_freq_csv_path
        current_runs += max(1, int(math.floor(current_runs * increment)))

    logger.info("=== Stability pipeline finished ===")
