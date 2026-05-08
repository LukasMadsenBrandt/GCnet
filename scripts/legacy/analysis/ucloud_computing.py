"""Legacy consensus/coassociation implementation kept for compatibility."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.cluster import AgglomerativeClustering
from community import community_louvain
import concurrent.futures
import networkx as nx
import numpy as np
import os

from gene_analysis_common.network import create_network
from project_paths import results_path
from gene_analysis_benito.granger_causality import filter_gene_pairs as filter_gene_pairs_benito
from gene_analysis_benito.granger_causality import collect_significant_edges as collect_significant_edges_benito


from gene_analysis_kutsche.granger_causality import filter_gene_pairs as filter_gene_pairs_kutsche
from gene_analysis_kutsche.granger_causality import collect_significant_edges as collect_significant_edges_kutsche



debugging = True
plots_dir = os.path.join(os.getcwd(), 'plots')

def debug_print(*args):
    """Print debug output when global debugging is enabled."""
    if debugging:
        print(" ".join(map(str, args)))

def debug_log(*args, log_file=results_path("logs", "debug_log.txt")):
    """Print and append debug output when global debugging is enabled."""
    if debugging:
        message = " ".join(map(str, args))
        print(message)  # Keep console output
        with open(log_file, "a") as f:
            f.write(message + "\n")

def run_louvain_once(G, seed):
    """Run one seeded Louvain partition on an undirected graph view."""
    partition = community_louvain.best_partition(G.to_undirected(), random_state=seed)
    return partition

def run_multiple_louvain_parallel(G, n_runs=100, existing_partitions=[]):
    """Run multiple Louvain partitions and preserve existing partitions."""
    partitions = existing_partitions.copy()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_louvain_once, G, seed) for seed in range(len(existing_partitions), n_runs)]
        for future in concurrent.futures.as_completed(futures):
            partitions.append(future.result())
    debug_print(f"Consensus detection complete: {len(partitions)} partitions run (expected {n_runs}).")
    return partitions


def build_coassociation_matrix(G, partitions):
    """Build a coassociation matrix from repeated community partitions."""
    nodes = list(G.nodes())
    n = len(nodes)
    coassoc = np.zeros((n, n))
    for part in partitions:
        for i in range(n):
            for j in range(i, n):
                if part.get(nodes[i]) == part.get(nodes[j]):
                    coassoc[i, j] += 1
                    if i != j:
                        coassoc[j, i] += 1
    coassoc /= len(partitions)
    return nodes, coassoc

def consensus_partition(G, n_runs=20, n_clusters=None, gene_of_interest=None, plot = None, existing_partitions=[]):
    """Run repeated Louvain consensus and return GOI coassociation metadata."""
    partitions = run_multiple_louvain_parallel(G, n_runs, existing_partitions)
    nodes, coassoc = build_coassociation_matrix(G, partitions)
    
    plot = True
    # If n_clusters is not provided, compute it as the average number of communities over all partitions.
    if n_clusters is None:
        total_communities = sum(len(set(partition.values())) for partition in partitions)
        avg_communities = total_communities / len(partitions)
        n_clusters = int(round(avg_communities)) if avg_communities > 1 else 1  # Ensure at least 1 cluster
    
    # Convert coassociation to a distance matrix (1 - coassociation)
    distance_matrix = 1 - coassoc
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    consensus = {node: labels[i] for i, node in enumerate(nodes)}
    
    # Compute the union of all genes that have ever been in the same community as gene_of_interest
    union_genes = set()
    if gene_of_interest is not None:
        for partition in partitions:
            if gene_of_interest in partition:
                current_comm = partition[gene_of_interest]
                union_genes.update([node for node, comm in partition.items() if comm == current_comm])
        # Plot the consensus matrix, I.E the freaquencies of two genes being in the same community
        # Only plot the matrix with the pair of the gene of interest and the rest of the genes
        if plot:
            plot_coassociation_for_gene(coassoc, nodes, gene_of_interest, n_runs, plots_dir)
        num_genes_same_comm = len(union_genes)
    else:
        num_genes_same_comm = None
    return consensus, coassoc, partitions, num_genes_same_comm, nodes
    

import plotly.express as px
import plotly.io as pio

def plot_coassociation_for_gene(coassoc, nodes, gene_of_interest, n_runs, save_dir):
    """
    Plot an interactive scatter plot showing the coassociation frequency for each gene 
    with respect to the gene_of_interest, based on the coassociation matrix.
    
    Parameters:
        coassoc (np.ndarray): The coassociation matrix (assumed symmetric).
        nodes (list): List of gene names corresponding to the rows/columns of coassoc.
        gene_of_interest (str): The gene to inspect (e.g. "ZEB2").
        n_runs (int): Number of consensus runs (used in the filename).
        save_dir (str): Directory to save the plot.

    Returns:
        fig (plotly.graph_objects.Figure): The interactive figure.
    """
    #gene_of_interest = "RPL8"
    # Find the index of the gene of interest in the nodes list.
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create directory if it does not exist
    
    try:
        idx = nodes.index(gene_of_interest)
    except ValueError:
        raise ValueError(f"Gene {gene_of_interest} not found in the nodes list.")

    # Extract the row corresponding to the gene of interest.
    frequencies = coassoc[idx, :]

    # Create a DataFrame for plotting.
    import pandas as pd
    df = pd.DataFrame({
        'Gene': nodes,
        'Coassociation Frequency': frequencies,
        'Index': list(range(len(nodes)))
    })
    # Remove the gene of interest itself.
    df = df[df['Gene'] != gene_of_interest]
    
    # Create an interactive scatter plot using Plotly Express.
    fig = px.scatter(
        df,
        x='Index',
        y='Coassociation Frequency',
        hover_data={'Gene': True, 'Coassociation Frequency': ':.4f'},
        title=f'Coassociation Frequencies for Genes with {gene_of_interest}',
    )
    
    # Remove x-axis tick labels since we don't need gene names there.
    fig.update_layout(
        xaxis=dict(
            showticklabels=False,
            title="",
        ),
        yaxis_title="Coassociation Frequency",
    )

    # Save the interactive plot as an HTML file.
    html_filename = f"coassoc_{gene_of_interest}_n_runs_{n_runs}.html"
    html_path = os.path.join(save_dir, html_filename)
    pio.write_html(fig, file=html_path, auto_open=False)
    debug_print(f"Interactive coassociation plot saved to {html_path}")

    # Additionally, save as a static image (PNG) if needed.
    png_filename = f"coassoc_{gene_of_interest}_n_runs_{n_runs}.png"
    png_path = os.path.join(save_dir, png_filename)
    try:
        fig.write_image(png_path)
        debug_print(f"Static coassociation plot saved to {png_path}")
    except Exception as e:
        debug_print(f"Error saving static image: {e}")

    return fig

import pandas as pd

def save_gene_frequencies_to_csv(nodes, coassoc, gene_of_interest, run_count, save_dir="gene_frequencies"):
    """
    Saves the coassociation frequencies of all genes with respect to the given gene_of_interest to a CSV file.
    
    Parameters:
        nodes (list): List of gene names.
        coassoc (np.ndarray): Coassociation matrix.
        gene_of_interest (str): The target gene for which frequencies are extracted.
        run_count (int): The number of Louvain runs for this iteration.
        save_dir (str): Directory where the CSV file will be saved.
    
    Returns:
        None (saves the file in the specified directory).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create directory if it does not exist

    try:
        gene_index = nodes.index(gene_of_interest)
        gene_frequencies = [
            {"Gene": gene, "Coassociation Frequency": coassoc[gene_index, i]}
            for i, gene in enumerate(nodes)
        ]
        
        # Convert to a DataFrame and sort by frequency (descending)
        df = pd.DataFrame(gene_frequencies)

        df = df.sort_values(by="Coassociation Frequency", ascending=False)

        # Define CSV file name
        csv_filename = f"{gene_of_interest}_coassoc_{run_count}_runs.csv"
        csv_path = os.path.join(save_dir, csv_filename)

        # Save DataFrame as CSV
        df.to_csv(csv_path, index=False)
        print(f"Gene frequencies saved to {csv_path}")

    except ValueError:
        print(f"Gene {gene_of_interest} not found in the network.")

def compute_quantile_thresholds(coassoc, nodes, quantiles=[0.90, 0.95]):
    """
    Compute the coassociation frequency thresholds for given quantiles.

    Parameters:
        coassoc (np.ndarray): The coassociation matrix.
        nodes (list): List of gene names corresponding to coassoc.
        quantiles (list): List of quantiles to compute thresholds for.

    Returns:
        dict: A dictionary of quantile thresholds.
    """
    # Flatten coassociation values (excluding self-associations)
    freq_values = coassoc.flatten()
    freq_values = freq_values[freq_values > 0]  # Remove zero values

    # Compute quantiles
    thresholds = {q: np.quantile(freq_values, q) for q in quantiles}

    return thresholds

if __name__ == '__main__':
    # Change working directory to the directory of the script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Parameters
    p_threshold = 0.0015
    genelist_global = ["MECP2"]
    gene_of_interest = "MECP2"
    #output_dir = f"1st_{gene_of_interest}_freq_{p_threshold}"
    output_dir = results_path("coassociation", f"2nd_{gene_of_interest}_freq_{p_threshold}")

    debug_log(f"Starting stability check for {genelist_global}")

    # Load gene pairs and build the network
    filtered_pairs = filter_gene_pairs_kutsche(filepath="granger_results_p00015_MECP2_433406of202677932.csv",
                                               p_threshold=p_threshold,
                                               starting_genes=genelist_global,
                                               higher_threshold_for_starting_genes=p_threshold)

    significant_edges = collect_significant_edges_kutsche(filtered_pairs,
                                                           p_value_threshold=p_threshold,
                                                           file=True,
                                                           filepath=filtered_pairs,
                                                           starting_genes=genelist_global,
                                                           higher_threshold_for_starting_genes=p_threshold)
    G = create_network(significant_edges)

    # Initialize parameters
    current_runs = 10  # Start at 5000
    increment = 0.2     # Increase in steps of 1000
    tolerance = 0.05      # 5% threshold for stability based on 90% quantile

    # Initialize previous run results
    previous_coassoc = None
    previous_thresholds = None
    previous_top_genes = set()

    # Track previous runs for stability checks
    previous_runs = set()

    partitions = []  # Initialize partitions
    # Start stability search
    while True:
        if current_runs in previous_runs:
            continue  # Skip if we've already computed for this run count

        previous_runs.add(current_runs)  # Store the current run count
        debug_log(f"\n🔄 Running Louvain clustering for {current_runs} runs...")

        # Run Louvain clustering
        consensus, coassoc, partitions, n_of_genes_in_interest_gene_community, nodes = consensus_partition(
            G, n_runs=current_runs, gene_of_interest=gene_of_interest, plot=True, existing_partitions=partitions
        )

        # Compute quantile thresholds
        current_thresholds = compute_quantile_thresholds(coassoc, nodes)

        # Save gene frequencies to CSV
        save_gene_frequencies_to_csv(nodes, coassoc, gene_of_interest, current_runs, save_dir=output_dir)

        # Identify top 5% genes (Ensure sorting is consistent)
        num_top_genes = max(1, int(0.05 * len(nodes)))  # Ensure at least 1 gene
        try:
            gene_index = nodes.index(gene_of_interest)
            current_top_genes = sorted(
                [(gene, coassoc[gene_index, i]) for i, gene in enumerate(nodes)],
                key=lambda x: x[1],
                reverse=True
            )[:num_top_genes]
        except ValueError:
            raise ValueError(f"Gene {gene_of_interest} not found in the network.")

        # Extract only gene names (ignore frequencies)
        current_top_genes_names = {gene for gene, _ in current_top_genes}

        # If we have a previous run, compute stability based on 90% quantile relative differences
        if previous_coassoc is not None:
            # Compute relative differences
            diff_matrix = np.abs(previous_coassoc - coassoc)
            relative_diff = diff_matrix / np.clip(previous_coassoc, a_min=1e-9, a_max=None)  # Avoid division by zero
            quantile_90_rel_diff = np.quantile(relative_diff, 0.90)

            # Compute overlap of top 5% genes
            overlapping_genes = previous_top_genes.intersection(current_top_genes_names)
            overlap_percentage = len(overlapping_genes) / num_top_genes * 100  # Convert to percentage

            # Log results
            debug_log(f"📊 Run: {current_runs}")
            debug_log(f"   - 90% Quantile of Relative Differences: {quantile_90_rel_diff:.6f} (Threshold: {tolerance})")
            debug_log(f"   - Top 5% Genes Overlap: {overlap_percentage:.2f}% (Threshold: 95%)")
            debug_log(f"   - Overlapping Genes ({len(overlapping_genes)}/{num_top_genes})")
            #debug_log(f"   - Overlapping Genes ({len(overlapping_genes)}/{num_top_genes}): {sorted(overlapping_genes)}")

            # Check stability criterion
            if quantile_90_rel_diff <= tolerance and overlap_percentage >= 95:
                debug_log(f"✅ Stability detected at {current_runs} runs.")
                break  # Stop when we reach stability

        # Store current results for next iteration
        previous_coassoc = coassoc
        previous_thresholds = current_thresholds
        previous_top_genes = current_top_genes_names

        # Increase by 20%
        current_runs += math.floor(int(current_runs*increment))

        # Increase run count by 1000
        #current_runs += increment
        debug_log(f"🔼 Increasing runs to {current_runs}")
