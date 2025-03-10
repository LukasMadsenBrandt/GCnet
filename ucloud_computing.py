from sklearn.cluster import AgglomerativeClustering
from community import community_louvain
import concurrent.futures
import networkx as nx
import numpy as np
import os

from gene_analysis_benito.granger_causality import filter_gene_pairs as filter_gene_pairs_benito
from gene_analysis_benito.granger_causality import collect_significant_edges as collect_significant_edges_benito


from gene_analysis_kutsche.granger_causality import filter_gene_pairs as filter_gene_pairs_kutsche
from gene_analysis_kutsche.granger_causality import collect_significant_edges as collect_significant_edges_kutsche



debugging = True
plots_dir = os.path.join(os.getcwd(), 'plots')
def create_network(significant_edges):
    G = nx.DiGraph()
    for edge in significant_edges:
        if len(edge) == 3:  # For benito and kutsche datasets
            (source, lag), (target, _), p_value = edge
            G.add_edge(source, target, lag=lag, p_value=p_value)
        elif len(edge) == 4:  # For intersection dataset
            (source, lag), (target, _), kutsche_p_value, benito_p_value = edge
            avg_p_value = (kutsche_p_value + benito_p_value) / 2
            G.add_edge(source, target, lag=lag, kutsche_p_value=kutsche_p_value, benito_p_value=benito_p_value, p_value=avg_p_value)
    return G

def debug_print(*args):
    if debugging:
        print(" ".join(map(str, args)))

def debug_log(*args, log_file="debug_log.txt"):
    if debugging:
        message = " ".join(map(str, args))
        print(message)  # Keep console output
        with open(log_file, "a") as f:
            f.write(message + "\n")

def run_louvain_once(G, seed):
    partition = community_louvain.best_partition(G.to_undirected(), random_state=seed)
    return partition

def run_multiple_louvain_parallel(G, n_runs=100):
    partitions = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_louvain_once, G, seed) for seed in range(n_runs)]
        for future in concurrent.futures.as_completed(futures):
            partitions.append(future.result())
    debug_print(f"Consensus detection complete: {len(partitions)} partitions run (expected {n_runs}).")
    return partitions


def build_coassociation_matrix(G, partitions):
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

def consensus_partition(G, n_runs=20, n_clusters=None, gene_of_interest=None, plot = None):
    partitions = run_multiple_louvain_parallel(G, n_runs)
    nodes, coassoc = build_coassociation_matrix(G, partitions)
    
    plot = True
    # If n_clusters is not provided, compute it as the average number of communities over all partitions.
    if n_clusters is None:
        total_communities = sum(len(set(partition.values())) for partition in partitions)
        avg_communities = total_communities / len(partitions)
        n_clusters = int(round(avg_communities))
    
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
    return consensus, coassoc, partitions, num_genes_same_comm
    

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

if __name__ == '__main__':
    # Change working directory to the directory of the script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    nruns = [2,3,4,5]
    #nruns = [100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
    pvalue_global = 0.0004
    p_threshold = 0.0004
    genelist_global = ["ZEB2"]
    debug_print(os.getcwd())
    filtered_pairs = filter_gene_pairs_kutsche(filepath="granger_causality_results.csv",
                                                    p_threshold=pvalue_global,
                                                    starting_genes=genelist_global,
                                                    higher_threshold_for_starting_genes=pvalue_global)
    significant_edges = collect_significant_edges_kutsche(filtered_pairs,
                                                                p_value_threshold=pvalue_global,
                                                                file=True,
                                                                filepath=filtered_pairs,
                                                                starting_genes=genelist_global,
                                                                higher_threshold_for_starting_genes=p_threshold)
    G = create_network(significant_edges)

    gene_of_interest="ZEB2" 
    for run in nruns:
        consensus, coassoc, partitions, n_of_genes_in_interest_gene_community = consensus_partition(G, n_runs=run, gene_of_interest=gene_of_interest, plot=True)
        partition = consensus
        if n_of_genes_in_interest_gene_community is not None:
            debug_log(f"There were {n_of_genes_in_interest_gene_community}/{len(partition)} unique genes in the community across {run} runs of Louvain in the same community as {gene_of_interest}")