# create_gene_network.py


import pandas as pd
import networkx as nx
from typing import List, Union, Tuple

from community import community_louvain
from collections import Counter

from app import (
    assign_colors,
    create_graphviz_dot,
    consensus_partition
)


def build_gene_network(
    gene_list: List[str],
    csv_file: str,
    p_threshold: float = 0.05
) -> nx.DiGraph:
    """
    Build a gene network from a CSV file given a list of starting genes and a p-value threshold.

    Parameters:
        gene_list (List[str]): Genes to start the search from (source).
        csv_file (str): Path to CSV file with columns: Gene1, Gene2, Lag, P_Value.
        p_threshold (float): Maximum p-value for edge inclusion.

    Returns:
        networkx.DiGraph: Directed graph of genes.
    """
    df = pd.read_csv(csv_file)
    required_columns = {'gene1', 'gene2', 'lag', 'p-value'}
    assert required_columns.issubset(df.columns), f"Missing required columns: {required_columns - set(df.columns)}"
    
    # Filter
    #df_filtered = df[(df['gene1'].isin(gene_list)) & (df['gene2'].isin(gene_list)) & (df['p-value'] <= p_threshold)]

    # Filter for edges involving "ZEB2" and genes in gene_list
    df_filtered = df[(((df['gene1'] == "ZEB2") & (df['gene2'].isin(gene_list))) | ((df['gene1'].isin(gene_list)) & (df['gene2'] == "ZEB2"))) & (df['p-value'] <= p_threshold)]

    # If you want to filter by gene1 only, uncomment the next line:
    # df_filtered = df[(df['gene1'].isin(gene_list)) & (df['p-value'] < p_threshold)]

    # Build graph
    G = nx.DiGraph()
    for _, row in df_filtered.iterrows():
        source, target, lag, p = row['gene1'], row['gene2'], row['lag'], row['p-value']
        G.add_edge(source, target, lag=lag, p_value=p)
    
    return G

# Optional: Export
def export_graph(G: nx.DiGraph, path: str, format: str = "gexf"):
    if format == "gexf":
        nx.write_gexf(G, path)
    elif format == "graphml":
        nx.write_graphml(G, path)
    elif format == "edgelist":
        nx.write_edgelist(G, path, data=['lag', 'p_value'])
    else:
        raise ValueError(f"Unsupported format: {format}")



def visualize_gene_network(
    gene_list,
    csv_file,
    p_threshold=0.05,
    layout="dot",
    highlight_node=None,
    output_path="output_graph",
    number_of_runs=None,  # Optional for consensus partitioning
    simple_layout=True  # Use simple layout for better readability
):
    # Build graph
    G = build_gene_network(gene_list, csv_file, p_threshold)

    if G.number_of_nodes() == 0:
        print("No edges found. Empty graph.")
        return

    # Community detection
    if number_of_runs != None:
        # Use cached_assign_colors for consensus partitioning
        consensus, coassoc, partitions, n_of_genes_in_interest_gene_community = consensus_partition(G, n_runs=number_of_runs, gene_of_interest="ZEB2")
        partition = consensus
    else:
        partition = community_louvain.best_partition(G.to_undirected(), random_state=42)

    # Assign colors
    community_colors = assign_colors(partition)

    # Render DOT
    dot = create_graphviz_dot(
        G,
        partition=partition,
        community_colors=community_colors,
        highlight_node=highlight_node,
        layout=layout,
        simple_layout=simple_layout  # Use simple layout for better readability
    )

    # Save to SVG
    svg_path = f"{output_path}"
    dot.render(svg_path, format='svg', cleanup=True)
    print(f"Saved graph SVG to {svg_path}")




if __name__ == "__main__":
    genes_00004 = ["ZEB2", "CCND2", "RIC1", "DHDDS", "RPL8"]
    genes_0004 = [
    'AIMP1',
    'ZEB2',
    'CACNA1B',
    'CCND2',
    'CEP295',
    'DHDDS',
    'FZD2',
    'LCA5',
    'RIC1',
    'SNAPC4',
    'CNOT1',
    'COLEC11',
    'EMC10',
    'FBXL3',
    'FBXW11',
    'FMN2',
    'MAST1',
    'MN1',
    'MTHFD1',
    'RPL8',
    'RPS19',
    'SERAC1'
]
    top3_in = [
    'ZEB2',
    'CCND2',
    'RIC1',
    'DHDDS',
]
    top3_out = [
        'ZEB2', 'RPL8', 'FBXW11', 'MTHFD1'
    ]
    csv_path = "granger_causality_results_truncated.csv"
    visualize_gene_network(
        gene_list=genes_0004,
        csv_file=csv_path,
        p_threshold=0.004,
        layout="sfdp",
        highlight_node=None,
        output_path="network_zeb2_0004_sfdp",
        number_of_runs=1000,  # Optional, set to None if not needed
        simple_layout=True  # Use simple layout for better readability
    )
