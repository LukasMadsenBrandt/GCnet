"""Render gene-network SVG figures from Granger edge CSVs."""


from itertools import chain
import os
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from networkx import minimum_spanning_tree
import pandas as pd
import networkx as nx
from typing import List, Union, Tuple

from community import community_louvain
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.legacy.network.induced_mst_st import (
    to_undirected_weighted,
    directed_edges_from_undirected_pairs,
)

from app import (
    create_graphviz_dot,
    consensus_partition
)
from gene_analysis.io.paths import data_path, resolve_existing_path, results_path

def assign_colors(partition, primary_gene="ZEB2"):
    """Assign display colors to Louvain/consensus community ids."""
    print(f"Assigning colors to communities: {set(partition.values())}")
    cmaps = ['tab20', 'tab20b']
    all_colors = list(chain.from_iterable(plt.get_cmap(c).colors for c in cmaps))
    community_colors = {}

    # Ensure the community of the primary_gene is first in order
    if primary_gene and primary_gene in partition:
        primary_community = partition[primary_gene]
        unique_communities = [primary_community] + [c for c in sorted(set(partition.values())) if c != primary_community]
    else:
        unique_communities = sorted(set(partition.values()))  # Sort to maintain consistency

    for idx, community in enumerate(unique_communities):
        color = all_colors[idx % len(all_colors)]
        color_hex = '#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3])
        community_colors[community] = (color_hex, str(idx + 1))  # Label communities as 1, 2, 3, ...

    print(f"Assigned community colors: {community_colors}")
    return community_colors

def build_gene_network(
    gene_list: List[str],
    csv_file: str,
    p_threshold: float = 0.05,
    all_connections: bool = False,
    mst: bool = False
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
    df = pd.read_csv(resolve_existing_path(csv_file))
    required_columns = {'gene1', 'gene2', 'lag', 'p-value'}
    assert required_columns.issubset(df.columns), f"Missing required columns: {required_columns - set(df.columns)}"

    # Filter
    if all_connections:
        # If all_connections is True, include all edges involving genes in gene_list
        df_filtered = df[(df['gene1'].isin(gene_list) & df['gene2'].isin(gene_list)) & (df['p-value'] <= p_threshold)]
    else:
        df_filtered = df[(((df['gene1'] == "ZEB2") & (df['gene2'].isin(gene_list))) | ((df['gene1'].isin(gene_list)) & (df['gene2'] == "ZEB2"))) & (df['p-value'] <= p_threshold)]
    print("Filtered edges:", len(df_filtered))
    # Build graph
    G = nx.DiGraph()
    for _, row in df_filtered.iterrows():
        source, target, lag, p = row['gene1'], row['gene2'], row['lag'], row['p-value']
        G.add_edge(source, target, lag=lag, p_value=p)

    print(f"Graph built, Nodes: {len(G.nodes)}, Edges: {len(G.edges)}" )

    return G

def build_mst(G_dir: nx.DiGraph, genes: list[str]) -> nx.DiGraph:
    """Build a directed minimum spanning forest for selected genes."""
    U = to_undirected_weighted(G_dir, weight_attr="p_value")
    U_sub = U.subgraph([g for g in genes if g in U]).copy()
    if U_sub.number_of_nodes() == 0:
        return nx.DiGraph()

    undirected_mst_edges = set()

    for comp_nodes in nx.connected_components(U_sub):
        comp = U_sub.subgraph(comp_nodes)
        if comp.number_of_edges() == 0:
            continue
        T = minimum_spanning_tree(comp, weight="weight")
        undirected_mst_edges.update(tuple(sorted(e)) for e in T.edges())

    H = directed_edges_from_undirected_pairs(G_dir, undirected_mst_edges)
    return H

def visualize_gene_network(
    gene_list,
    mst: bool,
    csv_file,
    p_threshold=0.05,
    layout="dot",
    graph_attr={},
    highlight_node=None,
    output_path="output_graph",
    number_of_runs=None,  # Optional for consensus partitioning
    simple_layout=True  # Use simple layout for better readability
):
    """Build, optionally simplify, and render a gene network to SVG."""
    # Build graph
    if isinstance(gene_list, str):
        all_connections = True
        # If gene_list is a file path, read the file
        with open(resolve_existing_path(gene_list), 'r') as f:
            gene_list = [line.strip() for line in f if line.strip()]
    else:
        all_connections = False
    
    G = build_gene_network(gene_list, csv_file, p_threshold, all_connections=all_connections)

    if mst:
        G = build_mst(G, gene_list)

    if G.number_of_nodes() == 0:
        print("No edges found. Empty graph.")
        return

    # Community detection
    if number_of_runs != None:
        # Use cached_assign_colors for consensus partitioning
        consensus, coassoc, partitions, n_of_genes_in_interest_gene_community = consensus_partition(G, n_runs=number_of_runs, gene_of_interest="ZEB2")
        partition = consensus
    else:
        # Use Louvain method for community detection
        ##partition = community_louvain.best_partition(G.to_undirected(), resolution=1.0, random_state=42)
        partition = {node: 0 for node in G.nodes()}


    community_colors = assign_colors(partition, primary_gene=goi)
    

    # Render DOT
    dot = create_graphviz_dot(
        G,
        partition=partition,
        community_colors=community_colors,
        highlight_node=highlight_node,
        layout=layout,
        graph_attr=graph_attr,
        simple_layout=simple_layout,  # Use simple layout for better readability
        highlight_new_genes=highlight_new_genes
    )
    print("Figure Created")

    # Save to SVG
    svg_path = f"{output_path}"  # e.g. "network/.../network_ZEB2_..."
    out_dir = os.path.dirname(svg_path)
    os.makedirs(out_dir, exist_ok=True)
    dot.render(svg_path, format='svg', cleanup=True)
    print(f"Saved graph SVG to {svg_path}")



if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
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
    pvalue_threshold = 0.0015  # Adjusted p-value threshold for filtering edges
    string_pvalue_threshold = str(pvalue_threshold).replace(".", "")
    number_of_genes = 115  # Number of genes to consider
    #file containing list of genes

    global highlight_new_genes
    highlight_new_genes = (False, ["CCND2", "DHX30", "NFIA", "PAK1", "SLC3A1", "SMARCAL1", "TEFM", "VPS4A", "YIF1B", "ZEB2"])  # Highlight these genes in the network

  # Highlight these genes in the network
    # Layout for Graphviz, can be 'dot', 'neato', 'fdp', 'circo', 'osage', 'sfdp', 'twopi', 'patchwork', etc.
    layout="sfdp"
    goi = "MECP2"
    graph_attr = {
        "nodesep": "3.2",  # minimal horizontal distance between nodes
        "ranksep": "3.2",  # minimal vertical distance between ranks (layers
        "overlap": "prism",
        "overlap_scaling": "1",
        "pad": "0.2",  # extra whitespace around the entire drawing
        'outputorder': 'edgesfirst',
        #"splines": "true",
        
    }
    if (False):
        csv_path = f"granger_causality_results_truncated.csv"
        consensus_step = "1st_consensus_clustering"  # Step in the analysis, can be used for naming output files
        genes_file = data_path("Kutsche", string_pvalue_threshold, f"gene_names_{string_pvalue_threshold}_{number_of_genes}.txt")

    else:
        #genes_file = f"Data/Kutsche/{string_pvalue_threshold}_explore/gene_names_{string_pvalue_threshold}_{number_of_genes}.txt"  # Path to the file containing all gene names
        #genes_file = data_path("Kutsche", "deliverables", "2", "00015", "ZEB2_115_p_00015.txt")
        genes_file = data_path("Kutsche", "00015", "gene_names_MECP2_00015_115.txt")
        consensus_step = "BeforeExploreCore115/"
        #csv_path = data_path("Kutsche", "deliverables", "5_6_7", "p00015", "mst", "zeb2_community", "edges.csv")
        #csv_path = os.path.join("network", "2nd_explore_23", "p0002", "SM9_0002_7200_runs", "toppct_005", "mst", "zeb2_community", "edges.csv")
        csv_path = "granger_causality_results_truncated.csv"
        #csv_path = f"granger_causality_results_explore_top5_{string_pvalue_threshold}.csv"
        #csv_path = f"granger_results_p00005_26295of26806506.csv"
        #csv_path = f"granger_results_p0001_134186of88877756.csv" 


    output_path = results_path("figures", consensus_step, string_pvalue_threshold, f"{number_of_genes} genes", f"network_{goi}_{string_pvalue_threshold}_{number_of_genes}_{layout}_3")

    visualize_gene_network(
        gene_list=genes_file,
        csv_file=csv_path,
        mst=True,
        p_threshold=pvalue_threshold,
        layout=layout,
        graph_attr=graph_attr,
        highlight_node=None,
        output_path=output_path,
        number_of_runs=5000,  # Optional, set to None if not needed
        simple_layout=False  # Use simple layout for better readability
    )
