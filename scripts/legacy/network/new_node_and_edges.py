"""Compare original and exploration networks to report newly added structure."""

import math
import os
import sys
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gene_analysis_common.network import create_network
from gene_analysis_kutsche.granger_causality import filter_gene_pairs as filter_gene_pairs_kutsche
from gene_analysis_kutsche.granger_causality import collect_significant_edges as collect_significant_edges_kutsche

def network_stats(G, label, top5_genes=None):
    """Print basic network size statistics for one graph."""
    print(f"\nAnalysing: {label}")
    print(f"Number of edges: {len(G.edges)}")
    print(f"Number of nodes: {len(G.nodes)}")
    if top5_genes:
        print(f"Number of genes in Top 5% + ZEB2: {len(top5_genes)}")
    else:
        print(f"Number of genes in Top 5% + ZEB2: {math.floor(len(G.nodes) * 0.05) + 1}")

def get_new_nodes_and_edges(G_explore, G_orig):
    """Return new exploration nodes and new edges that touch those nodes."""
    nodes_orig = set(G_orig.nodes)
    nodes_explore = set(G_explore.nodes)
    new_nodes = nodes_explore - nodes_orig

    edges_orig = set(G_orig.edges)
    edges_explore = set(G_explore.edges)
    new_edges = edges_explore - edges_orig

    # (Optional) Focus only on edges that involve new nodes
    new_edges_involving_new_nodes = {e for e in new_edges if e[0] in new_nodes or e[1] in new_nodes}
    return new_nodes, new_edges_involving_new_nodes

if __name__ == '__main__':
    # Change working directory to the directory of the script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Example: Original and Explore files
    orig_filepath = "granger_causality_results_truncated.csv"
    orig_threshold = 0.002
    explore_filepath = "granger_results_p0002_25939of10544120.csv"
    explore_threshold = 0.002
    genelist_global = ["ZEB2"]

    # Load original
    filtered_pairs_orig = filter_gene_pairs_kutsche(
        filepath=orig_filepath,
        p_threshold=orig_threshold,
        starting_genes=genelist_global,
        higher_threshold_for_starting_genes=orig_threshold,
    )
    significant_edges_orig = collect_significant_edges_kutsche(
        filtered_pairs_orig,
        p_value_threshold=orig_threshold,
        file=True,
        filepath=filtered_pairs_orig,
        starting_genes=genelist_global,
        higher_threshold_for_starting_genes=orig_threshold,
    )
    G_orig = create_network(significant_edges_orig)
    network_stats(G_orig, orig_filepath)

    # Load explore
    filtered_pairs_explore = filter_gene_pairs_kutsche(
        filepath=explore_filepath,
        p_threshold=explore_threshold,
        starting_genes=genelist_global,
        higher_threshold_for_starting_genes=explore_threshold,
    )
    significant_edges_explore = collect_significant_edges_kutsche(
        filtered_pairs_explore,
        p_value_threshold=explore_threshold,
        file=True,
        filepath=filtered_pairs_explore,
        starting_genes=genelist_global,
        higher_threshold_for_starting_genes=explore_threshold,
    )
    G_explore = create_network(significant_edges_explore)
    network_stats(G_explore, explore_filepath)

    # Save original nodes to a file
    with open("explore_nodes_0002.txt", "a") as f:
        for gene in G_explore.nodes():
            f.write(f"{gene}\n")

    # New nodes/edges
    new_nodes, new_edges_involving_new_nodes = get_new_nodes_and_edges(G_explore, G_orig)
    print(f"\nNEW NODES (not in original): {len(new_nodes)}")
    print(f"NEW EDGES involving at least one NEW NODE: {len(new_edges_involving_new_nodes)}")
