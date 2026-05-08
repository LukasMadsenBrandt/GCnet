"""Render induced neighborhoods around a gene of interest."""

import os
import sys
from pathlib import Path
import pandas as pd
import networkx as nx
from itertools import chain
from matplotlib import pyplot as plt
from networkx import minimum_spanning_tree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.legacy.network.induced_mst_st import to_undirected_weighted, directed_edges_from_undirected_pairs
from app import create_graphviz_dot, consensus_partition


def assign_colors(partition, primary_gene=None):
    """Assign stable display colors to community ids, prioritizing the GOI community."""
    cmaps = ["tab20", "tab20b"]
    all_colors = list(chain.from_iterable(plt.get_cmap(c).colors for c in cmaps))
    community_colors = {}

    if primary_gene is not None and primary_gene in partition:
        primary_community = partition[primary_gene]
        unique_communities = [primary_community] + [
            c for c in sorted(set(partition.values())) if c != primary_community
        ]
    else:
        unique_communities = sorted(set(partition.values()))

    for idx, community in enumerate(unique_communities):
        color = all_colors[idx % len(all_colors)]
        color_hex = "#%02x%02x%02x" % tuple(int(255 * c) for c in color[:3])
        community_colors[community] = (color_hex, str(idx + 1))

    return community_colors


def load_edge_table(csv_file: str) -> pd.DataFrame:
    """Load a Granger edge CSV and validate the required columns."""
    df = pd.read_csv(csv_file)
    required_columns = {"gene1", "gene2", "lag", "p-value"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # optional normalization if needed
    df["gene1"] = df["gene1"].astype(str).str.strip()
    df["gene2"] = df["gene2"].astype(str).str.strip()
    return df


def get_neighbors(df: pd.DataFrame, gene: str, p_threshold: float) -> set[str]:
    """Return direct neighbors of ``gene`` below the p-value threshold."""
    df_filt = df[df["p-value"] <= p_threshold]

    mask = (df_filt["gene1"] == gene) | (df_filt["gene2"] == gene)
    sub = df_filt.loc[mask, ["gene1", "gene2"]]

    neighbors = set(sub["gene1"]).union(set(sub["gene2"]))
    neighbors.discard(gene)
    return neighbors


def build_induced_neighborhood(
    csv_file: str,
    gene: str,
    p_threshold: float = 0.05
) -> tuple[nx.DiGraph, set[str]]:
    """
    Build the induced subgraph on {gene} U neighbors(gene),
    including edges among neighbors themselves.
    """
    df = load_edge_table(csv_file)
    df_filt = df[df["p-value"] <= p_threshold].copy()

    neighbors = get_neighbors(df_filt, gene, p_threshold)
    node_set = set([gene]) | neighbors

    # induced subgraph on node_set
    df_sub = df_filt[
        df_filt["gene1"].isin(node_set) & df_filt["gene2"].isin(node_set)
    ]

    G = nx.DiGraph()
    for _, row in df_sub.iterrows():
        G.add_edge(
            row["gene1"],
            row["gene2"],
            lag=row["lag"],
            p_value=row["p-value"]
        )

    # ensure isolated central node is present even if weird edge case
    if gene not in G:
        G.add_node(gene)

    return G, neighbors


from typing import List, Optional, Tuple, Dict

def build_mst(G_dir: nx.DiGraph, genes: Optional[List[str]] = None) -> nx.DiGraph:
    """Build a directed minimum spanning forest from a directed gene network."""
    U = to_undirected_weighted(G_dir, weight_attr="p_value")

    if genes is not None:
        U = U.subgraph([g for g in genes if g in U]).copy()

    if U.number_of_nodes() == 0:
        return nx.DiGraph()

    undirected_mst_edges = set()

    for comp_nodes in nx.connected_components(U):
        comp = U.subgraph(comp_nodes)
        if comp.number_of_edges() == 0:
            continue
        T = minimum_spanning_tree(comp, weight="weight")
        undirected_mst_edges.update(tuple(sorted(e)) for e in T.edges())

    H = directed_edges_from_undirected_pairs(G_dir, undirected_mst_edges)
    return H


def render_gene_neighborhood(
    csv_file: str,
    gene: str,
    output_path: str,
    p_threshold: float = 0.05,
    layout: str = "sfdp",
    mst: bool = False,
    number_of_runs: Optional[int] = None,
    graph_attr: Optional[Dict] = None,
    highlight_new_genes: Optional[Tuple[bool, List[str]]] = None,
    simple_layout: bool = False
):
    """Render a GOI neighborhood graph to SVG and return the graph and neighbors."""
    if graph_attr is None:
        graph_attr = {
            "overlap": "prism",
            "overlap_scaling": "1",
            "pad": "0.2",
            "outputorder": "edgesfirst",
        }

    G, neighbors = build_induced_neighborhood(
        csv_file=csv_file,
        gene=gene,
        p_threshold=p_threshold
    )

    if mst:
        G = build_mst(G, genes=list(G.nodes()))

    if G.number_of_nodes() == 0:
        print(f"No edges found for {gene}.")
        return G, neighbors

    if number_of_runs is not None:
        consensus, coassoc, partitions, n_of_genes_in_interest_gene_community = consensus_partition(
            G,
            n_runs=number_of_runs,
            gene_of_interest=gene
        )
        partition = consensus
    else:
        partition = {node: 0 for node in G.nodes()}

    community_colors = assign_colors(partition, primary_gene=gene)

    dot = create_graphviz_dot(
        G,
        partition=partition,
        community_colors=community_colors,
        highlight_node=gene,   # <- center gene highlighted
        layout=layout,
        graph_attr=graph_attr,
        simple_layout=simple_layout,
        highlight_new_genes=highlight_new_genes
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dot.render(output_path, format="svg", cleanup=True)
    print(f"Saved graph SVG to {output_path}.svg")
    print(f"{gene}: {len(neighbors)} neighbors, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, neighbors


from typing import Dict, Set

def compare_gene_neighbors(
    csv_file_a: str,
    gene_a: str,
    csv_file_b: str,
    gene_b: str,
    p_threshold: float = 0.05
) -> Dict[str, Set[str]]:
    """Compare significant neighbors for two GOI/dataset combinations."""
    df_a = load_edge_table(csv_file_a)
    df_b = load_edge_table(csv_file_b)

    neighbors_a = get_neighbors(df_a, gene_a, p_threshold)
    neighbors_b = get_neighbors(df_b, gene_b, p_threshold)

    shared = neighbors_a & neighbors_b
    only_a = neighbors_a - neighbors_b
    only_b = neighbors_b - neighbors_a

    return {
        "neighbors_a": neighbors_a,
        "neighbors_b": neighbors_b,
        "shared": shared,
        "only_a": only_a,
        "only_b": only_b,
    }


if __name__ == "__main__":
    pvalue_threshold = 0.0015
    string_pvalue_threshold = str(pvalue_threshold).replace(".", "")

    csv_path_gene1 = "gc_r_p00015_ZEB2_503811of190674672.csv"
    csv_path_gene2 = "gc_r_p00015_MECP2_433406of202677932.csv"

    graph_attr = {
        "overlap": "prism",
        "overlap_scaling": "1",
        "pad": "0.2",
        "outputorder": "edgesfirst",
        "splines": "true",
    }

    out_base = os.path.join(
        "Final_Figures_SM",
        "Neighborhoods",
        string_pvalue_threshold
    )

    G_zeb2, neigh_zeb2 = render_gene_neighborhood(
        csv_file=csv_path_gene1,
        gene="ZEB2",
        output_path=os.path.join(out_base, "ZEB2_neighborhood"),
        p_threshold=pvalue_threshold,
        layout="sfdp",
        mst=False,
        number_of_runs=None,
        graph_attr=graph_attr,
        highlight_new_genes=(False, []),
        simple_layout=False
    )

    G_mecp2, neigh_mecp2 = render_gene_neighborhood(
        csv_file=csv_path_gene2,
        gene="MECP2",
        output_path=os.path.join(out_base, "MECP2_neighborhood"),
        p_threshold=pvalue_threshold,
        layout="sfdpy",
        mst=False,
        number_of_runs=None,
        graph_attr=graph_attr,
        highlight_new_genes=(False, []),
        simple_layout=False
    )

    comparison = compare_gene_neighbors(
        csv_file_a=csv_path_gene1,
        gene_a="ZEB2",
        csv_file_b=csv_path_gene2,
        gene_b="MECP2",
        p_threshold=pvalue_threshold
    )

    print(f"ZEB2 CSV: {csv_path_gene1}")
    print(f"MECP2 CSV: {csv_path_gene2}")
    print("ZEB2 neighbors:", len(comparison["neighbors_a"]))
    print("MECP2 neighbors:", len(comparison["neighbors_b"]))
    print("Shared neighbors across the two CSVs:", len(comparison["shared"]))
    print("Shared neighbor genes:", sorted(comparison["shared"]))
