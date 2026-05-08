"""Write inspectable network artifacts from significant Granger edges."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx

from gene_analysis.analysis.granger import collect_significant_edges
from gene_analysis.analysis.network import create_network
from gene_analysis.io.paths import resolve_existing_path


def build_significant_network(
    gc_csv: str | Path,
    *,
    p_threshold: float,
    gene_of_interest: str,
) -> nx.DiGraph:
    """Build a directed significant-edge network from a Granger CSV."""
    edges = collect_significant_edges(
        None,
        p_value_threshold=p_threshold,
        file=True,
        filepath=resolve_existing_path(gc_csv),
        starting_genes=[gene_of_interest],
        higher_threshold_for_starting_genes=p_threshold,
    )
    return create_network(edges)


def summarize_network(graph: nx.DiGraph, *, gene_of_interest: str) -> dict[str, int | float | str | bool]:
    """Compute compact network metrics for pipeline audit manifests."""
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    weak_components = list(nx.weakly_connected_components(graph)) if node_count else []
    strong_components = list(nx.strongly_connected_components(graph)) if node_count else []
    goi_present = gene_of_interest in graph
    density = nx.density(graph) if node_count > 1 else 0.0

    return {
        "genes_total": node_count,
        "edges_total": edge_count,
        "density": density,
        "weak_components": len(weak_components),
        "strong_components": len(strong_components),
        "largest_weak_component_genes": max((len(component) for component in weak_components), default=0),
        "largest_strong_component_genes": max((len(component) for component in strong_components), default=0),
        "average_in_degree": edge_count / node_count if node_count else 0.0,
        "average_out_degree": edge_count / node_count if node_count else 0.0,
        "gene_of_interest": gene_of_interest,
        "gene_of_interest_present": goi_present,
        "gene_of_interest_in_degree": graph.in_degree(gene_of_interest) if goi_present else 0,
        "gene_of_interest_out_degree": graph.out_degree(gene_of_interest) if goi_present else 0,
        "gene_of_interest_total_degree": graph.degree(gene_of_interest) if goi_present else 0,
    }


def write_network_artifacts(
    gc_csv: str | Path,
    *,
    p_threshold: float,
    gene_of_interest: str,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Path]:
    """Write edges, nodes, GraphML, and summary files for a significant network."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = build_significant_network(
        gc_csv,
        p_threshold=p_threshold,
        gene_of_interest=gene_of_interest,
    )

    edge_csv = out_dir / f"{prefix}_network_edges.csv"
    node_txt = out_dir / f"{prefix}_network_nodes.txt"
    graphml = out_dir / f"{prefix}_network.graphml"
    summary_json = out_dir / f"{prefix}_network_summary.json"

    with open(edge_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gene1", "gene2", "lag", "p-value"])
        for source, target, data in graph.edges(data=True):
            writer.writerow([source, target, data.get("lag", 1), data.get("p_value", "")])

    with open(node_txt, "w", encoding="utf-8") as fh:
        for node in sorted(map(str, graph.nodes())):
            fh.write(f"{node}\n")

    nx.write_graphml(graph, graphml)

    metrics = summarize_network(graph, gene_of_interest=gene_of_interest)
    summary = {
        "source_gc_csv": str(gc_csv),
        "gene_of_interest": gene_of_interest,
        "p_value_threshold": p_threshold,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "metrics": metrics,
        "graphml": str(graphml),
        "edge_csv": str(edge_csv),
        "node_txt": str(node_txt),
    }
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return {
        "edge_csv": edge_csv,
        "node_txt": node_txt,
        "graphml": graphml,
        "summary_json": summary_json,
    }
