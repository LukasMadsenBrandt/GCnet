"""Reusable consensus/coassociation stage for seed and expanded networks."""

from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path
from dataclasses import dataclass

from gene_analysis.analysis.granger import collect_significant_edges
from gene_analysis.analysis.network import create_network
from gene_analysis.pipeline.config import ConsensusConfig
from gene_analysis.analysis.stability import GeneStabilityConfig, compute_csv_stability_metrics
from gene_analysis.analysis.consensus_backend import (
    consensus_partition,
    save_gene_frequencies_to_csv,
    setup_logging,
)


@dataclass(frozen=True)
class StableConsensusResult:
    """Result metadata returned after a stability-controlled consensus run."""

    output_dir: Path
    final_frequency_csv: Path
    final_runs: int
    stable: bool
    metrics: dict[str, int | float | str | bool | None]
    history: list[dict[str, int | float | bool | None]]


def _partition_metrics(partitions: list[dict], gene_of_interest: str) -> dict[str, int | float]:
    """Summarize Louvain partition variability across repeated runs."""
    if not partitions:
        return {
            "partitions_total": 0,
            "average_communities_per_partition": 0.0,
            "min_communities_per_partition": 0,
            "max_communities_per_partition": 0,
            "average_goi_louvain_community_size": 0.0,
            "min_goi_louvain_community_size": 0,
            "max_goi_louvain_community_size": 0,
        }

    community_counts = [len(set(partition.values())) for partition in partitions]
    goi_sizes = []
    for partition in partitions:
        if gene_of_interest not in partition:
            continue
        goi_community = partition[gene_of_interest]
        goi_sizes.append(sum(1 for community in partition.values() if community == goi_community))

    return {
        "partitions_total": len(partitions),
        "average_communities_per_partition": sum(community_counts) / len(community_counts),
        "min_communities_per_partition": min(community_counts),
        "max_communities_per_partition": max(community_counts),
        "average_goi_louvain_community_size": sum(goi_sizes) / len(goi_sizes) if goi_sizes else 0.0,
        "min_goi_louvain_community_size": min(goi_sizes) if goi_sizes else 0,
        "max_goi_louvain_community_size": max(goi_sizes) if goi_sizes else 0,
    }


def _consensus_metrics(consensus: dict, gene_of_interest: str) -> dict[str, int | float | bool]:
    """Summarize the final agglomerative consensus clustering."""
    if not consensus:
        return {
            "consensus_communities": 0,
            "largest_consensus_community_genes": 0,
            "gene_of_interest_present": False,
            "gene_of_interest_consensus_community": -1,
            "gene_of_interest_consensus_community_genes": 0,
        }

    counts = Counter(consensus.values())
    goi_present = gene_of_interest in consensus
    goi_label = consensus[gene_of_interest] if goi_present else -1
    return {
        "consensus_communities": len(counts),
        "largest_consensus_community_genes": max(counts.values(), default=0),
        "gene_of_interest_present": goi_present,
        "gene_of_interest_consensus_community": int(goi_label),
        "gene_of_interest_consensus_community_genes": counts.get(goi_label, 0) if goi_present else 0,
    }


def run(config: ConsensusConfig) -> Path:
    """Backward-compatible lightweight entrypoint: validate and prepare output folder."""
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_stable_consensus(
    config: ConsensusConfig,
    *,
    initial_runs: int,
    run_increment_fraction: float,
    stability_quantile: float,
    top_overlap_threshold_percent: float,
    max_workers: int | None = None,
) -> StableConsensusResult:
    """
    Run Louvain consensus until CSV stability criteria are reached.

    This wraps the existing optimized consensus implementation so the pipeline
    can call it consistently for both seed and expanded GC results.
    """
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=str(output_dir / "consensus.log"))

    significant_edges = collect_significant_edges(
        None,
        p_value_threshold=config.p_threshold,
        file=True,
        filepath=config.gc_result_file,
        starting_genes=[config.gene_of_interest],
        higher_threshold_for_starting_genes=config.p_threshold,
    )
    graph = create_network(significant_edges)
    undirected_graph = graph.to_undirected(as_view=False)

    workers = max_workers or max(1, (os.cpu_count() or 1) - 1)
    current_runs = int(initial_runs)
    previous_freq_csv_path = None
    partitions = []
    coassoc_state = None
    history = []
    csv_cfg = GeneStabilityConfig(
        quantile_p=float(stability_quantile),
        top_fraction=0.05,
        top_k=None,
    )

    while True:
        consensus, coassoc, partitions, unique_goi_genes, nodes, coassoc_state = consensus_partition(
            graph,
            undirected_graph,
            n_runs=current_runs,
            gene_of_interest=config.gene_of_interest,
            plot=False,
            existing_partitions=partitions,
            coassoc_state=coassoc_state,
            n_threads=workers,
            n_louvain_workers=workers,
        )
        save_gene_frequencies_to_csv(
            nodes,
            coassoc,
            config.gene_of_interest,
            current_runs,
            save_dir=str(output_dir),
        )
        current_freq_csv_path = output_dir / f"{config.gene_of_interest}_coassoc_{current_runs}_runs.csv"

        q_rel = None
        overlap_pct = None
        top_k = None
        if previous_freq_csv_path is not None:
            q_rel, overlap_pct, top_k = compute_csv_stability_metrics(
                str(previous_freq_csv_path),
                str(current_freq_csv_path),
                cfg=csv_cfg,
            )
        iteration_metrics = {
            "runs": current_runs,
            "frequency_csv": str(current_freq_csv_path),
            "stability_quantile_relative_change": q_rel,
            "top_gene_overlap_percent": overlap_pct,
            "top_gene_overlap_k": top_k,
            "stable": bool(
                previous_freq_csv_path is not None
                and q_rel <= config.stability_tolerance
                and overlap_pct >= top_overlap_threshold_percent
            ),
            "unique_genes_ever_with_goi_louvain": unique_goi_genes,
            **_partition_metrics(partitions, config.gene_of_interest),
            **_consensus_metrics(consensus, config.gene_of_interest),
        }
        history.append(iteration_metrics)

        if iteration_metrics["stable"]:
            metrics = {
                "final_runs": current_runs,
                "stable": True,
                "stability_tolerance": config.stability_tolerance,
                "stability_quantile": stability_quantile,
                "top_overlap_threshold_percent": top_overlap_threshold_percent,
                **{key: value for key, value in iteration_metrics.items() if key not in {"frequency_csv", "stable"}},
            }
            return StableConsensusResult(output_dir, current_freq_csv_path, current_runs, True, metrics, history)

        previous_freq_csv_path = current_freq_csv_path
        current_runs += max(1, int(math.floor(current_runs * run_increment_fraction)))
