"""Configuration objects used by the guided gene-expansion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gene_analysis.io.paths import results_path
from gene_analysis.pipeline.probe_selection import ProbeSelectionConfig


@dataclass(frozen=True)
class SeedGeneConfig:
    """Settings for running Granger causality within the curated seed set."""

    seed_gene_file: Path
    dataset: str = "kutsche"
    gene_of_interest: str = "ZEB2"
    p_threshold: float = 0.05
    quantile: float = 0.0005
    output_file: Path = results_path("pipeline", "01_seed_gc", "seed_gc_results.csv")

    def validate(self) -> None:
        """Raise ``ValueError`` when seed-stage settings are incomplete or invalid."""
        if not self.seed_gene_file:
            raise ValueError("seed_gene_file is required.")
        if not self.gene_of_interest:
            raise ValueError("gene_of_interest is required.")
        if not 0 < float(self.p_threshold) <= 1:
            raise ValueError("p_threshold must be in (0, 1].")
        if not 0 < float(self.quantile) <= 1:
            raise ValueError("quantile must be in (0, 1].")


@dataclass(frozen=True)
class ProbeConfig:
    """Settings for designing or running full-dataset probe pairs."""

    seed_gene_file: Path
    full_gene_file: Optional[Path] = None
    seed_gc_result_file: Optional[Path] = None
    expression_file: Optional[Path] = None
    dataset: str = "kutsche"
    gene_of_interest: str = "ZEB2"
    p_threshold: float = 0.05
    output_file: Path = results_path("pipeline", "03_probe", "probe_gc_results.csv")
    probe_pairs_file: Path = results_path("pipeline", "03_probe", "probe_pairs.csv")

    def validate(self) -> None:
        """Raise ``ValueError`` when probe-stage settings are incomplete or invalid."""
        if not self.seed_gene_file:
            raise ValueError("seed_gene_file is required.")
        if not 0 < float(self.p_threshold) <= 1:
            raise ValueError("p_threshold must be in (0, 1].")


@dataclass(frozen=True)
class ExpansionConfig:
    """Settings for extracting expanded candidate genes from a probe network."""

    candidate_network_csv: Path
    gene_of_interest: str = "ZEB2"
    p_threshold: float = 0.05
    seed_gene_file: Optional[Path] = None
    output_gene_list: Path = results_path("pipeline", "04_expanded_genes", "expanded_genes.txt")

    def validate(self) -> None:
        """Raise ``ValueError`` when expansion-stage settings are incomplete or invalid."""
        if not self.candidate_network_csv:
            raise ValueError("candidate_network_csv is required.")
        if not self.gene_of_interest:
            raise ValueError("gene_of_interest is required.")
        if not 0 < float(self.p_threshold) <= 1:
            raise ValueError("p_threshold must be in (0, 1].")


@dataclass(frozen=True)
class ConsensusConfig:
    """Settings for one consensus/coassociation calculation."""

    gc_result_file: Path
    gene_of_interest: str = "ZEB2"
    n_runs: int = 100
    p_threshold: float = 0.05
    stability_tolerance: float = 0.05
    output_dir: Path = results_path("pipeline", "06_consensus")

    def validate(self) -> None:
        """Raise ``ValueError`` when consensus-stage settings are incomplete or invalid."""
        if not self.gc_result_file:
            raise ValueError("gc_result_file is required.")
        if not self.gene_of_interest:
            raise ValueError("gene_of_interest is required.")
        if int(self.n_runs) < 1:
            raise ValueError("n_runs must be at least 1.")
        if not 0 < float(self.p_threshold) <= 1:
            raise ValueError("p_threshold must be in (0, 1].")


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset paths required by heavy Granger stages."""

    name: str
    expression_file: Path
    full_gene_file: Path

    def validate(self) -> None:
        """Raise ``ValueError`` when dataset metadata is incomplete."""
        if not self.name:
            raise ValueError("dataset.name is required.")
        if not self.expression_file:
            raise ValueError("dataset.expression_file is required.")
        if not self.full_gene_file:
            raise ValueError("dataset.full_gene_file is required.")


@dataclass(frozen=True)
class NetworkConfig:
    """Network threshold settings shared by seed and expanded stages."""

    p_value_threshold: float = 0.05

    def validate(self) -> None:
        """Raise ``ValueError`` when the p-value threshold is outside ``(0, 1]``."""
        if not 0 < float(self.p_value_threshold) <= 1:
            raise ValueError("network.p_value_threshold must be in (0, 1].")


@dataclass(frozen=True)
class ConsensusSettings:
    """Stability-control settings for repeated Louvain consensus runs."""

    initial_runs: int = 5000
    run_increment_fraction: float = 0.20
    stability_quantile: float = 0.90
    stability_tolerance: float = 0.05
    top_overlap_threshold_percent: float = 95.0

    def validate(self) -> None:
        """Raise ``ValueError`` when consensus stability settings are invalid."""
        if int(self.initial_runs) < 1:
            raise ValueError("consensus.initial_runs must be at least 1.")
        if float(self.run_increment_fraction) <= 0:
            raise ValueError("consensus.run_increment_fraction must be positive.")
        if not 0 < float(self.stability_quantile) <= 1:
            raise ValueError("consensus.stability_quantile must be in (0, 1].")
        if float(self.stability_tolerance) < 0:
            raise ValueError("consensus.stability_tolerance must be non-negative.")
        if not 0 <= float(self.top_overlap_threshold_percent) <= 100:
            raise ValueError("consensus.top_overlap_threshold_percent must be in [0, 100].")


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime controls for heavy Granger and consensus jobs."""

    max_workers: Optional[int] = None
    chunk_size: int = 1_000_000
    resume: bool = True

    def validate(self) -> None:
        """Raise ``ValueError`` when worker or chunk-size settings are invalid."""
        if self.max_workers is not None and int(self.max_workers) < 1:
            raise ValueError("execution.max_workers must be at least 1 when set.")
        if int(self.chunk_size) < 1:
            raise ValueError("execution.chunk_size must be at least 1.")
