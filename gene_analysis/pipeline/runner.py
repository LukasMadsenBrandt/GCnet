"""YAML-driven orchestration for the guided neurodeficiency expansion pipeline."""

from __future__ import annotations

import json
import shutil
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from gene_analysis.analysis.backends import backend_metadata, require_available_backend
from gene_analysis.analysis.preprocessing import apply_expression_preprocessing
from gene_analysis.io.paths import resolve_existing_path, results_path
from gene_analysis.pipeline.config import (
    ConsensusConfig,
    ConsensusSettings,
    DatasetConfig,
    ExecutionConfig,
    ExpansionConfig,
    NetworkConfig,
    PreprocessingConfig,
    ProbeSelectionConfig,
    SeedGeneConfig,
)
from gene_analysis.pipeline.consensus import run_stable_consensus
from gene_analysis.pipeline.network_expansion import run as run_expansion
from gene_analysis.pipeline.network_artifacts import write_network_artifacts, write_top_consensus_network_artifacts
from gene_analysis.pipeline.probe_selection import select_probe_genes, write_probe_genes
from gene_analysis.pipeline.seed_genes import load_seed_genes
from gene_analysis.pipeline.validation import (
    validate_expression_dataframe,
    validate_frequency_csv,
    validate_gc_csv,
    validate_gene_list_file,
    validate_network_artifact_bundle,
)
from gene_analysis.datasets.kutsche import preprocess_pipeline
from gene_analysis.datasets.kutsche import load_and_preprocess_data
from gene_analysis.analysis.granger_runner import perform_gc
from gene_analysis.analysis.granger_cuda import perform_gc_cuda


STAGES = (
    "01_seed_gc",
    "02_seed_consensus",
    "03_probe_selection",
    "04_dataset_probe",
    "05_expanded_genes",
    "06_expanded_gc",
    "07_expanded_consensus",
)

GENERIC_DATASET_NAMES = {"generic_expression", "preprocessed", "sample_real_gc", "real_gc_sample"}
MIN_RECOMMENDED_TIMEPOINTS = 5
NETWORK_SVG_ARTIFACT_KEYS = {
    "seed_network_svg",
    "seed_top_consensus_network_svg",
    "probe_network_svg",
    "expanded_network_svg",
    "expanded_top_consensus_network_svg",
    "figures_dir",
    "seed_network_figure_svg",
    "seed_top_consensus_network_figure_svg",
    "probe_network_figure_svg",
    "expanded_network_figure_svg",
    "expanded_top_consensus_network_figure_svg",
}


def _count_text_lines(path: str | Path) -> int:
    """Count non-empty lines in a small manifest/input text file."""
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _count_csv_rows(path: str | Path) -> int:
    """Count data rows in a CSV without loading the full file into memory."""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return sum(1 for row in reader if row)


def _read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _network_metrics(summary_json: str | Path) -> dict[str, Any]:
    """Read network metrics written by ``write_network_artifacts``."""
    summary = _read_json(summary_json)
    return dict(summary.get("metrics") or {})


def _top_consensus_network_metrics(summary_json: str | Path) -> dict[str, Any]:
    """Return top-consensus network metrics with stable manifest prefixes."""
    metrics = _network_metrics(summary_json)
    selected = {
        "genes_total": "top_consensus_genes_total",
        "edges_total": "top_consensus_edges_total",
        "density": "top_consensus_density",
        "subcommunity_count": "top_consensus_subcommunities",
        "largest_subcommunity_genes": "top_consensus_largest_subcommunity_genes",
        "gene_of_interest_subcommunity": "top_consensus_gene_of_interest_subcommunity",
        "gene_of_interest_subcommunity_genes": "top_consensus_gene_of_interest_subcommunity_genes",
        "top_consensus_fraction": "top_consensus_fraction",
        "top_consensus_gene_count": "top_consensus_selected_gene_count",
        "top_consensus_network_gene_count": "top_consensus_network_gene_count",
    }
    return {target: metrics[source] for source, target in selected.items() if source in metrics}


def _frequency_csv_metrics(path: str | Path, *, gene_of_interest: str) -> dict[str, Any]:
    """Summarize a GOI coassociation frequency CSV."""
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    frequencies = []
    goi_frequency = None
    for row in rows:
        gene = (row.get("Gene") or row.get("gene") or "").strip()
        value = row.get("Coassociation Frequency") or row.get("coassociation frequency") or ""
        try:
            frequency = float(value)
        except ValueError:
            continue
        frequencies.append(frequency)
        if gene.upper() == gene_of_interest.upper():
            goi_frequency = frequency

    frequencies.sort(reverse=True)
    top_n = max(1, round(len(frequencies) * 0.05)) if frequencies else 0
    return {
        "gene_count": len(rows),
        "max_coassociation_frequency": max(frequencies, default=0.0),
        "mean_coassociation_frequency": sum(frequencies) / len(frequencies) if frequencies else 0.0,
        "top_5_percent_gene_count": top_n,
        "top_5_percent_mean_frequency": sum(frequencies[:top_n]) / top_n if top_n else 0.0,
        "gene_of_interest_frequency": goi_frequency,
    }


def _relative_to_run(path: str | Path, run_dir: str | Path) -> str:
    """Return a compact run-relative path when possible."""
    path = Path(path)
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _top_priority_genes(path: str | Path | None, *, limit: int) -> list[dict[str, Any]]:
    """Read the first rows from a priority CSV for the run summary."""
    if path is None or not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gene = (row.get("Gene") or "").strip()
            if not gene:
                continue
            try:
                frequency = float(row.get("Coassociation Frequency") or 0.0)
            except ValueError:
                frequency = 0.0
            rows.append({"Gene": gene, "Coassociation Frequency": frequency})
            if len(rows) >= limit:
                break
    return rows


def _gc_total_pairs(result: dict[str, Any], output: str | Path) -> int:
    """Return the intended GC pair count from either real or fixture GC results."""
    return int(result.get("total_pairs_all") or result.get("total_pairs") or _count_csv_rows(output))


def _run_configured_gc(
    expression_df,
    *,
    genes_file: str | Path,
    output_file: str | Path,
    p_threshold: float,
    chunk_size: int,
    list_to_kutsche: bool,
    max_workers: int | None,
    progress: bool,
    resume: bool,
    rename_at_end: bool,
    backend: str,
    gpu_device: int | None,
) -> dict[str, Any]:
    """Dispatch a GC job to the configured compute backend."""
    if backend == "gpu_cuda":
        return perform_gc_cuda(
            expression_df,
            genes_file=str(genes_file),
            output_file=str(output_file),
            p_threshold=p_threshold,
            chunk_size=chunk_size,
            list_to_kutsche=list_to_kutsche,
            max_workers=max_workers,
            progress=progress,
            resume=resume,
            rename_at_end=rename_at_end,
            gpu_device=gpu_device,
        )
    return perform_gc(
        expression_df,
        genes_file=str(genes_file),
        output_file=str(output_file),
        p_threshold=p_threshold,
        chunk_size=chunk_size,
        list_to_kutsche=list_to_kutsche,
        max_workers=max_workers,
        progress=progress,
        resume=resume,
        rename_at_end=rename_at_end,
    )


@dataclass(frozen=True)
class PipelineConfig:
    """Complete run configuration loaded from the canonical pipeline YAML file."""

    run_name: str
    dataset: DatasetConfig
    gene_of_interest: str
    seed_gene_file: Path
    network: NetworkConfig = field(default_factory=NetworkConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    consensus: ConsensusSettings = field(default_factory=ConsensusSettings)
    probe_selection: ProbeSelectionConfig = field(default_factory=ProbeSelectionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    artifacts: Dict[str, Path] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        """Return the run-specific output directory under ``results/pipeline``."""
        return results_path("pipeline", self.run_name)

    def validate(self) -> None:
        """Validate nested settings before the runner starts creating artifacts."""
        if not self.run_name:
            raise ValueError("run_name is required.")
        if not self.gene_of_interest:
            raise ValueError("gene_of_interest is required.")
        if not self.seed_gene_file:
            raise ValueError("seed_gene_file is required.")
        self.dataset.validate()
        self.preprocessing.validate()
        self.network.validate()
        self.consensus.validate()
        self.probe_selection.validate()
        self.execution.validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load, parse, and validate a pipeline config from YAML."""
        config_path = resolve_existing_path(path)
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(raw)
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        """Build a pipeline config from a raw dictionary."""
        dataset_raw = raw.get("dataset") or {}
        dataset_name = str(dataset_raw.get("name") or "")
        preprocessing_raw = raw.get("preprocessing")
        network_raw = raw.get("network") or {}
        consensus_raw = raw.get("consensus") or {}
        selection_raw = raw.get("probe_selection") or {}
        execution_raw = raw.get("execution") or {}
        artifacts_raw = raw.get("artifacts") or {}

        return cls(
            run_name=str(raw.get("run_name") or ""),
            dataset=DatasetConfig(
                name=dataset_name,
                expression_file=Path(dataset_raw.get("expression_file") or ""),
                full_gene_file=Path(dataset_raw.get("full_gene_file") or ""),
            ),
            gene_of_interest=str(raw.get("gene_of_interest") or ""),
            seed_gene_file=Path(raw.get("seed_gene_file") or ""),
            preprocessing=_preprocessing_from_raw(preprocessing_raw, dataset_name=dataset_name),
            network=NetworkConfig(
                p_value_threshold=float(network_raw.get("p_value_threshold", 0.05)),
                write_svg=_bool_from_raw(network_raw.get("write_svg", network_raw.get("write_artifacts", True))),
                svg_renderer=str(network_raw.get("svg_renderer", "networkx")),
                svg_layout=str(network_raw.get("svg_layout", "dot")),
            ),
            consensus=ConsensusSettings(
                initial_runs=int(consensus_raw.get("initial_runs", 5000)),
                run_increment_fraction=float(consensus_raw.get("run_increment_fraction", 0.20)),
                stability_quantile=float(consensus_raw.get("stability_quantile", 0.90)),
                stability_tolerance=float(consensus_raw.get("stability_tolerance", 0.05)),
                top_overlap_threshold_percent=float(consensus_raw.get("top_overlap_threshold_percent", 95.0)),
            ),
            probe_selection=ProbeSelectionConfig(
                mode=str(selection_raw.get("mode", "top_percent")),
                top_percent=selection_raw.get("top_percent", 5.0),
                min_frequency=selection_raw.get("min_frequency"),
            ),
            execution=ExecutionConfig(
                max_workers=execution_raw.get("max_workers"),
                chunk_size=int(execution_raw.get("chunk_size", 1_000_000)),
                resume=bool(execution_raw.get("resume", True)),
                gc_backend=str(execution_raw.get("gc_backend", "cpu_statsmodels")),
                consensus_backend=str(execution_raw.get("consensus_backend", "cpu_louvain")),
                gpu_device=execution_raw.get("gpu_device"),
            ),
            artifacts={key: Path(value) for key, value in artifacts_raw.items() if value is not None},
        )


def normalize_stage(stage: str) -> str:
    """Resolve numeric or prefix stage input to a canonical stage name."""
    if stage in STAGES:
        return stage
    if stage.isdigit():
        matches = [candidate for candidate in STAGES if candidate.startswith(stage.zfill(2))]
        if len(matches) == 1:
            return matches[0]
    matches = [candidate for candidate in STAGES if candidate.startswith(stage)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unknown pipeline stage: {stage}. Expected one of: {', '.join(STAGES)}")


def _bool_from_raw(value: Any) -> bool:
    """Parse YAML booleans and common string booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
    raise ValueError("network.write_svg must be true or false.")


def _preprocessing_from_raw(raw: Optional[dict[str, Any]], *, dataset_name: str) -> PreprocessingConfig:
    """Build preprocessing settings, preserving historical dataset defaults."""
    if raw is None:
        return PreprocessingConfig()
    return PreprocessingConfig(
        normalize=str(raw.get("normalize", "none")),
        transform=str(raw.get("transform", "none")),
        aggregation=str(raw.get("aggregation", "robust")),
    )


def _pre_aggregation_normalize(normalize: str) -> str | None:
    """Return normalization that should happen before replicate aggregation."""
    normalize = str(normalize).lower()
    if normalize in {"deseq", "deseq2", "size_factors"}:
        return "deseq"
    return None


def _pre_aggregation_transform(transform: str) -> str | bool:
    """Return transform that should happen before replicate aggregation."""
    transform = str(transform).lower()
    if transform in {"log1p", "log+1"}:
        return "log+1"
    if transform == "sqrt":
        return "sqrt"
    return False


def _apply_remaining_preprocessing(
    df,
    preprocessing: PreprocessingConfig,
):
    """Apply preprocessing options not already handled before aggregation."""
    normalize = str(preprocessing.normalize).lower()
    transform = str(preprocessing.transform).lower()
    remaining_normalize = "none" if normalize in {"none", "deseq", "deseq2", "size_factors"} else normalize
    already_transformed = transform in {"log1p", "log+1", "sqrt"}
    remaining_transform = "none" if already_transformed else transform
    return apply_expression_preprocessing(df, normalize=remaining_normalize, transform=remaining_transform)


class PipelineRunner:
    """Run or resume the seven-stage guided gene-expansion workflow."""

    def __init__(self, config: PipelineConfig):
        config.validate()
        self.config = config
        self.run_dir = config.run_dir
        self.artifacts: dict[str, Path] = {}

    def stage_dir(self, stage: str) -> Path:
        """Create and return the output folder for one stage."""
        path = self.run_dir / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def default_artifacts(self) -> dict[str, Path]:
        """Return predictable artifact paths for every pipeline stage."""
        return {
            "run_summary_md": self.run_dir / "RUN_SUMMARY.md",
            "figures_dir": self.run_dir / "figures",
            "seed_network_figure_svg": self.run_dir / "figures" / "seed_network.svg",
            "seed_top_consensus_network_figure_svg": self.run_dir / "figures" / "seed_top_consensus_network.svg",
            "probe_network_figure_svg": self.run_dir / "figures" / "probe_network.svg",
            "expanded_network_figure_svg": self.run_dir / "figures" / "expanded_network.svg",
            "expanded_top_consensus_network_figure_svg": self.run_dir
            / "figures"
            / "expanded_top_consensus_network.svg",
            "seed_gc_csv": self.run_dir / "01_seed_gc" / "seed_gc.csv",
            "seed_frequency_csv": self.run_dir / "02_seed_consensus" / "priority_genes.csv",
            "seed_consensus_history_json": self.run_dir / "02_seed_consensus" / "consensus_history.json",
            "seed_network_graphml": self.run_dir / "02_seed_consensus" / "seed_network.graphml",
            "seed_network_svg": self.run_dir / "02_seed_consensus" / "seed_network.svg",
            "seed_network_edges_csv": self.run_dir / "02_seed_consensus" / "seed_network_edges.csv",
            "seed_network_nodes_file": self.run_dir / "02_seed_consensus" / "seed_network_nodes.txt",
            "seed_network_summary_json": self.run_dir / "02_seed_consensus" / "seed_network_summary.json",
            "seed_top_consensus_network_graphml": self.run_dir
            / "02_seed_consensus"
            / "seed_top_consensus_network.graphml",
            "seed_top_consensus_network_svg": self.run_dir
            / "02_seed_consensus"
            / "seed_top_consensus_network.svg",
            "seed_top_consensus_network_edges_csv": self.run_dir
            / "02_seed_consensus"
            / "seed_top_consensus_network_edges.csv",
            "seed_top_consensus_network_nodes_file": self.run_dir
            / "02_seed_consensus"
            / "seed_top_consensus_network_nodes.txt",
            "seed_top_consensus_network_summary_json": self.run_dir
            / "02_seed_consensus"
            / "seed_top_consensus_network_summary.json",
            "probe_genes_file": self.run_dir / "03_probe_selection" / "probe_genes.txt",
            "probe_gc_csv": self.run_dir / "04_dataset_probe" / "probe_gc.csv",
            "expanded_genes_file": self.run_dir / "05_expanded_genes" / "expanded_genes.txt",
            "probe_network_graphml": self.run_dir / "05_expanded_genes" / "probe_network.graphml",
            "probe_network_svg": self.run_dir / "05_expanded_genes" / "probe_network.svg",
            "probe_network_edges_csv": self.run_dir / "05_expanded_genes" / "probe_network_edges.csv",
            "probe_network_nodes_file": self.run_dir / "05_expanded_genes" / "probe_network_nodes.txt",
            "probe_network_summary_json": self.run_dir / "05_expanded_genes" / "probe_network_summary.json",
            "expanded_gc_csv": self.run_dir / "06_expanded_gc" / "expanded_gc.csv",
            "priority_genes_csv": self.run_dir / "07_expanded_consensus" / "priority_genes.csv",
            "expanded_consensus_history_json": self.run_dir / "07_expanded_consensus" / "consensus_history.json",
            "expanded_network_graphml": self.run_dir / "07_expanded_consensus" / "expanded_network.graphml",
            "expanded_network_svg": self.run_dir / "07_expanded_consensus" / "expanded_network.svg",
            "expanded_network_edges_csv": self.run_dir / "07_expanded_consensus" / "expanded_network_edges.csv",
            "expanded_network_nodes_file": self.run_dir / "07_expanded_consensus" / "expanded_network_nodes.txt",
            "expanded_network_summary_json": self.run_dir / "07_expanded_consensus" / "expanded_network_summary.json",
            "expanded_top_consensus_network_graphml": self.run_dir
            / "07_expanded_consensus"
            / "expanded_top_consensus_network.graphml",
            "expanded_top_consensus_network_svg": self.run_dir
            / "07_expanded_consensus"
            / "expanded_top_consensus_network.svg",
            "expanded_top_consensus_network_edges_csv": self.run_dir
            / "07_expanded_consensus"
            / "expanded_top_consensus_network_edges.csv",
            "expanded_top_consensus_network_nodes_file": self.run_dir
            / "07_expanded_consensus"
            / "expanded_top_consensus_network_nodes.txt",
            "expanded_top_consensus_network_summary_json": self.run_dir
            / "07_expanded_consensus"
            / "expanded_top_consensus_network_summary.json",
        }

    def resolve_required_artifact(self, key: str) -> Path:
        """Resolve a required upstream artifact or fail with a clear resume error."""
        if key in self.artifacts and Path(self.artifacts[key]).exists():
            return Path(self.artifacts[key])
        defaults = self.default_artifacts()
        if key in defaults and defaults[key].exists():
            return defaults[key]
        if key in self.config.artifacts:
            configured = resolve_existing_path(self.config.artifacts[key])
            if configured.exists():
                return configured
            raise FileNotFoundError(f"Configured artifact '{key}' does not exist: {configured}")
        raise FileNotFoundError(
            f"Missing required artifact '{key}'. Provide it in config.artifacts or run earlier stages first."
        )

    def write_manifest(
        self,
        stage: str,
        status: str,
        *,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        parameters: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
        started: str,
    ) -> None:
        """Write the per-stage manifest used for auditability and resuming."""
        manifest = {
            "stage": stage,
            "status": status,
            "started_at": started,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "inputs": {key: str(value) for key, value in inputs.items()},
            "outputs": {key: str(value) for key, value in outputs.items()},
            "parameters": parameters or {},
            "metrics": metrics or {},
        }
        with open(self.stage_dir(stage) / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    def read_stage_manifest(self, stage: str) -> dict[str, Any]:
        """Read one stage manifest if it exists."""
        path = self.run_dir / stage / "manifest.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def stage_metrics_summary(self, stages: Iterable[str]) -> dict[str, Any]:
        """Return metrics from completed stage manifests for run-level reporting."""
        return {
            stage: self.read_stage_manifest(stage).get("metrics", {})
            for stage in stages
            if self.read_stage_manifest(stage)
        }

    def reported_artifacts(self) -> dict[str, Path]:
        """Return artifact paths that are expected for the current config."""
        artifacts = {**self.default_artifacts(), **self.artifacts}
        if self.config.network.write_svg:
            return artifacts
        return {key: value for key, value in artifacts.items() if key not in NETWORK_SVG_ARTIFACT_KEYS}

    def write_network_artifact_bundle(self, gc_csv: str | Path, *, stage: str, prefix: str) -> dict[str, Path]:
        """Write machine-readable network files and optional SVG preview."""
        network_artifacts = write_network_artifacts(
            gc_csv,
            p_threshold=self.config.network.p_value_threshold,
            gene_of_interest=self.config.gene_of_interest,
            output_dir=self.stage_dir(stage),
            prefix=prefix,
            write_svg_preview=self.config.network.write_svg,
            svg_renderer=self.config.network.svg_renderer,
            svg_layout=self.config.network.svg_layout,
        )
        validate_network_artifact_bundle(network_artifacts, gene_of_interest=self.config.gene_of_interest)
        return network_artifacts

    def write_top_consensus_network_artifact_bundle(
        self,
        gc_csv: str | Path,
        frequency_csv: str | Path,
        *,
        stage: str,
        prefix: str,
    ) -> dict[str, Path]:
        """Write top-5%-consensus subnetwork files with visual subcommunities."""
        network_artifacts = write_top_consensus_network_artifacts(
            gc_csv,
            frequency_csv,
            p_threshold=self.config.network.p_value_threshold,
            gene_of_interest=self.config.gene_of_interest,
            output_dir=self.stage_dir(stage),
            prefix=prefix,
            top_fraction=0.05,
            write_svg_preview=self.config.network.write_svg,
            svg_renderer=self.config.network.svg_renderer,
            svg_layout=self.config.network.svg_layout,
        )
        validate_network_artifact_bundle(network_artifacts, gene_of_interest=self.config.gene_of_interest)
        return network_artifacts

    def pipeline_evolution(self, stages: Iterable[str]) -> list[dict[str, Any]]:
        """Build a compact table showing how the pipeline changes by stage."""
        evolution = []
        for stage in stages:
            manifest = self.read_stage_manifest(stage)
            if not manifest:
                continue
            metrics = manifest.get("metrics", {})
            row = {"stage": stage}
            for key in (
                "genes_total",
                "edges_total",
                "gene_count",
                "gc_pairs_total",
                "consensus_communities",
                "gene_of_interest_consensus_community_genes",
                "final_runs",
                "stability_quantile_relative_change",
                "top_gene_overlap_percent",
            ):
                if key in metrics:
                    row[key] = metrics[key]
            evolution.append(row)
        return evolution

    def write_pipeline_figure_bundle(self) -> None:
        """Collect stage SVG previews in one run-level figures folder."""
        if not self.config.network.write_svg:
            return
        figure_dir = self.run_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure_sources = {
            "seed_network_figure_svg": self.default_artifacts()["seed_network_svg"],
            "seed_top_consensus_network_figure_svg": self.default_artifacts()["seed_top_consensus_network_svg"],
            "probe_network_figure_svg": self.default_artifacts()["probe_network_svg"],
            "expanded_network_figure_svg": self.default_artifacts()["expanded_network_svg"],
            "expanded_top_consensus_network_figure_svg": self.default_artifacts()[
                "expanded_top_consensus_network_svg"
            ],
        }
        for key, source in figure_sources.items():
            if source.exists():
                target = self.default_artifacts()[key]
                shutil.copyfile(source, target)
                self.artifacts[key] = target
        if any(key in self.artifacts for key in figure_sources):
            self.artifacts["figures_dir"] = figure_dir

    def write_run_summary(self, run_manifest: dict[str, Any]) -> Path:
        """Write a researcher-facing Markdown summary for the completed run."""
        summary_path = self.default_artifacts()["run_summary_md"]
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts = {key: Path(value) for key, value in self.reported_artifacts().items()}
        lines = [
            f"# Pipeline Run Summary: {self.config.run_name}",
            "",
            "## Configuration",
            "",
            f"- Dataset: `{self.config.dataset.name}`",
            f"- Gene of interest: `{self.config.gene_of_interest}`",
            f"- Seed genes: `{self.config.seed_gene_file}`",
            f"- Expression matrix: `{self.config.dataset.expression_file}`",
            f"- Full gene list: `{self.config.dataset.full_gene_file}`",
            f"- P-value threshold: `{self.config.network.p_value_threshold}`",
            f"- Probe selection: `{self.config.probe_selection.mode}`",
            f"- GC backend: `{self.config.execution.gc_backend}`",
            f"- Consensus backend: `{self.config.execution.consensus_backend}`",
            f"- Max workers: `{self.config.execution.max_workers}`",
            f"- Network SVG previews: `{self.config.network.write_svg}`",
            f"- Network SVG renderer: `{self.config.network.svg_renderer}`",
            f"- Network SVG layout: `{self.config.network.svg_layout}`",
            "",
            "## Pipeline Evolution",
            "",
            "| Stage | Genes | Edges | GC pairs | Communities | GOI community genes | Runs |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in run_manifest.get("pipeline_evolution", []):
            lines.append(
                "| {stage} | {genes} | {edges} | {pairs} | {communities} | {goi_comm} | {runs} |".format(
                    stage=row.get("stage", ""),
                    genes=row.get("genes_total", row.get("gene_count", "")),
                    edges=row.get("edges_total", ""),
                    pairs=row.get("gc_pairs_total", ""),
                    communities=row.get("consensus_communities", ""),
                    goi_comm=row.get("gene_of_interest_consensus_community_genes", ""),
                    runs=row.get("final_runs", ""),
                )
            )
        lines.extend(["", "## Key Outputs", ""])
        output_keys = (
            "seed_network_graphml",
            "seed_network_figure_svg",
            "seed_top_consensus_network_graphml",
            "seed_top_consensus_network_figure_svg",
            "probe_genes_file",
            "probe_network_graphml",
            "probe_network_figure_svg",
            "expanded_genes_file",
            "expanded_network_graphml",
            "expanded_network_figure_svg",
            "expanded_top_consensus_network_graphml",
            "expanded_top_consensus_network_figure_svg",
            "priority_genes_csv",
            "run_summary_md",
        )
        for key in output_keys:
            value = artifacts.get(key)
            if value and value.exists():
                lines.append(f"- `{key}`: `{_relative_to_run(value, self.run_dir)}`")
        top_genes = _top_priority_genes(artifacts.get("priority_genes_csv"), limit=15)
        if top_genes:
            lines.extend(["", "## Top Priority Genes", "", "| Rank | Gene | Coassociation Frequency |", "| ---: | --- | ---: |"])
            for rank, row in enumerate(top_genes, start=1):
                lines.append(f"| {rank} | `{row['Gene']}` | {row['Coassociation Frequency']:.4f} |")
        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- `gpu_cuda` accelerates Granger causality calculations and should be benchmarked against `cpu_statsmodels` when used on new hardware.",
                "- `cpu_louvain` is the validated consensus backend. `gpu_cugraph` remains experimental until parity is demonstrated for the target dataset.",
                "- Stage-level `manifest.json` files contain the full inputs, outputs, parameters, and metrics for auditing or resuming.",
                "",
            ]
        )
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        self.artifacts["run_summary_md"] = summary_path
        return summary_path

    def load_expression_dataframe(self):
        """Load and preprocess the configured expression matrix for heavy GC stages."""
        dataset_name = self.config.dataset.name.lower()
        if dataset_name in GENERIC_DATASET_NAMES:
            import pandas as pd

            expression_file = resolve_existing_path(self.config.dataset.expression_file)
            df = pd.read_csv(expression_file, sep=None, engine="python")
            if "Gene" not in df.columns:
                raise ValueError("Generic expression file must contain a 'Gene' column.")
            df = df.set_index("Gene")
            if len(df.columns) < MIN_RECOMMENDED_TIMEPOINTS:
                raise ValueError(
                    "Generic expression file must contain at least "
                    f"{MIN_RECOMMENDED_TIMEPOINTS} ordered timepoint/condition columns for GC robustness."
                )
            df = apply_expression_preprocessing(
                df,
                normalize=self.config.preprocessing.normalize,
                transform=self.config.preprocessing.transform,
            )
            return self.restrict_expression_dataframe(df)
        if dataset_name in {"benito_human", "benito_gorilla"}:
            from gene_analysis.datasets.benito import preprocess_pipeline_benito

            df_mean_per_day, _, _ = preprocess_pipeline_benito(
                datafile=resolve_existing_path(self.config.dataset.expression_file),
                mappingfile=resolve_existing_path("Data/Benito/gene_id_to_gene_name.txt"),
                map_speciment_to_gene_file=resolve_existing_path("Data/Benito/map_speciment_to_gene.csv"),
                normalize=_pre_aggregation_normalize(self.config.preprocessing.normalize),
                transformed=_pre_aggregation_transform(self.config.preprocessing.transform),
                aggregation=self.config.preprocessing.aggregation,
            )
            df = _apply_remaining_preprocessing(df_mean_per_day, self.config.preprocessing)
            return self.restrict_expression_dataframe(df)
        if dataset_name != "kutsche":
            raise NotImplementedError(
                "The canonical runner currently supports dataset.name='kutsche', "
                "'benito_human', 'benito_gorilla', and 'generic_expression' for heavy GC stages."
            )
        df = load_and_preprocess_data(resolve_existing_path(self.config.dataset.expression_file))
        df_filtered, _, _ = preprocess_pipeline(
            df,
            normalize=_pre_aggregation_normalize(self.config.preprocessing.normalize),
            transformed=_pre_aggregation_transform(self.config.preprocessing.transform),
            aggregation=self.config.preprocessing.aggregation,
        )
        df = _apply_remaining_preprocessing(df_filtered, self.config.preprocessing)
        return self.restrict_expression_dataframe(df)

    def restrict_expression_dataframe(self, df):
        """Restrict expression data to configured full-dataset and seed genes when present."""
        full_genes = load_seed_genes(self.config.dataset.full_gene_file)
        seed_genes = load_seed_genes(self.config.seed_gene_file)
        requested = list(dict.fromkeys([*full_genes, *seed_genes]))
        available = [gene for gene in requested if gene in df.index]
        if not available:
            raise ValueError(
                "None of the genes from dataset.full_gene_file or seed_gene_file "
                "were found in the loaded expression matrix."
            )
        return df.loc[available]

    def run_fixture_gc(self, genes_file: str | Path, output_file: str | Path, *, list_to_full_dataset: bool) -> dict[str, str]:
        """
        Write deterministic significant GC rows for the sample pipeline.

        This mode exercises orchestration, manifests, selection, network
        expansion, and consensus without launching the full statistical GC job.
        It is intended only for examples and fast integration tests.

        For the probe stage, the fixture intentionally marks only a bounded
        discoverable subnetwork as significant. That keeps the sample aligned
        with the real guided-expansion idea: probe a large dataset, then run
        all-pairs GC only on the smaller network-supported candidate set.
        """
        output = Path(output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        genes = load_seed_genes(genes_file)
        if list_to_full_dataset:
            dataset_genes = load_seed_genes(self.config.dataset.full_gene_file)
            discoverable_genes = dataset_genes[: min(len(dataset_genes), 150)]
            pairs = [(gene, target) for gene in genes for target in discoverable_genes if target != gene]
            pairs += [(source, gene) for source in discoverable_genes for gene in genes if source != gene]
        else:
            pairs = [(source, target) for source in genes for target in genes if source != target]
        pairs = list(dict.fromkeys(pairs))
        with open(output, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["gene1", "gene2", "lag", "p-value"])
            for source, target in pairs:
                writer.writerow([source, target, 1, "0.001"])
        return {"output_file": str(output), "total_pairs": str(len(pairs))}

    def run(self, *, start_at: str = "01_seed_gc", stop_after: Optional[str] = None) -> dict[str, Path]:
        """Run a contiguous section of the pipeline and return known artifacts."""
        start = normalize_stage(start_at)
        stop = normalize_stage(stop_after) if stop_after else STAGES[-1]
        start_idx = STAGES.index(start)
        stop_idx = STAGES.index(stop)
        if start_idx > stop_idx:
            raise ValueError("start_at must be before or equal to stop_after.")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "pipeline_config.yml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(_config_to_dict(self.config), fh, sort_keys=False)

        for stage in STAGES[start_idx : stop_idx + 1]:
            getattr(self, f"run_{stage}")()

        self.write_pipeline_figure_bundle()
        run_manifest = {
            "run_name": self.config.run_name,
            "stages": list(STAGES[start_idx : stop_idx + 1]),
            "settings": _config_to_dict(self.config),
            "artifacts": {key: str(value) for key, value in self.reported_artifacts().items()},
            "stage_metrics": self.stage_metrics_summary(STAGES[start_idx : stop_idx + 1]),
            "pipeline_evolution": self.pipeline_evolution(STAGES[start_idx : stop_idx + 1]),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        summary_path = self.write_run_summary(run_manifest)
        run_manifest["artifacts"]["run_summary_md"] = str(summary_path)
        with open(self.run_dir / "run_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(run_manifest, fh, indent=2)
        return self.reported_artifacts()

    def run_01_seed_gc(self) -> None:
        """Run directed GC among curated seed genes."""
        stage = "01_seed_gc"
        started = datetime.now().isoformat(timespec="seconds")
        output = self.default_artifacts()["seed_gc_csv"]
        seed_genes = validate_gene_list_file(
            self.config.seed_gene_file,
            label="seed_gene_file",
            min_genes=2,
            required_gene=self.config.gene_of_interest,
        )
        if self.config.dataset.name.lower() == "fixture":
            result = self.run_fixture_gc(self.config.seed_gene_file, output, list_to_full_dataset=False)
        else:
            require_available_backend("gc", self.config.execution.gc_backend, device=self.config.execution.gpu_device)
            expression_df = self.load_expression_dataframe()
            validate_expression_dataframe(
                expression_df,
                gene_of_interest=self.config.gene_of_interest,
                seed_genes=seed_genes,
                transform=self.config.preprocessing.transform,
                min_timepoints=MIN_RECOMMENDED_TIMEPOINTS,
                min_present_genes=2,
            )
            result = _run_configured_gc(
                expression_df,
                genes_file=str(resolve_existing_path(self.config.seed_gene_file)),
                output_file=str(output),
                p_threshold=self.config.network.p_value_threshold,
                chunk_size=self.config.execution.chunk_size,
                list_to_kutsche=False,
                max_workers=self.config.execution.max_workers,
                progress=True,
                resume=self.config.execution.resume,
                rename_at_end=False,
                backend=self.config.execution.gc_backend,
                gpu_device=self.config.execution.gpu_device,
            )
        validate_gc_csv(output)
        self.artifacts["seed_gc_csv"] = Path(result["output_file"])
        self.write_manifest(
            stage,
            "completed",
            inputs={"seed_gene_file": self.config.seed_gene_file, "expression_file": self.config.dataset.expression_file},
            outputs={"seed_gc_csv": output},
            parameters={
                "p_value_threshold": self.config.network.p_value_threshold,
                "chunk_size": self.config.execution.chunk_size,
                "max_workers": self.config.execution.max_workers,
                "resume": self.config.execution.resume,
                "gc_backend": self.config.execution.gc_backend,
                "gpu_device": self.config.execution.gpu_device,
            },
            metrics={
                "backend": self.config.execution.gc_backend,
                "backend_metadata": backend_metadata(
                    "gc",
                    self.config.execution.gc_backend,
                    device=self.config.execution.gpu_device,
                ).to_dict(),
                "seed_gene_count": _count_text_lines(resolve_existing_path(self.config.seed_gene_file)),
                "gc_pairs_total": _gc_total_pairs(result, output),
                "gc_rows_written": _count_csv_rows(output),
                "gc_elapsed_seconds": result.get("elapsed_seconds"),
            },
            started=started,
        )

    def run_02_seed_consensus(self) -> None:
        """Build the seed network and run stability-controlled GOI consensus."""
        stage = "02_seed_consensus"
        started = datetime.now().isoformat(timespec="seconds")
        seed_gc = self.resolve_required_artifact("seed_gc_csv")
        network_artifacts = self.write_network_artifact_bundle(seed_gc, stage=stage, prefix="seed")
        result = run_stable_consensus(
            ConsensusConfig(
                gc_result_file=seed_gc,
                gene_of_interest=self.config.gene_of_interest,
                p_threshold=self.config.network.p_value_threshold,
                stability_tolerance=self.config.consensus.stability_tolerance,
                output_dir=self.stage_dir(stage),
            ),
            initial_runs=self.config.consensus.initial_runs,
            run_increment_fraction=self.config.consensus.run_increment_fraction,
            stability_quantile=self.config.consensus.stability_quantile,
            top_overlap_threshold_percent=self.config.consensus.top_overlap_threshold_percent,
            max_workers=self.config.execution.max_workers,
            backend=self.config.execution.consensus_backend,
            gpu_device=self.config.execution.gpu_device,
        )
        priority = self.default_artifacts()["seed_frequency_csv"]
        history_json = self.default_artifacts()["seed_consensus_history_json"]
        shutil.copyfile(result.final_frequency_csv, priority)
        with open(history_json, "w", encoding="utf-8") as fh:
            json.dump(result.history, fh, indent=2)
        validate_frequency_csv(priority, gene_of_interest=self.config.gene_of_interest)
        top_network_artifacts = self.write_top_consensus_network_artifact_bundle(
            seed_gc,
            priority,
            stage=stage,
            prefix="seed_top_consensus",
        )
        self.artifacts["seed_frequency_csv"] = priority
        self.artifacts["seed_consensus_history_json"] = history_json
        self.artifacts["seed_network_graphml"] = network_artifacts["graphml"]
        self.artifacts["seed_network_edges_csv"] = network_artifacts["edge_csv"]
        self.artifacts["seed_network_nodes_file"] = network_artifacts["node_txt"]
        self.artifacts["seed_network_summary_json"] = network_artifacts["summary_json"]
        self.artifacts["seed_top_consensus_network_graphml"] = top_network_artifacts["graphml"]
        self.artifacts["seed_top_consensus_network_edges_csv"] = top_network_artifacts["edge_csv"]
        self.artifacts["seed_top_consensus_network_nodes_file"] = top_network_artifacts["node_txt"]
        self.artifacts["seed_top_consensus_network_summary_json"] = top_network_artifacts["summary_json"]
        if "svg" in network_artifacts:
            self.artifacts["seed_network_svg"] = network_artifacts["svg"]
        if "svg" in top_network_artifacts:
            self.artifacts["seed_top_consensus_network_svg"] = top_network_artifacts["svg"]
        outputs = {
            "seed_frequency_csv": priority,
            "seed_consensus_history_json": history_json,
            "seed_network_graphml": network_artifacts["graphml"],
            "seed_network_edges_csv": network_artifacts["edge_csv"],
            "seed_network_nodes_file": network_artifacts["node_txt"],
            "seed_network_summary_json": network_artifacts["summary_json"],
            "seed_top_consensus_network_graphml": top_network_artifacts["graphml"],
            "seed_top_consensus_network_edges_csv": top_network_artifacts["edge_csv"],
            "seed_top_consensus_network_nodes_file": top_network_artifacts["node_txt"],
            "seed_top_consensus_network_summary_json": top_network_artifacts["summary_json"],
        }
        if "svg" in network_artifacts:
            outputs["seed_network_svg"] = network_artifacts["svg"]
        if "svg" in top_network_artifacts:
            outputs["seed_top_consensus_network_svg"] = top_network_artifacts["svg"]
        metrics = {
            **_network_metrics(network_artifacts["summary_json"]),
            **_top_consensus_network_metrics(top_network_artifacts["summary_json"]),
            **result.metrics,
            **_frequency_csv_metrics(priority, gene_of_interest=self.config.gene_of_interest),
        }
        self.write_manifest(
            stage,
            "completed",
            inputs={"seed_gc_csv": seed_gc},
            outputs=outputs,
            parameters={
                "gene_of_interest": self.config.gene_of_interest,
                "p_value_threshold": self.config.network.p_value_threshold,
                "initial_runs": self.config.consensus.initial_runs,
                "run_increment_fraction": self.config.consensus.run_increment_fraction,
                "stability_quantile": self.config.consensus.stability_quantile,
                "stability_tolerance": self.config.consensus.stability_tolerance,
                "top_overlap_threshold_percent": self.config.consensus.top_overlap_threshold_percent,
                "consensus_backend": self.config.execution.consensus_backend,
                "gpu_device": self.config.execution.gpu_device,
                "write_network_svg": self.config.network.write_svg,
                "network_svg_renderer": self.config.network.svg_renderer,
                "network_svg_layout": self.config.network.svg_layout,
            },
            metrics=metrics,
            started=started,
        )

    def run_03_probe_selection(self) -> None:
        """Select high-frequency GOI-community genes for full-dataset probing."""
        stage = "03_probe_selection"
        started = datetime.now().isoformat(timespec="seconds")
        frequency_csv = self.resolve_required_artifact("seed_frequency_csv")
        genes = select_probe_genes(
            frequency_csv,
            gene_of_interest=self.config.gene_of_interest,
            selection=self.config.probe_selection,
        )
        output = self.default_artifacts()["probe_genes_file"]
        write_probe_genes(genes, output)
        validate_gene_list_file(
            output,
            label="probe_genes_file",
            min_genes=1,
            required_gene=self.config.gene_of_interest,
        )
        self.artifacts["probe_genes_file"] = output
        self.write_manifest(
            stage,
            "completed",
            inputs={"seed_frequency_csv": frequency_csv},
            outputs={"probe_genes_file": output},
            parameters={
                "gene_of_interest": self.config.gene_of_interest,
                "mode": self.config.probe_selection.mode,
                "top_percent": self.config.probe_selection.top_percent,
                "min_frequency": self.config.probe_selection.min_frequency,
            },
            metrics={
                "selected_probe_genes": len(genes),
                "gene_of_interest_included": self.config.gene_of_interest in genes,
            },
            started=started,
        )

    def run_04_dataset_probe(self) -> None:
        """Run bidirectional probe GC between selected genes and the full dataset."""
        stage = "04_dataset_probe"
        started = datetime.now().isoformat(timespec="seconds")
        probe_genes = self.resolve_required_artifact("probe_genes_file")
        selected_probe_genes = validate_gene_list_file(
            probe_genes,
            label="probe_genes_file",
            min_genes=1,
            required_gene=self.config.gene_of_interest,
        )
        output = self.default_artifacts()["probe_gc_csv"]
        if self.config.dataset.name.lower() == "fixture":
            result = self.run_fixture_gc(probe_genes, output, list_to_full_dataset=True)
        else:
            require_available_backend("gc", self.config.execution.gc_backend, device=self.config.execution.gpu_device)
            expression_df = self.load_expression_dataframe()
            validate_expression_dataframe(
                expression_df,
                gene_of_interest=self.config.gene_of_interest,
                seed_genes=selected_probe_genes,
                transform=self.config.preprocessing.transform,
                min_timepoints=MIN_RECOMMENDED_TIMEPOINTS,
                min_present_genes=1,
            )
            result = _run_configured_gc(
                expression_df,
                genes_file=str(probe_genes),
                output_file=str(output),
                p_threshold=self.config.network.p_value_threshold,
                chunk_size=self.config.execution.chunk_size,
                list_to_kutsche=True,
                max_workers=self.config.execution.max_workers,
                progress=True,
                resume=self.config.execution.resume,
                rename_at_end=False,
                backend=self.config.execution.gc_backend,
                gpu_device=self.config.execution.gpu_device,
            )
        validate_gc_csv(output)
        self.artifacts["probe_gc_csv"] = Path(result["output_file"])
        self.write_manifest(
            stage,
            "completed",
            inputs={"probe_genes_file": probe_genes, "expression_file": self.config.dataset.expression_file},
            outputs={"probe_gc_csv": output},
            parameters={
                "p_value_threshold": self.config.network.p_value_threshold,
                "chunk_size": self.config.execution.chunk_size,
                "max_workers": self.config.execution.max_workers,
                "resume": self.config.execution.resume,
                "gc_backend": self.config.execution.gc_backend,
                "gpu_device": self.config.execution.gpu_device,
            },
            metrics={
                "backend": self.config.execution.gc_backend,
                "backend_metadata": backend_metadata(
                    "gc",
                    self.config.execution.gc_backend,
                    device=self.config.execution.gpu_device,
                ).to_dict(),
                "probe_gene_count": _count_text_lines(probe_genes),
                "dataset_gene_count": _count_text_lines(resolve_existing_path(self.config.dataset.full_gene_file)),
                "gc_pairs_total": _gc_total_pairs(result, output),
                "gc_rows_written": _count_csv_rows(output),
                "gc_elapsed_seconds": result.get("elapsed_seconds"),
            },
            started=started,
        )

    def run_05_expanded_genes(self) -> None:
        """Extract the expanded candidate gene set from significant probe edges."""
        stage = "05_expanded_genes"
        started = datetime.now().isoformat(timespec="seconds")
        probe_gc = self.resolve_required_artifact("probe_gc_csv")
        output = self.default_artifacts()["expanded_genes_file"]
        network_artifacts = self.write_network_artifact_bundle(probe_gc, stage=stage, prefix="probe")
        run_expansion(
            ExpansionConfig(
                candidate_network_csv=probe_gc,
                gene_of_interest=self.config.gene_of_interest,
                p_threshold=self.config.network.p_value_threshold,
                seed_gene_file=self.config.seed_gene_file,
                output_gene_list=output,
            )
        )
        validate_gene_list_file(
            output,
            label="expanded_genes_file",
            min_genes=2,
            required_gene=self.config.gene_of_interest,
        )
        self.artifacts["expanded_genes_file"] = output
        self.artifacts["probe_network_graphml"] = network_artifacts["graphml"]
        self.artifacts["probe_network_edges_csv"] = network_artifacts["edge_csv"]
        self.artifacts["probe_network_nodes_file"] = network_artifacts["node_txt"]
        self.artifacts["probe_network_summary_json"] = network_artifacts["summary_json"]
        if "svg" in network_artifacts:
            self.artifacts["probe_network_svg"] = network_artifacts["svg"]
        outputs = {
            "expanded_genes_file": output,
            "probe_network_graphml": network_artifacts["graphml"],
            "probe_network_edges_csv": network_artifacts["edge_csv"],
            "probe_network_nodes_file": network_artifacts["node_txt"],
            "probe_network_summary_json": network_artifacts["summary_json"],
        }
        if "svg" in network_artifacts:
            outputs["probe_network_svg"] = network_artifacts["svg"]
        metrics = {**_network_metrics(network_artifacts["summary_json"]), "expanded_gene_count": _count_text_lines(output)}
        self.write_manifest(
            stage,
            "completed",
            inputs={"probe_gc_csv": probe_gc},
            outputs=outputs,
            parameters={
                "gene_of_interest": self.config.gene_of_interest,
                "p_value_threshold": self.config.network.p_value_threshold,
                "seed_gene_file": str(self.config.seed_gene_file),
                "write_network_svg": self.config.network.write_svg,
                "network_svg_renderer": self.config.network.svg_renderer,
                "network_svg_layout": self.config.network.svg_layout,
            },
            metrics=metrics,
            started=started,
        )

    def run_06_expanded_gc(self) -> None:
        """Run directed GC among the expanded candidate gene set."""
        stage = "06_expanded_gc"
        started = datetime.now().isoformat(timespec="seconds")
        expanded_genes = self.resolve_required_artifact("expanded_genes_file")
        expanded_gene_list = validate_gene_list_file(
            expanded_genes,
            label="expanded_genes_file",
            min_genes=2,
            required_gene=self.config.gene_of_interest,
        )
        output = self.default_artifacts()["expanded_gc_csv"]
        if self.config.dataset.name.lower() == "fixture":
            result = self.run_fixture_gc(expanded_genes, output, list_to_full_dataset=False)
        else:
            require_available_backend("gc", self.config.execution.gc_backend, device=self.config.execution.gpu_device)
            expression_df = self.load_expression_dataframe()
            validate_expression_dataframe(
                expression_df,
                gene_of_interest=self.config.gene_of_interest,
                seed_genes=expanded_gene_list,
                transform=self.config.preprocessing.transform,
                min_timepoints=MIN_RECOMMENDED_TIMEPOINTS,
                min_present_genes=2,
            )
            result = _run_configured_gc(
                expression_df,
                genes_file=str(expanded_genes),
                output_file=str(output),
                p_threshold=self.config.network.p_value_threshold,
                chunk_size=self.config.execution.chunk_size,
                list_to_kutsche=False,
                max_workers=self.config.execution.max_workers,
                progress=True,
                resume=self.config.execution.resume,
                rename_at_end=False,
                backend=self.config.execution.gc_backend,
                gpu_device=self.config.execution.gpu_device,
            )
        validate_gc_csv(output)
        self.artifacts["expanded_gc_csv"] = Path(result["output_file"])
        self.write_manifest(
            stage,
            "completed",
            inputs={"expanded_genes_file": expanded_genes, "expression_file": self.config.dataset.expression_file},
            outputs={"expanded_gc_csv": output},
            parameters={
                "p_value_threshold": self.config.network.p_value_threshold,
                "chunk_size": self.config.execution.chunk_size,
                "max_workers": self.config.execution.max_workers,
                "resume": self.config.execution.resume,
                "gc_backend": self.config.execution.gc_backend,
                "gpu_device": self.config.execution.gpu_device,
            },
            metrics={
                "backend": self.config.execution.gc_backend,
                "backend_metadata": backend_metadata(
                    "gc",
                    self.config.execution.gc_backend,
                    device=self.config.execution.gpu_device,
                ).to_dict(),
                "expanded_gene_count": _count_text_lines(expanded_genes),
                "gc_pairs_total": _gc_total_pairs(result, output),
                "gc_rows_written": _count_csv_rows(output),
                "gc_elapsed_seconds": result.get("elapsed_seconds"),
            },
            started=started,
        )

    def run_07_expanded_consensus(self) -> None:
        """Run expanded consensus and write the final priority gene list."""
        stage = "07_expanded_consensus"
        started = datetime.now().isoformat(timespec="seconds")
        expanded_gc = self.resolve_required_artifact("expanded_gc_csv")
        network_artifacts = self.write_network_artifact_bundle(expanded_gc, stage=stage, prefix="expanded")
        result = run_stable_consensus(
            ConsensusConfig(
                gc_result_file=expanded_gc,
                gene_of_interest=self.config.gene_of_interest,
                p_threshold=self.config.network.p_value_threshold,
                stability_tolerance=self.config.consensus.stability_tolerance,
                output_dir=self.stage_dir(stage),
            ),
            initial_runs=self.config.consensus.initial_runs,
            run_increment_fraction=self.config.consensus.run_increment_fraction,
            stability_quantile=self.config.consensus.stability_quantile,
            top_overlap_threshold_percent=self.config.consensus.top_overlap_threshold_percent,
            max_workers=self.config.execution.max_workers,
            backend=self.config.execution.consensus_backend,
            gpu_device=self.config.execution.gpu_device,
        )
        priority = self.default_artifacts()["priority_genes_csv"]
        history_json = self.default_artifacts()["expanded_consensus_history_json"]
        shutil.copyfile(result.final_frequency_csv, priority)
        with open(history_json, "w", encoding="utf-8") as fh:
            json.dump(result.history, fh, indent=2)
        validate_frequency_csv(priority, gene_of_interest=self.config.gene_of_interest)
        top_network_artifacts = self.write_top_consensus_network_artifact_bundle(
            expanded_gc,
            priority,
            stage=stage,
            prefix="expanded_top_consensus",
        )
        self.artifacts["priority_genes_csv"] = priority
        self.artifacts["expanded_consensus_history_json"] = history_json
        self.artifacts["expanded_network_graphml"] = network_artifacts["graphml"]
        self.artifacts["expanded_network_edges_csv"] = network_artifacts["edge_csv"]
        self.artifacts["expanded_network_nodes_file"] = network_artifacts["node_txt"]
        self.artifacts["expanded_network_summary_json"] = network_artifacts["summary_json"]
        self.artifacts["expanded_top_consensus_network_graphml"] = top_network_artifacts["graphml"]
        self.artifacts["expanded_top_consensus_network_edges_csv"] = top_network_artifacts["edge_csv"]
        self.artifacts["expanded_top_consensus_network_nodes_file"] = top_network_artifacts["node_txt"]
        self.artifacts["expanded_top_consensus_network_summary_json"] = top_network_artifacts["summary_json"]
        if "svg" in network_artifacts:
            self.artifacts["expanded_network_svg"] = network_artifacts["svg"]
        if "svg" in top_network_artifacts:
            self.artifacts["expanded_top_consensus_network_svg"] = top_network_artifacts["svg"]
        outputs = {
            "priority_genes_csv": priority,
            "expanded_consensus_history_json": history_json,
            "expanded_network_graphml": network_artifacts["graphml"],
            "expanded_network_edges_csv": network_artifacts["edge_csv"],
            "expanded_network_nodes_file": network_artifacts["node_txt"],
            "expanded_network_summary_json": network_artifacts["summary_json"],
            "expanded_top_consensus_network_graphml": top_network_artifacts["graphml"],
            "expanded_top_consensus_network_edges_csv": top_network_artifacts["edge_csv"],
            "expanded_top_consensus_network_nodes_file": top_network_artifacts["node_txt"],
            "expanded_top_consensus_network_summary_json": top_network_artifacts["summary_json"],
        }
        if "svg" in network_artifacts:
            outputs["expanded_network_svg"] = network_artifacts["svg"]
        if "svg" in top_network_artifacts:
            outputs["expanded_top_consensus_network_svg"] = top_network_artifacts["svg"]
        metrics = {
            **_network_metrics(network_artifacts["summary_json"]),
            **_top_consensus_network_metrics(top_network_artifacts["summary_json"]),
            **result.metrics,
            **_frequency_csv_metrics(priority, gene_of_interest=self.config.gene_of_interest),
        }
        self.write_manifest(
            stage,
            "completed",
            inputs={"expanded_gc_csv": expanded_gc},
            outputs=outputs,
            parameters={
                "gene_of_interest": self.config.gene_of_interest,
                "p_value_threshold": self.config.network.p_value_threshold,
                "initial_runs": self.config.consensus.initial_runs,
                "run_increment_fraction": self.config.consensus.run_increment_fraction,
                "stability_quantile": self.config.consensus.stability_quantile,
                "stability_tolerance": self.config.consensus.stability_tolerance,
                "top_overlap_threshold_percent": self.config.consensus.top_overlap_threshold_percent,
                "consensus_backend": self.config.execution.consensus_backend,
                "gpu_device": self.config.execution.gpu_device,
                "write_network_svg": self.config.network.write_svg,
                "network_svg_renderer": self.config.network.svg_renderer,
                "network_svg_layout": self.config.network.svg_layout,
            },
            metrics=metrics,
            started=started,
        )


def _config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "dataset": {
            "name": config.dataset.name,
            "expression_file": str(config.dataset.expression_file),
            "full_gene_file": str(config.dataset.full_gene_file),
        },
        "gene_of_interest": config.gene_of_interest,
        "seed_gene_file": str(config.seed_gene_file),
        "preprocessing": {
            "normalize": config.preprocessing.normalize,
            "transform": config.preprocessing.transform,
            "aggregation": config.preprocessing.aggregation,
        },
        "network": {
            "p_value_threshold": config.network.p_value_threshold,
            "write_svg": config.network.write_svg,
            "svg_renderer": config.network.svg_renderer,
            "svg_layout": config.network.svg_layout,
        },
        "consensus": {
            "initial_runs": config.consensus.initial_runs,
            "run_increment_fraction": config.consensus.run_increment_fraction,
            "stability_quantile": config.consensus.stability_quantile,
            "stability_tolerance": config.consensus.stability_tolerance,
            "top_overlap_threshold_percent": config.consensus.top_overlap_threshold_percent,
        },
        "probe_selection": {
            "mode": config.probe_selection.mode,
            "top_percent": config.probe_selection.top_percent,
            "min_frequency": config.probe_selection.min_frequency,
        },
        "execution": {
            "max_workers": config.execution.max_workers,
            "chunk_size": config.execution.chunk_size,
            "resume": config.execution.resume,
            "gc_backend": config.execution.gc_backend,
            "consensus_backend": config.execution.consensus_backend,
            "gpu_device": config.execution.gpu_device,
        },
        "artifacts": {key: str(value) for key, value in config.artifacts.items()},
    }


def run_pipeline(config: PipelineConfig, *, start_at: str = "01_seed_gc", stop_after: Optional[str] = None) -> dict[str, Path]:
    """Convenience function for CLI wrappers and tests."""
    return PipelineRunner(config).run(start_at=start_at, stop_after=stop_after)
