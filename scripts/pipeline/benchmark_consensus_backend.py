#!/usr/bin/env python3
"""Run consensus-backend parity and benchmark checks for one pipeline config."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.analysis.parity import build_benchmark_report, compare_frequency_csvs
from gene_analysis.io.paths import results_path
from gene_analysis.pipeline.runner import PipelineConfig, PipelineRunner


CONSENSUS_ARTIFACT_KEYS = ("seed_frequency_csv", "priority_genes_csv")
CONSENSUS_STAGE_BY_ARTIFACT = {
    "seed_frequency_csv": "02_seed_consensus",
    "priority_genes_csv": "07_expanded_consensus",
}


def main() -> None:
    """Parse CLI arguments, run backend variants, and write a parity report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pipeline YAML config to run twice.")
    parser.add_argument(
        "--candidate-backend",
        default="gpu_cugraph",
        help="Consensus backend to compare against cpu_louvain.",
    )
    parser.add_argument("--output-file", default=None, help="Optional JSON report path.")
    parser.add_argument("--run-prefix", default=None, help="Optional run-name prefix for generated runs.")
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=None,
        help="Optional minimum candidate speedup required for each benchmarked consensus stage.",
    )
    parser.add_argument(
        "--top-overlap-threshold-percent",
        type=float,
        default=95.0,
        help="Minimum CPU/candidate top-gene overlap required for consensus parity.",
    )
    parser.add_argument("--stop-after", default=None, help="Optional stage to stop both runs after.")
    parser.add_argument("--skip-runs", action="store_true", help="Only compare existing benchmark run folders.")
    args = parser.parse_args()

    report_path = benchmark_consensus_backend(
        args.config,
        candidate_backend=args.candidate_backend,
        output_file=args.output_file,
        run_prefix=args.run_prefix,
        min_speedup=args.min_speedup,
        top_overlap_threshold_percent=args.top_overlap_threshold_percent,
        stop_after=args.stop_after,
        skip_runs=args.skip_runs,
    )
    print(f"Consensus benchmark report written to {report_path}")


def benchmark_consensus_backend(
    config_file: str | Path,
    *,
    candidate_backend: str = "gpu_cugraph",
    output_file: str | Path | None = None,
    run_prefix: str | None = None,
    min_speedup: float | None = None,
    top_overlap_threshold_percent: float = 95.0,
    stop_after: str | None = None,
    skip_runs: bool = False,
) -> Path:
    """Run comparable consensus backend variants and write a JSON report."""
    base = PipelineConfig.from_yaml(config_file)
    prefix = run_prefix or base.run_name
    cpu_cfg = _variant_config(base, run_name=f"{prefix}_cpu_consensus_benchmark", consensus_backend="cpu_louvain")
    candidate_cfg = _variant_config(
        base,
        run_name=f"{prefix}_{candidate_backend}_consensus_benchmark",
        consensus_backend=candidate_backend,
    )

    cpu_runner = PipelineRunner(cpu_cfg)
    candidate_runner = PipelineRunner(candidate_cfg)
    cpu_artifacts = cpu_runner.reported_artifacts() if skip_runs else cpu_runner.run(stop_after=stop_after)
    candidate_artifacts = (
        candidate_runner.reported_artifacts() if skip_runs else candidate_runner.run(stop_after=stop_after)
    )

    frequency_reports = {}
    benchmarks = {}
    for key in CONSENSUS_ARTIFACT_KEYS:
        if key not in cpu_artifacts or key not in candidate_artifacts:
            continue
        if not Path(cpu_artifacts[key]).exists() or not Path(candidate_artifacts[key]).exists():
            continue
        frequency_reports[key] = asdict(compare_frequency_csvs(cpu_artifacts[key], candidate_artifacts[key]))
        stage = CONSENSUS_STAGE_BY_ARTIFACT[key]
        cpu_seconds = _stage_consensus_seconds(cpu_cfg.run_dir / stage / "manifest.json")
        candidate_seconds = _stage_consensus_seconds(candidate_cfg.run_dir / stage / "manifest.json")
        work_units = _stage_consensus_work_units(candidate_cfg.run_dir / stage / "manifest.json")
        if cpu_seconds is not None and candidate_seconds is not None:
            benchmarks[stage] = asdict(
                build_benchmark_report(
                    stage=stage,
                    cpu_backend="cpu_louvain",
                    candidate_backend=candidate_backend,
                    cpu_seconds=cpu_seconds,
                    candidate_seconds=candidate_seconds,
                    work_units=work_units,
                    min_expected_speedup=min_speedup,
                )
            )

    report = {
        "config": str(config_file),
        "base_run_name": base.run_name,
        "cpu_run_name": cpu_cfg.run_name,
        "candidate_run_name": candidate_cfg.run_name,
        "candidate_backend": candidate_backend,
        "min_speedup": min_speedup,
        "stability_tolerance": base.consensus.stability_tolerance,
        "top_overlap_threshold_percent": top_overlap_threshold_percent,
        "frequency_parity": frequency_reports,
        "benchmarks": benchmarks,
        "passed": _report_passed(
            frequency_reports,
            benchmarks,
            stability_tolerance=base.consensus.stability_tolerance,
            top_overlap_threshold_percent=top_overlap_threshold_percent,
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    output = Path(output_file) if output_file else results_path("pipeline", f"{prefix}_consensus_benchmark_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def _variant_config(base: PipelineConfig, *, run_name: str, consensus_backend: str) -> PipelineConfig:
    execution = replace(
        base.execution,
        consensus_backend=consensus_backend,
        resume=False,
    )
    return replace(base, run_name=run_name, execution=execution)


def _stage_consensus_seconds(manifest_path: str | Path) -> float | None:
    manifest = _read_manifest(manifest_path)
    if not manifest:
        return None
    value = (manifest.get("metrics") or {}).get("consensus_total_seconds")
    if value is not None:
        return float(value)
    try:
        started = datetime.fromisoformat(manifest["started_at"])
        finished = datetime.fromisoformat(manifest["finished_at"])
    except (KeyError, ValueError):
        return None
    return max((finished - started).total_seconds(), 0.0)


def _stage_consensus_work_units(manifest_path: str | Path) -> int | None:
    metrics = (_read_manifest(manifest_path).get("metrics") or {})
    nodes = metrics.get("nodes_total") or metrics.get("genes_total")
    partitions = metrics.get("partitions_total") or metrics.get("final_runs")
    if nodes is None or partitions is None:
        return None
    return int(nodes) * int(partitions)


def _read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _report_passed(
    frequency_reports: dict[str, dict[str, Any]],
    benchmarks: dict[str, dict[str, Any]],
    *,
    stability_tolerance: float,
    top_overlap_threshold_percent: float,
) -> bool:
    if not frequency_reports:
        return False
    for report in frequency_reports.values():
        if report["top_gene_overlap_percent"] < top_overlap_threshold_percent:
            return False
        if report["quantile_relative_change"] > stability_tolerance:
            return False
    for benchmark in benchmarks.values():
        if benchmark.get("speedup_passed") is False:
            return False
    return True


if __name__ == "__main__":
    main()
