#!/usr/bin/env python3
"""Run CPU-vs-CUDA GC parity and benchmark checks for one pipeline config."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.analysis.cuda_environment import collect_cuda_environment_report
from gene_analysis.analysis.parity import build_benchmark_report, compare_frequency_csvs, compare_gc_csvs
from gene_analysis.io.paths import results_path
from gene_analysis.pipeline.config import ExecutionConfig
from gene_analysis.pipeline.runner import PipelineConfig, PipelineRunner


GC_ARTIFACT_KEYS = ("seed_gc_csv", "probe_gc_csv", "expanded_gc_csv")
GC_STAGE_BY_ARTIFACT = {
    "seed_gc_csv": "01_seed_gc",
    "probe_gc_csv": "04_dataset_probe",
    "expanded_gc_csv": "06_expanded_gc",
}


def main() -> None:
    """Parse CLI arguments, run CPU/CUDA variants, and write a parity report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pipeline YAML config to run twice.")
    parser.add_argument("--output-file", default=None, help="Optional JSON report path.")
    parser.add_argument("--run-prefix", default=None, help="Optional run-name prefix for generated CPU/CUDA runs.")
    parser.add_argument("--p-value-tolerance", type=float, default=1e-6, help="Allowed p-value difference.")
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=None,
        help="Optional minimum CUDA speedup required for each benchmarked GC stage.",
    )
    parser.add_argument("--stop-after", default=None, help="Optional stage to stop both runs after.")
    parser.add_argument("--skip-runs", action="store_true", help="Only compare existing CPU/CUDA run folders.")
    args = parser.parse_args()

    report_path = benchmark_cuda_gc(
        args.config,
        output_file=args.output_file,
        run_prefix=args.run_prefix,
        p_value_tolerance=args.p_value_tolerance,
        min_speedup=args.min_speedup,
        stop_after=args.stop_after,
        skip_runs=args.skip_runs,
    )
    print(f"CUDA GC benchmark report written to {report_path}")


def benchmark_cuda_gc(
    config_file: str | Path,
    *,
    output_file: str | Path | None = None,
    run_prefix: str | None = None,
    p_value_tolerance: float = 1e-6,
    min_speedup: float | None = None,
    stop_after: str | None = None,
    skip_runs: bool = False,
) -> Path:
    """Run comparable CPU and CUDA pipeline variants and write a JSON report."""
    base = PipelineConfig.from_yaml(config_file)
    prefix = run_prefix or base.run_name
    cpu_cfg = _variant_config(base, run_name=f"{prefix}_cpu_benchmark", gc_backend="cpu_statsmodels", gpu_device=None)
    cuda_cfg = _variant_config(base, run_name=f"{prefix}_cuda_benchmark", gc_backend="gpu_cuda", gpu_device=0)

    cuda_environment = collect_cuda_environment_report().to_dict()
    if not cuda_environment["cupy_gc_ready"]:
        raise RuntimeError("CuPy GC backend is not ready on this machine. Run scripts/pipeline/check_cuda.py first.")

    cpu_artifacts = PipelineRunner(cpu_cfg).reported_artifacts() if skip_runs else PipelineRunner(cpu_cfg).run(stop_after=stop_after)
    cuda_artifacts = (
        PipelineRunner(cuda_cfg).reported_artifacts() if skip_runs else PipelineRunner(cuda_cfg).run(stop_after=stop_after)
    )

    gc_reports = {}
    benchmarks = {}
    for key in GC_ARTIFACT_KEYS:
        if key not in cpu_artifacts or key not in cuda_artifacts:
            continue
        if not Path(cpu_artifacts[key]).exists() or not Path(cuda_artifacts[key]).exists():
            continue
        gc_reports[key] = asdict(
            compare_gc_csvs(
                cpu_artifacts[key],
                cuda_artifacts[key],
                p_value_tolerance=p_value_tolerance,
                significance_threshold=base.network.p_value_threshold,
            )
        )
        stage = GC_STAGE_BY_ARTIFACT[key]
        cpu_seconds = _stage_seconds(cpu_cfg.run_dir / stage / "manifest.json")
        cuda_seconds = _stage_seconds(cuda_cfg.run_dir / stage / "manifest.json")
        work_units = _stage_work_units(cuda_cfg.run_dir / stage / "manifest.json")
        if cpu_seconds is not None and cuda_seconds is not None:
            benchmarks[stage] = asdict(
                build_benchmark_report(
                    stage=stage,
                    cpu_backend="cpu_statsmodels",
                    candidate_backend="gpu_cuda",
                    cpu_seconds=cpu_seconds,
                    candidate_seconds=cuda_seconds,
                    work_units=work_units,
                    min_expected_speedup=min_speedup,
                )
            )

    frequency_report = None
    if "priority_genes_csv" in cpu_artifacts and "priority_genes_csv" in cuda_artifacts:
        if Path(cpu_artifacts["priority_genes_csv"]).exists() and Path(cuda_artifacts["priority_genes_csv"]).exists():
            frequency_report = asdict(
                compare_frequency_csvs(cpu_artifacts["priority_genes_csv"], cuda_artifacts["priority_genes_csv"])
            )

    report = {
        "config": str(config_file),
        "base_run_name": base.run_name,
        "cpu_run_name": cpu_cfg.run_name,
        "cuda_run_name": cuda_cfg.run_name,
        "cuda_environment": cuda_environment,
        "p_value_tolerance": p_value_tolerance,
        "min_speedup": min_speedup,
        "significance_threshold": base.network.p_value_threshold,
        "gc_parity": gc_reports,
        "frequency_parity": frequency_report,
        "benchmarks": benchmarks,
        "passed": _report_passed(gc_reports, frequency_report, benchmarks),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    output = Path(output_file) if output_file else results_path("pipeline", f"{prefix}_cuda_benchmark_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def _variant_config(base: PipelineConfig, *, run_name: str, gc_backend: str, gpu_device: int | None) -> PipelineConfig:
    execution = replace(
        base.execution,
        gc_backend=gc_backend,
        consensus_backend="cpu_louvain",
        gpu_device=gpu_device,
        resume=False,
    )
    return replace(base, run_name=run_name, execution=execution)


def _stage_seconds(manifest_path: str | Path) -> float | None:
    manifest = _read_manifest(manifest_path)
    if not manifest:
        return None
    gc_elapsed = (manifest.get("metrics") or {}).get("gc_elapsed_seconds")
    if gc_elapsed is not None:
        return float(gc_elapsed)
    try:
        started = datetime.fromisoformat(manifest["started_at"])
        finished = datetime.fromisoformat(manifest["finished_at"])
    except (KeyError, ValueError):
        return None
    return max((finished - started).total_seconds(), 0.0)


def _stage_work_units(manifest_path: str | Path) -> int | None:
    manifest = _read_manifest(manifest_path)
    if not manifest:
        return None
    value = (manifest.get("metrics") or {}).get("gc_pairs_total")
    return int(value) if value is not None else None


def _read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _report_passed(
    gc_reports: dict[str, dict[str, Any]],
    frequency_report: dict[str, Any] | None,
    benchmarks: dict[str, dict[str, Any]] | None = None,
) -> bool:
    if not gc_reports:
        return False
    for report in gc_reports.values():
        if report["missing_in_candidate"] or report["extra_in_candidate"]:
            return False
        if report["p_value_disagreements"] or report["significant_decision_disagreements"]:
            return False
    if frequency_report is not None and frequency_report["top_gene_overlap_percent"] < 100.0:
        return False
    for benchmark in (benchmarks or {}).values():
        if benchmark.get("speedup_passed") is False:
            return False
    return True


if __name__ == "__main__":
    main()
