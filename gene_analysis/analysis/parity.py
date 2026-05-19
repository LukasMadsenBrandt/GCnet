"""CPU/GPU parity and benchmark report helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from gene_analysis.analysis.stability import GeneStabilityConfig, compute_csv_stability_metrics
from gene_analysis.io.paths import resolve_existing_path


@dataclass(frozen=True)
class GcParityReport:
    """Differences between reference and candidate Granger result CSV files."""

    reference_csv: str
    candidate_csv: str
    reference_rows: int
    candidate_rows: int
    shared_pairs: int
    missing_in_candidate: int
    extra_in_candidate: int
    p_value_tolerance: float
    p_value_disagreements: int
    max_abs_p_value_diff: float
    significance_threshold: float
    significant_decision_disagreements: int


@dataclass(frozen=True)
class FrequencyParityReport:
    """Differences between reference and candidate coassociation frequency CSVs."""

    reference_csv: str
    candidate_csv: str
    quantile_relative_change: float
    top_gene_overlap_percent: float
    top_gene_overlap_k: int


@dataclass(frozen=True)
class BenchmarkReport:
    """Runtime and speedup metadata for comparable backend runs."""

    stage: str
    cpu_backend: str
    candidate_backend: str
    cpu_seconds: float
    candidate_seconds: float
    speedup: float
    work_units: int | None = None
    work_units_per_second_cpu: float | None = None
    work_units_per_second_candidate: float | None = None
    min_expected_speedup: float | None = None
    speedup_passed: bool | None = None


def compare_gc_csvs(
    reference_csv: str | Path,
    candidate_csv: str | Path,
    *,
    p_value_tolerance: float = 1e-6,
    significance_threshold: float = 0.05,
) -> GcParityReport:
    """Compare two GC CSVs by ordered pair, lag, p-value, and significance decision."""
    reference_path = resolve_existing_path(reference_csv)
    candidate_path = resolve_existing_path(candidate_csv)
    reference = _read_gc_rows(reference_path)
    candidate = _read_gc_rows(candidate_path)
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    shared = reference_keys & candidate_keys

    p_value_disagreements = 0
    max_abs_diff = 0.0
    significance_disagreements = 0
    for key in shared:
        ref = reference[key]
        cand = candidate[key]
        if ref is None or cand is None:
            if ref != cand:
                p_value_disagreements += 1
                significance_disagreements += 1
            continue
        abs_diff = abs(ref - cand)
        max_abs_diff = max(max_abs_diff, abs_diff)
        if abs_diff > p_value_tolerance:
            p_value_disagreements += 1
        if (ref <= significance_threshold) != (cand <= significance_threshold):
            significance_disagreements += 1

    return GcParityReport(
        reference_csv=str(reference_path),
        candidate_csv=str(candidate_path),
        reference_rows=len(reference),
        candidate_rows=len(candidate),
        shared_pairs=len(shared),
        missing_in_candidate=len(reference_keys - candidate_keys),
        extra_in_candidate=len(candidate_keys - reference_keys),
        p_value_tolerance=float(p_value_tolerance),
        p_value_disagreements=p_value_disagreements,
        max_abs_p_value_diff=max_abs_diff,
        significance_threshold=float(significance_threshold),
        significant_decision_disagreements=significance_disagreements,
    )


def compare_frequency_csvs(
    reference_csv: str | Path,
    candidate_csv: str | Path,
    *,
    top_fraction: float = 0.05,
    top_k: int | None = None,
    quantile_p: float = 0.90,
) -> FrequencyParityReport:
    """Compare two coassociation frequency CSVs using ranking stability metrics."""
    reference_path = resolve_existing_path(reference_csv)
    candidate_path = resolve_existing_path(candidate_csv)
    q_rel, overlap_pct, k = compute_csv_stability_metrics(
        str(reference_path),
        str(candidate_path),
        cfg=GeneStabilityConfig(quantile_p=quantile_p, top_fraction=top_fraction, top_k=top_k),
    )
    return FrequencyParityReport(
        reference_csv=str(reference_path),
        candidate_csv=str(candidate_path),
        quantile_relative_change=q_rel,
        top_gene_overlap_percent=overlap_pct,
        top_gene_overlap_k=k,
    )


def build_benchmark_report(
    *,
    stage: str,
    cpu_backend: str,
    candidate_backend: str,
    cpu_seconds: float,
    candidate_seconds: float,
    work_units: int | None = None,
    min_expected_speedup: float | None = None,
) -> BenchmarkReport:
    """Build a normalized runtime/speedup report for one comparable stage."""
    speedup = float("inf") if candidate_seconds == 0 else cpu_seconds / candidate_seconds
    cpu_rate = None if work_units is None or cpu_seconds == 0 else work_units / cpu_seconds
    candidate_rate = None if work_units is None or candidate_seconds == 0 else work_units / candidate_seconds
    speedup_passed = None if min_expected_speedup is None else speedup >= min_expected_speedup
    return BenchmarkReport(
        stage=stage,
        cpu_backend=cpu_backend,
        candidate_backend=candidate_backend,
        cpu_seconds=float(cpu_seconds),
        candidate_seconds=float(candidate_seconds),
        speedup=speedup,
        work_units=work_units,
        work_units_per_second_cpu=cpu_rate,
        work_units_per_second_candidate=candidate_rate,
        min_expected_speedup=min_expected_speedup,
        speedup_passed=speedup_passed,
    )


def write_report_json(report, output_file: str | Path) -> Path:
    """Write a dataclass parity or benchmark report as JSON."""
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2)
    return output


def _read_gc_rows(path: Path) -> dict[tuple[str, str, int], float | None]:
    rows: dict[tuple[str, str, int], float | None] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [column for column in ("gene1", "gene2", "lag", "p-value") if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}.")
        for row in reader:
            gene1 = (row.get("gene1") or "").strip()
            gene2 = (row.get("gene2") or "").strip()
            lag = _parse_int_or_none(row.get("lag"))
            p_value = _parse_float_or_none(row.get("p-value"))
            if not gene1 or not gene2 or lag is None:
                continue
            rows[(gene1, gene2, lag)] = p_value
    return rows


def _parse_int_or_none(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_float_or_none(value: str | None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
