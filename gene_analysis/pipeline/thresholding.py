"""P-value threshold helpers for Granger result CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from gene_analysis.io.paths import resolve_existing_path, results_path


@dataclass(frozen=True)
class ThresholdResult:
    """Computed lower-quantile threshold for one Granger result CSV."""

    csv_path: Path
    quantile: float
    threshold: float


def compute_lower_quantile_threshold(csv_path: str | Path, quantile: float = 0.0005) -> ThresholdResult:
    """Compute the requested lower quantile from a CSV ``p-value`` column."""
    if not 0 < float(quantile) <= 1:
        raise ValueError("quantile must be in (0, 1].")
    path = resolve_existing_path(csv_path)
    df = pd.read_csv(path)
    if "p-value" not in df.columns:
        raise ValueError("CSV must contain a 'p-value' column.")
    return ThresholdResult(path, float(quantile), float(df["p-value"].quantile(float(quantile))))


def run(csv_paths: Iterable[str | Path], quantile: float = 0.0005, output_file: str | Path | None = None) -> List[ThresholdResult]:
    """Compute thresholds for one or more CSVs and write a summary table."""
    results = [compute_lower_quantile_threshold(path, quantile) for path in csv_paths]
    out = Path(output_file) if output_file else results_path("pipeline", "02_thresholds", "thresholds.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["csv_path", "quantile", "threshold"])
        for result in results:
            writer.writerow([result.csv_path, result.quantile, result.threshold])
    return results
