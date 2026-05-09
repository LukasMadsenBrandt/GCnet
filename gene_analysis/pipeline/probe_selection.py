"""Select high-confidence genes for probing the full expression dataset."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from gene_analysis.io.paths import resolve_existing_path


@dataclass(frozen=True)
class ProbeSelectionConfig:
    """Selection rule for choosing genes from a GOI coassociation CSV."""

    mode: str = "top_percent"
    top_percent: Optional[float] = 5.0
    min_frequency: Optional[float] = None

    def validate(self) -> None:
        """Raise ``ValueError`` unless exactly one supported selection mode is active."""
        if self.mode not in {"top_percent", "min_frequency"}:
            raise ValueError("probe_selection.mode must be 'top_percent' or 'min_frequency'.")
        if self.mode == "top_percent":
            if self.top_percent is None or not 0 < float(self.top_percent) <= 100:
                raise ValueError("top_percent mode requires top_percent in (0, 100].")
            if self.min_frequency is not None:
                raise ValueError("top_percent mode cannot also set min_frequency.")
        if self.mode == "min_frequency":
            if self.min_frequency is None or not 0 <= float(self.min_frequency) <= 1:
                raise ValueError("min_frequency mode requires min_frequency in [0, 1].")
            if self.top_percent is not None:
                raise ValueError("min_frequency mode cannot also set top_percent.")


def select_probe_genes(
    frequency_csv: str | Path,
    *,
    gene_of_interest: str,
    selection: ProbeSelectionConfig,
) -> List[str]:
    """
    Select genes from a sorted or unsorted coassociation-frequency CSV.

    The CSV must contain ``Gene`` and ``Coassociation Frequency`` columns. The
    gene of interest is inserted if absent so downstream probing always anchors
    on the study target.
    """
    selection.validate()
    path = resolve_existing_path(frequency_csv)
    rows: list[tuple[str, float]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{path}: Missing CSV header.")
        if "Gene" not in reader.fieldnames or "Coassociation Frequency" not in reader.fieldnames:
            raise ValueError(f"{path}: expected columns Gene and Coassociation Frequency.")
        for row in reader:
            gene = (row.get("Gene") or "").strip()
            if not gene:
                continue
            try:
                freq = float(row.get("Coassociation Frequency") or "")
            except ValueError:
                continue
            rows.append((gene, freq))

    rows.sort(key=lambda item: item[1], reverse=True)
    if selection.mode == "top_percent":
        n = max(1, math.ceil(len(rows) * (float(selection.top_percent) / 100.0)))
        selected = [gene for gene, _ in rows[:n]]
    else:
        selected = [gene for gene, freq in rows if freq >= float(selection.min_frequency)]

    if gene_of_interest not in selected:
        selected.insert(0, gene_of_interest)
    return list(dict.fromkeys(selected))


def write_probe_genes(genes: list[str], output_file: str | Path) -> Path:
    """Write a one-gene-per-line probe list and return its path."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for gene in genes:
            fh.write(f"{gene}\n")
    return output_path
