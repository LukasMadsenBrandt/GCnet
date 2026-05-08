#!/usr/bin/env python3
"""
Config-editable script for gene list comparison.

How to use:
1) Open this file in VS Code.
2) Edit the CONFIG section below (paths, X values, headers, etc.).
3) Run the script (no CLI args). It writes outputs and prints a short summary.

Output per input file and X:
- {basename}_top{X}_annotated.csv  (columns: Gene, Coassociation Frequency, New)
- {basename}_top{X}_summary.txt    (counts of new vs known)
"""

from __future__ import annotations
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from project_paths import data_path, resolve_existing_path, results_path

# ===========================
# ========= CONFIG ==========
# Edit these variables directly.

# Ranked CSV files (with headers) to process.
INPUT_FILES = [
    #"SM7_00015_7200_runs.csv",
    "MECP2_coassoc_00015_6000_runs.csv",
]

# Known genes text file (one gene symbol per line, NO header).
KNOWN_GENES_FILE = data_path("Kutsche", "unique_genes.txt")

# Produce outputs for each X in this list (e.g., 690 and 804 for main text).
X_VALUES = [14237]

# Directory to save outputs (created if needed).
OUTPUT_DIR = results_path("gene_list_compare")

# Column names in your ranked CSVs. Case-insensitive header matching is supported.
GENE_COL = "Gene"
FREQ_COL = "Coassociation Frequency"

# Compare gene membership case-insensitively (recommended).
CASE_INSENSITIVE = True

# Number formatting for frequency in outputs.
FREQ_FORMAT = ".12g"

# ========= END CONFIG ======
# ===========================


@dataclass(frozen=True)
class RankedGene:
    """One gene and its ranking frequency from a coassociation CSV."""

    gene: str
    freq: float


def load_known_genes(filepath: str, case_insensitive: bool = False) -> Set[str]:
    """Load known/seed genes from a one-gene-per-line text file."""
    genes: Set[str] = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            g = line.strip()
            if not g:
                continue
            genes.add(g.lower() if case_insensitive else g)
    return genes


def read_ranked_csv(filepath: str, gene_col: str, freq_col: str) -> List[RankedGene]:
    """Read and sort ranked genes from a coassociation-frequency CSV."""
    rows: List[RankedGene] = []
    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{filepath}: Missing CSV header.")
        headers = {h.lower(): h for h in reader.fieldnames}
        gcol = headers.get(gene_col.lower(), gene_col)
        fcol = headers.get(freq_col.lower(), freq_col)
        if gcol not in reader.fieldnames:
            raise ValueError(f"{filepath}: Gene column '{gene_col}' not found. Columns: {reader.fieldnames}")
        if fcol not in reader.fieldnames:
            raise ValueError(f"{filepath}: Frequency column '{freq_col}' not found. Columns: {reader.fieldnames}")
        for row in reader:
            gene = (row.get(gcol) or "").strip()
            freq_raw = (row.get(fcol) or "").strip()
            if not gene:
                continue
            try:
                freq = float(freq_raw)
            except ValueError:
                # Skip rows with non-numeric frequency
                continue
            rows.append(RankedGene(gene, freq))
    rows.sort(key=lambda x: x.freq, reverse=True)
    return rows


def write_outputs(
    input_path: str,
    top_rows: List[RankedGene],
    known: Set[str],
    out_dir: str,
    x: int,
    case_insensitive: bool,
    freq_format: str = ".12g",
) -> None:
    """Write annotated top-X CSV and summary files for one ranked gene list."""
    base = os.path.basename(input_path)
    stem, _ = os.path.splitext(base)
    csv_out = os.path.join(out_dir, f"{stem}_top{x}_annotated.csv")
    txt_out = os.path.join(out_dir, f"{stem}_top{x}_summary.txt")

    new_count = 0
    known_count = 0
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Gene", "Coassociation Frequency", "New"])
        for rg in top_rows:
            key = rg.gene.lower() if case_insensitive else rg.gene
            is_new = 0 if key in known else 1
            if is_new:
                new_count += 1
            else:
                known_count += 1
            w.writerow([rg.gene, format(rg.freq, freq_format), is_new])

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(f"File: {input_path}\n")
        f.write(f"Top X: {x}\n")
        f.write(f"New genes (1): {new_count}\n")
        f.write(f"Known genes (0): {known_count}\n")

    print(f"Wrote: {csv_out}")
    print(f"Wrote: {txt_out}")


def main() -> None:
    """Run the configured gene-list comparison."""
    if not INPUT_FILES:
        raise SystemExit("CONFIG ERROR: Please set INPUT_FILES to your ranked CSV paths.")
    if not KNOWN_GENES_FILE:
        raise SystemExit("CONFIG ERROR: Please set KNOWN_GENES_FILE to your known gene list path.")
    if not X_VALUES:
        raise SystemExit("CONFIG ERROR: Please set X_VALUES to a list of integers (e.g., [690, 804]).")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    known = load_known_genes(KNOWN_GENES_FILE, case_insensitive=CASE_INSENSITIVE)

    i = 0
    for input_csv in INPUT_FILES:
        input_csv = resolve_existing_path(input_csv)
        rows = read_ranked_csv(str(input_csv), GENE_COL, FREQ_COL)
        top_rows = rows[:X_VALUES[i]]
        write_outputs(
            input_path=input_csv,
            top_rows=top_rows,
            known=known,
            out_dir=OUTPUT_DIR,
            x=X_VALUES[i],
            case_insensitive=CASE_INSENSITIVE,
            freq_format=FREQ_FORMAT,
        )
        i += 1

    print("Done.")


if __name__ == "__main__":
    main()
