#!/usr/bin/env python3
"""
Compare per-gene frequencies between two ranked CSVs and compute differences.

Inputs (edit CONFIG below):
- PREVIOUS_FILE: CSV with headers (e.g., from a prior run)
- CURRENT_FILE:  CSV with headers (e.g., from the new run)
- Each CSV must have: Gene column + Frequency column (names configurable)

Outputs:
- DIFF_CSV: CSV with columns:
    Gene, PrevFreq, CurrFreq, AbsDiff, RelDiff, InPrev, InCurr
  Sorted by RelDiff (desc) by default.

- SUMMARY_TXT: text file with:
    - Number of genes in each file
    - Size of intersection used for relative diffs
    - 90% quantile of relative diffs
    - Top-K overlap (K from TOP_FRACTION or TOP_K)
    - Optional stability check against thresholds

Relative difference definition (default):
    RelDiff = |prev - curr| / max(prev, EPS)
You can change DENOMINATOR to "previous" | "max" | "mean" in CONFIG.
"""

from __future__ import annotations
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gene_analysis.io.paths import resolve_existing_path, results_path

# ===========================
# ========= CONFIG ==========
# Paths to the two CSVs to compare
PREVIOUS_FILE = "BASE_BENITO_GORILLA_1st_ZEB2_freq_0.0015/ZEB2_coassoc_5000_runs.csv"  # e.g. "/path/to/previous.csv"
CURRENT_FILE  = "BASE_BENITO_GORILLA_1st_ZEB2_freq_0.0015/ZEB2_coassoc_6000_runs.csv"  # e.g. "/path/to/current.csv"

# Output paths
DIFF_CSV    = results_path("comparisons", "gene_freq_diff.csv")
SUMMARY_TXT = results_path("comparisons", "gene_freq_diff_summary.txt")

# Column names
GENE_COL = "Gene"
FREQ_COL = "Coassociation Frequency"

# Case-insensitive gene matching
CASE_INSENSITIVE = True

# How to handle genes missing from one file when building the per-gene diff table:
# "union"  -> include all genes from both files (missing freq treated as 0.0)
# "inter"  -> include only genes present in both (recommended if you don't want zeros)
ROWSET_MODE = "inter"  # "union" or "inter"

# Denominator for relative difference:
# - "previous": |prev - curr| / max(prev, EPS)    (matches your snippet’s intent)
# - "max":      |prev - curr| / max(prev, curr, EPS)
# - "mean":     |prev - curr| / max((prev + curr)/2, EPS)
DENOMINATOR = "previous"

# Small positive value to avoid division by zero
EPS = 1e-9

# Sorting of the output table
SORT_BY = "RelDiff"  # "RelDiff" | "AbsDiff" | "Gene"
SORT_DESC = True

# Quantile for the summary statistic (e.g., 0.90 for 90%)
QUANTILE_P = 0.80

# Top-overlap settings
# Either set TOP_FRACTION (e.g., 0.05 for top 5%) or TOP_K (integer).
TOP_FRACTION = 0.05
TOP_K = None  # e.g., 690 or 804; if set, takes precedence over TOP_FRACTION

# Optional stability checks (printed in summary)
REL_DIFF_TOLERANCE = None   # e.g., 0.05 means <= 5% at the chosen quantile
OVERLAP_THRESHOLD_PERCENT = 95.0  # % overlap threshold for top set
# ========= END CONFIG ======
# ===========================


@dataclass(frozen=True)
class GeneFreq:
    """One gene and its coassociation frequency."""

    gene: str
    freq: float


def _norm_gene(g: str) -> str:
    return g.lower() if CASE_INSENSITIVE else g


def read_gene_freq_csv(path: str, gene_col: str, freq_col: str) -> Dict[str, float]:
    """Read gene frequencies from a CSV into a normalized mapping."""
    data: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path}: Missing CSV header.")
        headers = {h.lower(): h for h in reader.fieldnames}
        gcol = headers.get(gene_col.lower(), gene_col)
        fcol = headers.get(freq_col.lower(), freq_col)
        if gcol not in reader.fieldnames:
            raise ValueError(f"{path}: Gene column '{gene_col}' not found. Columns: {reader.fieldnames}")
        if fcol not in reader.fieldnames:
            raise ValueError(f"{path}: Frequency column '{freq_col}' not found. Columns: {reader.fieldnames}")
        for row in reader:
            gene = (row.get(gcol) or "").strip()
            freq_raw = (row.get(fcol) or "").strip()
            if not gene:
                continue
            try:
                freq = float(freq_raw)
            except ValueError:
                # skip non-numeric frequency
                continue
            data[_norm_gene(gene)] = freq
    return data


def _rel_diff(prev: float, curr: float) -> float:
    num = abs(prev - curr)
    if DENOMINATOR == "previous":
        den = max(prev, EPS)
    elif DENOMINATOR == "max":
        den = max(prev, curr, EPS)
    elif DENOMINATOR == "mean":
        den = max((prev + curr) / 2.0, EPS)
    else:
        raise ValueError(f"Invalid DENOMINATOR: {DENOMINATOR}")
    return num / den


def _quantile(values: List[float], p: float) -> float:
    """Linear-interpolated quantile in pure Python; p in [0,1]."""
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 1:
        return max(values)
    xs = sorted(values)
    pos = p * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _top_k_from_freqs(freqs: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    # returns list of (gene, freq) sorted desc by freq
    return sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:k]


def main() -> None:
    """Run the configured frequency-difference comparison."""
    if not PREVIOUS_FILE or not CURRENT_FILE:
        raise SystemExit("CONFIG ERROR: Set PREVIOUS_FILE and CURRENT_FILE.")

    previous_file = resolve_existing_path(PREVIOUS_FILE)
    current_file = resolve_existing_path(CURRENT_FILE)

    prev = read_gene_freq_csv(str(previous_file), GENE_COL, FREQ_COL)
    curr = read_gene_freq_csv(str(current_file),  GENE_COL, FREQ_COL)

    prev_genes = set(prev.keys())
    curr_genes = set(curr.keys())
    union_genes = prev_genes | curr_genes
    inter_genes = prev_genes & curr_genes

    # Build rows for output
    if ROWSET_MODE.lower() == "union":
        genes_for_rows = union_genes
        missing_as_zero = True
    elif ROWSET_MODE.lower() == "inter":
        genes_for_rows = inter_genes
        missing_as_zero = False
    else:
        raise SystemExit("CONFIG ERROR: ROWSET_MODE must be 'union' or 'inter'.")

    rows = []
    rel_diffs_for_stats: List[float] = []
    for g in genes_for_rows:
        p = prev.get(g, 0.0)
        c = curr.get(g, 0.0)
        abs_diff = abs(p - c)
        rel_diff = _rel_diff(p, c)

        # Only include in quantile stats if in intersection (mirrors typical practice)
        if g in inter_genes:
            rel_diffs_for_stats.append(rel_diff)

        gene_out = g if not CASE_INSENSITIVE else next((x for x in (g,) ), g)  # keep as normalized key
        rows.append({
            "Gene": gene_out if not CASE_INSENSITIVE else g,  # already normalized; change if you want original case
            "PrevFreq": p,
            "CurrFreq": c,
            "AbsDiff": abs_diff,
            "RelDiff": rel_diff,
            "InPrev": 1 if g in prev_genes else 0,
            "InCurr": 1 if g in curr_genes else 0,
        })

    # Sort rows
    if SORT_BY not in {"RelDiff", "AbsDiff", "Gene"}:
        raise SystemExit("CONFIG ERROR: SORT_BY must be 'RelDiff', 'AbsDiff', or 'Gene'.")
    rows.sort(key=lambda r: r[SORT_BY], reverse=SORT_DESC)

    # Write DIFF_CSV
    os.makedirs(os.path.dirname(DIFF_CSV) or ".", exist_ok=True)
    with open(DIFF_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Gene", "PrevFreq", "CurrFreq", "AbsDiff", "RelDiff", "InPrev", "InCurr"])
        for r in rows:
            w.writerow([
                r["Gene"],
                f"{r['PrevFreq']:.12g}",
                f"{r['CurrFreq']:.12g}",
                f"{r['AbsDiff']:.12g}",
                f"{r['RelDiff']:.12g}",
                r["InPrev"],
                r["InCurr"],
            ])

    # Quantile stat
    q_val = _quantile(rel_diffs_for_stats, QUANTILE_P)

    # Top-overlap
    n_prev = len(prev)
    n_curr = len(curr)
    if TOP_K is not None:
        k = int(TOP_K)
    else:
        # Use min size so both sets have same K
        k = max(1, int(round(TOP_FRACTION * min(n_prev, n_curr))))
    top_prev = set(g for g, _ in _top_k_from_freqs(prev, k))
    top_curr = set(g for g, _ in _top_k_from_freqs(curr, k))
    overlap = top_prev & top_curr
    overlap_pct = (len(overlap) / k) * 100.0 if k > 0 else float("nan")

    # Optional stability check (only if threshold provided)
    stability_lines = []
    if REL_DIFF_TOLERANCE is not None:
        cond_rel = (q_val <= REL_DIFF_TOLERANCE)
        stability_lines.append(
            f" - 90% Quantile of Relative Differences: {q_val:.6f} "
            f"(Tolerance: {REL_DIFF_TOLERANCE}) -> {'OK' if cond_rel else 'NOT OK'}"
        )
    else:
        stability_lines.append(f" - 90% Quantile of Relative Differences: {q_val:.6f}")

    cond_overlap = (overlap_pct >= OVERLAP_THRESHOLD_PERCENT)
    stability_lines.append(
        f" - Top-{k} Overlap: {overlap_pct:.2f}% "
        f"(Threshold: {OVERLAP_THRESHOLD_PERCENT:.2f}%) -> {'OK' if cond_overlap else 'NOT OK'}"
    )

    # Write summary
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("Gene Frequency Comparison Summary\n")
        f.write("================================\n")
        f.write(f"Previous file : {previous_file}\n")
        f.write(f"Current file  : {current_file}\n\n")
        f.write(f"Prev genes: {n_prev}\n")
        f.write(f"Curr genes: {n_curr}\n")
        f.write(f"Intersection used for rel-diff stats: {len(inter_genes)} genes\n\n")
        f.write("\n".join(stability_lines) + "\n")
        f.write(f" - Overlapping Genes ({len(overlap)}/{k}):\n")
        # Uncomment the next line to print the actual overlapping genes:
        # f.write(\"  \" + \", \".join(sorted(overlap)) + \"\\n\")
        f.write("\n")
        f.write(f"Diff table written to: {DIFF_CSV}\n")

    # Console prints
    print(f"Wrote: {DIFF_CSV}")
    print(f"Wrote: {SUMMARY_TXT}")
    print(f"90% quantile rel-diff: {q_val:.6f}")
    print(f"Top-{k} overlap: {overlap_pct:.2f}% ({len(overlap)}/{k})")


if __name__ == "__main__":
    main()
