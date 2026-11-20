#!/usr/bin/env python3
"""Summarize raw-count expression-filter sensitivity for the Kutsche WT samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.analysis.preprocessing import aggregate_duplicate_genes
from gene_analysis.datasets.kutsche import load_and_preprocess_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression-file", type=Path,
                        default=Path("Data/Kutsche/Kutsche_Counts.txt"))
    parser.add_argument("--seed-file", type=Path,
                        default=Path("Data/Kutsche/unique_genes.txt"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/pipeline/expression_filter_sensitivity.csv"))
    parser.add_argument("--counts", nargs="+", type=float, default=[1, 5, 10, 20])
    parser.add_argument("--samples", nargs="+", type=int, default=[1, 3, 5, 7, 10])
    parser.add_argument("--markers", nargs="+", default=["ZEB2", "ZBTB18", "MECP2"])
    args = parser.parse_args()

    raw = load_and_preprocess_data(args.expression_file)
    if raw.index.has_duplicates:
        raw = aggregate_duplicate_genes(raw, method="mean")
    wt = raw[[column for column in raw.columns if "WT" in column]]
    seeds = set(args.seed_file.read_text(encoding="utf-8").split())
    nonzero_genes = set(wt.index[(wt != 0).any(axis=1)].astype(str))
    baseline_seeds = seeds & nonzero_genes
    rows = []
    for count in args.counts:
        prevalence = (wt >= count).sum(axis=1)
        for samples in args.samples:
            retained = set(prevalence.index[prevalence >= samples].astype(str))
            n = len(retained)
            rows.append({
                "min_raw_count": count,
                "min_wt_samples": samples,
                "retained_genes": n,
                "retained_seed_genes": len(retained & seeds),
                "excluded_from_previous_nonzero_seed_set": len(baseline_seeds - retained),
                "previous_nonzero_seed_genes": len(baseline_seeds),
                "seed_names_in_input_file": len(seeds),
                "directed_nonself_tests": n * (n - 1),
                "missing_markers": ",".join(marker for marker in args.markers
                                              if marker not in retained),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
