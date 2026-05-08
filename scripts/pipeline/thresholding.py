#!/usr/bin/env python3
"""Compute lower p-value quantile thresholds for GC result CSVs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.pipeline.thresholding import run


def main() -> None:
    """Parse CLI arguments and print/write p-value threshold summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", help="GC CSV files with a p-value column.")
    parser.add_argument("--quantile", type=float, default=0.0005)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    results = run(args.csv_paths, quantile=args.quantile, output_file=args.output_file)
    for result in results:
        print(f"{result.csv_path}: quantile={result.quantile} threshold={result.threshold:.12g}")


if __name__ == "__main__":
    main()
