#!/usr/bin/env python3
"""Build global BH/Storey tables and connectivity diagnostics from saved tests."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.experiments.qvalue_connectivity import run_qvalue_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    run_qvalue_comparison(args.run_dir, output_dir=args.output_dir, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
