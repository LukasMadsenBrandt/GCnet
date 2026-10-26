#!/usr/bin/env python3
"""Export small failure diagnostics from a completed FDR checkpoint (CPU only)."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.pipeline.fdr_diagnostics import run_failure_diagnostics
from gene_analysis.pipeline.runner import PipelineConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", choices=["anchor", "all_pairs"], default="anchor")
    parser.add_argument("--run-dir", type=Path, help="Override the recorded run directory, optionally ending in fdr.")
    parser.add_argument("--samples-per-category", type=int, default=25)
    parser.add_argument("--fetch-size", type=int, default=100_000)
    args = parser.parse_args()
    run_failure_diagnostics(PipelineConfig.from_yaml(args.config), family=args.family,
                            run_dir=args.run_dir, samples_per_category=args.samples_per_category,
                            fetch_size=args.fetch_size)


if __name__ == "__main__":
    main()
