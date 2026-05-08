#!/usr/bin/env python3
"""Prepare/run consensus analysis for a GC result CSV."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.pipeline.config import ConsensusConfig
from gene_analysis.pipeline.consensus import run


def main() -> None:
    """Parse CLI arguments and prepare a consensus output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gc-result-file", required=True)
    parser.add_argument("--gene-of-interest", default="ZEB2")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    default_config = ConsensusConfig(gc_result_file=args.gc_result_file)
    config = ConsensusConfig(
        gc_result_file=args.gc_result_file,
        gene_of_interest=args.gene_of_interest,
        n_runs=args.n_runs,
        p_threshold=args.p_threshold,
        output_dir=args.output_dir or default_config.output_dir,
    )
    output = run(config)
    print(f"Consensus output directory: {output}")


if __name__ == "__main__":
    main()
