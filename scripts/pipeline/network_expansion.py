#!/usr/bin/env python3
"""Extract expanded candidate genes from a probe/network GC CSV."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.pipeline.config import ExpansionConfig
from gene_analysis.pipeline.network_expansion import run


def main() -> None:
    """Parse CLI arguments and write an expanded candidate gene list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-network-csv", required=True)
    parser.add_argument("--gene-of-interest", default="ZEB2")
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--seed-gene-file", default=None)
    parser.add_argument("--output-gene-list", default=None)
    args = parser.parse_args()

    default_config = ExpansionConfig(candidate_network_csv=args.candidate_network_csv)
    config = ExpansionConfig(
        candidate_network_csv=args.candidate_network_csv,
        gene_of_interest=args.gene_of_interest,
        p_threshold=args.p_threshold,
        seed_gene_file=args.seed_gene_file,
        output_gene_list=args.output_gene_list or default_config.output_gene_list,
    )
    output = run(config)
    print(f"Wrote expanded genes: {output}")


if __name__ == "__main__":
    main()
