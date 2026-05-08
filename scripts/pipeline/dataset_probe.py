#!/usr/bin/env python3
"""Generate full-dataset probe pairs from curated seed genes."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.pipeline.config import ProbeConfig
from gene_analysis.pipeline.dataset_probe import run


def main() -> None:
    """Parse CLI arguments and write a probe-pair design file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-gene-file", required=True)
    parser.add_argument("--full-gene-file", required=True)
    parser.add_argument("--output-file", default=None, help="Probe-pair CSV output path.")
    args = parser.parse_args()

    default_config = ProbeConfig(seed_gene_file=args.seed_gene_file, full_gene_file=args.full_gene_file)
    config = ProbeConfig(
        seed_gene_file=args.seed_gene_file,
        full_gene_file=args.full_gene_file,
        probe_pairs_file=args.output_file or default_config.probe_pairs_file,
    )
    output = run(config)
    print(f"Wrote probe pairs: {output}")


if __name__ == "__main__":
    main()
