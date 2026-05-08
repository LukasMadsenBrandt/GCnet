#!/usr/bin/env python3
"""Run or resume the YAML-configured neurodeficiency gene-expansion pipeline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.pipeline.runner import PipelineConfig, run_pipeline


def main() -> None:
    """Parse CLI arguments and run or resume the configured pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pipeline YAML config.")
    parser.add_argument("--start-at", default="01_seed_gc", help="Stage number/name to start at.")
    parser.add_argument("--stop-after", default=None, help="Optional stage number/name to stop after.")
    args = parser.parse_args()

    artifacts = run_pipeline(
        PipelineConfig.from_yaml(args.config),
        start_at=args.start_at,
        stop_after=args.stop_after,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
