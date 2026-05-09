#!/usr/bin/env python3
"""Run GC among curated neurodeficiency seed genes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.pipeline.seed_gc import run
from gene_analysis.pipeline.config import SeedGeneConfig

__all__ = ["SeedGeneConfig", "run"]


if __name__ == "__main__":
    raise SystemExit(
        "Seed GC needs a preprocessed expression dataframe. Import "
        "`gene_analysis.pipeline.seed_gc.run` from a dataset-specific runner."
    )
