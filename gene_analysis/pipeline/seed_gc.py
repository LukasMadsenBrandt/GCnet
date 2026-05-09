"""Seed-stage Granger execution for curated genes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gene_analysis.analysis.granger import perform_granger_causality_tests, save_results_to_csv
from gene_analysis.pipeline.config import SeedGeneConfig


def run(config: SeedGeneConfig, expression_df: Any | None = None) -> Path:
    """
    Run Granger causality among curated seed genes.

    Pass `expression_df` from the dataset preprocessing layer. The cleanup keeps
    heavy dataset loading outside this function so tests and wrappers stay fast.
    """
    config.validate()
    if expression_df is None:
        raise ValueError("expression_df is required for seed GC execution.")
    results = perform_granger_causality_tests(
        expression_df,
        genes_file=config.seed_gene_file,
        progress=False,
    )
    save_results_to_csv(results, config.output_file)
    return Path(config.output_file)
