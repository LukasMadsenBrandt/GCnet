#!/usr/bin/env python3
"""Diagnose adjusted-R2 ties at one caused by machine-precision residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import numpy as np
import pandas as pd

from gene_analysis.pipeline.fdr import load_samples
from gene_analysis.pipeline.runner import PipelineConfig


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def genome_directory(path):
    path = Path(path).resolve()
    return path / "genome_fdr" if (path / "genome_fdr").is_dir() else path


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    levels = (0, .01, .1, .5, .9, .99, 1)
    return {str(level): float(value) for level, value in
            zip(levels, np.quantile(values, levels))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Model configuration used to prepare expression data.")
    parser.add_argument("--comparison", type=Path, required=True,
                        help="BH or Storey comparison output directory.")
    parser.add_argument("--model-run", type=Path, required=True,
                        help="Completed model0 run or genome_fdr directory.")
    parser.add_argument("--method", choices=("bh", "storey"), default="storey")
    args = parser.parse_args()

    comparison = args.comparison.resolve()
    model = genome_directory(args.model_run)
    common = comparison / "common_significant_pairs.parquet"
    significant = model / "model_comparison" / f"significant_pairs_{args.method}.parquet"
    genes_path = model / "genes.csv"
    query = f"""
        SELECT c.pair_id, c.source_id, c.target_id, c.Genefrom, c.Geneto,
               m.coefficient, m.standard_error, m.statistic, m.raw_p_value,
               m.residual_sum_squares, g.target_sum_squares
        FROM read_parquet({quote(common)}) c
        JOIN read_parquet({quote(significant)}) m USING (pair_id)
        JOIN read_csv_auto({quote(genes_path)}, header=true) g
          ON g.gene_id = c.target_id
        WHERE c.best_adjusted_r_squared = 'tie'
          AND c.model0_adjusted_r_squared = 1.0
          AND c.model1_adjusted_r_squared = 1.0
          AND c.model2_adjusted_r_squared = 1.0
        ORDER BY c.pair_id
    """
    frame = duckdb.sql(query).df()
    if frame.empty:
        raise ValueError("No three-model adjusted-R2 ties at one were found.")

    config = PipelineConfig.from_yaml(args.config)
    values, _, genes, _, _ = load_samples(config)
    metadata = pd.read_csv(genes_path)
    if genes != metadata.gene.astype(str).tolist():
        raise ValueError("Prepared expression genes do not match the completed run.")
    source = frame.source_id.to_numpy(dtype=np.int64)
    target = frame.target_id.to_numpy(dtype=np.int64)
    source_values, target_values = values[:, source], values[:, target]
    difference = source_values - target_values
    frame["expression_profiles_identical"] = np.all(difference == 0, axis=0)
    frame["expression_profiles_close_1e_12"] = np.all(
        np.abs(difference) <= 1e-12 + 1e-12*np.abs(target_values), axis=0)
    frame["source_nonzero_samples"] = np.count_nonzero(source_values, axis=0)
    frame["target_nonzero_samples"] = np.count_nonzero(target_values, axis=0)
    frame["rss_over_target_tss"] = (
        frame.residual_sum_squares.to_numpy() / frame.target_sum_squares.to_numpy())

    ratio = frame.rss_over_target_tss.to_numpy()
    report = {
        "ties": len(frame),
        "unique_source_genes": int(frame.source_id.nunique()),
        "unique_target_genes": int(frame.target_id.nunique()),
        "identical_expression_profiles": int(frame.expression_profiles_identical.sum()),
        "near_identical_expression_profiles_1e_12": int(
            frame.expression_profiles_close_1e_12.sum()),
        "rss_over_target_tss_at_or_below": {
            str(level): int(np.sum(ratio <= level)) for level in (1e-12, 1e-14, 1e-16, 1e-18)
        },
        "rss_over_target_tss_quantiles": quantiles(ratio),
        "absolute_t_statistic_quantiles": quantiles(np.abs(frame.statistic.to_numpy())),
        "raw_p_value_quantiles": quantiles(frame.raw_p_value.to_numpy()),
        "standard_error_quantiles": quantiles(frame.standard_error.to_numpy()),
        "source_nonzero_sample_quantiles": quantiles(frame.source_nonzero_samples.to_numpy()),
        "target_nonzero_sample_quantiles": quantiles(frame.target_nonzero_samples.to_numpy()),
    }
    output_json = comparison / "perfect_fit_ties_diagnostic.json"
    output_csv = comparison / "perfect_fit_ties_diagnostic.csv"
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    frame.to_csv(output_csv, index=False)
    print(json.dumps(report, indent=2))
    print(f"rows: {output_csv}")


if __name__ == "__main__":
    main()
