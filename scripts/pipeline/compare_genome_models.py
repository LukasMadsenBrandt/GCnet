#!/usr/bin/env python3
"""Compare three completed genome-FDR model runs at their automatic cutoffs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import numpy as np
import pandas as pd


MODELS = ("model0", "model1", "model2")


def _quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _run_directory(path):
    path = Path(path).resolve()
    return path / "genome_fdr" if (path / "genome_fdr").is_dir() else path


def _load_run(path, expected_model, method):
    root = _run_directory(path)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    timing = json.loads((root / "timing.json").read_text())
    if manifest.get("interaction_model") != expected_model:
        raise ValueError(f"{root} is {manifest.get('interaction_model')}, expected {expected_model}.")
    significant = root / "model_comparison" / f"significant_pairs_{method}.parquet"
    if not significant.is_file():
        raise ValueError(f"Missing comparison export: {significant}")
    genes = pd.read_csv(root / "genes.csv")
    return dict(root=root, manifest=manifest, summary=summary, selection=selection, timing=timing,
                significant=significant, genes=genes)


def _best_count_table(connection, relation):
    rows = []
    total = connection.execute(f"SELECT count(*) FROM {relation}").fetchone()[0]
    for model in (*MODELS, "tie"):
        rows.append(dict(
            model=model,
            common_pairs=total,
            best_aic=connection.execute(
                f"SELECT count(*) FROM {relation} WHERE best_aic=?", [model]).fetchone()[0],
            best_bic=connection.execute(
                f"SELECT count(*) FROM {relation} WHERE best_bic=?", [model]).fetchone()[0],
            best_adjusted_r_squared=connection.execute(
                f"SELECT count(*) FROM {relation} WHERE best_adjusted_r_squared=?", [model]).fetchone()[0],
        ))
    return pd.DataFrame(rows)


def compare(paths, method, output):
    if method not in {"bh", "storey"}:
        raise ValueError("method must be bh or storey")
    runs = {model: _load_run(paths[model], model, method) for model in MODELS}
    reference = runs["model0"]
    for model in MODELS[1:]:
        current = runs[model]
        for key in ("tests", "samples", "family"):
            if current["manifest"].get(key) != reference["manifest"].get(key):
                raise ValueError(f"Runs differ in manifest field {key}.")
        if not current["genes"].gene.equals(reference["genes"].gene):
            raise ValueError("Runs do not contain the same ordered gene universe.")

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    database = output / "comparison.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("SET preserve_insertion_order=false")
        for model in MODELS:
            connection.execute(
                f"CREATE OR REPLACE VIEW {model} AS SELECT * FROM read_parquet({_quote(runs[model]['significant'])})")
        select = """
            SELECT m0.pair_id, m0.source_id, m0.target_id, m0.Genefrom, m0.Geneto,
                   m0.source_is_reference_seed, m0.target_is_reference_seed,
                   m0.aic AS model0_aic, m1.aic AS model1_aic, m2.aic AS model2_aic,
                   m0.bic AS model0_bic, m1.bic AS model1_bic, m2.bic AS model2_bic,
                   m0.adjusted_r_squared AS model0_adjusted_r_squared,
                   m1.adjusted_r_squared AS model1_adjusted_r_squared,
                   m2.adjusted_r_squared AS model2_adjusted_r_squared,
                   CASE WHEN m0.aic < m1.aic AND m0.aic < m2.aic THEN 'model0'
                        WHEN m1.aic < m0.aic AND m1.aic < m2.aic THEN 'model1'
                        WHEN m2.aic < m0.aic AND m2.aic < m1.aic THEN 'model2' ELSE 'tie' END AS best_aic,
                   CASE WHEN m0.bic < m1.bic AND m0.bic < m2.bic THEN 'model0'
                        WHEN m1.bic < m0.bic AND m1.bic < m2.bic THEN 'model1'
                        WHEN m2.bic < m0.bic AND m2.bic < m1.bic THEN 'model2' ELSE 'tie' END AS best_bic,
                   CASE WHEN m0.adjusted_r_squared > m1.adjusted_r_squared AND m0.adjusted_r_squared > m2.adjusted_r_squared THEN 'model0'
                        WHEN m1.adjusted_r_squared > m0.adjusted_r_squared AND m1.adjusted_r_squared > m2.adjusted_r_squared THEN 'model1'
                        WHEN m2.adjusted_r_squared > m0.adjusted_r_squared AND m2.adjusted_r_squared > m1.adjusted_r_squared THEN 'model2'
                        ELSE 'tie' END AS best_adjusted_r_squared
            FROM model0 m0 JOIN model1 m1 USING (pair_id) JOIN model2 m2 USING (pair_id)
        """
        connection.execute("CREATE OR REPLACE TABLE common_pairs AS " + select)
        connection.execute(
            f"COPY common_pairs TO {_quote(output/'common_significant_pairs.parquet')} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)")
        connection.execute("CREATE OR REPLACE VIEW common_seed_pairs AS SELECT * FROM common_pairs "
                           "WHERE source_is_reference_seed AND target_is_reference_seed")
        connection.execute(
            f"COPY common_seed_pairs TO {_quote(output/'common_significant_seed_pairs.parquet')} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)")
        all_counts = _best_count_table(connection, "common_pairs")
        seed_counts = _best_count_table(connection, "common_seed_pairs")
    database.unlink()
    all_counts.to_csv(output / "best_model_counts_all_common_pairs.csv", index=False)
    seed_counts.to_csv(output / "best_model_counts_common_seed_pairs.csv", index=False)

    model_rows = []
    for model, run in runs.items():
        selection = run["selection"]["methods"][method]
        cutoff = selection["cutoff"]
        graph = pd.read_csv(run["root"] / "anchor_cutoff_comparison.csv")
        match = graph[(graph.method == method) & np.isclose(graph.cutoff, cutoff, rtol=1e-12, atol=0)]
        if match.empty:
            raise ValueError(f"Automatic cutoff {cutoff} is absent from {model}'s graph report.")
        row = match.iloc[0]
        timing = run["timing"]
        model_rows.append(dict(
            model=model, equation=run["manifest"]["model"], parameters=run["manifest"]["model_parameters"],
            adjustment_method=method, automatic_cutoff=cutoff,
            significant_directed_pairs=selection["selected_directed_connections"],
            cutoff_times_pairs=selection["cutoff_times_selected"],
            largest_component_nodes=int(row.largest_component_nodes),
            reference_seed_genes=int(row.reference_seed_genes),
            reference_seed_genes_in_largest_component=int(row.anchor_component_seed_genes),
            failed_tests=run["summary"]["failed_tests"],
            planned_tests=run["summary"]["tests"],
            active_runtime_seconds=timing["active_runtime_seconds"],
            calendar_span_seconds=timing["calendar_span_seconds"],
            preprocessing_seconds=timing["stages"].get("preprocessing", 0),
            fit_seconds=timing["stages"].get("fit", 0),
            sort_seconds=timing["stages"].get("sort", 0),
            adjustment_seconds=timing["stages"].get("adjust", 0),
            selection_seconds=timing["stages"].get("selection", 0),
            report_seconds=timing["stages"].get("report", 0),
            export_seconds=timing["stages"].get("export", 0),
        ))
    pd.DataFrame(model_rows).to_csv(output / "model_level_comparison.csv", index=False)
    (output / "README.md").write_text(
        "# Genome model comparison\n\n"
        f"Adjustment used for automatic selection: {method}. Each model uses its own largest complete tied "
        "q-value block satisfying q times selected directed pairs < 1. Pair-level fit criteria are compared "
        "only on directed pairs significant in all three models. Lower AIC/BIC is better; higher adjusted R2 "
        "is better. Exact ties are reported separately. The seed comparison is a strict subset requiring both "
        "endpoints to belong to the retained reference list.\n",
        encoding="utf-8",
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for model in MODELS:
        parser.add_argument(f"--{model}", required=True, help=f"Completed {model} run or genome_fdr directory")
    parser.add_argument("--method", choices=["bh", "storey"], default="storey")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {model: getattr(args, model) for model in MODELS}
    print(compare(paths, args.method, args.output))


if __name__ == "__main__":
    main()
