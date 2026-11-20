#!/usr/bin/env python3
"""Export preliminary marker-neighbor lists from a completed genome FDR run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.analysis.preprocessing import aggregate_duplicate_genes
from gene_analysis.datasets.kutsche import load_and_preprocess_data


def _safe_name(gene: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in gene)


def export_marker_neighbors(
    run_directory: Path,
    expression_file: Path,
    output_directory: Path,
    markers: list[str],
    q_cutoff: float,
    min_count: float,
    min_samples: int,
) -> pd.DataFrame:
    raw = load_and_preprocess_data(expression_file)
    if raw.index.has_duplicates:
        raw = aggregate_duplicate_genes(raw, method="mean")
    wt_columns = [column for column in raw.columns if "WT" in column]
    if min_samples > len(wt_columns):
        raise ValueError("min_samples exceeds the number of WT samples")
    prevalence = (raw[wt_columns] >= min_count).sum(axis=1)
    eligible = set(prevalence.index[prevalence >= min_samples].astype(str))

    output_directory.mkdir(parents=True, exist_ok=True)
    combined = []
    for marker in markers:
        source = (run_directory / "model_comparison" /
                  f"marker_{_safe_name(marker)}_all_pairs.parquet")
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_parquet(source)
        selected = frame[
            (frame["bh_adjusted_p_value"] <= q_cutoff)
            & frame["Genefrom"].isin(eligible)
            & frame["Geneto"].isin(eligible)
        ].copy()
        selected["marker"] = marker
        selected["direction"] = selected["Genefrom"].eq(marker).map(
            {True: "outgoing", False: "incoming"}
        )
        selected["neighbor_gene"] = selected["Geneto"].where(
            selected["Genefrom"].eq(marker), selected["Genefrom"]
        )
        selected["neighbor_samples_at_or_above_min_count"] = (
            selected["neighbor_gene"].map(prevalence).astype(int)
        )
        columns = [
            "marker", "direction", "neighbor_gene", "Genefrom", "Geneto",
            "raw_p_value", "bh_adjusted_p_value", "storey_q_value",
            "coefficient", "standard_error", "statistic", "adjusted_r_squared",
            "aic", "bic", "neighbor_samples_at_or_above_min_count",
        ]
        selected = selected[columns].sort_values(
            ["bh_adjusted_p_value", "raw_p_value", "direction", "neighbor_gene"],
            kind="stable",
        )
        selected.to_csv(output_directory / f"{marker}_neighbors.csv", index=False)
        for direction in ("incoming", "outgoing"):
            selected[selected["direction"] == direction].to_csv(
                output_directory / f"{marker}_{direction}_neighbors.csv", index=False
            )
        combined.append(selected)

    result = pd.concat(combined, ignore_index=True).sort_values(
        ["marker", "bh_adjusted_p_value", "raw_p_value"], kind="stable"
    )
    result.to_csv(output_directory / "all_marker_neighbors.csv", index=False)
    counts = (result.groupby(["marker", "direction"], observed=True).size()
              .rename("neighbors").reset_index())
    counts.to_csv(output_directory / "marker_neighbor_counts.csv", index=False)
    (output_directory / "README.md").write_text(
        "# Preliminary marker-neighbor results\n\n"
        f"These tables contain directed marker connections with BH q <= {q_cutoff:g}. "
        f"Genes were retained when their raw count was at least {min_count:g} in at "
        f"least {min_samples} WT samples. Incoming means neighbor -> marker; outgoing "
        "means marker -> neighbor. Rows are sorted by BH q-value.\n\n"
        "The q-values come from the completed, larger genome-wide correction family. "
        "They are therefore conservative for this expression-filtered interim subset. "
        "Final tables must be regenerated after the filtered genome-wide analysis "
        "recalculates q-values over its complete tested family.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expression-file", type=Path,
                        default=Path("Data/Kutsche/Kutsche_Counts.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--markers", nargs="+", default=["ZEB2", "ZBTB18", "MECP2"])
    parser.add_argument("--q-cutoff", type=float, default=1e-7)
    parser.add_argument("--min-count", type=float, default=10)
    parser.add_argument("--min-samples", type=int, default=7)
    args = parser.parse_args()
    result = export_marker_neighbors(
        args.run_dir, args.expression_file, args.output_dir, args.markers,
        args.q_cutoff, args.min_count, args.min_samples,
    )
    print(f"Exported {len(result)} directed connections to {args.output_dir}")


if __name__ == "__main__":
    main()
