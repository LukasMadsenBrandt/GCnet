#!/usr/bin/env python3
"""
Python 3.9.6 compatible.

Edit the CONFIG section below, then run:
  python degree_quantile_hist_config.py

For each input file, creates:
  <OUTDIR>/<file_stem>/
    - edges_<file_stem>_qXX.csv         (raw filtered edges)
    - degrees_<file_stem>_qXX.csv       (degree table)
    - degrees_<file_stem>_qXX.html      (interactive stacked bar chart)
    - summary_<file_stem>_qXX.json      (metadata + stats)
"""

import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from gene_analysis.io.paths import resolve_existing_path, results_path


# =========================
# CONFIG (edit these)
# =========================

INPUT_FILES = [
    Path("benito_human_granger_results_p1_5633483of5647752.csv"),
    Path("benito_gorilla_granger_results_p1_5555398of5581406.csv"),
]

# Lower quantile x used as p-value cutoff (0 < x <= 1).
# Example: 0.05 means "keep the lowest 5% p-values" (per file)
QUANTILE_X = 0.05

# Root output directory
OUTDIR = results_path("degree_outputs", "quantile_5_percent")

# If not None: only plot top N genes by total degree (still saves full CSV tables)
PLOT_TOP_N = None  # e.g. 200

# If True: include genes from the ENTIRE input file in the output degree table
# (genes with 0 degree after threshold will appear)
INCLUDE_ALL_GENES_FROM_INPUT = True

# Plot readability: hide gene names on x-axis (recommended for large graphs)
HIDE_X_AXIS_GENE_LABELS = True

# If hiding x labels, keep a sane number of ticks if you still want some labels:
# Set to None to let plotly decide; set e.g. 50 to show ~50 ticks.
MAX_X_TICKS = None

# Summary: include top K genes (by total degree) in the per-file JSON
SUMMARY_TOP_K_GENES = 10

# =========================


def load_edges(csv_path: Path) -> pd.DataFrame:
    """Load and normalize a Granger edge CSV for degree analysis."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "gene1" not in df.columns or "gene2" not in df.columns:
        raise ValueError("CSV must contain columns: gene1, gene2 (header required).")

    pcol = None
    for cand in ["p-value", "p_value", "pval", "p-value "]:
        if cand in df.columns:
            pcol = cand
            break
    if pcol is None:
        raise ValueError("CSV must contain a p-value column named 'p-value' (or 'p_value').")

    df["gene1"] = df["gene1"].astype(str).str.strip()
    df["gene2"] = df["gene2"].astype(str).str.strip()

    df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
    df = df.dropna(subset=[pcol]).rename(columns={pcol: "p_value"})

    if "lag" in df.columns:
        df["lag"] = pd.to_numeric(df["lag"], errors="coerce")

    return df


def threshold_by_quantile(df: pd.DataFrame, quantile_x: float) -> float:
    """Return the p-value cutoff at the requested lower quantile."""
    qx = float(quantile_x)
    if not (0.0 < qx <= 1.0):
        raise ValueError("QUANTILE_X must be in (0, 1]. Got: %r" % (quantile_x,))
    if df.empty:
        raise ValueError("Input dataframe is empty after parsing.")
    return float(df["p_value"].quantile(qx))


def compute_degrees(df_selected_edges: pd.DataFrame, all_genes) -> pd.DataFrame:
    """Compute in-, out-, and total degree for each gene."""
    # out-degree: count occurrences as gene1 (source)
    out_deg = df_selected_edges.groupby("gene1").size()
    out_deg.name = "out_degree"

    # in-degree: count occurrences as gene2 (target)
    in_deg = df_selected_edges.groupby("gene2").size()
    in_deg.name = "in_degree"

    deg = pd.DataFrame(index=all_genes)
    deg = deg.join(in_deg, how="left").join(out_deg, how="left").fillna(0)

    deg["in_degree"] = deg["in_degree"].astype(int)
    deg["out_degree"] = deg["out_degree"].astype(int)
    deg["total_degree"] = deg["in_degree"] + deg["out_degree"]

    deg = deg.reset_index().rename(columns={"index": "gene"})
    deg = deg.sort_values(["total_degree", "gene"], ascending=[False, True]).reset_index(drop=True)
    return deg


def make_interactive_plot(deg_df: pd.DataFrame, title: str, top_n, out_html: Path) -> None:
    """Write an interactive stacked degree bar chart as HTML."""
    try:
        import plotly.express as px
    except ImportError:
        raise SystemExit("Missing dependency: plotly\n\nInstall with:\n  pip install plotly\n")

    plot_df = deg_df.copy()
    if top_n is not None and int(top_n) > 0:
        plot_df = plot_df.head(int(top_n))

    # Long form for stacked bars
    long_df = plot_df.melt(
        id_vars=["gene", "total_degree"],
        value_vars=["in_degree", "out_degree"],
        var_name="degree_type",
        value_name="degree",
    )

    long_df["degree_type"] = long_df["degree_type"].map(
        {"in_degree": "In-degree", "out_degree": "Out-degree"}
    )

    fig = px.bar(
        long_df,
        x="gene",
        y="degree",
        color="degree_type",
        barmode="stack",
        title=title,
        hover_data={"total_degree": True, "degree": True, "degree_type": True, "gene": True},
    )

    fig.update_layout(
        xaxis_title="Gene",
        yaxis_title="Degree",
        legend_title="",
        margin=dict(l=40, r=20, t=60, b=80),
    )

    # Readability tweaks for big plots
    if HIDE_X_AXIS_GENE_LABELS:
        fig.update_xaxes(showticklabels=False)
    else:
        fig.update_xaxes(tickangle=-45)

    if MAX_X_TICKS is not None:
        fig.update_xaxes(nticks=int(MAX_X_TICKS))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main() -> None:
    """Run configured degree-threshold summaries and figure exports."""
    OUTDIR.mkdir(parents=True, exist_ok=True)

    qtag = "q%02d" % int(round(float(QUANTILE_X) * 100))
    run_ts = datetime.now().isoformat(timespec="seconds")

    for f in INPUT_FILES:
        f = resolve_existing_path(f)
        df = load_edges(f)
        cutoff = threshold_by_quantile(df, QUANTILE_X)

        df_sel = df[df["p_value"] <= cutoff].copy()
        if df_sel.empty:
            raise SystemExit(
                "No edges selected.\n"
                "quantile=%s -> cutoff=%g\n"
                "Try increasing QUANTILE_X (e.g. 0.1, 0.2) or check input p-values."
                % (QUANTILE_X, cutoff)
            )

        # Degrees should be computed over either:
        # - all genes in the input file (include zeros) OR
        # - only genes present in selected edges
        if INCLUDE_ALL_GENES_FROM_INPUT:
            all_genes = sorted(set(df["gene1"]).union(set(df["gene2"])))
        else:
            all_genes = sorted(set(df_sel["gene1"]).union(set(df_sel["gene2"])))

        deg = compute_degrees(df_sel, all_genes)

        # Per-file output folder
        stem = f.stem
        file_outdir = OUTDIR / stem
        file_outdir.mkdir(parents=True, exist_ok=True)

        # Outputs
        edges_csv = file_outdir / ("edges_%s.csv" % qtag)
        degrees_csv = file_outdir / ("degrees_%s.csv" % qtag)
        plot_html = file_outdir / ("degrees_%s.html" % qtag)
        summary_json = file_outdir / ("summary_%s.json" % qtag)


        # Write raw selected edges + degree table
        df_sel.to_csv(edges_csv, index=False)
        deg.to_csv(degrees_csv, index=False)

        # Plot
        title = "Gene degrees (p <= lower %.3g-quantile cutoff=%.6g) — %s" % (QUANTILE_X, cutoff, stem)
        make_interactive_plot(deg, title, PLOT_TOP_N, plot_html)

        # Summary JSON (per file)
        unique_genes_total = int(len(set(df["gene1"]).union(set(df["gene2"]))))
        unique_genes_selected = int(len(set(df_sel["gene1"]).union(set(df_sel["gene2"]))))

        topk = deg.head(int(SUMMARY_TOP_K_GENES))
        topk_list = [
            {
                "gene": row["gene"],
                "in_degree": int(row["in_degree"]),
                "out_degree": int(row["out_degree"]),
                "total_degree": int(row["total_degree"]),
            }
            for _, row in topk.iterrows()
        ]

        summary = {
            "run": {
                "timestamp": run_ts,
                "quantile_x": float(QUANTILE_X),
                "quantile_tag": qtag,
                "include_all_genes_from_input": bool(INCLUDE_ALL_GENES_FROM_INPUT),
                "plot_top_n": PLOT_TOP_N,
                "hide_x_axis_gene_labels": bool(HIDE_X_AXIS_GENE_LABELS),
                "max_x_ticks": MAX_X_TICKS,
            },
            "input": {
                "file": str(f),
                "file_stem": stem,
            },
            "selection": {
                "cutoff": float(cutoff),
                "rows_total": int(len(df)),
                "rows_selected": int(len(df_sel)),
                "selected_fraction": float(len(df_sel)) / float(len(df)) if len(df) else 0.0,
            },
            "counts": {
                "unique_genes_total": unique_genes_total,
                "unique_genes_selected": unique_genes_selected,
                "unique_sources_selected": int(df_sel["gene1"].nunique()),
                "unique_targets_selected": int(df_sel["gene2"].nunique()),
            },
            "degree_stats": {
                "max_total_degree": int(deg["total_degree"].max()) if len(deg) else 0,
                "top_genes": topk_list,
            },
            "outputs": {
                "edges_csv": str(edges_csv),
                "degrees_csv": str(degrees_csv),
                "plot_html": str(plot_html),
            },
        }

        with open(summary_json, "w") as fp:
            json.dump(summary, fp, indent=2)

        print("[OK] Input:     %s" % f)
        print("[OK] Cutoff:    %.10g" % cutoff)
        print("[OK] Selected edges: %d / %d" % (len(df_sel), len(df)))
        print("[OK] Folder:    %s" % file_outdir)
        print("[OK] Wrote edges:   %s" % edges_csv)
        print("[OK] Wrote degrees: %s" % degrees_csv)
        print("[OK] Wrote plot:    %s" % plot_html)
        print("[OK] Wrote summary: %s" % summary_json)
        print()

if __name__ == "__main__":
    main()
