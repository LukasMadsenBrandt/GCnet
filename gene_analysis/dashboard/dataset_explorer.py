"""Responsive dashboard and metrics for expression and all-pairs seed GC exploration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_QUANTILES = (0.001, 0.005, 0.01, 0.025, 0.05, 0.10)


def load_expression_csv(path: str | Path) -> pd.DataFrame:
    """Load one preprocessing artifact using its required Gene index column."""
    frame = pd.read_csv(path)
    if "Gene" not in frame.columns:
        raise ValueError(f"{path}: expression CSV must contain a 'Gene' column.")
    frame = frame.set_index("Gene")
    return frame.apply(pd.to_numeric, errors="raise")


def load_gc_csv(path: str | Path) -> pd.DataFrame:
    """Load a canonical GC CSV and coerce its p-values to numeric values."""
    frame = pd.read_csv(path)
    required = {"gene1", "gene2", "lag", "p-value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing GC columns: {', '.join(sorted(missing))}.")
    frame = frame.copy()
    frame["p-value"] = pd.to_numeric(frame["p-value"], errors="coerce")
    return frame


def quantile_relationship_metrics(
    gc_frame: pd.DataFrame,
    gene_of_interest: str,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Count GOI relationships at lower-tail quantiles of all valid GC p-values."""
    valid = gc_frame.loc[gc_frame["p-value"].notna()].copy()
    rows = []
    for quantile in sorted({float(value) for value in quantiles}):
        if not 0 < quantile <= 1:
            raise ValueError("Quantiles must be in (0, 1].")
        threshold = float(valid["p-value"].quantile(quantile)) if not valid.empty else float("nan")
        selected = valid.loc[valid["p-value"] <= threshold]
        outgoing = selected.loc[selected["gene1"] == gene_of_interest]
        incoming = selected.loc[selected["gene2"] == gene_of_interest]
        related = set(outgoing["gene2"].astype(str)) | set(incoming["gene1"].astype(str))
        rows.append(
            {
                "Quantile": quantile,
                "P-value threshold": threshold,
                "All directed relationships": len(selected),
                "Outgoing": len(outgoing),
                "Incoming": len(incoming),
                "Directed relationships": len(outgoing) + len(incoming),
                "Unique related genes": len(related),
                "GOI share (%)": (
                    (len(outgoing) + len(incoming)) / len(selected) * 100.0 if len(selected) else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def pairwise_comparison(gc_frame: pd.DataFrame, gene_of_interest: str, genes: Iterable[str]) -> pd.DataFrame:
    """Return both directed GC results between a GOI and selected comparison genes."""
    comparisons = set(map(str, genes)) - {str(gene_of_interest)}
    if not comparisons:
        return pd.DataFrame(columns=["Source", "Target", "Lag", "P-value"])
    mask = (
        (gc_frame["gene1"].astype(str).eq(str(gene_of_interest)) & gc_frame["gene2"].astype(str).isin(comparisons))
        | (gc_frame["gene2"].astype(str).eq(str(gene_of_interest)) & gc_frame["gene1"].astype(str).isin(comparisons))
    )
    result = gc_frame.loc[mask, ["gene1", "gene2", "lag", "p-value"]].copy()
    result.columns = ["Source", "Target", "Lag", "P-value"]
    return result.sort_values(["P-value", "Source", "Target"], na_position="last")


def gc_coverage_summary(gc_frame: pd.DataFrame, manifest_path: str | Path | None = None) -> dict[str, int | float | bool]:
    """Describe whether the supplied result CSV appears to contain all attempted rows."""
    written = len(gc_frame)
    attempted = written
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        attempted = int((manifest.get("metrics") or {}).get("gc_pairs_total") or written)
    return {
        "rows_written": written,
        "pairs_attempted": attempted,
        "coverage_fraction": (written / attempted) if attempted else 0.0,
        "appears_complete": written == attempted,
    }


def create_dashboard(
    summarized: pd.DataFrame,
    gc_frame: pd.DataFrame,
    *,
    replicates: pd.DataFrame | None = None,
    default_gene: str | None = None,
    coverage: dict[str, int | float | bool] | None = None,
):
    """Create the responsive Dash application without starting its web server."""
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, dash_table, dcc, html

    genes = sorted(map(str, summarized.index.unique()))
    if not genes:
        raise ValueError("The summarized expression matrix has no genes.")
    selected_default = default_gene if default_gene in genes else genes[0]
    coverage = coverage or gc_coverage_summary(gc_frame)
    coverage_text = (
        f"GC result coverage: {coverage['rows_written']:,} of {coverage['pairs_attempted']:,} attempted pairs."
    )
    coverage_color = "success" if coverage["appears_complete"] else "warning"

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Gene dataset explorer")
    app.layout = dbc.Container(
        fluid=True,
        className="py-3 px-3 px-lg-4",
        children=[
            html.H2("Gene expression and Granger-causality explorer"),
            dbc.Alert(coverage_text, color=coverage_color, className="mt-3"),
            dbc.Row(
                className="g-3 mb-2",
                children=[
                    dbc.Col(dbc.Card(dbc.CardBody([html.Small("Genes"), html.H4(f"{len(genes):,}")])), sm=4),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [html.Small("Summarized timepoints / conditions"), html.H4(f"{len(summarized.columns):,}")]
                            )
                        ),
                        sm=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [html.Small("Valid GC p-values"), html.H4(f"{gc_frame['p-value'].notna().sum():,}")]
                            )
                        ),
                        sm=4,
                    ),
                ],
            ),
            dbc.Row(
                className="g-3",
                children=[
                    dbc.Col(
                        md=4,
                        children=[
                            dbc.Label("Gene of interest"),
                            dcc.Dropdown(
                                id="explorer-goi",
                                options=[{"label": gene, "value": gene} for gene in genes],
                                value=selected_default,
                                clearable=False,
                            ),
                        ],
                    ),
                    dbc.Col(
                        md=5,
                        children=[
                            dbc.Label("Genes to compare"),
                            dcc.Dropdown(
                                id="explorer-comparisons",
                                options=[{"label": gene, "value": gene} for gene in genes],
                                value=[],
                                multi=True,
                                placeholder="Select one or more genes",
                            ),
                        ],
                    ),
                    dbc.Col(
                        md=3,
                        children=[
                            dbc.Label("Lower p-value quantiles"),
                            dcc.Dropdown(
                                id="explorer-quantiles",
                                options=[
                                    {"label": f"{value:g} ({value * 100:g}%)", "value": value}
                                    for value in DEFAULT_QUANTILES
                                ],
                                value=list(DEFAULT_QUANTILES),
                                multi=True,
                                clearable=False,
                            ),
                        ],
                    ),
                ],
            ),
            dbc.Row(
                className="g-3 mt-1",
                children=[
                    dbc.Col(dcc.Graph(id="explorer-summary-plot"), lg=7),
                    dbc.Col(dcc.Graph(id="explorer-replicate-plot"), lg=5),
                ],
            ),
            dcc.Graph(id="explorer-pvalue-distribution"),
            html.H4("Relationships across p-value quantiles", className="mt-3"),
            dash_table.DataTable(
                id="explorer-quantile-table",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "right", "padding": "0.5rem"},
                style_header={"fontWeight": "bold"},
            ),
            html.H4("Selected directed pair results", className="mt-4"),
            dash_table.DataTable(
                id="explorer-pair-table",
                page_size=12,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "right", "padding": "0.5rem"},
                style_header={"fontWeight": "bold"},
            ),
        ],
    )

    @app.callback(
        Output("explorer-summary-plot", "figure"),
        Output("explorer-replicate-plot", "figure"),
        Output("explorer-pvalue-distribution", "figure"),
        Output("explorer-quantile-table", "data"),
        Output("explorer-quantile-table", "columns"),
        Output("explorer-pair-table", "data"),
        Output("explorer-pair-table", "columns"),
        Input("explorer-goi", "value"),
        Input("explorer-comparisons", "value"),
        Input("explorer-quantiles", "value"),
    )
    def update_dashboard(gene_of_interest, comparison_genes, quantiles):
        selected = [gene_of_interest, *(comparison_genes or [])]
        selected = list(dict.fromkeys(gene for gene in selected if gene in summarized.index))

        summary_figure = go.Figure()
        for gene in selected:
            summary_figure.add_scatter(
                x=list(map(str, summarized.columns)),
                y=summarized.loc[gene].to_numpy(),
                mode="lines+markers",
                name=gene,
            )
        summary_figure.update_layout(
            title="Summarized expression supplied to Granger causality",
            xaxis_title="Ordered timepoint / condition",
            yaxis_title="Expression",
            margin={"l": 50, "r": 20, "t": 60, "b": 50},
        )

        replicate_figure = go.Figure()
        if replicates is not None:
            for gene in selected:
                if gene not in replicates.index:
                    continue
                values = replicates.loc[[gene]]
                for row_number, (_label, row) in enumerate(values.iterrows(), start=1):
                    suffix = f" replicate row {row_number}" if len(values) > 1 else ""
                    replicate_figure.add_scatter(
                        x=list(map(str, replicates.columns)),
                        y=row.to_numpy(),
                        mode="markers+lines",
                        name=f"{gene}{suffix}",
                    )
        replicate_figure.update_layout(
            title="Pre-aggregation replicate-level expression",
            xaxis_title="Replicate / sample column",
            yaxis_title="Expression",
            margin={"l": 50, "r": 20, "t": 60, "b": 50},
        )

        metrics = quantile_relationship_metrics(gc_frame, gene_of_interest, quantiles or DEFAULT_QUANTILES)
        distribution_figure = go.Figure()
        valid_gc = gc_frame.loc[gc_frame["p-value"].notna()].copy()
        valid_gc["-log10(p)"] = -np.log10(valid_gc["p-value"].clip(lower=1e-300))
        goi_gc = valid_gc.loc[
            valid_gc["gene1"].astype(str).eq(str(gene_of_interest))
            | valid_gc["gene2"].astype(str).eq(str(gene_of_interest))
        ]
        distribution_figure.add_histogram(
            x=valid_gc["-log10(p)"],
            name="All valid directed pairs",
            opacity=0.65,
            nbinsx=60,
        )
        distribution_figure.add_histogram(
            x=goi_gc["-log10(p)"],
            name=f"Pairs involving {gene_of_interest}",
            opacity=0.75,
            nbinsx=60,
        )
        for threshold in metrics["P-value threshold"].dropna().unique():
            distribution_figure.add_vline(
                x=-np.log10(max(float(threshold), 1e-300)),
                line_dash="dot",
                line_color="#555",
                opacity=0.35,
            )
        distribution_figure.update_layout(
            title="P-value distribution and selected lower-tail quantiles",
            xaxis_title="-log10(p-value); farther right is stronger evidence",
            yaxis_title="Directed pair count",
            barmode="overlay",
            margin={"l": 50, "r": 20, "t": 60, "b": 50},
        )
        metrics["Quantile"] = metrics["Quantile"].map(lambda value: f"{value:g}")
        metrics["P-value threshold"] = metrics["P-value threshold"].map(lambda value: f"{value:.8g}")
        metrics["GOI share (%)"] = metrics["GOI share (%)"].map(lambda value: f"{value:.2f}")
        pairs = pairwise_comparison(gc_frame, gene_of_interest, comparison_genes or [])
        if not pairs.empty:
            pairs["P-value"] = pairs["P-value"].map(lambda value: "" if pd.isna(value) else f"{value:.8g}")
        metric_records = metrics.to_dict("records")
        pair_records = pairs.to_dict("records")
        return (
            summary_figure,
            replicate_figure,
            distribution_figure,
            metric_records,
            [{"name": column, "id": column} for column in metrics.columns],
            pair_records,
            [{"name": column, "id": column} for column in pairs.columns],
        )

    return app
