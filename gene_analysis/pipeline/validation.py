"""Input and output validation helpers for the canonical pipeline."""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import pandas as pd

from gene_analysis.io.paths import resolve_existing_path
from gene_analysis.pipeline.seed_genes import load_seed_genes


REQUIRED_GC_COLUMNS = ("gene1", "gene2", "lag", "p-value")
REQUIRED_FREQUENCY_COLUMNS = ("Gene", "Coassociation Frequency")


def validate_gene_list_file(
    path: str | Path,
    *,
    label: str = "gene list",
    min_genes: int = 1,
    required_gene: str | None = None,
) -> list[str]:
    """Validate a one-gene-per-line file and return its de-duplicated genes."""
    genes = load_seed_genes(path)
    resolved = resolve_existing_path(path)
    if len(genes) < min_genes:
        raise ValueError(f"{label} {resolved} must contain at least {min_genes} unique genes.")
    if required_gene and required_gene not in genes:
        raise ValueError(f"{label} {resolved} must include gene_of_interest '{required_gene}'.")
    return genes


def validate_expression_dataframe(
    df: pd.DataFrame,
    *,
    gene_of_interest: str,
    seed_genes: Sequence[str],
    transform: str = "none",
    min_timepoints: int = 5,
    min_present_genes: int = 2,
) -> None:
    """Validate an expression matrix before statistical GC stages use it."""
    if df.empty:
        raise ValueError("Expression matrix is empty.")
    if df.index.has_duplicates:
        duplicates = sorted(map(str, df.index[df.index.duplicated()].unique()))
        raise ValueError(f"Expression matrix contains duplicate gene names: {', '.join(duplicates[:5])}.")
    if any(str(gene).strip() == "" for gene in df.index):
        raise ValueError("Expression matrix contains empty gene names.")
    if len(df.columns) < min_timepoints:
        raise ValueError(f"Expression matrix must contain at least {min_timepoints} timepoint/condition columns.")
    numeric = df.apply(pd.to_numeric, errors="raise")
    if gene_of_interest not in numeric.index:
        raise ValueError(f"gene_of_interest '{gene_of_interest}' was not found in the expression matrix.")
    present_seed_genes = [gene for gene in seed_genes if gene in numeric.index]
    if len(present_seed_genes) < min_present_genes:
        raise ValueError(
            f"At least {min_present_genes} configured genes must be present in the expression matrix for this stage."
        )
    transform = str(transform).lower()
    if transform in {"log1p", "log+1", "sqrt"} and (numeric < 0).any().any():
        raise ValueError(f"preprocessing.transform '{transform}' cannot be applied to negative expression values.")


def validate_gc_csv(path: str | Path, *, expected_pairs: int | None = None) -> Path:
    """Validate a Granger-causality CSV schema and common invariants."""
    resolved = resolve_existing_path(path)
    with open(resolved, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        _require_columns(resolved, reader.fieldnames, REQUIRED_GC_COLUMNS)
        rows = list(reader)
    if expected_pairs is not None and len(rows) != expected_pairs:
        raise ValueError(f"{resolved}: expected {expected_pairs} GC rows, found {len(rows)}.")
    for line_number, row in enumerate(rows, start=2):
        gene1 = (row.get("gene1") or "").strip()
        gene2 = (row.get("gene2") or "").strip()
        if not gene1 or not gene2:
            raise ValueError(f"{resolved}:{line_number}: gene1 and gene2 are required.")
        if gene1 == gene2:
            raise ValueError(f"{resolved}:{line_number}: self-pairs are not allowed.")
        _validate_float_or_nan(row.get("lag"), resolved, line_number, "lag")
        _validate_float_or_nan(row.get("p-value"), resolved, line_number, "p-value")
    return resolved


def validate_frequency_csv(path: str | Path, *, gene_of_interest: str, require_sorted: bool = True) -> Path:
    """Validate a GOI coassociation-frequency CSV."""
    resolved = resolve_existing_path(path)
    frequencies: list[float] = []
    genes: list[str] = []
    with open(resolved, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        _require_columns(resolved, reader.fieldnames, REQUIRED_FREQUENCY_COLUMNS)
        for line_number, row in enumerate(reader, start=2):
            gene = (row.get("Gene") or "").strip()
            if not gene:
                raise ValueError(f"{resolved}:{line_number}: Gene is required.")
            frequency = _parse_float(row.get("Coassociation Frequency"), resolved, line_number, "Coassociation Frequency")
            if not 0 <= frequency <= 1:
                raise ValueError(f"{resolved}:{line_number}: Coassociation Frequency must be in [0, 1].")
            genes.append(gene)
            frequencies.append(frequency)
    if gene_of_interest not in genes:
        raise ValueError(f"{resolved}: gene_of_interest '{gene_of_interest}' is missing from frequency output.")
    if require_sorted and frequencies != sorted(frequencies, reverse=True):
        raise ValueError(f"{resolved}: Coassociation Frequency values must be sorted descending.")
    return resolved


def validate_network_artifact_bundle(artifacts: Mapping[str, str | Path], *, gene_of_interest: str) -> None:
    """Validate network artifact consistency, including SVG when present."""
    graphml = Path(artifacts["graphml"])
    edge_csv = Path(artifacts["edge_csv"])
    node_txt = Path(artifacts["node_txt"])
    summary_json = Path(artifacts["summary_json"])

    graph = nx.read_graphml(graphml)
    with open(edge_csv, newline="", encoding="utf-8") as fh:
        edge_rows = list(csv.DictReader(fh))
    nodes = [line.strip() for line in node_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    with open(summary_json, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    metrics = summary.get("metrics") or {}

    if len(edge_rows) != graph.number_of_edges():
        raise ValueError(f"{edge_csv}: edge row count does not match GraphML edge count.")
    if sorted(nodes) != sorted(map(str, graph.nodes())):
        raise ValueError(f"{node_txt}: node list does not match GraphML nodes.")
    if int(summary.get("nodes", -1)) != graph.number_of_nodes():
        raise ValueError(f"{summary_json}: node count does not match GraphML.")
    if int(summary.get("edges", -1)) != graph.number_of_edges():
        raise ValueError(f"{summary_json}: edge count does not match GraphML.")
    if int(metrics.get("genes_total", -1)) != graph.number_of_nodes():
        raise ValueError(f"{summary_json}: metrics.genes_total does not match GraphML.")
    if "svg" in artifacts:
        svg = Path(artifacts["svg"])
        svg_text = svg.read_text(encoding="utf-8")
        ET.parse(svg)
        if "<svg" not in svg_text:
            raise ValueError(f"{svg}: expected SVG markup.")
        if gene_of_interest in graph and gene_of_interest not in svg_text:
            raise ValueError(f"{svg}: gene_of_interest '{gene_of_interest}' is missing from SVG preview.")


def _require_columns(path: Path, fieldnames: Sequence[str] | None, required: Sequence[str]) -> None:
    if not fieldnames:
        raise ValueError(f"{path}: missing CSV header.")
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}.")


def _validate_float_or_nan(value: str | None, path: Path, line_number: int, column: str) -> None:
    _parse_float(value, path, line_number, column)


def _parse_float(value: str | None, path: Path, line_number: int, column: str) -> float:
    text = "" if value is None else str(value).strip()
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: {column} must be numeric or NaN.") from exc
    if math.isinf(parsed):
        raise ValueError(f"{path}:{line_number}: {column} must not be infinite.")
    return parsed
