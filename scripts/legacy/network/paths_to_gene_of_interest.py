#!/usr/bin/env python3
"""
paths_only_configurable.py

Goal
----
A small runner that computes ONLY paths-to-GOI from a Granger-causality CSV,
while REUSING the functions you already have in your big file (no code duplication).

What it does
------------
For each dataset config:
  1) Build candidate graph from CSV using your existing build_candidate_graph(...)
  2) Pick terminals:
       - from terminals_file (optional), otherwise ALL genes in CSV (gene1 ∪ gene2)
  3) For each gene_of_interest (GOI):
       - compute min-sum + min-max paths + path length (your existing function)
       - add quantile flags (keep_*_q) with configurable q
       - write:
           paths_to_<GOI>.summary.csv
           paths_to_<GOI>.full.csv

Configuration
-------------
You can configure everything INSIDE this file via DATASETS + GLOBALS below.
Optionally, you can still override dataset selection via CLI (--only-dataset).

Requirements
------------
Your existing "big" file must be importable as a Python module and must expose:
  - get_logger
  - build_candidate_graph
  - compute_and_save_paths_to_gene_of_interest
  - add_quantile_flags

Usage
-----
Run all configured datasets:
  python paths_only_configurable.py

Run only one dataset key:
  python paths_only_configurable.py --only-dataset mecp2_kutsche

"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.legacy.network.induced_mst_st import *


# =============================================================================
# CONFIG (edit here)
# =============================================================================

# Your existing big file as an importable module name (e.g. if file is gene_net.py -> "gene_net")

# Where to write results
OUT_ROOT = os.path.join("network", "paths_only2")

# Quantile cutoff for "keep" flags (top q = best/lowest scores)
DEFAULT_Q = 0.05

# Sort output tables by which metric
DEFAULT_SORT_BY = "minsum"  # "minsum" or "minmax"

# If you want to compute only on edges up to p-threshold (same as your pipeline)
# Provide per-dataset p-threshold in DATASETS.


DATASETS: Dict[str, Dict[str, Any]] = {
    "ZEB2_benito_human": {
        "csv": "granger_Benito_Human_results_p1_5633483of5647752.csv",
        "p_threshold": 0.05,
        "gene_of_interest": ["ZEB2"],
        # Terminals: if None -> all genes in CSV. Otherwise provide a file with one gene per line.
        "terminals_file": "Data/Kutsche/unique_genes.txt",
        # Optional: file with steiner-added genes (one per line) for hop counting columns
        "steiner_added_file": None,
        "q": 0.05,
        "sort_by": "minmax",
        # Optional: additional subfolder tag
        "tag": "",
    },
    "ZEB2_benito_gorilla": {
        "csv": "granger_Benito_Gorilla_results_p1_5555398of5581406.csv",
        "p_threshold": 0.05,
        "gene_of_interest": ["ZEB2"],
        "terminals_file": "Data/Kutsche/unique_genes.txt",
        "steiner_added_file": None,
        "q": 0.05,
        "sort_by": "minmax",
        "tag": "",
    },
}


# =============================================================================
# Helpers
# =============================================================================

def _read_gene_list_file(path: str) -> List[str]:
    genes: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            g = line.strip()
            if g and not g.startswith("#"):
                genes.append(g)
    # de-dup preserve order
    seen = set()
    out = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _all_genes_from_granger_csv(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    required = {"gene1", "gene2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns {missing}. Expected at least gene1,gene2.")
    genes = pd.concat([df["gene1"], df["gene2"]], axis=0).dropna().astype(str).unique().tolist()
    genes.sort()
    return genes

def _all_genes_from_graph_csv(csv_path: str) -> Set[str]:
    df = pd.read_csv(csv_path, usecols=["gene1", "gene2"])
    genes = set(df["gene1"].dropna().astype(str)) | set(df["gene2"].dropna().astype(str))
    return genes

# =============================================================================
# Main
# =============================================================================

def run_one_dataset(name: str, cfg: Dict[str, Any]) -> None:
    """Compute configured path tables for one dataset entry."""
    logger = get_logger()

    csv_path = cfg["csv"]
    genes_in_csv = _all_genes_from_graph_csv(csv_path)
    p_threshold = float(cfg["p_threshold"])
    gois: List[str] = list(cfg.get("gene_of_interest") or [])
    if not gois:
        raise ValueError(f"Dataset {name!r} must define gene_of_interest (list).")

    q = float(cfg.get("q", DEFAULT_Q))
    sort_by = str(cfg.get("sort_by", DEFAULT_SORT_BY))
    tag = cfg.get("tag", None)

    # output dir
    out_dir = os.path.join(OUT_ROOT, name if tag is None else f"{name}__{tag}")
    os.makedirs(out_dir, exist_ok=True)

    # terminals
    terminals_file = cfg.get("terminals_file", None)
    if terminals_file:
        terminals = _read_gene_list_file(terminals_file)
        logger.info("[%s] terminals from %s: n=%d", name, terminals_file, len(terminals))
    else:
        terminals = _all_genes_from_granger_csv(csv_path)
        logger.info("[%s] terminals from CSV gene1∪gene2: n=%d", name, len(terminals))

    # optional steiner-added set
    steiner_added_file = cfg.get("steiner_added_file", None)
    steiner_added: Set[str] = set()
    if steiner_added_file:
        steiner_added = set(_read_gene_list_file(steiner_added_file))
        logger.info("[%s] steiner-added from %s: n=%d", name, steiner_added_file, len(steiner_added))

    # Build candidate graph with your existing function (legacy behavior)
    G = build_candidate_graph(
        csv_file=csv_path,
        p_threshold=p_threshold,
        connectors_pmax=None,
        terminals=None,
        logger=logger,
    )
    G.add_nodes_from(terminals)

    logger.info(
        "[%s] candidate graph: nodes=%d edges=%d | p_threshold=%.6g",
        name, G.number_of_nodes(), G.number_of_edges(), p_threshold
    )

    # Compute paths per GOI
    for goi in gois:
        goi_dir = os.path.join(out_dir, goi)
        os.makedirs(goi_dir, exist_ok=True)

        summary_filename = f"paths_to_{goi}.summary.csv"
        full_filename = f"paths_to_{goi}.full.csv"

    (
        minsum_summary_path,
        minsum_full_path,
        minmax_summary_path,
        minmax_full_path,
        combined_summary_path,
        metrics,
    ) = compute_and_save_paths_to_gene_of_interest_split(
        G_view=G,
        terminals=terminals,
        output_dir=out_dir,
        gene_of_interest=goi,
        logger=logger,
        steiner_added=steiner_added,
    )

    logger.info(
        "DONE GOI=%s | minsum_summary=%s | minsum_full=%s | "
        "minmax_summary=%s | minmax_full=%s | combined_summary=%s | metrics=%s",
        goi,
        minsum_summary_path,
        minsum_full_path,
        minmax_summary_path,
        minmax_full_path,
        combined_summary_path,
        metrics,
    )



def main():
    """Run all configured datasets or one selected dataset."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-dataset", default=None, help="Run only one dataset key from DATASETS.")
    args = ap.parse_args()


    if args.only_dataset:
        if args.only_dataset not in DATASETS:
            raise SystemExit(
                f"--only-dataset {args.only_dataset!r} not found. "
                f"Available: {', '.join(sorted(DATASETS.keys()))}"
            )
        run_one_dataset(args.only_dataset, DATASETS[args.only_dataset])
        return

    for name, cfg in DATASETS.items():
        run_one_dataset(name, cfg)


if __name__ == "__main__":
    main()
