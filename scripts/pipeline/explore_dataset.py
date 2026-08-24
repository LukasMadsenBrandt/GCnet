#!/usr/bin/env python3
"""Launch the responsive dataset and all-pairs GC exploration dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gene_analysis.dashboard.dataset_explorer import (  # noqa: E402
    create_dashboard,
    gc_coverage_summary,
    load_expression_csv,
    load_gc_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument("--summarized-csv", help="Override the all-gene summarized expression CSV.")
    parser.add_argument("--replicates-csv", help="Override the all-gene replicate expression CSV.")
    parser.add_argument("--gc-csv", help="Override the seed all-pairs GC CSV.")
    parser.add_argument("--gene", help="Initial gene of interest.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summarized_path = Path(args.summarized_csv) if args.summarized_csv else run_dir / "00_preprocessing" / "all_genes_summarized.csv"
    replicates_path = Path(args.replicates_csv) if args.replicates_csv else run_dir / "00_preprocessing" / "all_genes_replicates.csv"
    if args.gc_csv:
        gc_path = Path(args.gc_csv)
    else:
        all_pairs_path = run_dir / "01_seed_gc" / "seed_gc_all_pairs.csv"
        gc_path = all_pairs_path if all_pairs_path.exists() else run_dir / "01_seed_gc" / "seed_gc.csv"
    manifest_path = run_dir / "01_seed_gc" / "manifest.json"

    summarized = load_expression_csv(summarized_path)
    replicates = load_expression_csv(replicates_path) if replicates_path.exists() else None
    gc_frame = load_gc_csv(gc_path)
    app = create_dashboard(
        summarized,
        gc_frame,
        replicates=replicates,
        default_gene=args.gene,
        coverage=gc_coverage_summary(gc_frame, manifest_path),
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
