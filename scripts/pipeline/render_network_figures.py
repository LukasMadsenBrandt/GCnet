#!/usr/bin/env python3
"""Render network SVG figures from an existing pipeline run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.pipeline.network_artifacts import GRAPHVIZ_LAYOUTS, render_network_artifacts_with_renderer


def main() -> None:
    """Parse CLI arguments and render network figures from existing GraphML files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing results/pipeline/<run_name> folder.")
    parser.add_argument(
        "--renderer",
        choices=("networkx", "graphviz"),
        default="graphviz",
        help="SVG renderer to use for the post-run figures.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=("seed", "seed_top_consensus", "probe", "expanded", "expanded_top_consensus"),
        help="Stage to render. Repeat to render multiple stages. Defaults to all network stages.",
    )
    parser.add_argument(
        "--layout",
        action="append",
        choices=GRAPHVIZ_LAYOUTS,
        help="Graphviz layout to render. Repeat for multiple layouts. Defaults to dot.",
    )
    parser.add_argument(
        "--all-layouts",
        action="store_true",
        help="Render all supported Graphviz layouts: dot, neato, fdp, sfdp, circo, twopi.",
    )
    args = parser.parse_args()
    layouts = GRAPHVIZ_LAYOUTS if args.all_layouts else tuple(args.layout or ("dot",))

    outputs = render_network_artifacts_with_renderer(
        run_dir=args.run_dir,
        renderer=args.renderer,
        stages=tuple(args.stage or ("seed", "seed_top_consensus", "probe", "expanded", "expanded_top_consensus")),
        layouts=layouts,
        skip_failed_layouts=args.all_layouts,
    )
    if not outputs:
        raise SystemExit("No matching GraphML network artifacts were found.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
