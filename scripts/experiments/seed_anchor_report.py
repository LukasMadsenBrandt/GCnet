#!/usr/bin/env python3
"""Report multiple anchors from a complete saved seed family without refitting."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from gene_analysis.analysis.anchor_graph import ThresholdGraph
from gene_analysis.analysis.fdr import benjamini_hochberg
from gene_analysis.analysis.storey import storey_qvalues
from gene_analysis.experiments.qvalue_connectivity import _read_saved, THRESHOLDS


def run(run_dir, output_dir, anchors, cutoff=1e-6):
    root, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Choose a separate output directory outside the source pipeline run.")
    if not np.isfinite(cutoff) or not 0 < cutoff < 1:
        raise ValueError("Cutoff must be in (0,1).")
    anchors = list(dict.fromkeys(anchors))
    with (root / "run.lock").open() as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        manifest = json.loads((root / "manifest.json").read_text())
        summary = json.loads((root / "all_pairs/overview/summary.json").read_text())
        genes, source, target, p, saved = _read_saved(root, manifest, summary, 100_000)
        for anchor in anchors:
            if anchor not in genes:
                raise ValueError(f"Anchor is not in the seed family: {anchor}")
        bh = benjamini_hochberg(p)
        if not np.allclose(bh, saved, rtol=1e-12, atol=1e-15, equal_nan=True):
            raise ValueError("Saved BH differs from complete-family recalculation.")
        del saved
        storey, details = storey_qvalues(p, bh_adjusted=bh)
        valid = np.flatnonzero(np.isfinite(bh))
        order = valid[np.argsort(bh[valid], kind="stable")]
        graph = ThresholdGraph(genes, anchors, details["pi0"], (*THRESHOLDS, cutoff),
                               reference_seeds=np.arange(len(genes)))
        for start in range(0, len(order), 100_000):
            ids = order[start:start+100_000]
            graph.consume(source[ids], target[ids], bh[ids])
            print(f"Anchor report: graph_checked={start+len(ids)}/{len(order)}", flush=True)
        comparison = pd.DataFrame(graph.finish()).sort_values(["anchor", "method", "cutoff"])
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".anchor-report-", dir=output) as temporary:
            tmp = Path(temporary)
            comparison.to_csv(tmp / "anchor_cutoff_comparison.csv", index=False)
            names = genes.to_numpy()
            for anchor in anchors:
                index = genes.get_loc(anchor)
                ids = np.flatnonzero((source == index) | (target == index))
                for method, q in (("bh", bh), ("storey", storey)):
                    frame = pd.DataFrame(dict(Genefrom=names[source[ids]], Geneto=names[target[ids]],
                        direction=np.where(source[ids] == index, "outgoing", "incoming"),
                        neighbor=np.where(source[ids] == index, names[target[ids]], names[source[ids]]),
                        raw_p_value=p[ids], q_value=q[ids], adjustment_method=method,
                        status=np.where(np.isfinite(p[ids]), "ok", "failed")))
                    frame = frame.sort_values(["q_value", "Genefrom", "Geneto"], na_position="last")
                    frame.to_csv(tmp / f"{anchor}_{method}_all_connections.csv", index=False)
                    selected = frame.loc[frame.q_value <= cutoff]
                    selected.to_csv(tmp / f"{anchor}_{method}_edges_q_{cutoff:g}.csv", index=False)
                    neighbors = []
                    for neighbor, links in selected.groupby("neighbor", sort=False):
                        incoming = links.loc[links.direction == "incoming", "q_value"]
                        outgoing = links.loc[links.direction == "outgoing", "q_value"]
                        neighbors.append(dict(neighbor=neighbor, best_q_value=links.q_value.min(),
                            outgoing_q_value=outgoing.min() if len(outgoing) else np.nan,
                            incoming_q_value=incoming.min() if len(incoming) else np.nan,
                            outgoing=bool(len(outgoing)), incoming=bool(len(incoming))))
                    columns = ["neighbor", "best_q_value", "outgoing_q_value", "incoming_q_value", "outgoing", "incoming"]
                    pd.DataFrame(neighbors, columns=columns).sort_values(["best_q_value", "neighbor"]).to_csv(
                        tmp / f"{anchor}_{method}_neighbors_q_{cutoff:g}.csv", index=False)
            digest = hashlib.sha256()
            with (root / "all_pairs/overview/all_tests.csv").open("rb") as handle:
                for chunk in iter(lambda: handle.read(8*1024*1024), b""):
                    digest.update(chunk)
            (tmp / "summary.json").write_text(json.dumps(dict(source=str(root),
                source_sha256=digest.hexdigest(), input_fingerprint=manifest["fingerprint"],
                seed_genes=len(genes), tests=len(p), failed_tests=int(np.isnan(p).sum()),
                anchors=anchors, selected_cutoff=cutoff, storey=details), indent=2) + "\n")
            (tmp / "README.md").write_text(
                "# Seed-network anchor reports\n\n"
                f"Complete fixed seed family: {len(genes):,} genes, {len(p):,} directed non-self tests. "
                "No regression or genome expansion was run. Global BH was recomputed and checked against saved values; "
                "Storey uses the same complete family and closed-form bootstrap estimator.\n\n"
                f"Selected exports use adjusted value <= {cutoff:g}, in increasing adjusted-value order. "
                "BH files contain BH-adjusted p-values; Storey files contain Storey q-values.\n\n"
                "- *_edges_q_*.csv: individual directed connections in either direction.\n"
                "- *_neighbors_q_*.csv: one row per distinct neighbor, ordered by its smallest selected q-value; "
                "blank directional values mean that direction did not pass the cutoff.\n"
                "- *_all_connections.csv: every tested pair involving the anchor; failed rows last with blank statistics.\n"
                "- anchor_cutoff_comparison.csv: global graph counts and each anchor's component and incoming/outgoing neighbors.\n"
                "- summary.json: provenance, complete-family denominator and Storey diagnostics.\n\n"
                "Components are weakly connected (direction ignored for paths), with all retained genes included. "
                "Incoming means other gene → anchor; outgoing means anchor → other gene. Reciprocal edges count "
                "twice as directed connections but once as distinct neighbors. An isolated anchor has component size one.\n\n"
                "directed_edges_times_cutoff is a descriptive product, not a validated expected false-edge count "
                "conditional on this observed network. FDR controls an expected proportion under statistical assumptions; "
                "a product below one does not establish absence of false edges. "
                "See https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html .\n",
                encoding="utf-8",
            )
            for path in tmp.iterdir():
                path.replace(output / path.name)
    print(comparison.loc[comparison.cutoff == cutoff,
          ["anchor", "method", "directed_connections", "anchor_component_nodes", "outgoing_neighbors", "incoming_neighbors", "distinct_neighbors"]].to_string(index=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchors", nargs="+", default=["ZEB2", "ZBTB18", "MECP2"])
    parser.add_argument("--cutoff", type=float, default=1e-6)
    args = parser.parse_args()
    run(args.run_dir, args.output_dir, args.anchors, args.cutoff)
