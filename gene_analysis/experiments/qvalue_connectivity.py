"""Compare BH and Storey on the complete saved directed seed-test family.

Build six-column tables and weak-connectivity/ZEB2 cutoff diagnostics.
All graph counts are evaluated after complete tied-q groups. No model is fitted.
"""

from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from gene_analysis.analysis.fdr import benjamini_hochberg
from gene_analysis.analysis.storey import storey_qvalues

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration only
    def njit(*args, **kwargs):
        return lambda function: function


HEADER = ["Genefrom", "Geneto", "p-value", "q-value", "total_connections_at_or_below_q",
          "largest_component_nodes"]
METRICS = ["directed_connections", "undirected_pairs", "genes_with_connections", "largest_component_nodes",
           "zeb2_component_nodes", "zeb2_in_largest_component", "zeb2_neighbors",
           "zeb2_outgoing_connections", "zeb2_incoming_connections"]
THRESHOLDS = tuple(10.0 ** exponent for exponent in range(-14, -4)) + (
    0.0001, 0.001, 0.002, 0.005, 0.01, 0.05,
)


@njit(cache=True)
def _root(parent, node):
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


@njit(cache=True)
def _advance(source, target, ends, begin_group, finish_group, parent, sizes, active,
             pair_bits, neighbors, anchor, state, output):
    """Update union-find in bounded groups so Python can print progress."""
    largest, connected, undirected, neighbor_count, outgoing, incoming = state
    for group in range(begin_group, finish_group):
        start = 0 if group == 0 else ends[group - 1]
        finish = ends[group]
        for i in range(start, finish):
            a, b = source[i], target[i]
            if not active[a]:
                active[a] = True
                connected += 1
            if not active[b]:
                active[b] = True
                connected += 1
            key = min(a, b) * len(parent) + max(a, b)
            byte, bit = key // 8, 1 << (key % 8)
            if not (pair_bits[byte] & bit):
                pair_bits[byte] |= bit
                undirected += 1
            ra, rb = _root(parent, a), _root(parent, b)
            if ra != rb:
                if sizes[ra] < sizes[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                sizes[ra] += sizes[rb]
                largest = max(largest, sizes[ra])
            other = -1
            if a == anchor:
                outgoing += 1
                other = b
            elif b == anchor:
                incoming += 1
                other = a
            if other >= 0 and not neighbors[other]:
                neighbors[other] = True
                neighbor_count += 1
        anchor_size = sizes[_root(parent, anchor)]
        output[group, 0] = finish
        output[group, 1] = undirected
        output[group, 2] = connected
        output[group, 3] = largest
        output[group, 4] = anchor_size
        output[group, 5] = anchor_size == largest
        output[group, 6] = neighbor_count
        output[group, 7] = outgoing
        output[group, 8] = incoming
    state[:] = largest, connected, undirected, neighbor_count, outgoing, incoming


def connectivity_events(source, target, q, gene_count, anchor):
    """Return one event per unique sorted q, including reciprocal-pair counts."""
    source, target, q = np.asarray(source), np.asarray(target), np.asarray(q)
    if (not len(q) or len(source) != len(q) or len(target) != len(q)
            or not np.isfinite(q).all() or np.any(np.diff(q) < 0)
            or np.any((q < 0) | (q > 1))):
        raise ValueError("Expected nonempty, sorted finite q-values and aligned pairs.")
    if (not 0 <= anchor < gene_count or np.any(source < 0) or np.any(target < 0)
            or np.any(source >= gene_count) or np.any(target >= gene_count) or np.any(source == target)):
        raise ValueError("Invalid gene indices or self-pairs.")
    ends = np.r_[np.flatnonzero(q[1:] != q[:-1]) + 1, len(q)].astype(np.int64)
    values = q[ends - 1]
    counts = np.empty((len(ends), len(METRICS)), dtype=np.int32)
    parent = np.arange(gene_count, dtype=np.int32)
    sizes = np.ones(gene_count, dtype=np.int32)
    active = np.zeros(gene_count, dtype=np.bool_)
    pair_bits = np.zeros((gene_count * gene_count + 7) // 8, dtype=np.uint8)
    neighbors = np.zeros(gene_count, dtype=np.bool_)
    state = np.array([1, 0, 0, 0, 0, 0], dtype=np.int64)
    for start in range(0, len(ends), 100_000):
        stop = min(start + 100_000, len(ends))
        _advance(source, target, ends, start, stop, parent, sizes, active, pair_bits, neighbors,
                 anchor, state, counts)
        print(f"Q-value connectivity: checked_links={ends[stop-1]}/{len(q)} "
              f"largest_component={state[0]}", flush=True)
    return values, ends, counts


def _cutoff_row(method, criterion, cutoff, index, counts):
    metrics = counts[index].tolist() if index >= 0 else [0, 0, 0, 1, 1, 1, 0, 0, 0]
    row = dict(method=method, criterion=criterion, cutoff=cutoff, **dict(zip(METRICS, metrics)))
    row["zeb2_in_largest_component"] = bool(row["zeb2_in_largest_component"])
    return row


def summarize_cutoffs(method, q, counts, gene_count, fraction=0.95, desired_neighbors=5):
    target = math.ceil(gene_count * fraction)
    matches = np.flatnonzero((counts[:, 3] >= target) & (counts[:, 5] == 1))
    milestones = {}
    rows = []
    for threshold in THRESHOLDS:
        i = int(np.searchsorted(q, threshold, side="right") - 1)
        rows.append(_cutoff_row(method, "fixed_threshold", threshold, i, counts))
    for label, indices, pick in [
        ("minimum_95pct_including_zeb2", matches, "first"),
        ("first_at_least_5_zeb2_neighbors", np.flatnonzero(counts[:, 6] >= desired_neighbors), "first"),
        ("last_at_most_5_zeb2_neighbors", np.flatnonzero(counts[:, 6] <= desired_neighbors), "last"),
    ]:
        if len(indices):
            i = int(indices[0] if pick == "first" else indices[-1])
            row = _cutoff_row(method, label, float(q[i]), i, counts)
            rows.append(row)
            milestones[label] = row
        else:
            milestones[label] = None
    reached = milestones["minimum_95pct_including_zeb2"]
    return rows, {"target_component_nodes": target, "milestones": milestones,
                  "at_most_five_neighbors_compatible_with_95pct":
                      bool(reached["zeb2_neighbors"] <= desired_neighbors) if reached else None}


def _read_saved(root, manifest, summary, chunk_size):
    genes = pd.Index([manifest["genes"][i] for i in manifest["subset"]])
    n = summary["family_tests"]
    if not genes.is_unique or n != len(genes) * (len(genes) - 1):
        raise ValueError("Expected the complete unique directed non-self seed family.")
    source, target = np.empty(n, np.int32), np.empty(n, np.int32)
    p, saved_bh = np.empty(n), np.empty(n)
    seen = np.zeros(len(genes)**2, dtype=np.bool_)
    path = root / "all_pairs/overview/all_tests.csv"
    position = 0
    for chunk in pd.read_csv(path, usecols=["predictor_gene", "target_gene", "raw_p_value",
                                           "bh_adjusted_p_value", "status"],
                             dtype={"predictor_gene": str, "target_gene": str, "status": str,
                                    "raw_p_value": float, "bh_adjusted_p_value": float},
                             na_values={"raw_p_value": [""], "bh_adjusted_p_value": [""]},
                             keep_default_na=False, float_precision="round_trip", chunksize=chunk_size):
        a, b = genes.get_indexer(chunk.predictor_gene), genes.get_indexer(chunk.target_gene)
        if np.any(a < 0) or np.any(b < 0) or np.any(a == b):
            raise ValueError("Unknown seed genes or self-tests in saved export.")
        keys = a * len(genes) + b
        if seen[keys].any() or len(np.unique(keys)) != len(keys):
            raise ValueError("Duplicate directed test in saved export.")
        seen[keys] = True
        stop = position + len(chunk)
        if stop > n:
            raise ValueError("Saved export has more rows than its completed summary.")
        raw = chunk.raw_p_value.to_numpy(float)
        adjusted = chunk.bh_adjusted_p_value.to_numpy(float)
        valid = np.isfinite(raw)
        if (np.isinf(raw).any() or np.isinf(adjusted).any() or np.any((raw[valid] < 0) | (raw[valid] > 1))
                or not np.array_equal(np.isfinite(adjusted), valid)
                or not np.array_equal(chunk.status.to_numpy(), np.where(valid, "ok", "failed"))):
            raise ValueError("Invalid statistics or failed-test flags in saved export.")
        source[position:stop], target[position:stop] = a, b
        p[position:stop], saved_bh[position:stop] = raw, adjusted
        position = stop
        print(f"Q-value comparison: reading_saved_tests={position}/{n}", flush=True)
    if position != n or int(np.isnan(p).sum()) != summary["failed_tests"]:
        raise ValueError("Incomplete saved export or inconsistent failure count.")
    return genes, source, target, p, saved_bh


def _export_method(directory, names, source, target, p, q, group_q, ends, counts, failures, chunk_size):
    directory.mkdir()
    anchor_pairs = (names[source] == "ZEB2") | (names[target] == "ZEB2")
    with gzip.open(directory / "connections_sorted.csv.gz", "wt", compresslevel=1, newline="") as complete:
        with (directory / "zeb2_connections.csv").open("w", newline="") as local:
            csv.writer(complete).writerow(HEADER)
            csv.writer(local).writerow(HEADER)
            for start in range(0, len(q), chunk_size):
                stop = min(start + chunk_size, len(q))
                groups = np.searchsorted(ends, np.arange(start, stop), side="right")
                frame = pd.DataFrame({HEADER[0]: names[source[start:stop]], HEADER[1]: names[target[start:stop]],
                                      HEADER[2]: p[start:stop], HEADER[3]: q[start:stop],
                                      HEADER[4]: ends[groups], HEADER[5]: counts[groups, 3]})
                frame.to_csv(complete, index=False, header=False)
                frame.loc[anchor_pairs[start:stop]].to_csv(local, index=False, header=False)
                print(f"Q-value comparison: method={directory.name} exported={stop}/{len(q)}", flush=True)
            # Invalid tests are retained at the bottom, with no fabricated q or
            # cutoff counts. The separate failure file explains these rows.
            failed_frame = pd.DataFrame({HEADER[0]: names[failures[0]], HEADER[1]: names[failures[1]],
                                         HEADER[2]: "", HEADER[3]: "", HEADER[4]: "", HEADER[5]: ""})
            failed_frame.to_csv(complete, index=False, header=False)
            failed_frame.loc[(failed_frame[HEADER[0]] == "ZEB2") | (failed_frame[HEADER[1]] == "ZEB2")].to_csv(
                local, index=False, header=False)
    # Keep full event counts in binary form; use a small curve CSV for plotting
    # and convenient spreadsheet inspection without millions of extra rows.
    np.savez_compressed(directory / "connectivity_events.npz", q=group_q, counts=counts,
                        metric_names=np.asarray(METRICS))
    linear = np.linspace(0, len(group_q)-1, min(1500, len(group_q))).astype(int)
    logarithmic = (np.searchsorted(group_q, np.geomspace(max(group_q[0], 1e-300), group_q[-1], 500))
                   if group_q[-1] > 0 else np.array([], dtype=int))
    sample = np.unique(np.r_[linear, logarithmic])
    sample = sample[sample < len(group_q)]
    curve = pd.DataFrame(counts[sample], columns=METRICS)
    curve.insert(0, "q_cutoff", group_q[sample])
    curve.to_csv(directory / "connectivity_curve.csv", index=False)


def _milestone_neighbors(method, milestones, names, source, target, adjusted, anchor):
    local = (source == anchor) | (target == anchor)
    a, b, q = source[local], target[local], adjusted[local]
    result = []
    for criterion, row in milestones["milestones"].items():
        if row is None:
            continue
        selected = q <= row["cutoff"]
        pairs = pd.DataFrame({"gene": names[np.where(a[selected] == anchor, b[selected], a[selected])],
                              "minimum_global_adjusted_value": q[selected],
                              "zeb2_to_gene": a[selected] == anchor,
                              "gene_to_zeb2": b[selected] == anchor})
        if pairs.empty:
            continue
        neighbors = pairs.groupby("gene", as_index=False).agg({"minimum_global_adjusted_value": "min",
                                                                "zeb2_to_gene": "max", "gene_to_zeb2": "max"})
        neighbors.insert(0, "criterion", criterion)
        neighbors.insert(0, "method", method)
        result.extend(neighbors.to_dict("records"))
    return result


def run_qvalue_comparison(run_dir, *, output_dir=None, chunk_size=100_000):
    """Use a completed all-pairs stage even if the later genome stage is incomplete."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    root = Path(run_dir).resolve()
    if root.name != "fdr" and (root / "fdr").is_dir():
        root = root / "fdr"
    output = Path(output_dir).resolve() if output_dir else root / "seed_qvalue_comparison"
    if output == root or output in root.parents or any(
        output == root / name or root / name in output.parents for name in ("anchor", "all_pairs")
    ):
        raise ValueError("Output must be outside the source export directories.")
    with (root / "run.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("The source pipeline is still writing; wait until it stops.") from exc
        manifest = json.loads((root / "manifest.json").read_text())
        summary = json.loads((root / "all_pairs/overview/summary.json").read_text())
        if manifest["schema"] != 1 or manifest["anchor"] != "ZEB2":
            raise ValueError("Expected a schema-1 seed run centered on ZEB2.")
        source_path = root / "all_pairs/overview/all_tests.csv"
        before = source_path.stat()
        genes, source, target, p, saved_bh = _read_saved(root, manifest, summary, chunk_size)
        anchor = genes.get_loc("ZEB2")
        print("Q-value comparison: verifying_saved_bh_and_estimating_storey_pi0", flush=True)
        bh = benjamini_hochberg(p)
        if not np.allclose(bh, saved_bh, rtol=1e-12, atol=1e-15, equal_nan=True):
            raise ValueError("Saved BH-adjusted values do not match the full raw-p family.")
        del saved_bh
        storey, pi0 = storey_qvalues(p, bh_adjusted=bh)
        valid = np.isfinite(p)
        order = np.flatnonzero(valid)
        order = order[np.lexsort((target[order], source[order], p[order]))]
        a, b, raw = source[order], target[order], p[order]
        failures = (source[~valid], target[~valid])
        output.mkdir(parents=True, exist_ok=True)
        comparison_rows, methods, neighbor_rows = [], {}, []
        curves = {}
        with tempfile.TemporaryDirectory(prefix=".qvalue-comparison-", dir=output) as tmp:
            tmp = Path(tmp)
            for method, values in [("bh", bh), ("storey", storey)]:
                adjusted = values[order]
                group_q, ends, counts = connectivity_events(a, b, adjusted, len(genes), anchor)
                rows, milestones = summarize_cutoffs(method, group_q, counts, len(genes))
                comparison_rows.extend(rows)
                methods[method] = milestones
                neighbor_rows.extend(_milestone_neighbors(method, milestones, genes.to_numpy(), a, b, adjusted, anchor))
                _export_method(tmp / method, genes.to_numpy(), a, b, raw, adjusted, group_q, ends,
                               counts, failures, chunk_size)
                curves[method] = pd.read_csv(tmp / method / "connectivity_curve.csv")
            pd.DataFrame(comparison_rows).to_csv(tmp / "cutoff_comparison.csv", index=False)
            pd.DataFrame(neighbor_rows, columns=["method", "criterion", "gene", "minimum_global_adjusted_value",
                                                 "zeb2_to_gene", "gene_to_zeb2"]).to_csv(
                tmp / "zeb2_neighbors_at_milestones.csv", index=False)
            pd.DataFrame({"Genefrom": genes.to_numpy()[failures[0]], "Geneto": genes.to_numpy()[failures[1]],
                          "status": "failed_non_estimable_saved_test"}).to_csv(tmp / "failed_tests.csv", index=False)
            digest = hashlib.sha256()
            with source_path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            after = source_path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise RuntimeError("Source export changed during reporting.")
            report = {"status": "complete", "source": str(source_path), "source_sha256": digest.hexdigest(),
                      "input_fingerprint": manifest["fingerprint"], "seed_genes": len(genes),
                      "tests": len(p), "estimable_tests": int(valid.sum()), "failed_tests": int((~valid).sum()),
                      "self_tests": "excluded; denominator is G*(G-1)", "saved_bh_verified": True,
                      "connectivity": "weak; links traversed in either direction; ties added together",
                      "storey": pi0, "methods": methods,
                      "ranking": "Storey pfdr=False multiplies BH values by a single estimated pi0; "
                                 "the edge order and networks at corresponding cutoffs are unchanged.",
                      "failure_policy": "Failed tests kept as p=1 in both adjustments and in pi0 estimation. "
                                        "Their exported p/q/cutoff counts stay blank. "
                                        "This changes the upper tail; pi0 excluding placeholders is reported as sensitivity.",
                      "cutoff_selection": "Exploratory connectivity-driven cutoff; not an additional FDR guarantee."}
            (tmp / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            _plot_comparison(curves, tmp, len(genes))
            (tmp / "README.md").write_text(_readme(report, comparison_rows))
            for path in tmp.iterdir():
                if path.is_dir():
                    destination = output / path.name
                    destination.mkdir(exist_ok=True)
                    for child in path.iterdir():
                        child.replace(destination / child.name)
                else:
                    path.replace(output / path.name)
    print(f"Q-value comparison: status=complete report={output / 'README.md'}", flush=True)
    return output


def _plot_comparison(curves, output, gene_count):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4), layout="constrained")
    axes = figure.subplots(1, 3)
    for method, frame in curves.items():
        frame = frame[frame.q_cutoff > 0]
        for axis, column in zip(axes, ["largest_component_nodes", "zeb2_neighbors", "directed_connections"]):
            axis.plot(frame.q_cutoff, frame[column], label=method.upper())
            axis.set_xscale("log")
            axis.set_xlabel("BH-adjusted p / Storey q cutoff")
            axis.grid(alpha=0.25)
    axes[0].axhline(math.ceil(0.95 * gene_count), color="gray", linestyle="--", label="95% of seed genes")
    axes[1].axhline(5, color="gray", linestyle="--", label="5 neighbors")
    for axis, title in zip(axes, ["Largest weak component", "Distinct ZEB2 neighbors", "Directed connections"]):
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8)
    figure.savefig(output / "cutoff_comparison.svg")
    figure.savefig(output / "cutoff_comparison.png", dpi=160)


def _readme(report, rows):
    lines = ["# Global seed-network q-values and connectivity", "",
             f"Seed genes: {report['seed_genes']:,}; directed non-self tests: {report['tests']:,}; "
             f"non-estimable tests: {report['failed_tests']:,}. Saved BH values were independently verified.", "",
             "Both methods use the same complete global seed family. The ZEB2 subset keeps the global "
             "adjustments; it is not corrected separately. No model fits or genome expansion were run.", "",
             f"Storey pi0 = {report['storey']['pi0']:.12g}, estimated with the closed-form bootstrap "
             "method over lambda=0.05,0.10,...,0.95 (pfdr=False). This is an explicit estimator choice, "
             "not the smoother default of R qvalue or py-qvalue. No local FDR is calculated.", "",
             report["failure_policy"], "",
             f"Sensitivity pi0 excluding failed-test placeholders: "
             f"{report['storey']['pi0_sensitivity_excluding_failed_tests']}. "
             "The primary comparison retains them to match the original planned family.", "",
             report["ranking"], "",
             "A 95% target means at least ceil(0.95*G) genes in one weakly connected component "
             "that includes ZEB2. It does not mean 95% of all possible edges. ZEB2 neighbors are "
             "distinct genes with a link in either direction. Reciprocal directions count as two "
             "connections and one undirected pair. Selecting a cutoff for connectivity is exploratory; "
             "it does not verify p-value calibration or establish an additional adaptive FDR guarantee.", "",
             "## Cutoff milestones", "",
             "| Method | Criterion | Cutoff | Directed connections | Largest component | ZEB2 neighbors |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["criterion"] != "fixed_threshold":
            lines.append(f"| {row['method']} | {row['criterion']} | {row['cutoff']:.12g} | "
                         f"{row['directed_connections']:,} | {row['largest_component_nodes']:,} | "
                         f"{row['zeb2_neighbors']:,} |")
    for method, info in report["methods"].items():
        if info["milestones"]["minimum_95pct_including_zeb2"] is None:
            lines += ["", f"{method}: the 95% connectivity target is not reached."]
        elif not info["at_most_five_neighbors_compatible_with_95pct"]:
            lines += ["", f"{method}: 95% connectivity and at most five ZEB2 neighbors cannot "
                      "be achieved together by a single cutoff; neighbor counts only increase as links are added."]
    lines += ["", "## Files", "",
              "- `bh/connections_sorted.csv.gz` and `storey/connections_sorted.csv.gz`: complete "
              "six-column tables, ordered by increasing adjusted value. Failed rows are last with blank values.",
              "- Each method's `zeb2_connections.csv`: the same six columns for all ZEB2 pairs.",
              "- `cutoff_comparison.csv`: fixed adjusted-value thresholds 10^-14 through 10^-5, "
              "0.0001, 0.001, 0.002, 0.005, 0.01 and 0.05, plus exact connectivity/neighbor milestones. "
              "Thresholds apply to BH-adjusted p-values or Storey q-values, not raw p-values.",
              "- `zeb2_neighbors_at_milestones.csv`: neighbor names and directions at each milestone.",
              "- `cutoff_comparison.svg` and `.png`: threshold curves (sampled for display).",
              "- Each method's `connectivity_events.npz`: exact counts for every distinct cutoff. "
              "`connectivity_curve.csv` is a smaller sampled version.",
              "- `failed_tests.csv`: original non-estimable pair identifiers.",
              "- `summary.json`: parameters, pi0 diagnostics, source hash and exact milestone results.", "",
              "Table columns: `Genefrom`, `Geneto`, `p-value`, `q-value`, "
              "`total_connections_at_or_below_q`, `largest_component_nodes`. The q-value column in "
              "the BH table means BH-adjusted p-value. Counts use <= and include all tied links, "
              "so every row with the same adjusted value has the same counts.", "",
              "References: [Storey pi0 estimator](https://github.com/jdstorey/qvalue/blob/master/R/pi0est.R), "
              "[q-value calculation](https://github.com/jdstorey/qvalue/blob/master/R/qvalue.R).", ""]
    return "\n".join(lines)
