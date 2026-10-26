"""Read-only investigation of saved failed FDR tests, with bounded CPU rechecks.

The original checkpoint did not store failure reasons. Counts below distinguish
design defects found in the exact input from causes that require a pair recheck.
They are not invented historical error messages.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
import fcntl
import json
from pathlib import Path
import sqlite3
import tarfile
import time

import numpy as np
import pandas as pd

from gene_analysis.analysis.fdr import QuadraticInteraction
from gene_analysis.pipeline.fdr import _atomic_json, input_fingerprint, load_samples


def _pair_rechecks(model, records, genes):
    """Recheck only a bounded illustrative sample, retaining raw SE diagnostics."""
    if not records:
        return pd.DataFrame(columns=["checkpoint_row_id", "predictor_gene", "target_gene", "cpu_recheck_reason"])
    source = np.array([record[1] for record in records], dtype=int)
    target = np.array([record[2] for record in records], dtype=int)
    basis, response = model.basis[source], model.residual[target]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projection = np.einsum("bni,bn->bi", basis, response)
        error = response - np.einsum("bni,bi->bn", basis, projection)
        rss = np.sum(error**2, axis=1)
        se = np.sqrt(rss / model.df) * model.se_scale[source]
        _, _, _, p_value = model.fit(source, target)
    output = []
    for i, (row_id, a, b, broad_reason) in enumerate(records):
        if a == b:
            reason = "self_pair_excluded_on_cpu_recheck"
        elif not model.predictor_valid[a]:
            reason = "rank_deficient_predictor_design_on_cpu_recheck"
        elif se[i] == 0:
            reason = "zero_standard_error_on_cpu_recheck"
        elif not np.isfinite(se[i]):
            reason = "nonfinite_standard_error_on_cpu_recheck"
        elif not np.isfinite(p_value[i]):
            reason = "nonfinite_p_value_on_cpu_recheck"
        else:
            reason = "valid_on_cpu_recheck_original_failure_unresolved"
        output.append({
            "checkpoint_row_id": row_id, "predictor_gene": genes[a], "target_gene": genes[b],
            "broad_category": broad_reason, "cpu_recheck_reason": reason,
            "residual_sum_squares_on_recheck": rss[i], "standard_error_on_recheck": se[i],
            "p_value_on_recheck": p_value[i],
        })
    return pd.DataFrame(output)


def _design_table(values, times, genes, model):
    singular = model.predictor_singular_values
    threshold = model.predictor_rank_tolerance
    rank = np.sum(singular > threshold[:, None], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = np.where(singular[:, 1] > 0, singular[:, 0] / singular[:, 1], np.inf)
    days = np.unique(times)
    return pd.DataFrame({
        "gene": genes, "predictor_design_valid_on_cpu": model.predictor_valid,
        "residual_predictor_design_rank": rank, "full_design_rank_on_cpu": 3 + rank,
        "largest_residual_singular_value": singular[:, 0],
        "smallest_residual_singular_value": singular[:, 1], "rank_tolerance": threshold,
        "residual_design_condition_number": condition,
        "nonzero_processed_samples": np.count_nonzero(values, axis=0),
        "days_with_nonzero_processed_expression": np.sum(
            np.stack([np.any(values[times == day] != 0, axis=0) for day in days]), axis=0),
        "processed_expression_min": np.min(values, axis=0),
        "processed_expression_max": np.max(values, axis=0),
        "time_only_residual_sum_squares": np.sum(model.residual**2, axis=1),
    })


def run_failure_diagnostics(config, *, family="anchor", run_dir=None,
                            samples_per_category=25, fetch_size=100_000):
    """Scan NULL-p rows once and export a compact report without changing SQLite."""
    if family not in {"anchor", "all_pairs"}:
        raise ValueError("family must be anchor or all_pairs.")
    if samples_per_category < 0 or fetch_size < 1:
        raise ValueError("Need nonnegative samples_per_category and positive fetch_size.")
    root = Path(run_dir) if run_dir else config.run_dir / "fdr"
    if root.name != "fdr" and (root / "fdr").is_dir():
        root = root / "fdr"
    with (root / "run.lock").open("r", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("The pipeline is still writing this run; wait until it finishes.") from exc
        return _diagnose_locked(config, root, family, samples_per_category, fetch_size)


def _diagnose_locked(config, root, family, samples_per_category, fetch_size):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "complete":
        raise ValueError("This diagnostic expects a completed pipeline run.")
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported manifest schema.")
    print("Failure diagnostics: loading and verifying expression input (CPU only)", flush=True)
    values, times, genes, subset, _missing = load_samples(config)
    _, fingerprint = input_fingerprint(config, values, times, genes, subset)
    if fingerprint != manifest["fingerprint"]:
        raise ValueError("Input/settings fingerprint mismatch. Use the original config, data and Python environment.")
    model = QuadraticInteraction(values, times, backend="cpu_numpy")
    designs = _design_table(values, times, genes, model)
    n = len(genes)
    by_source, by_target = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    zero_target = np.sum(model.residual**2, axis=1) == 0
    invalid_scale = ~np.isfinite(model.se_scale)
    categories, samples = Counter(), defaultdict(list)
    directory = root / f"{family}_failure_diagnostics"
    directory.mkdir(exist_ok=True)
    started = time.monotonic()
    scanned = 0
    last_notification = started

    def notify(force=False):
        nonlocal last_notification
        now = time.monotonic()
        if force or now - last_notification >= 5:
            status = {"stage": "scanning_failed_tests", "failed_rows_scanned": scanned,
                      "elapsed_seconds": round(now - started, 2)}
            _atomic_json(directory / "progress.json", status)
            print(f"Failure diagnostics: scanned {scanned:,} failed rows; elapsed {now - started:.1f}s", flush=True)
            last_notification = now
        return 0

    database = root / family / "tests.sqlite"
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA cache_size=-32768")
        recorded = db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
        if not recorded or recorded[0] != fingerprint:
            raise ValueError("Checkpoint fingerprint does not match the run manifest.")
        completed = [json.loads(row[0]) for row in db.execute("SELECT result FROM layers WHERE result IS NOT NULL")]
        if not completed:
            raise ValueError("No completed test family found in the checkpoint.")
        expected = max(completed, key=lambda row: row["family_tests"])
        print(f"Failure diagnostics: sequential read-only scan of {database}; "
              f"expecting {expected['failed_tests']:,} failed tests", flush=True)
        # Report progress even while SQLite passes through long stretches of
        # successful rows. No secondary index or temporary sort is created.
        db.set_progress_handler(notify, 1_000_000)
        cursor = db.execute("SELECT id,source,target FROM tests NOT INDEXED WHERE p IS NULL")
        while rows := cursor.fetchmany(fetch_size):
            ids = np.asarray(rows, dtype=np.int64)
            source, target = ids[:, 1], ids[:, 2]
            if np.any(ids[:, 1:] < 0) or np.any(ids[:, 1:] >= n):
                raise ValueError("Checkpoint gene indices are outside the recorded gene list.")
            by_source += np.bincount(source, minlength=n)
            by_target += np.bincount(target, minlength=n)
            reason = np.full(len(rows), "other_requires_pair_recheck", dtype=object)
            reason[invalid_scale[source]] = "nonfinite_predictor_se_scale_on_recheck"
            reason[zero_target[target]] = "zero_time_adjusted_target_on_recheck"
            reason[~model.predictor_valid[source]] = "rank_deficient_predictor_design_on_recheck"
            reason[source == target] = "self_pair_in_checkpoint"
            unique, counts = np.unique(reason, return_counts=True)
            for name, count in zip(unique, counts):
                categories[str(name)] += int(count)
                available = samples_per_category - len(samples[str(name)])
                if available > 0:
                    for index in np.flatnonzero(reason == name)[:available]:
                        row_id, a, b = rows[int(index)]
                        samples[str(name)].append((row_id, a, b, str(name)))
            scanned += len(rows)
            notify()
        db.set_progress_handler(None, 0)
    notify(force=True)
    if scanned != expected["failed_tests"]:
        raise ValueError(f"Scanned {scanned} failures but the saved completed summary reports {expected['failed_tests']}.")
    rechecks = _pair_rechecks(model, [row for sample in samples.values() for row in sample], genes)
    designs["failed_tests_as_predictor"] = by_source
    designs["failed_tests_as_target"] = by_target
    affected = designs.loc[by_source > 0].sort_values("failed_tests_as_predictor", ascending=False)
    affected.to_csv(directory / "failed_predictors.csv", index=False)
    designs.loc[by_target > 0, ["gene", "failed_tests_as_target", "time_only_residual_sum_squares"]].sort_values(
        "failed_tests_as_target", ascending=False).to_csv(directory / "failed_targets.csv", index=False)
    rechecks.to_csv(directory / "failed_pair_samples.csv", index=False)
    summary = {
        "status": "complete", "family": family, "input_fingerprint": fingerprint,
        "samples": len(times), "residual_degrees_of_freedom": model.df,
        "normalization": config.preprocessing.normalize, "transformation": config.preprocessing.transform,
        "checkpoint_opened_read_only": True, "saved_failure_count_verified": True,
        "family_tests_from_saved_summary": expected["family_tests"], "failed_tests": scanned,
        "failed_test_fraction": scanned / max(1, expected["family_tests"]),
        "predictors_with_failures": int(np.count_nonzero(by_source)),
        "targets_with_failures": int(np.count_nonzero(by_target)),
        "all_genes_with_rank_deficient_predictor_design_on_cpu": int(np.count_nonzero(~model.predictor_valid)),
        "failed_tests_by_current_input_recheck_category": dict(categories),
        "sampled_pairs_rechecked": len(rechecks),
        "sampled_pair_cpu_recheck_reasons": dict(Counter(rechecks.cpu_recheck_reason)),
        "sampling": f"First up to {samples_per_category} rows per broad category; not a random sample",
        "limits": "Original failures stored no reason. Categories describe the current input/CPU recheck; "
                  "sample findings must not be extrapolated to unexamined pairs.",
        "scan_and_report_seconds": round(time.monotonic() - started, 2),
    }
    _atomic_json(directory / "summary.json", summary)
    lines = ["# Failed FDR test diagnostics", "",
             f"Family: {family}. Failed tests: {scanned:,} / {expected['family_tests']:,} "
             f"({100 * summary['failed_test_fraction']:.2f}%).", "",
             "The checkpoint was read-only. Input/settings and the saved failure count were verified.", "",
             "## Categories established by rechecking the input design", ""]
    for name, count in categories.most_common():
        lines.append(f"- {name}: {count:,} ({100 * count / max(1, scanned):.2f}% of failures)")
    lines.extend(["", "## Predictors with most failed tests", "",
                  "| Predictor | Failed tests | Residual design rank (needs 2) | Nonzero samples |",
                  "| --- | ---: | ---: | ---: |"])
    for row in affected.head(15).itertuples():
        lines.append(f"| {row.gene} | {row.failed_tests_as_predictor:,} | "
                     f"{row.residual_predictor_design_rank} | {row.nonzero_processed_samples} |")
    lines.extend(["", "## Limits", "", summary["limits"], "",
                  "A rank-deficient predictor design cannot identify separate predictor and time-interaction "
                  "effects. This is a model-identifiability issue, not evidence that the gene has no biological effect.", "",
                  "Failure counts by target include failures caused by an invalid predictor; they do not by themselves "
                  "identify a problem with that target gene.", "",
                  "Only the small illustrative sample was recomputed pair by pair on CPU. The full pairwise analysis "
                  "and BH correction were not rerun. No GPU is needed.", ""])
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")
    _atomic_json(directory / "progress.json", {"stage": "diagnostics", "status": "complete", "failed_rows_scanned": scanned})
    archive_path = root / f"{family}_failure_diagnostics.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in {".json", ".csv", ".md"}:
                archive.add(path, arcname=str(Path(directory.name) / path.name))
    print(f"Failure diagnostics complete. Send: {archive_path}", flush=True)
    return directory, archive_path
