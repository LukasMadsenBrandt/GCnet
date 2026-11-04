"""Disk-backed exhaustive genome interaction tests and global BH/Storey.

Only the sample matrix, one fitting shard, or one Parquet row group is loaded.
Fitting and adjustment commit immutable chunks; external sorting and the small
graph report restart their current stage after interruption.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from gene_analysis.analysis.fdr import QuadraticInteraction
from gene_analysis.pipeline.fdr import FDRRunner, _run_lock, input_fingerprint, load_samples

GRID = np.arange(1, 20, dtype=float) / 20
RAW_COLUMNS = ("coefficient", "standard_error", "statistic", "raw_p_value")


def _commit(temporary, destination):
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    _commit(temporary, path)


def decode_pairs(pair_ids, genes):
    """Int64 IDs enumerate source-major non-self pairs without storing a grid."""
    source = pair_ids // (genes - 1)
    offset = pair_ids % (genes - 1)
    return source, offset + (offset >= source)


def pi0_from_counts(tails, count, maximum):
    """Exactly the bootstrap estimator, using additive sufficient statistics."""
    tails = np.asarray(tails, dtype=np.float64)
    if count <= 0 or maximum < GRID[-1]:
        raise ValueError("Storey estimation requires nonempty, untruncated p-values.")
    estimates = tails / (count * (1 - GRID))
    reference = np.quantile(estimates, .1)
    loss = tails / (float(count)**2 * (1 - GRID)**2) * (1 - tails / count) + (estimates - reference)**2
    best = loss == loss.min()
    pi0 = float(min(1, estimates[best].min()))
    if pi0 <= 0:
        raise ValueError("Nonpositive pi0: inspect the complete p-value distribution.")
    return dict(pi0=pi0, method="bootstrap_closed_form", lambda_grid=GRID.tolist(),
                pi0_lambda=estimates.tolist(), bootstrap_mse=loss.tolist(),
                selected_lambda=GRID[best].tolist(), p_values_used=int(count), pfdr=False)


class GenomeFDRRunner(FDRRunner):
    def __init__(self, config):
        super().__init__(config)
        self.directory = config.run_dir / "genome_fdr"

    def _metadata(self, path):
        metadata = self.pq.read_metadata(path)
        info = json.loads(metadata.metadata[b"genome_fdr"])
        if info["fingerprint"] != self.fingerprint:
            raise ValueError(f"Checkpoint fingerprint mismatch: {path}")
        if metadata.num_rows != info["rows"]:
            raise ValueError(f"Checkpoint row count mismatch: {path}")
        return info

    def _parquet(self, path, arrays, **info):
        table = self.pa.table(arrays)
        info.update(fingerprint=self.fingerprint, rows=table.num_rows)
        table = table.replace_schema_metadata({b"genome_fdr": json.dumps(info).encode()})
        temporary = path.with_suffix(".parquet.tmp")
        self.pq.write_table(table, temporary, compression="zstd", row_group_size=131072)
        _commit(temporary, path)

    def run(self):
        self.settings.validate()
        try:
            import duckdb
            import pyarrow as pa
            import pyarrow.parquet as pq
            import numba  # noqa: F401 -- required for genome-scale graph loops
        except ImportError as exc:
            raise RuntimeError("Install requirements-genome.txt before the genome workflow.") from exc
        self.duckdb, self.pa, self.pq = duckdb, pa, pq
        self.directory.mkdir(parents=True, exist_ok=True)
        with _run_lock(self.directory):
            try:
                return self._run_genome()
            except KeyboardInterrupt:
                self.progress("pipeline", status="interrupted", resumable=True)
                raise
            except Exception as exc:
                self.progress("pipeline", status="failed", error=str(exc), resumable=True)
                raise

    def _run_genome(self):
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists() and not self.config.execution.resume:
            raise ValueError("Existing genome run with resume=false; choose a new run_name.")
        self.progress("preprocessing", status="loading_samples")
        values, times, self.genes, subset, missing = load_samples(self.config)
        self.n = len(self.genes)
        self.total = self.n * (self.n - 1)
        specification, self.fingerprint = input_fingerprint(self.config, values, times, self.genes, subset)
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["fingerprint"] != self.fingerprint or previous.get("genome_schema") != 1:
                raise ValueError("Resume refused: genome input/settings/schema changed; choose a new run_name.")
        else:
            _json(manifest_path, dict(specification, fingerprint=self.fingerprint, genome_schema=1,
                  tests=self.total, samples=len(times), family="all retained directed non-self genome pairs",
                  model="target ~ 1 + time + time^2 + predictor + time:predictor",
                  tested_coefficient="time:predictor", failure_policy="p=1 internally; NaN in raw output",
                  pair_encoding="source=pair_id//(G-1); offset=pair_id%(G-1); target=offset+(offset>=source)"))
        self.artifacts.update(manifest=manifest_path, progress=self.directory / "progress.json")
        for name in ("raw", "adjusted", "scratch"):
            (self.directory / name).mkdir(exist_ok=True)
        self._write_genes()
        if not self._fit(values, times):
            self.progress("pipeline", status="paused", resumable=True)
            return self.artifacts
        if self.settings.genome_stage == "fit":
            self.progress("pipeline", status="fit_complete", next_stage="adjust")
            return self.artifacts
        if self.expired() or not self._sort():
            self.progress("pipeline", status="paused", resumable=True)
            return self.artifacts
        if not self._adjust():
            self.progress("pipeline", status="paused", resumable=True)
            return self.artifacts
        if self.settings.genome_stage == "adjust":
            self.progress("pipeline", status="adjust_complete", next_stage="report")
            return self.artifacts
        if not self._report():
            self.progress("pipeline", status="paused", resumable=True)
            return self.artifacts
        self.progress("pipeline", status="complete", family_tests=self.total)
        return self.artifacts

    def _write_genes(self):
        path = self.directory / "genes.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["gene_id", "gene"])
            writer.writerows(enumerate(self.genes))
        _commit(temporary, path)

    def _fit(self, values, times):
        next_pair = failed = 0
        tails = np.zeros(len(GRID), dtype=np.int64)
        maximum = 0.0
        # Footer-only validation; never reread all fitted statistics to resume.
        for path in sorted((self.directory / "raw").glob("*.parquet")):
            info = self._metadata(path)
            if info["start"] != next_pair or info["stop"] - info["start"] != info["rows"]:
                raise ValueError(f"Gap/overlap in committed raw shards: {path}")
            next_pair = info["stop"]
            failed += info["failed"]
            tails += np.asarray(info["tails"], dtype=np.int64)
            maximum = max(maximum, info["maximum"])
        if next_pair > self.total:
            raise ValueError("Raw checkpoint exceeds planned family.")
        model = None
        invocation_start, initial = time.monotonic(), next_pair
        shards_written = 0
        while next_pair < self.total:
            if self.expired():
                return False
            if model is None:
                model = QuadraticInteraction(values, times, backend=self.settings.backend,
                                             gpu_device=self.config.execution.gpu_device)
            stop = min(self.total, next_pair + self.settings.genome_shard_pairs)
            ids = np.arange(next_pair, stop, dtype=np.int64)
            result = np.empty((4, len(ids)), dtype=np.float64)
            for begin in range(0, len(ids), self.settings.batch_size):
                end = min(begin + self.settings.batch_size, len(ids))
                source, target = decode_pairs(ids[begin:end], self.n)
                result[:, begin:end] = model.fit(source, target)
                self.progress("fit", status="fitting", committed_tests=next_pair,
                              checked_tests=next_pair + end, family_tests=self.total,
                              percent=round(100 * (next_pair + end) / self.total, 3))
            p = result[3]
            valid = np.isfinite(p)
            ordered = np.sort(p[valid])
            shard_failed = int((~valid).sum())
            shard_tails = len(ordered) - np.searchsorted(ordered, GRID, side="left")
            shard_maximum = float(ordered[-1]) if len(ordered) else 0.0
            self._parquet(self.directory / "raw" / f"part-{next_pair:012d}.parquet",
                          dict(pair_id=ids, **dict(zip(RAW_COLUMNS, result))),
                          start=next_pair, stop=stop, failed=shard_failed,
                          tails=shard_tails.tolist(), maximum=shard_maximum)
            failed += shard_failed
            tails += shard_tails
            maximum = max(maximum, shard_maximum)
            next_pair = stop
            shards_written += 1
            elapsed = time.monotonic() - invocation_start
            rate = (next_pair - initial) / max(elapsed, 1e-9)
            raw_bytes = sum(path.stat().st_size for path in (self.directory / "raw").glob("*.parquet"))
            _json(self.directory / "fit_metrics.json", dict(committed_tests=next_pair,
                  family_tests=self.total, new_tests=next_pair-initial, seconds=elapsed,
                  fit_and_checkpoint_pairs_per_second=rate, raw_bytes=raw_bytes,
                  estimated_raw_bytes=raw_bytes * self.total / next_pair,
                  estimated_remaining_fit_seconds=(self.total-next_pair)/rate,
                  estimate_scope="This invocation's fitting/checkpoint rate; excludes sort, adjustment and report."))
            self.progress("fit", status="checkpointed", committed_tests=next_pair,
                          family_tests=self.total, pairs_per_second=round(rate, 1),
                          eta_fit_seconds=round((self.total-next_pair)/rate))
            limit = self.settings.genome_shards_per_invocation
            if limit is not None and shards_written >= limit and next_pair < self.total:
                return False
        if failed == self.total:
            raise ValueError("No estimable genome tests.")
        self.failed = failed
        estimate = pi0_from_counts(tails + failed, self.total, 1.0 if failed else maximum)
        try:
            sensitivity = pi0_from_counts(tails, self.total - failed, maximum)["pi0"]
        except ValueError:
            sensitivity = None
        estimate.update(failed_tests_as_p_one=failed, pi0_sensitivity_excluding_failed_tests=sensitivity)
        self.estimate = estimate
        _json(self.directory / "storey.json", estimate)
        return True

    def _sort(self):
        path = self.directory / "sorted_p.parquet"
        expected = self.total - self.failed
        if path.exists():
            if self.pq.read_metadata(path).num_rows != expected:
                raise ValueError("Sorted checkpoint has an incorrect row count.")
            return True
        quote = lambda value: "'" + str(value).replace("'", "''") + "'"
        temporary = path.with_suffix(".parquet.tmp")
        with self.duckdb.connect() as connection:
            connection.execute("SET memory_limit = " + quote(self.settings.genome_sort_memory))
            connection.execute(f"SET threads = {int(self.settings.genome_sort_threads)}")
            connection.execute("SET temp_directory = " + quote(self.directory / "scratch"))
            connection.execute("SET enable_progress_bar = false")
            query = (f"COPY (SELECT pair_id, raw_p_value FROM read_parquet({quote(self.directory / 'raw/*.parquet')}) "
                     f"WHERE isfinite(raw_p_value) ORDER BY raw_p_value, pair_id) TO {quote(temporary)} "
                     "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)")
            self.progress("sort", status="sorting_on_disk", estimable_tests=expected)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(connection.execute, query)
                try:
                    while True:
                        try:
                            future.result(timeout=5)
                            break
                        except TimeoutError:
                            self.progress("sort", status="sorting_on_disk",
                                          query_percent=connection.query_progress(),
                                          scratch_bytes=sum(p.stat().st_size for p in (self.directory / "scratch").glob("*") if p.is_file()))
                            if self.expired():
                                connection.interrupt()
                                try:
                                    future.result()
                                except self.duckdb.InterruptException:
                                    pass
                                return False
                except BaseException:
                    connection.interrupt()
                    raise
        if self.pq.read_metadata(temporary).num_rows != expected:
            raise ValueError("External sort lost rows.")
        _commit(temporary, path)
        return True

    def _adjust(self):
        parquet = self.pq.ParquetFile(self.directory / "sorted_p.parquet")
        counts = [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]
        starts = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        carry, previous_p = 1.0, 1.0
        pi0 = self.estimate["pi0"]
        for group in reversed(range(parquet.num_row_groups)):
            if self.expired():
                return False
            path = self.directory / "adjusted" / f"part-{group:08d}.parquet"
            if path.exists():
                info = self._metadata(path)
                if (info["start"] != int(starts[group]) or info["rows"] != counts[group]
                        or info["incoming_min"] != carry or info["pi0"] != pi0
                        or info["max_p"] > previous_p):
                    raise ValueError(f"Adjusted checkpoint is inconsistent: {path}")
                carry, previous_p = info["outgoing_min"], info["min_p"]
                continue
            table = parquet.read_row_group(group)
            p = table["raw_p_value"].to_numpy()
            if np.any(np.diff(p) < 0) or p[-1] > previous_p:
                raise ValueError("External-sort output is not globally ordered.")
            ranks = np.arange(starts[group] + 1, starts[group + 1] + 1, dtype=np.float64)
            q = np.minimum(np.minimum.accumulate((p * self.total / ranks)[::-1])[::-1], carry)
            ids = table["pair_id"].to_numpy()
            source, target = decode_pairs(ids, self.n)
            self._parquet(path, dict(pair_id=ids, source_id=source.astype(np.int32),
                          target_id=target.astype(np.int32), raw_p_value=p,
                          bh_adjusted_p_value=q, storey_q_value=q*pi0),
                          start=int(starts[group]), incoming_min=carry, outgoing_min=float(q[0]),
                          pi0=pi0, min_p=float(p[0]), max_p=float(p[-1]))
            carry, previous_p = float(q[0]), float(p[0])
            self.progress("adjust", status="checkpointed", groups_completed=parquet.num_row_groups-group,
                          total_groups=parquet.num_row_groups)
        return True

    def _report(self):
        from gene_analysis.experiments.qvalue_connectivity import THRESHOLDS, METRICS, _advance
        # One ascending pass for both methods; a Storey cutoff maps to BH/ pi0.
        requests = sorted((threshold / scale, method, threshold)
                          for method, scale in (("bh", 1.0), ("storey", self.estimate["pi0"]))
                          for threshold in sorted(set((*THRESHOLDS, self.settings.alpha, *self.settings.comparison_alphas))))
        parent = np.arange(self.n, dtype=np.int32)
        sizes = np.ones(self.n, dtype=np.int64)
        active = np.zeros(self.n, dtype=bool)
        bits = np.zeros((self.n*self.n+7)//8, dtype=np.uint8)
        neighbors = np.zeros(self.n, dtype=bool)
        anchor = self.genes.index(self.config.gene_of_interest)
        state = np.array([1, 0, 0, 0, 0, 0], dtype=np.int64)
        metrics = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int64)
        total_edges, request = 0, 0
        rows = []
        zeb_path = self.directory / "anchor_connections.csv"
        zeb_tmp = zeb_path.with_suffix(".csv.tmp")

        def record():
            _, method, cutoff = requests[request]
            row = dict(method=method, criterion="fixed_threshold", cutoff=cutoff,
                       **dict(zip(METRICS, map(int, metrics[0]))))
            row["directed_connections"] = total_edges
            row["zeb2_in_largest_component"] = bool(row["zeb2_in_largest_component"])
            rows.append(row)

        with zeb_tmp.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Genefrom", "Geneto", "p-value", "bh_adjusted_p_value", "storey_q_value"])
            for path in sorted((self.directory / "adjusted").glob("*.parquet")):
                if self.expired():
                    return False
                table = self.pq.read_table(path)
                source = table["source_id"].to_numpy()
                target = table["target_id"].to_numpy()
                q = table["bh_adjusted_p_value"].to_numpy()
                storey_q = table["storey_q_value"].to_numpy()
                p = table["raw_p_value"].to_numpy()
                for i in np.flatnonzero((source == anchor) | (target == anchor)):
                    writer.writerow([self.genes[source[i]], self.genes[target[i]], p[i], q[i], q[i]*self.estimate["pi0"]])
                position = 0
                while request < len(requests):
                    _, method, threshold = requests[request]
                    # Compare the actual exported values, avoiding rounding at
                    # an inclusive Storey boundary from dividing by pi0.
                    selected_q = q if method == "bh" else storey_q
                    end = int(np.searchsorted(selected_q, threshold, side="right"))
                    if end > position:
                        a, b = source[position:end], target[position:end]
                        _advance(a, b, np.array([len(a)], dtype=np.int64), 0, 1,
                                 parent, sizes, active, bits, neighbors, anchor, state, metrics)
                        total_edges += end-position
                        position = end
                    if end == len(q):
                        break  # Complete ties can cross shard boundaries.
                    record()
                    request += 1
                self.progress("report", status="scanning", shard=path.name,
                              selected_edges_scanned=total_edges, thresholds_completed=request)
        while request < len(requests):
            record()
            request += 1
        _commit(zeb_tmp, zeb_path)
        path = self.directory / "cutoff_comparison.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["method"], row["cutoff"])))
        _commit(temporary, path)
        _json(self.directory / "summary.json", dict(status="complete", fingerprint=self.fingerprint,
              genes=self.n, tests=self.total, failed_tests=self.failed, storey=self.estimate,
              family="all retained directed non-self pairs", connectivity="weak; all retained genes in denominator",
              anchor=self.config.gene_of_interest, anchor_export="estimable pairs only; failures retained in raw shards",
              cutoff_rows=len(rows)))
        readme = self.directory / "README.md"
        temporary = readme.with_suffix(".md.tmp")
        temporary.write_text(
            "# Exhaustive genome interaction results\n\n"
            f"Genes: {self.n:,}; directed non-self tests: {self.total:,}; "
            f"non-estimable tests: {self.failed:,}.\n\n"
            "The family includes every retained gene in both predictor and target roles. "
            "BH and Storey are adjusted globally, never per shard or anchor subset. "
            "Failed tests remain in the denominator as p=1 placeholders; actual raw values are NaN.\n\n"
            f"Storey pi0: {self.estimate['pi0']:.12g}; closed-form bootstrap, pfdr=False. "
            "Storey q = pi0 * BH-adjusted p. Sensitivity excluding failures is in storey.json.\n\n"
            "- raw/: committed Parquet fitting shards, including failed tests and coefficients.\n"
            "- adjusted/: estimable pairs with raw p, global BH and Storey values. "
            "Read files in filename order for increasing adjusted value.\n"
            "- genes.csv: integer gene IDs and symbols. pair_id joins raw and adjusted data.\n"
            "- cutoff_comparison.csv: fixed adjusted-value cutoffs and graph counts. "
            "Components ignore arrow direction; neighbors are distinct directly linked genes. "
            "The legacy zeb2_* column names refer to the configured anchor.\n"
            "- anchor_connections.csv: all estimable anchor pairs with global adjustments. "
            "No separate anchor correction is applied.\n"
            "- manifest.json, storey.json, summary.json: provenance and statistical definitions.\n"
            "- fit_metrics.json: measured fitting/checkpoint throughput; excludes downstream stages.\n\n"
            "Thresholds are fractions (0.001 = 0.1%) and use adjusted value <= cutoff. "
            "The graph statistics are descriptive; they do not validate p-value calibration "
            "or establish causal regulation. This workflow does not perform community detection "
            "or select a biologically optimal cutoff.\n"
        )
        _commit(temporary, readme)
        self.artifacts.update(cutoffs=path, anchor_connections=zeb_path, summary=self.directory / "summary.json",
                              adjusted=self.directory / "adjusted", raw=self.directory / "raw", readme=readme)
        return True
