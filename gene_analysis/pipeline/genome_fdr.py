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


def family_pair_ids(ordinals, genes, seed_ids=None):
    """Map a contiguous family ordinal to a unique global non-self pair ID.

    Seed family: all seed→genome pairs, followed by nonseed→seed pairs.
    Seed→seed pairs are included once, with both ordered directions retained.
    """
    if seed_ids is None:
        return ordinals
    seeds = np.asarray(seed_ids, dtype=np.int64)
    nonseeds = np.setdiff1d(np.arange(genes, dtype=np.int64), seeds)
    boundary = len(seeds)*(genes-1)
    first = ordinals < boundary
    source, target = np.empty_like(ordinals), np.empty_like(ordinals)
    source[first] = seeds[ordinals[first]//(genes-1)]
    offset = ordinals[first] % (genes-1)
    target[first] = offset + (offset >= source[first])
    remaining = ordinals[~first] - boundary
    source[~first] = nonseeds[remaining//len(seeds)]
    target[~first] = seeds[remaining % len(seeds)]
    return source*(genes-1) + target - (target > source)


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
        self.family_seeds = np.asarray(subset, dtype=np.int64) if self.settings.workflow == "seed_genome" else None
        self.total = (len(subset)*(2*self.n-len(subset)-1) if self.family_seeds is not None
                      else self.n * (self.n - 1))
        self.family_name = ("all directed non-self pairs incident to a retained seed" if self.family_seeds is not None
                            else "all retained directed non-self genome pairs")
        self.anchors = list(dict.fromkeys((self.config.gene_of_interest, *self.settings.report_anchors)))
        if any(anchor not in self.genes for anchor in self.anchors):
            raise ValueError("All report anchors must be retained expression genes.")
        self.reference_seeds = self.family_seeds
        if self.reference_seeds is None and self.config.seed_gene_file.is_file():
            requested = set(self.config.seed_gene_file.read_text().split())
            self.reference_seeds = np.array([i for i, gene in enumerate(self.genes) if gene in requested], dtype=np.int64)
        specification, self.fingerprint = input_fingerprint(self.config, values, times, self.genes, subset)
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["fingerprint"] != self.fingerprint or previous.get("genome_schema") != 1:
                raise ValueError("Resume refused: genome input/settings/schema changed; choose a new run_name.")
        else:
            _json(manifest_path, dict(specification, fingerprint=self.fingerprint, genome_schema=1,
                  tests=self.total, samples=len(times), family=self.family_name,
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
            ids = family_pair_ids(np.arange(next_pair, stop, dtype=np.int64), self.n, self.family_seeds)
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
        from gene_analysis.analysis.anchor_graph import ThresholdGraph
        from gene_analysis.experiments.qvalue_connectivity import THRESHOLDS
        graph = ThresholdGraph(self.genes, self.anchors, self.estimate["pi0"],
                               (*THRESHOLDS, self.settings.alpha, *self.settings.comparison_alphas),
                               reference_seeds=self.reference_seeds)
        primary = self.config.gene_of_interest
        primary_path = self.directory / "anchor_connections.csv"
        all_path = self.directory / "anchor_connections_all.csv"
        primary_tmp = primary_path.with_suffix(".csv.tmp")
        all_tmp = all_path.with_suffix(".csv.tmp")
        with primary_tmp.open("w", newline="") as first, all_tmp.open("w", newline="") as all_handle:
            writer, all_writer = csv.writer(first), csv.writer(all_handle)
            header = ["Genefrom", "Geneto", "p-value", "bh_adjusted_p_value", "storey_q_value"]
            writer.writerow(header)
            all_writer.writerow(["anchor", "direction", *header])
            for path in sorted((self.directory / "adjusted").glob("*.parquet")):
                if self.expired():
                    return False
                table = self.pq.read_table(path)
                source, target = table["source_id"].to_numpy(), table["target_id"].to_numpy()
                q, p = table["bh_adjusted_p_value"].to_numpy(), table["raw_p_value"].to_numpy()
                storey = table["storey_q_value"].to_numpy()
                for anchor, node in zip(graph.anchors, graph.anchor_ids):
                    for i in np.flatnonzero((source == node) | (target == node)):
                        row = [self.genes[source[i]], self.genes[target[i]], p[i], q[i], storey[i]]
                        all_writer.writerow([anchor, "outgoing" if source[i] == node else "incoming", *row])
                        if anchor == primary:
                            writer.writerow(row)
                graph.consume(source, target, q)
                self.progress("report", status="scanning", shard=path.name,
                              selected_edges_scanned=int(graph.state[0]),
                              thresholds_completed=graph.next_request)
        rows = sorted(graph.finish(), key=lambda row: (row["anchor"], row["method"], row["cutoff"]))
        _commit(primary_tmp, primary_path)
        _commit(all_tmp, all_path)

        def write_csv(path, records):
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            _commit(temporary, path)

        comparison = self.directory / "anchor_cutoff_comparison.csv"
        write_csv(comparison, rows)
        # Retain the existing primary-anchor table/column names for consumers.
        legacy = []
        for row in rows:
            if row["anchor"] != primary:
                continue
            mapped = dict(row)
            mapped.pop("anchor")
            mapped["criterion"] = "fixed_threshold"
            for before, after in (("anchor_component_nodes", "zeb2_component_nodes"),
                                  ("anchor_in_largest_component", "zeb2_in_largest_component"),
                                  ("distinct_neighbors", "zeb2_neighbors"),
                                  ("outgoing_neighbors", "zeb2_outgoing_connections"),
                                  ("incoming_neighbors", "zeb2_incoming_connections")):
                mapped[after] = mapped.pop(before)
            legacy.append(mapped)
        path = self.directory / "cutoff_comparison.csv"
        write_csv(path, legacy)
        _json(self.directory / "summary.json", dict(status="complete", fingerprint=self.fingerprint,
              genes=self.n, tests=self.total, failed_tests=self.failed, storey=self.estimate,
              family=self.family_name, connectivity="weak; all retained genes in denominator",
              anchor=primary, report_anchors=self.anchors, report_schema=2,
              reference_seed_genes=None if self.reference_seeds is None else [self.genes[i] for i in self.reference_seeds],
              anchor_export="estimable pairs only; failures retained in raw shards",
              cutoff_rows=len(rows)))
        readme = self.directory / "README.md"
        temporary = readme.with_suffix(".md.tmp")
        temporary.write_text(
            "# Global interaction results\n\n"
            f"Family: {self.family_name}. Genes: {self.n:,}; directed tests: {self.total:,}; "
            f"non-estimable tests: {self.failed:,}.\n\n"
            "Each ordered pair is counted once. Seed-to-seed tests are included in the seed-to-genome "
            "family. BH and Storey use every planned test in the family, never separate shard corrections. "
            "Failed tests are p=1 placeholders internally, NaN in raw output and omitted from adjusted output.\n\n"
            f"Storey pi0: {self.estimate['pi0']:.12g}; bootstrap estimator, pfdr=False. "
            "Storey q = pi0 * BH-adjusted p. See storey.json for sensitivity excluding failures.\n\n"
            "- raw/: committed model statistics including failures; adjusted/: global p/BH/Storey values.\n"
            "- genes.csv: IDs and symbols; pair_id joins raw and adjusted data.\n"
            "- anchor_cutoff_comparison.csv: all anchors, global graph counts, in/out/distinct neighbors, "
            "weak component sizes, and counts outside the reference seed list when available.\n"
            "- cutoff_comparison.csv: primary-anchor view with legacy zeb2_* names.\n"
            "- anchor_connections_all.csv: estimable directed pairs for every requested anchor; "
            "anchor_connections.csv retains only the primary anchor. Global adjustments are preserved.\n"
            "- manifest.json, summary.json, storey.json: provenance and definitions.\n"
            "- fit_metrics.json: fitting/checkpoint throughput only.\n\n"
            "Thresholds are fractions, selected with adjusted value <= cutoff. Connected components "
            "ignore direction; neighbors are distinct directly linked genes. Outside-seed counts concern "
            "the reference list after input filtering. directed_edges_times_cutoff is descriptive, "
            "not a validated conditional expected false-edge count or proof of zero errors. "
            "Community detection and causal interpretation are separate downstream questions.\n"
        )
        _commit(temporary, readme)
        self.artifacts.update(cutoffs=path, anchor_cutoffs=comparison, anchor_connections=primary_path,
                              all_anchor_connections=all_path, summary=self.directory / "summary.json",
                              adjusted=self.directory / "adjusted", raw=self.directory / "raw", readme=readme)
        return True
