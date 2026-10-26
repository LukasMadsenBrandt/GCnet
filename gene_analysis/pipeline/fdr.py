"""Resumable BH overview and cumulative anchor exploration.

SQLite transactions commit statistics and pair identity together, so interruption
cannot leave a checkpoint ahead of its results. Each family has a separate DB.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import csv
import fcntl
import hashlib
from itertools import islice
import json
from pathlib import Path
import sqlite3
import time as clock

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from gene_analysis.analysis.fdr import QuadraticInteraction, benjamini_hochberg
from gene_analysis.analysis.cuda_environment import require_cuda_kernels
from gene_analysis.analysis.preprocessing import aggregate_duplicate_genes, apply_expression_preprocessing
from gene_analysis.datasets.kutsche import filter_data_wt, load_and_preprocess_data


@dataclass(frozen=True)
class FDRSettings:
    """Selection and traversal choices; all-pairs and local families are separate."""

    workflow: str = "both"
    alpha: float = 0.05
    comparison_alphas: tuple[float, ...] = (0.10,)
    backend: str = "gpu_cuda"
    batch_size: int = 4096
    direction: str = "both"
    max_depth: int = 10
    max_nodes: int = 50
    max_seconds: float | None = None
    genome_expansion: bool = True
    duplicate_gene_method: str = "error"

    def validate(self):
        if self.workflow not in {"both", "all_pairs", "anchor"}:
            raise ValueError("analysis.workflow must be both, all_pairs or anchor.")
        if self.backend not in {"cpu_numpy", "gpu_cuda"}:
            raise ValueError("analysis.backend must be cpu_numpy or gpu_cuda.")
        if self.direction not in {"both", "outgoing", "incoming"}:
            raise ValueError("analysis.direction must be both, outgoing or incoming.")
        if self.duplicate_gene_method not in {"error", "mean", "median", "robust"}:
            raise ValueError("duplicate_gene_method must be error, mean, median or robust.")
        if not isinstance(self.genome_expansion, bool):
            raise ValueError("genome_expansion must be true or false.")
        if any(not np.isfinite(a) or not 0 < a < 1 for a in (self.alpha, *self.comparison_alphas)):
            raise ValueError("BH alpha levels must be in (0, 1).")
        if self.batch_size < 1 or self.max_depth < 1 or self.max_nodes < 1:
            raise ValueError("batch_size, max_depth and max_nodes must be positive.")
        if self.max_seconds is not None and (not np.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be positive and finite.")

    @classmethod
    def from_mapping(cls, raw):
        raw = dict(raw or {})
        raw.pop("mode", None)
        if "comparison_alphas" in raw:
            raw["comparison_alphas"] = tuple(raw["comparison_alphas"])
        settings = cls(**raw)
        settings.validate()
        return settings


def _atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _run_lock(directory):
    with (directory / "run.lock").open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is writing this FDR run directory.") from exc
        yield


def load_samples(config):
    """Retain sample alignment and replicates; never use GC's day aggregation.

    Kutsche matrices use the existing WT filter. Generic input is a long CSV:
    Gene, Sample, Day, Expression. Duplicate labels/sample cells are rejected.
    """
    if config.dataset.name.lower() == "kutsche":
        raw = load_and_preprocess_data(config.dataset.expression_file)
        if raw.index.has_duplicates:
            if config.analysis.duplicate_gene_method == "error":
                raise ValueError("Duplicate gene symbols in input; set analysis.duplicate_gene_method explicitly.")
            raw = aggregate_duplicate_genes(raw, method=config.analysis.duplicate_gene_method)
        expression, days, columns = filter_data_wt(
            raw, normalize=config.preprocessing.normalize,
            transformed=config.preprocessing.transform,
        )
        times = np.array([days[column] for column in columns], dtype=float)
        unique_days, counts = np.unique(times, return_counts=True)
        if not np.array_equal(unique_days, np.arange(5)) or not np.all(counts == 7):
            raise ValueError("Kutsche FDR expects seven unaggregated WT samples at each of days 0-4.")
    else:
        raw = pd.read_csv(config.dataset.expression_file)
        required = {"Gene", "Sample", "Day", "Expression"}
        if not required.issubset(raw.columns):
            raise ValueError("FDR sample CSV requires Gene, Sample, Day, Expression columns.")
        if raw[list(required)].isna().any().any():
            raise ValueError("Sample CSV contains missing values.")
        if (raw.groupby("Sample").Day.nunique() != 1).any():
            raise ValueError("Each sample must have exactly one numeric day.")
        expression = raw.pivot(index="Gene", columns="Sample", values="Expression")
        times = raw.groupby("Sample").Day.first().loc[expression.columns].to_numpy(float)
        if not np.isfinite(expression.to_numpy(float)).all():
            raise ValueError("All genes must have the same complete sample measurements.")
        expression = expression.loc[(expression != 0).any(axis=1)]
        expression = apply_expression_preprocessing(
            expression, normalize=config.preprocessing.normalize,
            transform=config.preprocessing.transform,
        )
    genes = expression.index.astype(str).tolist()
    if len(set(genes)) != len(genes) or any(not gene.strip() for gene in genes):
        raise ValueError("Gene identifiers must be unique and nonempty.")
    requested = list(dict.fromkeys(config.seed_gene_file.read_text(encoding="utf-8").split()))
    lookup = {gene: i for i, gene in enumerate(genes)}
    subset = [lookup[gene] for gene in requested if gene in lookup]
    missing = [gene for gene in requested if gene not in lookup]
    if len(subset) < 2 or config.gene_of_interest not in {genes[i] for i in subset}:
        raise ValueError("Need at least two retained seed genes, including the anchor.")
    return expression.to_numpy(float).T, times, genes, subset, missing


class Family:
    """Disk-backed cumulative unique directed test family."""

    def __init__(self, directory, fingerprint):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "tests.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY, source INTEGER NOT NULL, target INTEGER NOT NULL,
                coefficient REAL, standard_error REAL, statistic REAL, p REAL,
                UNIQUE(source, target));
            CREATE INDEX IF NOT EXISTS tests_target ON tests(target);
            CREATE TABLE IF NOT EXISTS layers (name TEXT PRIMARY KEY, plan TEXT NOT NULL,
                result TEXT);
        """)
        old = self.db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
        if old and old[0] != fingerprint:
            self.db.close()
            raise ValueError("Resume refused: input data or analysis settings changed. Use a new run_name.")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('fingerprint', ?)", (fingerprint,))

    def plan(self, name, sources, universe, direction):
        plan = {"sources": sources, "universe": universe, "direction": direction}
        row = self.db.execute("SELECT plan, result FROM layers WHERE name=?", (name,)).fetchone()
        if row:
            if json.loads(row[0]) != plan:
                raise ValueError(f"Resume plan changed for {name}.")
            return json.loads(row[1]) if row[1] else None
        with self.db:
            self.db.execute("INSERT INTO layers VALUES (?, ?, NULL)", (name, json.dumps(plan)))
        return None

    def blocks(self, sources, universe, direction, batch_size):
        """Generate jobs without materializing all G*(G-1) pair identities."""
        source_set, universe_set = set(sources), set(universe)

        def pairs():
            if direction in {"both", "outgoing"}:
                for source in sources:
                    for target in universe:
                        if source != target:
                            yield source, target
            if direction in {"both", "incoming"}:
                for target in sources:
                    for source in universe:
                        if source != target and not (
                            direction == "both" and source in source_set and target in universe_set
                        ):
                            yield source, target

        iterator = pairs()
        while block := list(islice(iterator, batch_size)):
            yield block

    def count(self):
        return self.db.execute("SELECT count(*) FROM tests").fetchone()[0]

    def pending(self, block):
        if not block:
            return []
        # Batches can span multiple predictor rows. Use whichever indexed
        # orientation needs fewer lookups, never one query per pair.
        sources, targets = {pair[0] for pair in block}, {pair[1] for pair in block}
        column, keys = ("source", sources) if len(sources) <= len(targets) else ("target", targets)
        wanted = set(block)
        existing = {pair for key in keys
                    for pair in self.db.execute(f"SELECT source,target FROM tests WHERE {column}=?", (key,))
                    if pair in wanted}
        return [pair for pair in block if pair not in existing]

    def save(self, pairs, statistics):
        values = np.column_stack(statistics)
        clean = values.astype(object)
        clean[~np.isfinite(values)] = None

        def rows():
            for pair, numbers in zip(pairs, clean.tolist()):
                yield (*pair, *numbers)
        with self.db:
            self.db.executemany(
                "INSERT INTO tests(source,target,coefficient,standard_error,statistic,p) VALUES(?,?,?,?,?,?)",
                rows(),
            )


class FDRRunner:
    def __init__(self, config):
        self.config = config
        self.settings = config.analysis
        self.directory = config.run_dir / "fdr"
        self.started = clock.monotonic()
        self.artifacts = {}

    def progress(self, stage, **values):
        data = {"stage": stage, "elapsed_seconds": round(clock.monotonic() - self.started, 2), **values}
        _atomic_json(self.directory / "progress.json", data)
        if self.config.execution.progress:
            details = " ".join(f"{key}={value}" for key, value in values.items())
            print(f"FDR {stage}: {details}", flush=True)

    def expired(self):
        limit = self.settings.max_seconds
        return limit is not None and clock.monotonic() - self.started >= limit

    def fit_layer(self, family, name, sources, universe, direction, *, alphas):
        previous = family.plan(name, sources, universe, direction)
        if previous is not None:
            return previous
        initial = family.count()
        source_set = set(sources)
        outgoing = len(sources) * len(universe) - len(source_set.intersection(universe))
        # Both directions overlap on ordered pairs internal to the frontier.
        overlap = len(source_set.intersection(universe))
        planned = outgoing if direction != "both" else 2 * outgoing - overlap * (overlap - 1)
        self.progress(name, status="fitting", family_tests=initial, layer_planned_tests=planned)
        new, checked = 0, 0
        fitting_seconds, checkpoint_seconds, lookup_seconds = 0.0, 0.0, 0.0
        layer_started = clock.monotonic()
        for block in family.blocks(sources, universe, direction, self.settings.batch_size):
            checked += len(block)
            tick = clock.monotonic()
            pairs = family.pending(block)
            lookup_seconds += clock.monotonic() - tick
            if not pairs:
                continue
            if self.expired():
                self.progress(name, status="paused", reason="time_budget", family_tests=initial + new)
                return None
            source, target = np.asarray(pairs, dtype=int).T
            tick = clock.monotonic()
            statistics = self.model.fit(source, target)
            fitting_seconds += clock.monotonic() - tick
            tick = clock.monotonic()
            family.save(pairs, statistics)
            checkpoint_seconds += clock.monotonic() - tick
            new += len(pairs)
            self.progress(name, status="fitting", family_tests=initial + new,
                          new_tests_this_invocation=new, layer_planned_tests=planned,
                          layer_checked_tests=checked, layer_percent=round(100 * checked / max(1, planned), 1),
                          fit_seconds=round(fitting_seconds, 3), checkpoint_seconds=round(checkpoint_seconds, 3),
                          lookup_seconds=round(lookup_seconds, 3),
                          new_tests_per_second=round(new / max(clock.monotonic() - layer_started, 1e-9), 1))
        result = self.export(family, name, alphas, universe)
        with family.db:
            family.db.execute("UPDATE layers SET result=? WHERE name=?", (json.dumps(result), name))
        return result

    def export(self, family, name, alphas, universe):
        n = family.count()
        self.progress(name, status="adjusting_bh", family_tests=n)
        p = np.fromiter((row[0] if row[0] is not None else np.nan
                         for row in family.db.execute("SELECT p FROM tests ORDER BY id")), float, count=n)
        q = benjamini_hochberg(p)
        source = np.fromiter((row[0] for row in family.db.execute("SELECT source FROM tests ORDER BY id")), int, count=n)
        target = np.fromiter((row[0] for row in family.db.execute("SELECT target FROM tests ORDER BY id")), int, count=n)
        output = family.directory / name
        output.mkdir(exist_ok=True)
        valid = np.isfinite(q)
        summaries = {}
        for alpha in dict.fromkeys(alphas):
            selected = valid & (q <= alpha)
            rows, cols = source[selected], target[selected]
            graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(self.genes), len(self.genes))).tocsr()
            _, labels = connected_components(graph, directed=False)
            nodes = np.unique(np.concatenate([rows, cols])).tolist()
            eligible = np.union1d(universe, np.union1d(source, target))
            sizes = np.bincount(labels[eligible]) if len(eligible) else np.array([], dtype=int)
            undirected = np.unique(np.minimum(rows, cols) * len(self.genes) + np.maximum(rows, cols))
            key = f"{alpha:g}"
            summaries[key] = {
                "selected_directed_edges": int(selected.sum()),
                "selected_undirected_pairs": len(undirected), "connected_genes": len(nodes),
                "eligible_genes": len(eligible), "isolated_genes": len(eligible) - len(nodes),
                "components_including_isolates": int(np.count_nonzero(sizes)),
                "largest_component_genes": int(sizes.max()) if len(sizes) else 0,
                "directed_density": float(selected.sum() / max(1, len(eligible) * (len(eligible) - 1))),
                "raw_p_cutoff": float(p[selected].max()) if selected.any() else None,
                "nodes": nodes,
            }
        self.progress(name, status="exporting", family_tests=n)
        # Stream raw statistics and adjusted p-values; never prefilter before BH.
        paths = {"all": output / "all_tests.csv"}
        paths.update({f"{a:g}": output / f"selected_q_{a:g}.csv" for a in dict.fromkeys(alphas)})
        handles = {key: path.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") for key, path in paths.items()}
        try:
            writers = {key: csv.writer(handle) for key, handle in handles.items()}
            header = ["predictor_gene", "target_gene", "interaction_coefficient", "standard_error",
                      "t_statistic", "raw_p_value", "bh_adjusted_p_value", "status"]
            for writer in writers.values():
                writer.writerow(header)
            for index, row in enumerate(family.db.execute(
                "SELECT source,target,coefficient,standard_error,statistic,p FROM tests ORDER BY id"
            )):
                a, b, *numbers = row
                record = [self.genes[a], self.genes[b], *numbers,
                          float(q[index]) if valid[index] else "", "ok" if valid[index] else "failed"]
                writers["all"].writerow(record)
                if valid[index]:
                    for alpha in dict.fromkeys(alphas):
                        if q[index] <= alpha:
                            writers[f"{alpha:g}"].writerow(record)
                if (index + 1) % 100_000 == 0:
                    self.progress(name, status="exporting", exported_tests=index + 1, family_tests=n)
        finally:
            for handle in handles.values():
                handle.close()
        for path in paths.values():
            path.with_suffix(".csv.tmp").replace(path)
        primary = summaries[f"{self.settings.alpha:g}"]
        # Compatible graph input: p-value is explicitly BH-adjusted; lag is N/A.
        network_path = output / "network_edges.csv"
        with network_path.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["gene1", "gene2", "lag", "p-value", "raw_p_value", "analysis"])
            for i in np.flatnonzero(valid & (q <= self.settings.alpha)):
                writer.writerow([self.genes[source[i]], self.genes[target[i]], "NaN", q[i], p[i], "quadratic_interaction_bh"])
        network_path.with_suffix(".csv.tmp").replace(network_path)
        self.artifacts[f"{family.directory.name}_{name}_network"] = network_path
        result = {"family_tests": n, "failed_tests": int((~valid).sum()),
                  "bh_denominator": n, "summaries": summaries,
                  "selected_nodes": primary["nodes"], "network_edges": str(network_path)}
        _atomic_json(output / "summary.json", result)
        self.plot_overview(output, source, target, q, alphas)
        self.progress(name, status="complete", family_tests=n,
                      selected_edges=primary["selected_directed_edges"])
        return result

    def plot_overview(self, directory, source, target, q, alphas):
        """Compare network density without rendering millions of edge strokes."""
        from matplotlib.figure import Figure

        figure = Figure(figsize=(9, 4))
        axes = figure.subplots(1, 2)
        levels = list(dict.fromkeys(alphas))
        for alpha in levels:
            selected = np.isfinite(q) & (q <= alpha)
            degree = np.bincount(np.concatenate([source[selected], target[selected]]), minlength=len(self.genes))
            # Only include genes eligible for this family, not unrelated full-data genes.
            eligible = np.unique(np.concatenate([source, target]))
            axes[0].hist(degree[eligible], bins=30, histtype="step", label=f"BH {alpha:g}")
        axes[0].set(xlabel="Incident directed edges per gene", ylabel="Genes")
        axes[0].legend()
        counts = [int(np.sum(np.isfinite(q) & (q <= a))) for a in levels]
        axes[1].bar(range(len(levels)), counts)
        axes[1].set_xticks(range(len(levels)))
        axes[1].set_xticklabels([f"{a:g}" for a in levels])
        axes[1].set(xlabel="BH threshold", ylabel="Selected directed edges")
        figure.tight_layout()
        figure.savefig(directory / "overview.svg")

    def run(self):
        self.settings.validate()
        self.directory.mkdir(parents=True, exist_ok=True)
        with _run_lock(self.directory):
            try:
                return self._run_locked()
            except KeyboardInterrupt:
                self.progress("pipeline", status="interrupted", resumable=True)
                raise
            except Exception as exc:
                self.progress("pipeline", status="failed", error=str(exc), resumable=True)
                raise

    def _run_locked(self):
        if not self.config.execution.resume and any(self.directory.glob("*/tests.sqlite")):
            raise ValueError("Existing FDR results found with resume=false; use a new run_name.")
        if self.settings.backend == "gpu_cuda":
            self.progress("cuda_preflight", status="checking_kernel_compilation")
            require_cuda_kernels(self.config.execution.gpu_device)
        self.progress("preprocessing", status="loading_samples")
        values, times, self.genes, subset, missing = load_samples(self.config)
        semantic = asdict(self.settings)
        for key in ("batch_size", "max_seconds"):
            semantic.pop(key)
        specification = {"schema": 1, "settings": semantic, "genes": self.genes, "subset": subset,
                         "anchor": self.config.gene_of_interest,
                         "normalize": self.config.preprocessing.normalize,
                         "transform": self.config.preprocessing.transform}
        digest = hashlib.sha256(json.dumps(specification, sort_keys=True).encode())
        digest.update(np.ascontiguousarray(values).tobytes())
        digest.update(np.ascontiguousarray(times).tobytes())
        fingerprint = digest.hexdigest()
        manifest = self.directory / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("Resume refused: input data or analysis settings changed. Use a new run_name.")
        self.model = QuadraticInteraction(values, times, backend=self.settings.backend,
                                          gpu_device=self.config.execution.gpu_device)
        _atomic_json(manifest, {**specification, "fingerprint": fingerprint,
                               "expression_file": str(self.config.dataset.expression_file),
                               "seed_gene_file": str(self.config.seed_gene_file),
                               "model": "target ~ 1 + time + time^2 + predictor + time:predictor",
                               "tested_coefficient": "time:predictor",
                               "samples": len(times), "missing_or_zero_seed_genes": missing,
                               "retained_subset_genes": len(subset),
                               "planned_subset_directed_tests": len(subset) * (len(subset) - 1),
                               "residual_degrees_of_freedom": self.model.df,
                               "adaptive_search": "exploratory cumulative BH; no global adaptive FDR guarantee"})
        completed = True
        if self.settings.workflow in {"both", "all_pairs"}:
            family = Family(self.directory / "all_pairs", fingerprint)
            try:
                result = self.fit_layer(family, "overview", subset, subset, "outgoing",
                                        alphas=(self.settings.alpha, *self.settings.comparison_alphas))
                completed = result is not None
            finally:
                family.db.close()
        if completed and self.settings.workflow in {"both", "anchor"}:
            completed = self.run_anchor(fingerprint, subset)
        self.artifacts.update(manifest=manifest, progress=self.directory / "progress.json")
        for summary_path in self.directory.glob("*/*/summary.json"):
            self.artifacts[f"{summary_path.parent.parent.name}_{summary_path.parent.name}_network"] = (
                summary_path.parent / "network_edges.csv"
            )
        self.write_run_summary(completed)
        self.progress("pipeline", status="complete" if completed else "paused")
        return self.artifacts

    def run_anchor(self, fingerprint, subset):
        family = Family(self.directory / "anchor", fingerprint)
        anchor = self.genes.index(self.config.gene_of_interest)
        discovered, expanded, frontier = {anchor}, set(), [anchor]
        reason = "max_depth"
        try:
            for depth in range(1, self.settings.max_depth + 1):
                result = self.fit_layer(family, f"depth_{depth:02d}", frontier, subset,
                                        self.settings.direction, alphas=(self.settings.alpha,))
                if result is None:
                    return False
                expanded.update(frontier)
                discovered.update(result["selected_nodes"])
                frontier = sorted(discovered - expanded)
                if len(discovered) >= self.settings.max_nodes:
                    reason = "max_nodes_after_complete_layer"
                    break
                if not frontier:
                    reason = "no_new_neighbors"
                    break
            core = sorted(discovered)
            core_path = family.directory / "core_genes.txt"
            core_path.write_text("".join(self.genes[i] + "\n" for i in core), encoding="utf-8")
            self.artifacts["core_genes"] = core_path
            _atomic_json(family.directory / "core_summary.json", {
                "stop_reason": reason, "core_genes": [self.genes[i] for i in core],
                "discovery_policy": "cumulative ever-selected nodes; current edges recalculated at each layer",
            })
            if self.settings.genome_expansion:
                result = self.fit_layer(family, "genome_probe", core, list(range(len(self.genes))),
                                        self.settings.direction, alphas=(self.settings.alpha,))
                if result is None:
                    return False
            return True
        finally:
            family.db.close()

    def write_run_summary(self, completed):
        path = self.directory / "RUN_SUMMARY.md"
        lines = ["# Interaction regression with BH selection", "",
                 f"Status: {'complete' if completed else 'paused; rerun the same command to resume'}.", "",
                 "Model: target ~ 1 + time + time² + predictor + time×predictor; two-sided OLS interaction t test.",
                 "Replicates are retained. No permutations or bootstrap are used for selection.", "",
                 "## Network summaries", ""]
        for summary_path in sorted(self.directory.glob("*/*/summary.json")):
            result = json.loads(summary_path.read_text(encoding="utf-8"))
            lines.append(f"- {summary_path.parent.relative_to(self.directory)}: N={result['family_tests']}, failed={result['failed_tests']}")
            for alpha, summary in result["summaries"].items():
                lines.append(f"  - BH {alpha}: {summary['selected_directed_edges']} directed edges, "
                             f"{summary['connected_genes']}/{summary['eligible_genes']} genes connected, "
                             f"density {summary['directed_density']:.2%}, "
                             f"largest component {summary['largest_component_genes']} genes.")
        lines.extend(["", "## Notes for the later result email to Jing", "",
                      "Lukas's concern is that BH may include nearly every gene and produce a dense network, "
                      "making it difficult to identify a focused set. Report both gene coverage and edge density; "
                      "including all genes does not mean selecting all pairs. Compare 5% with 10% where requested.", "",
                      "The denominator uses the actual unique directed non-self tests. Failed fits remain in N "
                      "as p=1 for adjustment and cannot be selected. Both directions are one family.", "",
                      "Anchor expansion follows the proposed cumulative-BH heuristic. Its adaptive choice of tests "
                      "does not establish global 5% FDR control. Regression arrows describe model direction, not causality.", "",
                      "Core genes are the anchor and all nodes discovered at any completed local layer. "
                      "The node limit stops after a complete layer and can be exceeded. The time budget pauses execution; "
                      "increase it and resume to complete remaining work.", "",
                      "network_edges.csv uses BH-adjusted values in the p-value column for existing graph readers; "
                      "raw_p_value is retained separately and lag is NaN. No Louvain consensus is run by this workflow.", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        self.artifacts["summary"] = path


def run_fdr_pipeline(config):
    return FDRRunner(config).run()
