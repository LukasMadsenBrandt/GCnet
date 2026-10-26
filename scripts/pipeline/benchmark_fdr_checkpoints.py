#!/usr/bin/env python3
"""Compare old and optimized FDR checkpoints with synthetic rows, without CUDA.

Only a temporary directory owned by this benchmark is used and removed. No
expression data, regression fitting or existing pipeline checkpoints are read.
Use --directory to choose the filesystem that holds the temporary databases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.pipeline.fdr import Family


class LegacyFamily(Family):
    """Reproduce the previous reverse-index/default-cache write pattern."""

    def __init__(self, directory):
        super().__init__(directory, "synthetic-checkpoint-benchmark")
        self.db.execute("PRAGMA cache_size=-2000")
        with self.db:
            self.db.execute("CREATE INDEX tests_target ON tests(target)")

    def pending(self, block):
        sources, targets = {pair[0] for pair in block}, {pair[1] for pair in block}
        column, keys = ("source", sources) if len(sources) <= len(targets) else ("target", targets)
        wanted = set(block)
        existing = {pair for key in keys
                    for pair in self.db.execute(f"SELECT source,target FROM tests WHERE {column}=?", (key,))
                    if pair in wanted}
        return [pair for pair in block if pair not in existing]


def profile(directory, source, target, batch_size, optimized):
    if optimized:
        family = Family(directory, "synthetic-checkpoint-benchmark")
        # The updated generator groups both directions by source. Preserve the
        # exact benchmark pair set while changing its insertion order.
        order = np.lexsort((target, source))
        source, target = source[order], target[order]
    else:
        family = LegacyFamily(directory)
    timings, lookup_seconds = [], 0.0
    try:
        for offset in range(0, len(source), batch_size):
            pairs = list(zip(source[offset:offset + batch_size].tolist(),
                             target[offset:offset + batch_size].tolist()))
            started = time.perf_counter()
            pending = family.pending(pairs)
            lookup_seconds += time.perf_counter() - started
            assert pending == pairs
            # Non-integer floats occupy the same SQL field sizes as model
            # statistics; zero/one values compress and understate write volume.
            batch_statistics = tuple(np.full(len(pairs), value) for value in
                                     (0.123456789, 0.987654321, 2.345678901, 0.234567891))
            started = time.perf_counter()
            family.save(pairs, batch_statistics)
            timings.append(time.perf_counter() - started)
        count, source_sum, target_sum = family.db.execute(
            "SELECT count(*),sum(source),sum(target) FROM tests"
        ).fetchone()
        assert count == len(source)
        assert source_sum == int(source.sum()) and target_sum == int(target.sum())
        # Previously saved pairs must still be recognized in either order.
        assert family.pending(list(zip(source[:100].tolist(), target[:100].tolist()))) == []
        elapsed = sum(timings) + lookup_seconds
        result = {"rows": count, "save_seconds": sum(timings), "lookup_seconds": lookup_seconds,
                  "save_and_lookup_rows_per_second": count / elapsed,
                  "first_batches_mean_save_seconds": statistics.mean(timings[:4]),
                  "last_batches_mean_save_seconds": statistics.mean(timings[-4:]),
                  "checkpoint_bytes_before_close": sum(p.stat().st_size for p in directory.glob("tests.sqlite*")),
                  "verified_row_counts_and_pair_sums": True}
        return result
    finally:
        family.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1_048_576)
    parser.add_argument("--universe", type=int, default=45_000)
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument("--direction", choices=["outgoing", "incoming", "both"], default="both")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.universe < 2 or args.pairs < 1 or args.batch_size < 1:
        parser.error("Need universe >= 2 and positive pairs/batch-size.")
    if args.pairs > args.universe * (args.universe - 1):
        parser.error("pairs exceeds the available unique directed non-self pairs.")
    if args.directory:
        args.directory.mkdir(parents=True, exist_ok=True)
    indices = np.arange(args.pairs)
    source = indices // (args.universe - 1)
    target = indices % (args.universe - 1)
    target += target >= source
    report = {"synthetic_rows": args.pairs, "universe": args.universe,
              "batch_size": args.batch_size, "durability": "WAL, synchronous=FULL in both versions",
              "timing_scope": "save and deduplication lookup; excludes row generation and database close",
              "filesystem_directory": str(args.directory or Path(tempfile.gettempdir()))}
    directions = ["outgoing", "incoming"] if args.direction == "both" else [args.direction]
    with tempfile.TemporaryDirectory(prefix="fdr-checkpoint-benchmark-", dir=args.directory) as temporary:
        for direction in directions:
            a, b = (source, target) if direction == "outgoing" else (target, source)
            old = profile(Path(temporary) / f"{direction}_legacy", a, b, args.batch_size, False)
            new = profile(Path(temporary) / f"{direction}_optimized", a, b, args.batch_size, True)
            speedup = (old["save_seconds"] + old["lookup_seconds"]) / (new["save_seconds"] + new["lookup_seconds"])
            report[direction] = {"legacy": old, "optimized": new, "save_and_lookup_speedup": speedup}
            print(f"{direction}: {speedup:.2f}x save+lookup speedup", file=sys.stderr, flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
