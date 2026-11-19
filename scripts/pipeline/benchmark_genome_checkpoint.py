#!/usr/bin/env python3
"""Benchmark genome checkpoint encodings on the filesystem used by a run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import pyarrow.parquet as pq


VARIANTS = {
    "current_zstd": dict(compression="zstd", row_group_size=131072),
    "snappy": dict(compression="snappy", row_group_size=131072),
    "uncompressed": dict(compression=None, row_group_size=131072),
    "fast_snappy": dict(compression="snappy", row_group_size=None,
                         use_dictionary=False, write_statistics=False),
    "fast_uncompressed": dict(compression=None, row_group_size=None,
                               use_dictionary=False, write_statistics=False),
}


def durable_write(table, temporary: Path, destination: Path, options):
    started = time.perf_counter()
    pq.write_table(table, temporary, **options)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(
        description="Compare durable Parquet writes using one completed raw genome shard.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Genome FDR directory containing raw/*.parquet.")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    sources = sorted((args.run_dir / "raw").glob("*.parquet"))
    if not sources:
        parser.error(f"No raw checkpoints found under {args.run_dir}")
    source = sources[0]
    table = pq.read_table(source)
    benchmark = args.run_dir / "checkpoint_benchmark"
    benchmark.mkdir(exist_ok=True)
    results = {}
    try:
        for name, options in VARIANTS.items():
            measurements = []
            size = None
            for repeat in range(args.repeats):
                temporary = benchmark / f"{name}-{repeat}.parquet.tmp"
                destination = benchmark / f"{name}-{repeat}.parquet"
                seconds = durable_write(table, temporary, destination, options)
                if pq.read_metadata(destination).num_rows != table.num_rows:
                    raise RuntimeError(f"Row-count verification failed for {name}")
                size = destination.stat().st_size
                measurements.append(seconds)
                destination.unlink()
            results[name] = {
                "median_seconds": statistics.median(measurements),
                "minimum_seconds": min(measurements),
                "bytes": size,
                "rows_per_second": table.num_rows / statistics.median(measurements),
                "megabytes_per_second": size / statistics.median(measurements) / 1_000_000,
            }
    finally:
        for path in benchmark.iterdir():
            path.unlink()
        benchmark.rmdir()
    current = results["current_zstd"]["median_seconds"]
    for result in results.values():
        result["speedup_vs_current"] = current / result["median_seconds"]
    print(json.dumps({
        "source": str(source),
        "rows": table.num_rows,
        "source_bytes": source.stat().st_size,
        "repeats": args.repeats,
        "timing_scope": "Parquet write, file fsync, atomic rename and directory fsync; source read excluded",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
