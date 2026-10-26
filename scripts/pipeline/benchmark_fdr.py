#!/usr/bin/env python3
"""Compare cached FDR regression with the previous per-pair SVD on tiny synthetic data.

No real inputs, BH searches, checkpoint databases or full analyses are used.
GPU timings include synchronous result transfers and CPU p-value calculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.analysis.fdr import QuadraticInteraction


def legacy_fit(values, times, source, target, xp):
    """Build the former fit callable, with preprocessing outside the timer."""
    times = times - times.mean()
    baseline = np.column_stack([np.ones(len(times)), times, times**2])
    q = xp.asarray(np.linalg.qr(baseline, mode="reduced")[0])
    values = xp.asarray(values - values.mean(axis=0))
    interaction = xp.asarray(times)[:, None] * values
    scale = xp.maximum(xp.linalg.norm(xp.asarray(baseline)), xp.sqrt(xp.sum(values**2 + interaction**2, axis=0)))
    residual = values - q @ (q.T @ values)
    interaction = interaction - q @ (q.T @ interaction)
    source, target = xp.asarray(source), xp.asarray(target)

    def fit():
        design = xp.stack([residual[:, source].T, interaction[:, source].T], axis=2)
        response = residual[:, target].T
        u, singular, vh = xp.linalg.svd(design, full_matrices=False)
        tolerance = np.finfo(float).eps * max(len(times), 5) * scale[source, None]
        valid = xp.all(singular > tolerance, axis=1)
        inverse = xp.where(singular > tolerance, 1 / xp.maximum(singular, np.finfo(float).tiny), 0)
        projection = xp.einsum("bni,bn->bi", u, response)
        beta = xp.einsum("bij,bj->bi", vh.transpose(0, 2, 1), inverse * projection)
        error = response - xp.einsum("bni,bi->bn", design, beta)
        variance = xp.sum(error**2, axis=1) / (len(times) - 5)
        se = xp.sqrt(variance * xp.sum((vh[:, :, 1] * inverse)**2, axis=1))
        valid &= (se > 0) & xp.isfinite(se)
        statistic = beta[:, 1] / xp.where(se > 0, se, 1)
        result = [xp.where(valid, value, xp.nan) for value in (beta[:, 1], se, statistic)]
        if xp is not np:
            result = [xp.asnumpy(value) for value in result]
        return (*result, 2 * student_t.sf(np.abs(result[2]), len(times) - 5))

    return fit


def measure(fit, repeats):
    reference = fit()  # Warm up compilation and library initialization.
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        fit()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations), reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    if args.genes < 2 or args.pairs < 1 or args.repeats < 1:
        parser.error("Need genes >= 2 and positive pairs/repeats.")
    rng = np.random.default_rng(85)
    times = np.repeat(np.arange(5), 7).astype(float)
    values = rng.normal(size=(35, args.genes))
    indices = np.arange(args.pairs)
    source = (indices // (args.genes - 1)) % args.genes
    target = (source + 1 + indices % (args.genes - 1)) % args.genes
    report = {"samples": 35, "genes": args.genes, "pairs": args.pairs, "repeats": args.repeats,
              "timing_scope": "warm fit, CPU p-values and result transfers; excludes preparation/checkpoints"}
    cpu = QuadraticInteraction(values, times)
    cached_cpu, reference = measure(lambda: cpu.fit(source, target), args.repeats)
    old_cpu, old_reference = measure(legacy_fit(values, times, source, target, np), args.repeats)
    for expected, actual in zip(reference, old_reference):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-11, equal_nan=True)
    report.update(cached_cpu_seconds=cached_cpu, legacy_cpu_seconds=old_cpu,
                  cached_cpu_pairs_per_second=args.pairs / cached_cpu)
    if not args.cpu_only:
        gpu = QuadraticInteraction(values, times, backend="gpu_cuda", gpu_device=args.gpu_device)
        cached_gpu, actual = measure(lambda: gpu.fit(source, target), args.repeats)
        old_gpu, old_actual = measure(legacy_fit(values, times, source, target, gpu.xp), args.repeats)
        for expected, new, old in zip(reference, actual, old_actual):
            np.testing.assert_allclose(new, expected, rtol=1e-8, atol=1e-11, equal_nan=True)
            np.testing.assert_allclose(old, expected, rtol=1e-8, atol=1e-11, equal_nan=True)
        report.update(cached_gpu_seconds=cached_gpu, legacy_gpu_seconds=old_gpu,
                      cached_gpu_pairs_per_second=args.pairs / cached_gpu,
                      gpu_cached_vs_legacy_speedup=old_gpu / cached_gpu,
                      cached_gpu_vs_cached_cpu_speedup=cached_cpu / cached_gpu,
                      gpu_name=str(gpu.xp.cuda.runtime.getDeviceProperties(args.gpu_device)["name"]))
    report["numerical_comparison"] = "passed"
    output = json.dumps(report, indent=2)
    print(output)
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
