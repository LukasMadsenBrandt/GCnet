"""Experimental CuPy-backed lag-1 Granger-causality runner."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator

from scipy.stats import f as f_distribution

from gene_analysis.analysis.cuda_environment import preload_cuda_wheel_libraries
from gene_analysis.analysis.granger_runner import (
    _append_significant_rows,
    _batched,
    _bit_count,
    _bit_get,
    _bit_set,
    _load_or_init_bitset,
    _pair_index,
    _prepare_fresh_run,
    _save_bitset,
)


def perform_gc_cuda(
    df_filtered_wt_weighted_mean,
    genes_file: str,
    output_file: str,
    p_threshold: float = 0.05,
    chunk_size: int = 25_000,
    list_to_kutsche: bool = False,
    max_workers=None,
    progress: bool = True,
    resume: bool = True,
    rename_at_end: bool = True,
    gpu_device: int | None = None,
    record_failed_pairs: bool = False,
):
    """
    Run lag-1 Granger causality using CuPy for the regression calculations.

    This experimental backend mirrors the canonical CSV/checkpoint contract and
    only writes significant rows. It intentionally supports the current
    pipeline's lag-1 use case; CPU remains the reference implementation.
    """
    del max_workers
    preload_cuda_wheel_libraries()
    import cupy as cp

    start = time.time()
    if gpu_device is not None:
        cp.cuda.Device(int(gpu_device)).use()

    ts_data = df_filtered_wt_weighted_mean.T
    with open(genes_file, "r", encoding="utf-8") as fh:
        requested = [line.strip() for line in fh if line.strip()]

    if list_to_kutsche:
        genes_of_interest = [gene for gene in requested if gene in ts_data.columns]
        if not genes_of_interest:
            raise ValueError("No genes_of_interest from file are present in the dataframe columns.")
        dataset_genes = sorted(set(map(str, ts_data.columns)))
        pairs = [(goi, gene) for goi in genes_of_interest for gene in dataset_genes if gene != goi]
        pairs += [(gene, goi) for gene in dataset_genes for goi in genes_of_interest if gene != goi]
        pairs = list(dict.fromkeys(pairs))
        total_pairs_all = len(pairs)

        def skipping_iter() -> Iterator[tuple[str, str, int]]:
            for idx, (gene1, gene2) in enumerate(pairs):
                if resume and _bit_get(bitset, idx):
                    continue
                yield gene1, gene2, idx

        genes = []
    else:
        genes = [gene for gene in requested if gene in ts_data.columns]
        if len(genes) < 2:
            raise ValueError("Need at least two genes (after filtering to dataframe columns).")
        total_pairs_all = len(genes) * (len(genes) - 1)

        def skipping_iter() -> Iterator[tuple[str, str, int]]:
            for i, gene1 in enumerate(genes):
                for j, gene2 in enumerate(genes):
                    if i == j:
                        continue
                    idx = _pair_index(i, j, len(genes))
                    if resume and _bit_get(bitset, idx):
                        continue
                    yield gene1, gene2, idx

    checkpoint_path = os.path.splitext(output_file)[0] + ".seen.bin"
    _prepare_fresh_run(output_file, checkpoint_path, resume)
    bitset = _load_or_init_bitset(checkpoint_path, total_pairs_all)
    processed_pairs = 0
    significant_count = 0
    failed_count = 0

    if list_to_kutsche:
        genes_to_load = sorted({gene for pair in pairs for gene in pair})
    else:
        genes_to_load = genes
    series_by_gene = {
        str(gene): cp.asarray(ts_data[gene].to_numpy(dtype=float), dtype=cp.float64)
        for gene in genes_to_load
    }

    for chunk in _batched(skipping_iter(), chunk_size):
        rows = []
        chunk_gene1 = [gene1 for gene1, _gene2, _idx in chunk]
        chunk_gene2 = [gene2 for _gene1, gene2, _idx in chunk]
        causes = cp.stack([series_by_gene[gene] for gene in chunk_gene1], axis=0)
        targets = cp.stack([series_by_gene[gene] for gene in chunk_gene2], axis=0)
        p_values = _lag1_granger_p_values_cupy(causes, targets, cp)
        for (gene1, gene2, idx), p_value in zip(chunk, p_values):
            _bit_set(bitset, idx)
            if p_value is not None and p_value <= p_threshold:
                row = [gene1, gene2, "1", format(p_value, ".17g")]
                if record_failed_pairs:
                    row.append("")
                rows.append(row)
            elif p_value is None and record_failed_pairs:
                rows.append([gene1, gene2, "1", "NaN", "no usable lag-1 F-test result"])
                failed_count += 1
        if rows:
            _append_significant_rows(output_file, rows, include_error=record_failed_pairs)
            significant_count += sum(1 for row in rows if str(row[3]).lower() != "nan")
        _save_bitset(checkpoint_path, bitset)
        processed_pairs += len(chunk)
        if progress:
            print(f"CUDA GC progress: {processed_pairs}/{total_pairs_all}")

    final_path = output_file
    if rename_at_end:
        threshold_str = str(p_threshold).replace(".", "")
        dir_, _base = os.path.split(output_file)
        final_path = os.path.join(dir_ or ".", f"granger_results_p{threshold_str}_{significant_count}of{total_pairs_all}.csv")
        try:
            os.replace(output_file, final_path)
        except FileNotFoundError:
            _append_significant_rows(output_file, [], include_error=record_failed_pairs)
            os.replace(output_file, final_path)
    elif not os.path.exists(output_file):
        _append_significant_rows(output_file, [], include_error=record_failed_pairs)

    print(f"Total pairs: {total_pairs_all}")
    print(f"Processed this run: {processed_pairs}")
    print(f"Significant edges saved: {significant_count}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {final_path}")
    elapsed = time.time() - start
    print(f"Total execution time of perform_gc_cuda: {elapsed} seconds")

    return {
        "total_pairs_all": total_pairs_all,
        "processed_this_run": processed_pairs,
        "significant_edges": significant_count,
        "failed_pairs": failed_count,
        "p_threshold": p_threshold,
        "output_file": final_path,
        "checkpoint_path": checkpoint_path,
        "backend": "gpu_cuda",
        "elapsed_seconds": elapsed,
        "seen_count": _bit_count(bytearray(Path(checkpoint_path).read_bytes())) if Path(checkpoint_path).exists() else 0,
    }


def _lag1_granger_p_value_cupy(cause, target, cp) -> float | None:
    """Return the lag-1 SSR F-test p-value for cause -> target."""
    return _lag1_granger_p_values_cupy(cp.expand_dims(cause, 0), cp.expand_dims(target, 0), cp)[0]


def _lag1_granger_p_values_cupy(causes, targets, cp) -> list[float | None]:
    """Return lag-1 SSR F-test p-values for a batch of cause -> target pairs."""
    if causes.ndim != 2 or targets.ndim != 2:
        raise ValueError("causes and targets must be 2D arrays with shape (pairs, timepoints).")
    if causes.shape != targets.shape:
        raise ValueError("causes and targets must have the same shape.")
    if causes.shape[1] < 4:
        return [None] * int(causes.shape[0])

    try:
        return _lag1_granger_p_values_cupy_vectorized(causes, targets, cp)
    except Exception:
        return [_lag1_granger_p_value_cupy_scalar(causes[i], targets[i], cp) for i in range(int(causes.shape[0]))]


def _lag1_granger_p_values_cupy_vectorized(causes, targets, cp) -> list[float | None]:
    """Vectorized CuPy implementation of the lag-1 Granger SSR F-test."""
    pair_count = int(causes.shape[0])
    valid = (cp.std(causes, axis=1) != 0) & (cp.std(targets, axis=1) != 0)

    y = targets[:, 1:]
    y_lag = targets[:, :-1]
    x_lag = causes[:, :-1]
    ones = cp.ones_like(y)
    unrestricted = cp.stack([ones, y_lag, x_lag], axis=2)
    restricted = cp.stack([ones, y_lag], axis=2)

    beta_unrestricted = _batched_ols_beta(unrestricted, y, cp)
    beta_restricted = _batched_ols_beta(restricted, y, cp)

    residual_unrestricted = y - cp.sum(unrestricted * beta_unrestricted[:, None, :], axis=2)
    residual_restricted = y - cp.sum(restricted * beta_restricted[:, None, :], axis=2)
    rss_unrestricted = cp.sum(residual_unrestricted**2, axis=1)
    rss_restricted = cp.sum(residual_restricted**2, axis=1)
    df_den = int(y.shape[1]) - int(unrestricted.shape[2])
    if df_den <= 0:
        return [None] * pair_count

    p_values: list[float | None] = [None] * pair_count
    valid_np = cp.asnumpy(valid)
    rss_unrestricted_np = cp.asnumpy(rss_unrestricted)
    rss_restricted_np = cp.asnumpy(rss_restricted)
    f_stats = ((rss_restricted_np - rss_unrestricted_np) / 1.0) / (rss_unrestricted_np / df_den)
    f_stats[f_stats < 0] = 0.0
    p_value_np = f_distribution.sf(f_stats, 1, df_den)

    for idx in range(pair_count):
        if not bool(valid_np[idx]):
            continue
        if rss_unrestricted_np[idx] <= 1e-12:
            p_values[idx] = 0.0 if rss_restricted_np[idx] > 1e-12 else 1.0
        else:
            p_values[idx] = float(p_value_np[idx])
    return p_values


def _batched_ols_beta(design, y, cp):
    """Solve many small OLS systems using batched normal equations."""
    xt = cp.swapaxes(design, 1, 2)
    xtx = xt @ design
    xty = xt @ y[:, :, None]
    return cp.linalg.solve(xtx, xty)[:, :, 0]


def _lag1_granger_p_value_cupy_scalar(cause, target, cp) -> float | None:
    """Scalar fallback for singular batched systems."""
    if cause.size < 4 or target.size < 4:
        return None
    if float(cp.std(cause).get()) == 0.0 or float(cp.std(target).get()) == 0.0:
        return None

    y = target[1:]
    y_lag = target[:-1]
    x_lag = cause[:-1]
    ones = cp.ones_like(y)
    unrestricted = cp.stack([ones, y_lag, x_lag], axis=1)
    restricted = cp.stack([ones, y_lag], axis=1)

    try:
        beta_unrestricted = cp.linalg.lstsq(unrestricted, y, rcond=None)[0]
        beta_restricted = cp.linalg.lstsq(restricted, y, rcond=None)[0]
    except Exception:
        return None

    residual_unrestricted = y - unrestricted @ beta_unrestricted
    residual_restricted = y - restricted @ beta_restricted
    rss_unrestricted = float(cp.sum(residual_unrestricted**2).get())
    rss_restricted = float(cp.sum(residual_restricted**2).get())
    df_den = int(y.size) - unrestricted.shape[1]
    if df_den <= 0:
        return None
    if rss_unrestricted <= 1e-12:
        if rss_restricted > 1e-12:
            return 0.0
        return 1.0

    f_stat = ((rss_restricted - rss_unrestricted) / 1.0) / (rss_unrestricted / df_den)
    if f_stat < 0:
        f_stat = 0.0
    return float(f_distribution.sf(f_stat, 1, df_den))
