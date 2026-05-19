"""Optimized chunked Granger-causality runner used by the YAML pipeline."""

import os
import csv
import itertools
import multiprocessing as mp
from typing import Iterable, Iterator, List, Tuple, Optional
import time
import warnings

# External deps
from statsmodels.tsa.stattools import grangercausalitytests

# Suppress FutureWarnings from statsmodels
warnings.simplefilter(action='ignore', category=FutureWarning)


def timing_decorator(func):
    """Print the runtime of the optimized Granger runner."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        if isinstance(result, dict):
            result["elapsed_seconds"] = elapsed
        print(f"Total execution time of {func.__name__}: {elapsed} seconds")
        return result

    return wrapper

# ============================================================
# Progress bar (queue-driven; no manual flush)
# ============================================================

def update_progress_bar_tqdm(total: int, queue: mp.Queue):
    """
    Runs in a separate process. Receives integer increments on `queue`.
    Stops when it receives the 'STOP' sentinel.
    """
    try:
        from tqdm import tqdm
        with tqdm(total=total, unit="pair") as pbar:
            while True:
                item = queue.get()
                if item == "STOP":
                    break
                pbar.update(int(item or 0))
    except Exception:
        processed = 0
        while True:
            item = queue.get()
            if item == "STOP":
                break
            processed += int(item or 0)
            if processed % 1000 == 0:
                print(f"Progress: {processed}/{total}")

# ============================================================
# Batching utilities
# ============================================================

def _batched(it: Iterable, n: int):
    it = iter(it)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

# ============================================================
# Tiny, scalable checkpoint: bitset for processed ordered pairs
# ============================================================

# Precompute popcount for bytes 0..255 for fast bit counting
_BITCOUNT_TABLE = bytes(bin(i).count("1") for i in range(256))

def _bit_len(n_bits: int) -> int:
    return (n_bits + 7) // 8

def _bit_get(buf: bytearray, k: int) -> bool:
    return (buf[k >> 3] >> (k & 7)) & 1

def _bit_set(buf: bytearray, k: int) -> None:
    buf[k >> 3] |= (1 << (k & 7))

def _bit_count(buf: bytearray) -> int:
    # Fast: sum popcounts over bytes via table lookup
    return sum(_BITCOUNT_TABLE[b] for b in buf)

def _load_or_init_bitset(path: str, n_bits: int) -> bytearray:
    n_bytes = _bit_len(n_bits)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = bytearray(f.read())
        if len(data) != n_bytes:
            # Genes changed between runs; start fresh
            data = bytearray(n_bytes)
        return data
    else:
        return bytearray(n_bytes)

def _save_bitset(path: str, buf: bytearray) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf)
    os.replace(tmp, path)

def _prepare_fresh_run(output_file: str, checkpoint_path: str, resume: bool) -> None:
    """Remove stale generated files when a GC run is explicitly non-resumable."""
    if resume:
        return
    for path in (output_file, checkpoint_path, checkpoint_path + ".tmp"):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

def _pair_index(i: int, j: int, n: int) -> int:
    """
    Deterministic index for ordered pairs without self-pairs.
    Order: for fixed i in [0..n-1], iterate j in [0..n-1], j!=i.
    Within block i, offset = j if j<i else j-1. Then k = i*(n-1) + offset.
    """
    return i * (n - 1) + (j if j < i else j - 1)

# ============================================================
# Shared dataframe in workers (sent once via Pool initializer)
# ============================================================

_TS_DATA = None  # set by _init_worker

def _init_worker(ts_data):
    global _TS_DATA
    _TS_DATA = ts_data

# Your worker logic (kept simple & robust)
def process_gene_combination(combination, time_series_data, progress_queue, lag = 1):
    """
    Run Granger for a single ordered pair (gene1 causes gene2).
    Returns: ((gene1, gene2), payload) where payload is either:
      - dict with results (as from statsmodels), or
      - {'error': '...'} on failure/constant data
    """
    gene1, gene2 = combination
    test_data = time_series_data[[gene2, gene1]]  # gene1 causes gene2
    try:
        if test_data.std(axis=0).eq(0).any():
            result = combination, {'error': 'constant data'}
        else:
            result = combination, grangercausalitytests(test_data, maxlag=lag, verbose=False)
    except Exception as e:
        result = combination, {'error': str(e)}
    finally:
        if progress_queue is not None:
            progress_queue.put(1)  # advance progress by one pair
    return result

def _worker_wrapper(args):
    combination, progress_queue = args
    return process_gene_combination(combination, _TS_DATA, progress_queue)

# ============================================================
# CSV append for significant rows only
# ============================================================

def _append_significant_rows(path: str, rows: List[Tuple[str, str, str, str]]):
    """
    Append rows with schema: gene1,gene2,lag,p-value (only significant entries should be passed here).
    """
    must_header = (not os.path.exists(path) or os.path.getsize(path) == 0)
    # Ensure directory exists (if path includes dirs)
    dir_ = os.path.dirname(os.path.abspath(path))
    if dir_ and not os.path.exists(dir_):
        os.makedirs(dir_, exist_ok=True)

    with open(path, mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if must_header:
            w.writerow(["gene1", "gene2", "lag", "p-value"])
        w.writerows(rows)

# ============================================================
# Main entry: chunked + streaming + threshold + bitset resume
# ============================================================

@timing_decorator
def perform_gc(
    df_filtered_wt_weighted_mean,
    genes_file: str,
    output_file: str,
    p_threshold: float = 0.05,
    chunk_size: int = 25_000,
    list_to_kutsche: bool = False,
    max_workers: Optional[int] = None,
    pool_chunksize: int = 64,
    progress: bool = True,
    resume: bool = True,
    rename_at_end: bool = True
):
    """
    Chunked, streaming Granger causality over ordered gene pairs.

    Modes
    -----
    - Full mode (list_to_kutsche=False): test all ordered pairs among genes listed in `genes_file`
      (filtered to columns present in the dataframe).
    - Kutsche mode (list_to_kutsche=True): test only
        (genes_of_interest x (dataset minus genes_of_interest)) +
        ((dataset minus genes_of_interest) x genes_of_interest)

      Here the bitset/checkpoint is sized exactly to that subset of pairs for precise resume.

    Writes only significant rows (p <= p_threshold) to CSV with schema: gene1,gene2,lag,p-value.
    """
    import pandas as pd

    # Prepare time series: columns are per-gene TS
    ts_data = df_filtered_wt_weighted_mean.T

    # Load requested genes from file
    with open(genes_file, "r", encoding="utf-8") as fh:
        requested = [ln.strip() for ln in fh if ln.strip()]

    # Build pair list depending on mode
    if list_to_kutsche:
        # Genes of interest present in the dataframe
        genes_of_interest = [g for g in requested if g in ts_data.columns]
        if not genes_of_interest:
            raise ValueError("No genes_of_interest from file are present in the dataframe columns.")

        # Background = dataset minus genes_of_interest
        dataset_genes = sorted(set(map(str, ts_data.columns)))
        background_genes = [g for g in dataset_genes if g not in genes_of_interest]
        if not background_genes:
            raise ValueError("Background set is empty (all dataset genes are 'genes_of_interest').")

        # All pairs with at least one side in GOI, excluding only GOI self-loops.
        pairs = (
            [(goi, d) for goi in genes_of_interest for d in dataset_genes if d != goi] +  # GOI -> all but self
            [(d, goi) for d in dataset_genes for goi in genes_of_interest if d != goi]    # all but self -> GOI
        )

        # Optionally de-duplicate (safe, though the two lists are disjoint by construction)
        pairs = list(dict.fromkeys(pairs))

        total_pairs_all = len(pairs)

        # Sanity check
        g = len(genes_of_interest)
        N = len(dataset_genes)
        expected = 2 * g * N - g * g - g
        assert total_pairs_all == expected, f"expected {expected}, got {total_pairs_all}"
        pair_to_idx = {p: idx for idx, p in enumerate(pairs)}  # for fast bit marking

        # Bitset checkpoint sized to subset length
        checkpoint_path = os.path.splitext(output_file)[0] + ".seen.bin"
        _prepare_fresh_run(output_file, checkpoint_path, resume)
        bitset = _load_or_init_bitset(checkpoint_path, total_pairs_all)

        def skipping_iter() -> Iterator[Tuple[str, str, int]]:
            """Yield unprocessed probe ordered pairs and their checkpoint index."""
            # Deterministic order over 'pairs' list
            for k, (g1, g2) in enumerate(pairs):
                if resume and _bit_get(bitset, k):
                    continue
                yield (g1, g2, k)

        # progress baseline
        seen_count_start = _bit_count(bitset) if resume else 0
        remaining_estimate = max(total_pairs_all - seen_count_start, 0)

        # We'll not use i/j-based indexing in this mode
        use_pair_index_only = True
        genes = None  # not used here

    else:
        # Full mode: all-by-all among requested genes that exist in the dataframe
        genes = [g for g in requested if g in ts_data.columns]
        if len(genes) < 2:
            raise ValueError("Need at least two genes (after filtering to dataframe columns).")
        n = len(genes)
        total_pairs_all = n * (n - 1)

        # Bitset checkpoint sized to full all-by-all
        checkpoint_path = os.path.splitext(output_file)[0] + ".seen.bin"
        _prepare_fresh_run(output_file, checkpoint_path, resume)
        bitset = _load_or_init_bitset(checkpoint_path, total_pairs_all)

        def skipping_iter() -> Iterator[Tuple[str, str, int, int, int]]:
            """Yield unprocessed all-pairs comparisons and their checkpoint index."""
            # Deterministic order: for each i, iterate j != i
            for i, g1 in enumerate(genes):
                for j, g2 in enumerate(genes):
                    if i == j:
                        continue
                    k = _pair_index(i, j, n)
                    if resume and _bit_get(bitset, k):
                        continue
                    yield (g1, g2, i, j, k)

        seen_count_start = _bit_count(bitset) if resume else 0
        remaining_estimate = max(total_pairs_all - seen_count_start, 0)

        use_pair_index_only = False
        gene_to_idx = {g: idx for idx, g in enumerate(genes)}

    max_workers = max_workers or mp.cpu_count()
    processed_pairs = 0
    significant_count = 0

    # Progress infra
    with mp.Manager() as manager:
        progress_queue = manager.Queue() if progress else None
        progress_proc = None
        if progress:
            progress_proc = mp.Process(
                target=update_progress_bar_tqdm, args=(remaining_estimate, progress_queue)
            )
            progress_proc.start()

        # Worker pool (shares dataframe once per worker)
        with mp.Pool(processes=max_workers, initializer=_init_worker, initargs=(ts_data,)) as pool:

            # Chunked iteration with resume-aware skipping iterator
            for chunk in _batched(skipping_iter(), chunk_size):
                if not chunk:
                    break

                # Prepare tasks + a way to recover the bit index k for each result
                if list_to_kutsche:
                    # chunk elements: (g1, g2, k)
                    tasks = (((g1, g2), progress_queue) for (g1, g2, _k) in chunk)
                else:
                    # chunk elements: (g1, g2, i, j, k)
                    tasks = (((g1, g2), progress_queue) for (g1, g2, _i, _j, _k) in chunk)

                rows_for_csv: List[Tuple[str, str, str, str]] = []

                for result in pool.imap_unordered(_worker_wrapper, tasks, chunksize=pool_chunksize):
                    if not result:
                        continue

                    (gene1, gene2), payload = result

                    # Mark this pair as seen in bitset (regardless of success/significance)
                    if use_pair_index_only:
                        k = pair_to_idx.get((gene1, gene2))
                        if k is not None:
                            _bit_set(bitset, k)
                    else:
                        i = gene_to_idx[gene1]
                        j = gene_to_idx[gene2]
                        _bit_set(bitset, _pair_index(i, j, n))

                    # Skip on error/empty
                    if not payload or (isinstance(payload, dict) and "error" in payload):
                        continue

                    # Extract only significant rows (p <= threshold)
                    try:
                        for lag, res in payload.items():
                            ssr_ftest = res[0].get('ssr_ftest') if isinstance(res, (list, tuple)) else res.get('ssr_ftest')
                            if ssr_ftest and len(ssr_ftest) > 1:
                                pval = float(ssr_ftest[1])
                                if pval <= p_threshold:
                                    rows_for_csv.append([gene1, gene2, str(lag), f"{pval:.6f}"])
                    except Exception:
                        # Ignore malformed result entries
                        pass

                # Persist significant rows for this chunk
                if rows_for_csv:
                    _append_significant_rows(output_file, rows_for_csv)
                    significant_count += len(rows_for_csv)

                # Persist checkpoint at chunk boundary
                _save_bitset(checkpoint_path, bitset)

                processed_pairs += len(chunk)

        if progress:
            progress_queue.put("STOP")
            progress_proc.join()

    # Build final, meaningful filename and rename
    final_path = output_file
    if rename_at_end:
        threshold_str = str(p_threshold).replace(".", "")
        dir_, _base = os.path.split(output_file)
        new_name = f"granger_results_p{threshold_str}_{significant_count}of{total_pairs_all}.csv"
        final_path = os.path.join(dir_ or ".", new_name)
        try:
            os.replace(output_file, final_path)
        except FileNotFoundError:
            _append_significant_rows(output_file, [])
            os.replace(output_file, final_path)
    elif not os.path.exists(output_file):
        _append_significant_rows(output_file, [])

    print(f"Total pairs: {total_pairs_all}")
    print(f"Processed this run: {processed_pairs}")
    print(f"Significant edges saved: {significant_count}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {final_path}")

    return {
        "total_pairs_all": total_pairs_all,
        "processed_this_run": processed_pairs,
        "significant_edges": significant_count,
        "p_threshold": p_threshold,
        "output_file": final_path,
        "checkpoint_path": checkpoint_path,
    }
