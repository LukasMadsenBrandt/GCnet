"""Shared Granger-causality implementation used by dataset compatibility wrappers."""

from __future__ import annotations

import csv
import itertools
import multiprocessing
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

from statsmodels.tsa.stattools import grangercausalitytests

from gene_analysis.io.paths import resolve_existing_path


warnings.simplefilter(action="ignore", category=FutureWarning)


def timing_decorator(func):
    """Print the runtime of legacy all-pairs Granger helpers."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Total execution time of {func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper


def process_gene_combination(combination, time_series_data, progress_queue):
    """Run one lag-1 Granger test for a directed gene pair."""
    gene1, gene2 = combination
    test_data = time_series_data[[gene2, gene1]]
    try:
        if test_data.std(axis=0).eq(0).any():
            result = combination, {"error": "constant data"}
        else:
            result = combination, grangercausalitytests(test_data, maxlag=1, verbose=False)
    except Exception as e:
        result = combination, {"error": str(e)}
    if progress_queue is not None:
        progress_queue.put(1)
    return result


def update_progress_bar(total_combinations, progress_queue):
    """Print multiprocessing progress for a batch of Granger tests."""
    processed_combinations = 0
    while processed_combinations < total_combinations:
        processed_combinations += progress_queue.get()
        percent_complete = (processed_combinations / total_combinations) * 100
        sys.stdout.write(
            f"\rProgress: {processed_combinations}/{total_combinations} gene pairs processed ({percent_complete:.2f}%)"
        )
        sys.stdout.flush()
    print()


@timing_decorator
def perform_granger_causality_tests(df_filtered_wt_weighted_mean, genes_file="gene_names.txt", progress=False):
    """
    Perform Granger causality tests on all ordered pairs of genes listed in genes_file.
    """
    time_series_data = df_filtered_wt_weighted_mean.T
    print(time_series_data.shape)

    with open(resolve_existing_path(genes_file), "r") as file:
        genes = [line.strip() for line in file.readlines()]

    genes = [gene for gene in genes if gene in time_series_data.columns]
    gene_combinations = list(itertools.permutations(genes, 2))
    total_combinations = len(gene_combinations)
    print(f"Total gene pairs to test: {total_combinations}")
    gc_results = {}

    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue() if progress else None

        with multiprocessing.Pool() as pool:
            if progress:
                progress_updater = multiprocessing.Process(
                    target=update_progress_bar,
                    args=(total_combinations, progress_queue),
                )
                progress_updater.start()

            results = pool.starmap(
                process_gene_combination,
                [(combination, time_series_data, progress_queue) for combination in gene_combinations],
            )

            for result in results:
                if result:
                    gc_results[result[0]] = result[1]

            if progress:
                progress_queue.put(total_combinations)
                progress_updater.join()

    return gc_results


def collect_significant_edges(
    gc_results,
    p_value_threshold=0.05,
    file=False,
    filepath=None,
    starting_genes=None,
    higher_threshold_for_starting_genes=0.001,
):
    """
    Collect significant Granger-causality edges from in-memory results or a stored CSV.
    """
    significant_edges = []
    starting_gene_set = set(starting_genes or [])

    if not file:
        for (gene1, gene2), results in gc_results.items():
            if "error" in results or not results:
                continue
            for lag, result in results.items():
                if "ssr_ftest" in result[0]:
                    p_value = result[0]["ssr_ftest"][1]
                    if p_value < p_value_threshold:
                        significant_edges.append(((gene1, lag), (gene2, 0), p_value))
        return significant_edges

    if filepath is None:
        raise ValueError("file_path must be provided when stored is True.")

    with open(resolve_existing_path(filepath), "r") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            gene1 = row["gene1"]
            gene2 = row["gene2"]
            try:
                lag = int(row["lag"])
                p_value = float(row["p-value"])
            except ValueError:
                continue

            starts_from_seed = gene1 in starting_gene_set or gene2 in starting_gene_set
            if p_value <= p_value_threshold or (
                starts_from_seed and p_value <= higher_threshold_for_starting_genes
            ):
                significant_edges.append(((gene1, lag), (gene2, 0), p_value))
    return significant_edges


def save_results_to_csv(gc_results, output_file):
    """
    Save Granger causality test results to a CSV file with headers:
    gene1, gene2, lag, p-value
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["gene1", "gene2", "lag", "p-value"])

            for (gene1, gene2), results in gc_results.items():
                try:
                    if not results or "error" in results:
                        writer.writerow([gene1, gene2, "NaN", "NaN"])
                        continue

                    for lag, result in results.items():
                        ssr_ftest = result[0].get("ssr_ftest")
                        if ssr_ftest and len(ssr_ftest) > 1:
                            p_value = round(ssr_ftest[1], 4)
                            writer.writerow([gene1, gene2, lag, p_value])
                        else:
                            writer.writerow([gene1, gene2, lag, "NaN"])
                except Exception as line_error:
                    print(f"Error writing results for {gene1}, {gene2}: {line_error}")
    except Exception as e:
        print(f"Failed to write to file {output_path}: {e}")
    else:
        print(f"Results successfully saved to {output_path}")


def filter_gene_pairs(
    filepath,
    p_threshold,
    starting_genes=None,
    download_path=None,
    higher_threshold_for_starting_genes=0.001,
):
    """
    Filter gene pairs by p-value and exhaustively expand from starting genes.
    """
    if starting_genes is None:
        raise ValueError("A list of starting genes must be provided.")

    all_related_genes = set(starting_genes)
    newly_added_genes = set(starting_genes)

    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="")
    temp_file_path = temp_file.name
    temp_file.close()
    filtered_edges = []

    with open(resolve_existing_path(filepath), "r") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = reader.fieldnames

        for row in reader:
            pvalue = float(row["p-value"])
            gene1 = row["gene1"]
            gene2 = row["gene2"]
            if pvalue <= p_threshold or (
                (gene1 in starting_genes or gene2 in starting_genes)
                and pvalue <= higher_threshold_for_starting_genes
            ):
                filtered_edges.append(row)

        filtered_rows = []
        seen_filtered_rows = set()

        while newly_added_genes:
            current_genes = newly_added_genes.copy()
            newly_added_genes.clear()

            for row in filtered_edges:
                gene1 = row["gene1"]
                gene2 = row["gene2"]

                if gene1 in current_genes or gene2 in current_genes:
                    row_key = tuple(row.get(name, "") for name in fieldnames or [])
                    if row_key not in seen_filtered_rows:
                        filtered_rows.append(row)
                        seen_filtered_rows.add(row_key)

                    if gene1 not in all_related_genes:
                        newly_added_genes.add(gene1)
                    if gene2 not in all_related_genes:
                        newly_added_genes.add(gene2)

                    all_related_genes.update([gene1, gene2])

    with open(temp_file_path, "w", newline="") as tmpfile:
        writer = csv.DictWriter(tmpfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    if download_path is not None:
        download_path = f"p_value_threshold_{p_threshold}.csv"
        shutil.move(temp_file_path, download_path)
        return download_path

    return temp_file_path
