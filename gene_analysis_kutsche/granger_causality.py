import itertools
import math
import os
import shutil
import sys
import warnings
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from gene_analysis_kutsche.decorators import timing_decorator
import multiprocessing
import csv
# Suppress FutureWarnings from statsmodels
warnings.simplefilter(action='ignore', category=FutureWarning)



def process_gene_combination(combination, time_series_data, progress_queue):
    gene1, gene2 = combination
    test_data = time_series_data[[gene2, gene1]]  # gene 1 causes gene 2
    try:
        if test_data.std(axis=0).eq(0).any():
            result = combination, {'error': 'constant data'}
        else:
            result = combination, grangercausalitytests(test_data, maxlag=1, verbose=False)
    except Exception as e:  # Replace InfeasibleTestError with generic Exception
        result = combination, {'error': str(e)}
    if progress_queue is not None:
        progress_queue.put(1)
    return result

def update_progress_bar(total_combinations, progress_queue):
    processed_combinations = 0
    while processed_combinations < total_combinations:
        processed_combinations += progress_queue.get()
        percent_complete = (processed_combinations / total_combinations) * 100
        sys.stdout.write(f"\rProgress: {processed_combinations}/{total_combinations} gene pairs processed ({percent_complete:.2f}%)")
        sys.stdout.flush()
    print()

@timing_decorator
def perform_granger_causality_tests(df_filtered_wt_weighted_mean, progress=False):
    """
    Perform Granger causality tests on all pairs of genes.
    """
    time_series_data = df_filtered_wt_weighted_mean.T  # To make each column a timeseries

    genes = df_filtered_wt_weighted_mean.index.tolist()
    # n * (n-1) combinations of genes
    gene_combinations = list(itertools.permutations(genes, 2))
    total_combinations = len(gene_combinations)

    gc_results = {}

    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue() if progress else None

        with multiprocessing.Pool() as pool:
            if progress:
                progress_updater = multiprocessing.Process(target=update_progress_bar, args=(total_combinations, progress_queue))
                progress_updater.start()
            
            results = pool.starmap(
                process_gene_combination,
                [(combination, time_series_data, progress_queue) for combination in gene_combinations]
            )

            for result in results:
                if result:
                    gc_results[result[0]] = result[1]

            if progress:
                progress_queue.put(total_combinations)  # Ensure progress updater finishes
                progress_updater.join()

    return gc_results



def collect_significant_edges(gc_results, p_value_threshold=0.05, file=False, filepath=None, starting_genes=None, higher_threshold_for_starting_genes = 0.001):
    """
    Collects gene pairs with significant Granger causality into a list of edges with their corresponding lag,
    and whether the relationship is positive or negative.
    """
    significant_edges = []
    # If the results are not stored, we can directly access the results
    if not file:
        for (gene1, gene2), results in gc_results.items():
            if 'error' in results or not results:
                continue
            for lag, result in results.items():
                if 'ssr_ftest' in result[0]:
                    p_value = result[0]['ssr_ftest'][1]  # Get the p-value of the F-test at this lag
                    if p_value < p_value_threshold:
                        # Append the edge with the corresponding lag
                        significant_edges.append(((gene1, lag), (gene2, 0), p_value))
    # If the results are stored, we need to read them from the file
    else:
        if filepath is None:
            raise ValueError("file_path must be provided when stored is True.")
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                gene1 = row['gene1']
                gene2 = row['gene2']
                try:
                    lag = int(row['lag'])
                    p_value = float(row['p-value'])
                except ValueError:
                    continue  # Skip rows with invalid lag or p-value
                
                if p_value <= p_value_threshold or ((gene1 in starting_genes or gene2 in starting_genes) and p_value <= higher_threshold_for_starting_genes):
                    significant_edges.append(((gene1, lag), (gene2, 0), p_value))
    return significant_edges


def save_results_to_csv(gc_results, output_file):
    """
    Save Granger causality test results to a CSV file with headers:
    gene1, gene2, lag, p-value

    Parameters:
    gc_results (dict): Dictionary with keys as (gene1, gene2) tuples and values as results.
    output_file (str): Path to the output CSV file.
    """
    try:
        with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['gene1', 'gene2', 'lag', 'p-value'])

            for (gene1, gene2), results in gc_results.items():
                try:
                    if not results or 'error' in results:
                        writer.writerow([gene1, gene2, 'NaN', 'NaN'])
                        continue

                    for lag, result in results.items():
                        ssr_ftest = result[0].get('ssr_ftest')
                        if ssr_ftest and len(ssr_ftest) > 1:
                            p_value = round(ssr_ftest[1], 4)  # Round p-value to 4 decimal places
                            writer.writerow([gene1, gene2, lag, p_value])
                        else:
                            writer.writerow([gene1, gene2, lag, 'NaN'])
                except Exception as line_error:
                    print(f"Error writing results for {gene1}, {gene2}: {line_error}")
    except Exception as e:
        print(f"Failed to write to file {output_file}: {e}")
    else:
        print(f"Results successfully saved to {output_file}")



import tempfile


def filter_gene_pairs(filepath, p_threshold, starting_genes=None, download_path=None, higher_threshold_for_starting_genes = 0.001):
    """
    Filter gene pairs based on the p-value threshold and the presence of starting genes,
    then exhaustively find all related genes that meet the criteria.

    Args:
    filepath (str): Path to the CSV file containing gene pairs.
    p_threshold (float): The threshold for the p-value to consider a gene pair significant.
    starting_genes (list): Initial list of gene names to start filtering by.
    download_path (str, optional): If provided, the path to save the filtered CSV file.

    Returns:
    str: The path to the temporary CSV file containing the filtered gene pairs.
    """
    if starting_genes is None:
        raise ValueError("A list of starting genes must be provided.")
    
    # Use sets for efficient lookups and to avoid duplicates
    all_related_genes = set(starting_genes)
    newly_added_genes = set(starting_genes)
    
    # Create a temporary file to store the filtered results
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
    temp_file_path = temp_file.name
    filtered_edges = []
    
    # First pass: filter out insignificant edges
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        for row in reader:
            pvalue = float(row['p-value'])
            gene1 = row['gene1']
            gene2 = row['gene2']
            if pvalue <= p_threshold or ((gene1 in starting_genes or gene2 in starting_genes) and pvalue <= higher_threshold_for_starting_genes):
                filtered_edges.append(row)
        
        # We will collect all rows that meet the criteria and write them at the end
        filtered_rows = []
        
        while newly_added_genes:
            current_genes = newly_added_genes.copy()
            newly_added_genes.clear()
            
            # Rewind the file to the start
            file.seek(0)
            next(reader)  # Skip the header row
            
            for row in filtered_edges:
                gene1 = row['gene1']
                gene2 = row['gene2']
                pvalue = float(row['p-value'])
                
                # If either gene1 or gene2 is in the current set of related genes
                if gene1 in current_genes or gene2 in current_genes:
                    # Add the row to the filtered results
                    filtered_rows.append(row)
                    
                    # Add the other gene to the set of all related genes
                    if gene1 not in all_related_genes:
                        newly_added_genes.add(gene1)
                    if gene2 not in all_related_genes:
                        newly_added_genes.add(gene2)
                    
                    # Update the set of all related genes
                    all_related_genes.update([gene1, gene2])
        
    # Write the filtered results to the temporary file
    with open(temp_file_path, 'w', newline='') as tmpfile:
        writer = csv.DictWriter(tmpfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    # If a download path is specified, move the temp file to the download path
    if download_path != None:
        download_path = f'p_value_threshold_{p_threshold}.csv'
        shutil.move(temp_file_path, download_path)
        return download_path
    
    return temp_file_path