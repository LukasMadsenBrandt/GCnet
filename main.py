import itertools
import multiprocessing
import os
import sys

import numpy as np
#Kutsche
from gene_analysis_kutsche.granger_causality import perform_granger_causality_tests as perform_gc_kutsche
from gene_analysis_kutsche.granger_causality import collect_significant_edges as collect_significant_edges_kutsche
from gene_analysis_kutsche.granger_causality import save_results_to_csv as save_results_to_csv_kutsche
from gene_analysis_kutsche.data_preprocessing import load_and_preprocess_data as load_and_preprocess_kutsche
from gene_analysis_kutsche.data_filtering import preprocess_pipeline as preprocess_pipeline


from gene_analysis_kutsche.data_filtering import filter_data_wt as filter_wt_kutsche

#Benito
from gene_analysis_benito.granger_causality import perform_granger_causality_tests as perform_gc_benito
from gene_analysis_benito.granger_causality import collect_significant_edges as collect_significant_edges_benito
from gene_analysis_benito.data_preprocessing import filter_data_proximity_based_weights as filter_proximity_benito
from gene_analysis_benito.data_preprocessing import filter_data_arithmetic_mean as filter_mean_benito
from gene_analysis_benito.data_preprocessing import filter_data_median as filter_median_benito
from gene_analysis_benito.data_filtering import filter_data as mapper_benito

def main():
    kutsche = True # Set to True for Kutsche data, False for Benito data
    if kutsche:            
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = load_and_preprocess_kutsche(os.path.join('Data', 'Kutsche', 'genes.txt'))

        print("Data loaded and preprocessed.")
        df_filtered, raw_data, day_map = preprocess_pipeline(df, normalize="deseq", logged=True, aggregation="robust")
        print("Data filtered.")
        print(df_filtered[:5])
        #print 5 lines of the data
        print(len(df_filtered))
        #gc_results = perform_gc_kutsche(df_filtered, progress=True)
        #gc_results = perform_granger_explore_new(df_filtered, progress=True)
        print("Granger causality tests performed.")

        # Save results
        #save_results_to_csv_kutsche(gc_results, "granger_causality_results_test.csv")
        print("Results saved to granger_causality_results_test.csv")
    else:
        df_human = mapper_benito(
            datafile=os.path.join('Data', 'Benito', 'Benito_Human'),
            mappingfile=os.path.join('Data', 'Benito', 'gene_id_to_gene_name.txt'),
            map_speciment_to_gene_file=os.path.join('Data', 'Benito', 'map_speciment_to_gene.csv')
            )
        print("Data loaded and preprocessed.")
        filter_function, _, _ = filter_proximity_benito(df_human)
        print("Data filtered.")
        print("Performing Granger causality tests...")
        gc_results = perform_gc_benito(filter_function, genes_file=os.path.join('Data', 'Benito', 'gene_names_all.txt'), progress=True)
        print("Granger causality tests performed.")

        save_results_to_csv_kutsche(gc_results, "granger_causality_results_benito2.csv")
        


def perform_granger_explore_new(df_filtered_wt_weighted_mean, progress=False):
    """
    Perform Granger causality tests on all pairs of genes.
    """
    time_series_data = df_filtered_wt_weighted_mean.T  # To make each column a timeseries
    filepath = os.path.join('Data', 'Kutsche', 'top_5%_stable_comm_ZEB2.txt')
    
    with open(filepath, 'r') as file:
        # Read gene names directly, one per line
        genes_of_interest = [line.strip() for line in file if line.strip()]
    
    genes = df_filtered_wt_weighted_mean.index.tolist()

    # Generate pairs involving genes_of_interest (no self-pairs)
    gene_combinations = [
        (goi, gene) for goi in genes_of_interest for gene in genes if goi != gene
    ] + [
        (gene, goi) for goi in genes_of_interest for gene in genes if goi != gene
    ]
    total_combinations = len(gene_combinations)
    print(f"Number of genes {len(genes)}")
    print(f"Number of combinations {total_combinations}")

    # Limit to top 10 combinations for testing
    gene_combinations = gene_combinations[:10]

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

def update_progress_bar(total_combinations, progress_queue):
    processed_combinations = 0
    while processed_combinations < total_combinations:
        processed_combinations += progress_queue.get()
        percent_complete = (processed_combinations / total_combinations) * 100
        sys.stdout.write(f"\rProgress: {processed_combinations}/{total_combinations} gene pairs processed ({percent_complete:.2f}%)")
        sys.stdout.flush()
    print()

from statsmodels.tsa.stattools import grangercausalitytests
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

if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    main()


