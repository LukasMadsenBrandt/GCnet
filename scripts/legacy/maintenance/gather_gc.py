"""Config-editable maintenance script for launching legacy GC collection jobs."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

#Kutsche
from gene_analysis_kutsche.granger_causality import perform_granger_causality_tests as perform_gc_kutsche
from gene_analysis_kutsche.granger_causality import collect_significant_edges as collect_significant_edges_kutsche
from gene_analysis_kutsche.granger_causality import save_results_to_csv as save_results_to_csv_kutsche
from gene_analysis_kutsche.data_preprocessing import load_and_preprocess_data as load_and_preprocess_kutsche
from gene_analysis_kutsche.data_filtering import preprocess_pipeline as preprocess_pipeline

from gene_analysis_kutsche.granger_new import perform_gc as perform_gc



from gene_analysis_kutsche.data_filtering import filter_data_wt as filter_wt_kutsche

#Benito
from gene_analysis_benito.utility import preprocess_pipeline_benito
from gene_analysis_benito.granger_causality import perform_granger_causality_tests as perform_gc_benito
from gene_analysis_benito.granger_causality import collect_significant_edges as collect_significant_edges_benito
from gene_analysis_benito.data_preprocessing import filter_data_proximity_based_weights as filter_proximity_benito
from gene_analysis_benito.data_preprocessing import filter_data_arithmetic_mean as filter_mean_benito
from gene_analysis_benito.data_preprocessing import filter_data_median as filter_median_benito
from gene_analysis_benito.data_filtering import filter_data as mapper_benito

def main():
    """Run the configured Kutsche or Benito GC collection branch."""

    kutsche = False # Set to True for Kutsche data, False for Benito data
    kutsche_explore = False # Set to True for Kutsche explore data

    benito = True # Set to True for Kutsche data, False for Benito data
    benito_explore = True # Set to True for Kutsche explore data
    suffix = "00015"


    if kutsche_explore:
        print("Loading and preprocessing data...")
        df = load_and_preprocess_kutsche(os.path.join('Data', 'Kutsche', 'Kutsche_Counts.txt'))
        df_filtered, raw_data, day_map = preprocess_pipeline(df, normalize=False, transformed=False, aggregation="robust")
        
        # 1 step
        """
        summary = new_perform_gc_kutsche(
            df_filtered_wt_weighted_mean=df_filtered,
            genes_file=os.path.join('Data', 'Kutsche', f'unique_genes.txt'),
            output_file=f"granger_causality_results_1st_step_only_sig_{suffix}.csv",
            p_threshold=0.002,             # <= only
            chunk_size=1000000,  # Adjusted for larger datasets
            max_workers=None,
            pool_chunksize=64,
            progress=True,
            resume=True
        )
        """
        # 1,5 step
        """
        summary = new_perform_gc_kutsche(
            df_filtered_wt_weighted_mean=df_filtered,
            genes_file=os.path.join('Data', 'Kutsche', f'{suffix}', f'gene_names_{suffix}_115.txt'),
            output_file=f"granger_causality_results_1_5_step_only_sig_{suffix}.csv",
            p_threshold=0.002,             # <= only
            chunk_size=1000000,  # Adjusted for larger datasets
            list_to_kutsche=True,
            max_workers=None,
            pool_chunksize=64,
            progress=True,
            resume=True
        )
        """
        # 2 step
        
        summary = perform_gc(
            df_filtered_wt_weighted_mean=df_filtered,
            genes_file=os.path.join('Data', 'Kutsche', f'explore_genes_{suffix}.txt'),
            output_file=f"kutsche_granger_causality_results_exploration_2nd_step_only_sig_{suffix}.csv",
            p_threshold=0.002,             # <= only
            chunk_size=1000000,  # Adjusted for larger datasets
            max_workers=None,
            pool_chunksize=64,
            progress=True,
            resume=True
        )
        

    elif benito_explore:
        species = "Gorilla"

        df_agg, df_filtered_reps, day_map = preprocess_pipeline_benito(
            datafile=os.path.join("Data", "Benito", f"Benito_{species}"),
            mappingfile=os.path.join("Data", "Benito", "gene_id_to_gene_name.txt"),
            map_speciment_to_gene_file=os.path.join("Data", "Benito", "map_speciment_to_gene.csv"),
            normalize=False,
            transformed=False,
            aggregation="robust",
        )

        # 1 step
        #"""
        summary = perform_gc(
            df_filtered_wt_weighted_mean=df_agg,
            genes_file=os.path.join('Data', 'Benito', f'unique_genes.txt'),
            output_file=f"benito_{species}_granger_causality_results_1st_step_only_sig_{suffix}.csv",
            p_threshold=1,             # <= only
            chunk_size=1000000,  # Adjusted for larger datasets
            max_workers=None,
            pool_chunksize=64,
            progress=True,
            resume=True
        )
        #"""
        # 1,5 step
        """
        summary = new_perform_gc_kutsche(
            df_filtered_wt_weighted_mean=df_filtered,
            genes_file=os.path.join('Data', 'Benito', f'{suffix}', f'gene_names_{suffix}_115.txt'),
            output_file=f"granger_causality_results_1_5_step_only_sig_{suffix}.csv",
            p_threshold=0.002,             # <= only
            chunk_size=1000000,  # Adjusted for larger datasets
            list_to_kutsche=True,
            max_workers=None,
            pool_chunksize=64,
            progress=True,
            resume=True
        )
        """
        # 2 step
        
        """        
            summary = new_perform_gc_kutsche(
            df_filtered_wt_weighted_mean=df_filtered,
            genes_file=os.path.join('Data', 'Benito', f'explore_genes_{suffix}.txt'),
            output_file=f"granger_causality_results_exploration_2nd_step_only_sig_{suffix}.csv",
            p_threshold=0.002,             # <= only
            chunk_size=1000000,  # Adjusted for larger datasets
            max_workers=None,
            pool_chunksize=64,
            progress=True,
            resume=True
        )
        """

        print("Done:", summary)
    elif kutsche:            
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = load_and_preprocess_kutsche(os.path.join('Data', 'Kutsche', 'Kutsche_Counts.txt'))
        suffix = "0001"
        print("Data loaded and preprocessed.")
        df_filtered, raw_data, day_map = preprocess_pipeline(df, normalize=False, transformed=False , aggregation="robust")
        print("Data filtered.")
        print(df_filtered[:5])
        #print 5 lines of the data
        print(len(df_filtered))
        #gc_results = perform_gc_kutsche(df_filtered, progress=True)
        #gc_results = perform_granger_explore_new(df_filtered, progress=True, filepath=os.path.join('Data', 'Kutsche', f'gene_names_{suffix}.txt'))
        #gc_results = perform_gc_kutsche(df_filtered, progress=True, genes_file=os.path.join('Data', 'Kutsche', f'explore_genes_{suffix}.txt'))
        
        print("Granger causality tests performed.")

        # Save results
        save_results_to_csv_kutsche(gc_results, f"granger_causality_results_exploration_{suffix}.csv")
    elif benito:
        df_human = mapper_benito(
            datafile=os.path.join('Data', 'Benito', 'Benito_Human'),
            mappingfile=os.path.join('Data', 'Benito', 'gene_id_to_gene_name.txt'),
            map_speciment_to_gene_file=os.path.join('Data', 'Benito', 'map_speciment_to_gene.csv')
            )
        print("Data loaded and preprocessed.")
        filter_function, _, _ = filter_proximity_benito(df_human)
        print("Performing Granger causality tests...")
        gc_results = perform_gc_benito(filter_function, genes_file=os.path.join('Data', 'Benito', 'unique_genes.txt'), progress=True)
        print("Granger causality tests performed.")

        save_results_to_csv_kutsche(gc_results, "granger_causality_results_truncated_benito_human.csv")
        


if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    main()
