"""Compare real gene expression Granger statistics against noise baselines."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from gene_analysis_kutsche.data_preprocessing import load_and_preprocess_data
from gene_analysis_kutsche.data_filtering import preprocess_pipeline
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define all p-value thresholds here
p_values = {
    "0001": 0.001,
    "00005": 0.0005,
    # add more thresholds as needed
}

def convert_tsv_to_csv(noise_dir, output_dir=None):
    """
    Converts all .tsv files in the given directory to .csv files with the same base name.
    """
    if output_dir is None:
        output_dir = noise_dir

    for filename in os.listdir(noise_dir):
        if filename.endswith(".tsv"):
            input_path = os.path.join(noise_dir, filename)
            df = pd.read_csv(input_path, sep='\t', header=None)

            output_filename = filename.replace(".tsv", ".csv")
            output_path = os.path.join(output_dir, output_filename)

            df.to_csv(output_path, index=False, header=False)
            print(f"Converted: {filename} -> {output_filename}")

    print("All TSV files converted to CSV.")


def compute_gene_gc_vs_noise(args):
    """
    Helper for parallel Granger-causality: returns median p-value and counts below each threshold.
    """
    gene_id, gseq, noise_df, p_values = args
    pvals = []

    for _, noise_row in noise_df.iterrows():
        df_pair = pd.DataFrame({
            "noise": noise_row.values,
            "gene":  gseq.flatten()
        })
        try:
            res = grangercausalitytests(df_pair, maxlag=1, verbose=False)
            pvals.append(res[1][0]['ssr_ftest'][1])
        except Exception:
            pvals.append(np.nan)

    arr = np.array(pvals, dtype=float)
    result = {"gene": gene_id, "median": np.nanmedian(arr)}
    for label, threshold in p_values.items():
        result[f"count_below_{label}"] = np.nansum(arr < threshold)
    return result


def compute_gc_noise_baseline(real_data, noise_matrices_path,
                              output_csv="noise_gc_summary_full.csv",
                              real_gc_path="Results/raw_counts/granger_causality_results_truncated.csv"):
    """
    Computes, for each gene, the median Granger-causality p-value against real data,
    and against _n_ noise matrices, counting how often p < each threshold.
    """
    gene_ids = real_data.index.tolist()
    n_noise_files = 5

    # load and align noise matrices
    noise_matrices = []
    min_len = len(gene_ids)
    for i in range(1, n_noise_files+1):
        path = os.path.join(noise_matrices_path, f"matNS{i}.csv")
        noise_df = pd.read_csv(path, header=None)
        noise_df.columns = [f"T{t}" for t in range(noise_df.shape[1])]
        noise_matrices.append(noise_df)
        min_len = min(min_len, len(noise_df))

    gene_ids = gene_ids[:min_len]
    real_data = real_data.loc[gene_ids]
    for i in range(n_noise_files):
        noise_matrices[i] = noise_matrices[i].iloc[:min_len]
        noise_matrices[i].index = gene_ids

    print(f"✅ Aligned to {min_len} genes")

    # real GC medians
    df_real = pd.read_csv(real_gc_path)
    df_real = df_real[df_real["p-value"].notna()]
    real_gc_median = df_real.groupby("gene1")["p-value"].median().rename("RealGC_Median")

    # summary container
    summary_data = {
        "Gene": gene_ids,
        "RealGC_Median": [real_gc_median.get(g, np.nan) for g in gene_ids]
    }
    matrix_medians = []

    for i, noise_df in enumerate(noise_matrices, start=1):
        print(f"\n⚙️  Processing Noise Matrix {i}/{n_noise_files}...")
        with Pool(cpu_count()) as pool:
            tasks = [
                (g, real_data.loc[g].values.reshape(1, -1), noise_df, p_values)
                for g in gene_ids
            ]
            results = list(tqdm(pool.imap(compute_gene_gc_vs_noise, tasks), total=len(gene_ids)))

        # collect
        medians = [r["median"] for r in results]
        summary_data[f"NoiseMatrix_{i}"] = medians
        matrix_medians.append(medians)
        for label in p_values:
            summary_data[f"NoiseMatrix_{i}_count_below_{label}"] = [r[f"count_below_{label}"] for r in results]

    # aggregate noise stats
    mat = np.array(matrix_medians)
    summary_data["MedianOfMedians"] = np.median(mat, axis=0)
    summary_data["StdOfMedians"]    = np.std(mat, axis=0)

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_csv, index=False)
    print(f"\n✅ Final output saved to: {output_csv}")
    return df_summary


def compute_noise_stats_for_gene(gene_name, gseq, noise_matrices, p_values):
    """
    For a single gene, return list of medians and dict of counts per threshold.
    """
    medians = []
    counts = {lbl: [] for lbl in p_values}

    for noise_df in noise_matrices:
        pvals = []
        for _, noise_row in noise_df.iterrows():
            df_pair = pd.DataFrame({
                "noise": noise_row.values,
                "gene":  gseq.flatten()
            })
            try:
                res = grangercausalitytests(df_pair, maxlag=1, verbose=False)
                pvals.append(res[1][0]['ssr_ftest'][1])
            except:
                pvals.append(np.nan)

        arr = np.array(pvals, dtype=float)
        medians.append(np.nanmedian(arr))
        for lbl, thr in p_values.items():
            counts[lbl].append(np.nansum(arr < thr))

    return medians, counts


def insert_zswim7_into_summary(summary_file, gene_expr_df, real_gc_file, noise_matrices):
    """
    Appends ZSWIM7 stats into an existing summary (if not present).
    """
    gene_name = "ZSWIM7"
    if gene_name not in gene_expr_df.index:
        raise ValueError("ZSWIM7 not found in expression matrix.")

    gseq = gene_expr_df.loc[gene_name].values.reshape(1, -1)
    real_med = get_real_gc_median_for_gene(gene_name, real_gc_file)
    meds, counts = compute_noise_stats_for_gene(gene_name, gseq, noise_matrices, p_values)

    row = {"Gene": gene_name, "RealGC_Median": real_med,
           "MedianOfMedians": np.median(meds), "StdOfMedians": np.std(meds)}
    for i, m in enumerate(meds, start=1):
        row[f"NoiseMatrix_{i}"] = m
        for lbl in p_values:
            row[f"NoiseMatrix_{i}_count_below_{lbl}"] = counts[lbl][i-1]

    df = pd.read_csv(summary_file)
    if gene_name in df["Gene"].values:
        print("ZSWIM7 already present — skipping.")
        return
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(summary_file, index=False)
    print(f"✅ ZSWIM7 added to {summary_file}")


def get_real_gc_median_for_gene(gene_name, real_gc_file):
    """Return the median real-data Granger p-value for one source gene."""
    df_real = pd.read_csv(real_gc_file)
    df_gene = df_real[df_real['gene1'] == gene_name]
    if df_gene.empty:
        return np.nan
    return df_gene['p-value'].astype(float).median()


if __name__ == "__main__":
    raw_df = load_and_preprocess_data("Data/Kutsche/genes_all.txt")
    agg_df, _, _ = preprocess_pipeline(raw_df, normalize=False, transformed=False, aggregation="robust")

    # compute baseline and save
    summary = compute_gc_noise_baseline(
        real_data=agg_df,
        noise_matrices_path="noise/csv",
        real_gc_path="granger_causality_results_truncated.csv",
        output_csv="noise/results/noise_gc_summary_full.csv"
    )

    # insert ZSWIM7
    noise_mats = [pd.read_csv(f"noise/csv/matNS{i}.csv", header=None) for i in range(1,6)]
    insert_zswim7_into_summary(
        summary_file="noise/results/noise_gc_summary_full.csv",
        gene_expr_df=agg_df,
        real_gc_file="granger_causality_results_truncated.csv",
        noise_matrices=noise_mats
    )
