"""Kutsche expression preprocessing and replicate aggregation helpers."""

import numpy as np
import pandas as pd

def compute_deseq_size_factors(counts_df):
    """
    Compute DESeq2-style size factors for normalization.
    """
    # Step 1: Compute gene-wise geometric means (ignoring zeros)
    # Replace zeros with NaN for log to avoid -inf
    counts_no_zeros = counts_df.replace(0, np.nan)
    log_counts = np.log(counts_no_zeros)
    geo_means = np.exp(log_counts.mean(axis=1, skipna=True))

    # Step 2: For each sample, compute ratios of counts to geometric means
    ratios = counts_df.div(geo_means, axis=0)

    # Step 3: Size factor = median of ratios (excluding NaNs)
    size_factors = ratios.median(axis=0, skipna=True)

    return size_factors

def normalize_with_size_factors(counts_df):
    """
    Normalize RNA-seq counts using DESeq2 median-ratio method.
    """
    size_factors = compute_deseq_size_factors(counts_df)
    normalized = counts_df.div(size_factors, axis=1)
    return normalized

def log_transform_data(df_selected):
    """ Apply safe natural log transformation using log1p (i.e., log(x + 1)) """
    return np.log1p(df_selected)

def filter_data_wt(df, transformed=True, normalize="deseq"):
    """
    Extract and optionally normalize and log-transform WT data.
    """
    wt_columns = [col for col in df.columns if 'WT' in col]
    day_map = {col: int(col.split('d')[1].split('_')[0]) for col in wt_columns}
    ordered_columns = sorted(wt_columns, key=lambda x: day_map[x])
    df_filtered_wt = df[ordered_columns]
    
    # Remove genes where all replicates are zero
    df_filtered_wt = df_filtered_wt.loc[(df_filtered_wt != 0).any(axis=1)]

    if normalize == 'deseq':
        df_filtered_wt = normalize_with_size_factors(df_filtered_wt)

    if transformed == "log+1":
        df_filtered_wt = log_transform_data(df_filtered_wt)
    elif transformed == "sqrt":
        df_filtered_wt = np.sqrt(df_filtered_wt)

    return df_filtered_wt, day_map, ordered_columns

def aggregate_replicates(df_filtered_wt, day_map, method="robust"):
    """
    Aggregates replicate data using mean, median, or robust weighted mean.
    """
    method = method.lower()
    aggregated_per_day = {}

    for day in sorted(set(day_map.values())):
        columns_for_day = [col for col in df_filtered_wt.columns if day_map[col] == day]
        daily_data = df_filtered_wt[columns_for_day]

        if method == "mean":
            aggregated_per_day[day] = daily_data.mean(axis=1)

        elif method == "median":
            aggregated_per_day[day] = daily_data.median(axis=1)

        elif method == "robust":
            median = daily_data.median(axis=1)
            weights = 1 / (1 + daily_data.subtract(median, axis=0).abs())
            weights = weights.div(weights.sum(axis=1), axis=0)
            weighted_mean = (daily_data * weights).sum(axis=1)
            aggregated_per_day[day] = weighted_mean

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return pd.DataFrame(aggregated_per_day)

def preprocess_pipeline(df, normalize="deseq", transformed="log+1", aggregation="robust"):
    """
    High-level pipeline: Filter, normalize, log-transform, and aggregate RNA-seq data.
    """
    df_filtered_wt, day_map, _ = filter_data_wt(df, transformed=transformed, normalize=normalize)
    df_aggregated = aggregate_replicates(df_filtered_wt, day_map, method=aggregation)
    return df_aggregated, df_filtered_wt, day_map
