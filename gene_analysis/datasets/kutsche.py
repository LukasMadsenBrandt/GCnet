"""Kutsche expression filtering and replicate aggregation helpers."""

import pandas as pd

from gene_analysis.analysis.preprocessing import apply_expression_preprocessing


def load_and_preprocess_data(filepath):
    """Load a Kutsche tab-separated count matrix indexed by gene symbol."""
    with open(filepath, 'r') as file:
        header_line = file.readline().strip()
        genes_data = file.readlines()

    columns = header_line.split('\t')
    data = [line.strip().split('\t') for line in genes_data]
    df = pd.DataFrame(data, columns=columns)
    df.set_index('Gene', inplace=True)

    for col in df.columns:
        df[col] = df[col].astype(float)

    return df


def filter_data_wt(df, transformed=True, normalize="deseq"):
    """
    Extract WT columns and apply shared preprocessing before aggregation.
    """
    wt_columns = [col for col in df.columns if 'WT' in col]
    day_map = {col: int(col.split('d')[1].split('_')[0]) for col in wt_columns}
    ordered_columns = sorted(wt_columns, key=lambda x: day_map[x])
    df_filtered_wt = df[ordered_columns]

    # Remove genes where all replicates are zero
    df_filtered_wt = df_filtered_wt.loc[(df_filtered_wt != 0).any(axis=1)]

    if transformed is True:
        transformed = "log+1"

    df_filtered_wt = apply_expression_preprocessing(
        df_filtered_wt,
        normalize=normalize,
        transform=transformed,
    )

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
