"""Benito expression preprocessing and replicate aggregation helpers."""

import numpy as np
import pandas as pd

def filter_data_srr(data):
    """
    Preprocess the data: set index, filter columns, and sort by day.
    """
    if 'Gene_Name' in data.columns:
        data = data.set_index('Gene_Name')
        
    data_cols = [col for col in data.columns if 'SRR' in col]
    day_map = {col: int(col.split('Day ')[1]) for col in data_cols}

    # Sort the columns by day
    ordered_columns = sorted(data_cols, key=lambda x: day_map[x])
    data_filtered = data[ordered_columns]

    # Remove all columns with only zeros
    data_filtered = data_filtered.loc[(data_filtered[data_cols] != 0).any(axis=1)]

    
    return log_transform_data(data_filtered), day_map

def filter_data_proximity_based_weights(data):
    """
    Filter, organize, and compute a weighted mean gene expression data by day.
    """
    data_filtered, day_map = filter_data_srr(data)
    
    robust_mean_per_day = {}
    for day in set(day_map.values()):
        columns_for_day = [col for col in data_filtered if day_map[col] == day]
        daily_data = data_filtered[columns_for_day]

        # Compute median and assign weights based on proximity to median
        median = daily_data.median(axis=1)
        weights = 1 / (1 + daily_data.subtract(median, axis=0).abs())
        sum_weights = weights.sum(axis=1)
        
        # Since DataFrame operations broadcast based on index and columns, ensure sum_weights is a DataFrame
        # This allows division to broadcast properly across the weights DataFrame
        sum_weights_df = sum_weights.to_frame('sum')
        normalized_weights = weights.div(sum_weights_df['sum'], axis=0)

        # Compute weighted mean using normalized weights
        weighted_mean = (daily_data.multiply(normalized_weights)).sum(axis=1)
        robust_mean_per_day[day] = weighted_mean

    df_mean_per_day = pd.DataFrame(robust_mean_per_day)
    return df_mean_per_day, data_filtered, day_map

def filter_data_arithmetic_mean(data):
    """
    Filter, organize, and compute an arithmetic mean gene expression data by day.
    """
    data_filtered, day_map = filter_data_srr(data)
    
    # Compute arithmetic mean per day
    mean_per_day = {}
    for day in set(day_map.values()):
        columns_for_day = [col for col in data_filtered if day_map[col] == day]
        daily_data = data_filtered[columns_for_day]
        mean_per_day[day] = daily_data.mean(axis=1)

    df_mean_per_day = pd.DataFrame(mean_per_day)
    return df_mean_per_day, data_filtered, day_map


def filter_data_median(data):
    """
    Filter, organize, and compute a median gene expression data by day.
    """
    data_filtered, day_map = filter_data_srr(data)
    
    # Compute median per day
    median_per_day = {}
    for day in set(day_map.values()):
        columns_for_day = [col for col in data_filtered if day_map[col] == day]
        daily_data = data_filtered[columns_for_day]
        median_per_day[day] = daily_data.median(axis=1)

    df_median_per_day = pd.DataFrame(median_per_day)

    log_df_median_per_day = log_transform_data(df_median_per_day)

    return log_df_median_per_day, data_filtered, day_map

def log_transform_data(df_selected):
    """Apply natural-log transformation to a dataframe."""
    return df_selected.apply(np.log)
