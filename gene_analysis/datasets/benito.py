"""Benito dataset loading and replicate aggregation."""

import pandas as pd

from gene_analysis.analysis.preprocessing import apply_expression_preprocessing

def map_gene_name_to_data(mappingfile):
    """Load a mapping from Ensembl gene IDs to gene symbols."""
    read_mappingfile = pd.read_csv(mappingfile, sep='\t', header=None)
    return {row[0]: row[1] for row in read_mappingfile.values}


def put_mapping_inside_of_datafile(datafile, mapping):
    """Load a Benito count file and add a mapped ``Gene_Name`` column."""
    df = pd.read_csv(datafile, sep='\t', header=1, on_bad_lines='skip')
    df['Gene_Name'] = df['Geneid'].map(mapping)
    return df


def replace_speciment_with_timepoint(run_to_time_point, column_name):
    """Replace an SRR identifier in a column name with the matching timepoint."""
    for run_id, time_point in run_to_time_point.items():
        if run_id in column_name:
            replaced = column_name.replace(run_id, f"{run_id}_{time_point}")
            return replaced.split('_REF')[0]
    return column_name.split('_REF')[0]


def map_speciment_to_species_and_day(data_with_gene_name, map_speciment_to_gene_file):
    """Rename Benito sample columns using their timepoint metadata."""
    meta = pd.read_csv(
        map_speciment_to_gene_file,
        usecols=['Run', 'Organism', 'Time_point']
    )
    run_to_time_point = meta.set_index('Run')['Time_point'].to_dict()

    renamed = data_with_gene_name.copy()
    renamed.columns = [
        replace_speciment_with_timepoint(run_to_time_point, c)
        for c in renamed.columns
    ]
    return renamed


def load_and_map_benito(datafile, mappingfile, map_speciment_to_gene_file):
    """Load Benito counts and apply gene-name and timepoint mappings."""
    mapping = map_gene_name_to_data(mappingfile)
    df = put_mapping_inside_of_datafile(datafile, mapping)
    df = map_speciment_to_species_and_day(df, map_speciment_to_gene_file)
    return df


# ============================================================
# Filtering + aggregation (Benito pipeline)
# ============================================================

def filter_benito_replicates(
    df,
    organism=None,
    transformed="log+1",
    normalize=None,
):
    """
    Filter replicate columns and optionally normalize/transform.
    Returns filtered dataframe and day_map.
    """

    if 'Gene_Name' in df.columns:
        df = df.set_index('Gene_Name')

    data_cols = [c for c in df.columns if 'SRR' in c]

    day_map = {}
    for col in data_cols:
        if "Day " not in col:
            continue
        try:
            day_map[col] = int(col.split("Day ")[1].split()[0])
        except Exception as e:
            raise ValueError(f"Could not parse day from column: {col}") from e

    ordered_columns = sorted(day_map.keys(), key=lambda c: day_map[c])
    df_filtered = df[ordered_columns]

    df_filtered = df_filtered.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df_filtered = df_filtered.loc[(df_filtered != 0).any(axis=1)]

    df_filtered = apply_expression_preprocessing(df_filtered, normalize=normalize, transform=transformed)

    return df_filtered, day_map


def aggregate_replicates_by_day(df_filtered, day_map, method="robust"):
    """
    Aggregate replicate columns into per-day columns.
    """
    method = method.lower()
    aggregated_per_day = {}

    for day in sorted(set(day_map.values())):
        cols_for_day = [c for c in df_filtered.columns if day_map.get(c) == day]
        daily_data = df_filtered[cols_for_day]

        if method == "mean":
            aggregated_per_day[day] = daily_data.mean(axis=1)

        elif method == "median":
            aggregated_per_day[day] = daily_data.median(axis=1)

        elif method == "robust":
            med = daily_data.median(axis=1)
            weights = 1 / (1 + daily_data.subtract(med, axis=0).abs())
            weights = weights.div(weights.sum(axis=1), axis=0)
            aggregated_per_day[day] = (daily_data * weights).sum(axis=1)

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return pd.DataFrame(aggregated_per_day)


def preprocess_pipeline_benito(
    datafile,
    mappingfile,
    map_speciment_to_gene_file,
    normalize="deseq",
    transformed="log+1",
    aggregation="robust",
    organism=None,
):
    """
    High-level Benito preprocessing pipeline.
    """
    df_raw = load_and_map_benito(
        datafile,
        mappingfile,
        map_speciment_to_gene_file
    )

    df_filtered, day_map = filter_benito_replicates(
        df_raw,
        organism=organism,
        normalize=normalize,
        transformed=transformed,
    )

    df_agg = aggregate_replicates_by_day(
        df_filtered,
        day_map,
        method=aggregation
    )

    return df_agg, df_filtered, day_map
