"""Benito dataset mapping helpers used before timepoint aggregation."""

import pandas as pd

def map_gene_name_to_data(mappingfile):
    """
    Load and preprocess data from a file to map specimen numbers to species and days.
    """
    # Load the data from the CSV file
    read_mappingfile = pd.read_csv(mappingfile, sep='\t', header=None)
    # For each row, map the value in the first column to the value in the second column
    mapping = {row[0]: row[1] for row in read_mappingfile.values}

    return mapping

def put_mapping_inside_of_datafile(datafile, mapping):
    """Load a Benito featureCounts file and add a mapped ``Gene_Name`` column."""
    # Load the data from the CSV file, tab-separated and skipping initial metadata lines
    df = pd.read_csv(datafile, sep='\t', header=1, on_bad_lines='skip')


    # Assuming 'mapping' is your dictionary mapping gene IDs to gene names
    # Add a new column to the DataFrame based on the mapping
    df['Gene_Name'] = df['Geneid'].map(mapping)

    return df

def map_speciment_to_species_and_day(data_with_gene_name, map_speciment_to_gene_file):
    """
    Load and preprocess data from a file to map specimen numbers to species and days.
    """
    # Load the data from the CSV file
    df = pd.read_csv(map_speciment_to_gene_file, usecols=['Run', 'Organism', 'Time_point'])

    # Create a mapping from Run to Organism and Time_point
    metadata_mapping = df.set_index('Run')[['Organism', 'Time_point']].to_dict('index')

    # Create a mapping from Run to Time_point
    run_to_time_point = {run_id: details['Time_point'] for run_id, details in metadata_mapping.items()}

    # Replace column names in the gene count DataFrame
    data_with_gene_name.columns = [replace_speciment_with_timepoint(run_to_time_point, col) for col in data_with_gene_name.columns]

    return data_with_gene_name


def replace_speciment_with_timepoint(run_to_time_point, column_name):
    """Replace SRR identifiers in a count column with their mapped timepoint."""
    # Check if the column name contains a Run identifier and replace it with Time_point
    for run_id, time_point in run_to_time_point.items():
        if run_id in column_name:
            # Replace Run identifier with Run_Time_point
            replaced_column = column_name.replace(run_id, f"{run_id}_{time_point}")
            # Remove everything after and including "_REF"
            cleaned_column = replaced_column.split('_REF')[0]
            return cleaned_column
    return column_name.split('_REF')[0]




def filter_data(datafile, mappingfile, map_speciment_to_gene_file):
    """Map gene IDs and SRR samples in a Benito count file to analysis labels."""

    # Using all three methods, filter the data.
    mapping = map_gene_name_to_data(mappingfile)
    data_with_gene_name = put_mapping_inside_of_datafile(datafile=datafile, mapping=mapping)
    day_mapped_to_speciment = map_speciment_to_species_and_day(data_with_gene_name, map_speciment_to_gene_file)
    return day_mapped_to_speciment
