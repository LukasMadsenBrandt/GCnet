"""Kutsche count-matrix loading helpers."""

import pandas as pd

def load_and_preprocess_data(filepath):
    """
    Load and preprocess gene expression data from a file.
    """
    with open(filepath, 'r') as file:
        header_line = file.readline().strip()  # Read the header line for column names
        genes_data = file.readlines()

    columns = header_line.split('\t')
    data = [line.strip().split('\t') for line in genes_data]
    df = pd.DataFrame(data, columns=columns)
    df.set_index('Gene', inplace=True)

    # Convert all columns to floats except the 'Gene' index
    for col in df.columns:
        df[col] = df[col].astype(float)

    return df
