import pandas as pd
import numpy as np
import argparse

def compute_lower_quantile_threshold(csv_path, quantile=0.005):
    """
    Load CSV with columns: gene1, gene2, lag, pvalue
    and compute the threshold corresponding to the lower quantile of p-values.
    
    Parameters:
        csv_path (str): Path to the input CSV file.
        quantile (float): The quantile to compute (e.g., 0.005 for 0.5%).
        
    Returns:
        float: The threshold value at the given quantile.
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Validate
    if 'p-value' not in df.columns:
        raise ValueError("CSV must contain a 'p-value' column.")

    # Compute quantile
    threshold = df['p-value'].quantile(quantile)
    print(f"Threshold at the {quantile*100:.2f}% quantile: {threshold:.6g}")
    return threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute p-value threshold for Granger results.")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("--quantile", type=float, default=0.005, help="Quantile to compute (default 0.005 for 0.5%)")
    args = parser.parse_args()

    compute_lower_quantile_threshold(args.csv_path, args.quantile)
