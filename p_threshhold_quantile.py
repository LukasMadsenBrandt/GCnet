import os
import pandas as pd

def compute_lower_quantile_threshold(csv_paths, quantile=0.0005):
    """
    Load CSV with columns: gene1, gene2, lag, pvalue
    and compute the threshold corresponding to the lower quantile of p-values.
    
    Parameters:
        csv_path (str): Path to the input CSV file.
        quantile (float): The quantile to compute (e.g., 0.0005 for 0.05%).
        
    Returns:
        float: The threshold value at the given quantile.
    """
    i = 1
    # Compute quantile
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue  # Skip if the file does not exist
  
        # Load data
        df = pd.read_csv(csv_path)

        # Validate
        if 'p-value' not in df.columns:
            raise ValueError("CSV must contain a 'p-value' column.")

        threshold = df['p-value'].quantile(quantile)
        print(
            f"\nProcessing File #{i}: {csv_path}\n"
            f"  ↳ Quantile: {quantile*100:.2f}%\n"
            f"  ↳ Raw threshold value: {threshold}\n"
            f"  ↳ Rounded (4 decimals): {round(threshold, 4)}\n"
        )
        i += 1


if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Define the paths to the CSV files and the quantile to compute
    csv_paths = ["granger_causality_results_log_e_plus_1.csv", "granger_causality_results_sqrt.csv", "granger_causality_results_truncated.csv", "granger_causality_results.csv"]
    quantile = 0.0005

    compute_lower_quantile_threshold(csv_paths, quantile)
