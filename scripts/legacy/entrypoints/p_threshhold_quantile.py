"""Backward-compatible misspelled threshold helper entrypoint."""

from gene_analysis.pipeline.thresholding import compute_lower_quantile_threshold as compute_one_lower_quantile_threshold
from gene_analysis.pipeline.thresholding import run


def compute_lower_quantile_threshold(csv_paths, quantile=0.0005):
    """Backward-compatible wrapper around the canonical thresholding module."""
    if isinstance(csv_paths, (str, bytes)):
        return compute_one_lower_quantile_threshold(csv_paths, quantile).threshold
    results = run(csv_paths, quantile)
    for result in results:
        print(
            f"\nProcessing File: {result.csv_path}\n"
            f"  Quantile: {result.quantile*100:.2f}%\n"
            f"  Rounded (4 decimals): {round(result.threshold, 4)}\n"
        )
    return [result.threshold for result in results]


if __name__ == "__main__":
    csv_paths = ["granger_causality_results_truncated_benito_gorilla.csv", "granger_causality_results_truncated_benito_human.csv", "granger_causality_results_truncated.csv"]
    quantile = 0.05

    compute_lower_quantile_threshold(csv_paths, quantile)
