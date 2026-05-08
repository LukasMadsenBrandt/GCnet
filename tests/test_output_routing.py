from pathlib import Path

from scripts.reporting import gene_list_compare, relative_difference


def test_lightweight_scripts_route_outputs_to_results():
    assert Path(gene_list_compare.OUTPUT_DIR).parts[-2:] == ("results", "gene_list_compare")
    assert Path(relative_difference.DIFF_CSV).parts[-3:] == ("results", "comparisons", "gene_freq_diff.csv")
    assert Path(relative_difference.SUMMARY_TXT).parts[-3:] == (
        "results",
        "comparisons",
        "gene_freq_diff_summary.txt",
    )
