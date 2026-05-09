import csv

from gene_analysis_common.granger_causality import collect_significant_edges, save_results_to_csv


def test_collect_significant_edges_from_memory():
    gc_results = {
        ("A", "B"): {1: [{"ssr_ftest": (3.1, 0.01)}]},
        ("B", "C"): {1: [{"ssr_ftest": (0.5, 0.20)}]},
        ("C", "D"): {"error": "constant data"},
    }

    edges = collect_significant_edges(gc_results, p_value_threshold=0.05)

    assert edges == [(("A", 1), ("B", 0), 0.01)]


def test_save_results_to_csv_preserves_columns_and_collects_from_file(tmp_path):
    output_file = tmp_path / "nested" / "gc.csv"
    gc_results = {
        ("A", "B"): {1: [{"ssr_ftest": (3.1, 0.01234)}]},
        ("C", "D"): {"error": "constant data"},
    }

    save_results_to_csv(gc_results, output_file)

    with output_file.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["gene1", "gene2", "lag", "p-value"]
    assert rows[1] == ["A", "B", "1", "0.0123"]
    assert rows[2] == ["C", "D", "NaN", "NaN"]

    edges = collect_significant_edges(
        None,
        p_value_threshold=0.05,
        file=True,
        filepath=output_file,
        starting_genes=["A"],
        higher_threshold_for_starting_genes=0.05,
    )

    assert edges == [(("A", 1), ("B", 0), 0.0123)]

