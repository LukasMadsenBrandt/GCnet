import pytest
import pandas as pd

from gene_analysis.pipeline.validation import (
    validate_expression_dataframe,
    validate_frequency_csv,
    validate_gc_csv,
    validate_gene_list_file,
)


pytestmark = pytest.mark.unit


def test_validate_gene_list_requires_gene_of_interest(tmp_path):
    gene_file = tmp_path / "genes.txt"
    gene_file.write_text("A\nB\n", encoding="utf-8")

    with pytest.raises(ValueError, match="gene_of_interest 'ZEB2'"):
        validate_gene_list_file(gene_file, min_genes=2, required_gene="ZEB2")


def test_validate_expression_dataframe_rejects_duplicate_genes():
    df = pd.DataFrame(
        {"T1": [1, 2], "T2": [2, 3], "T3": [3, 4], "T4": [4, 5], "T5": [5, 6]},
        index=["ZEB2", "ZEB2"],
    )

    with pytest.raises(ValueError, match="duplicate gene names"):
        validate_expression_dataframe(df, gene_of_interest="ZEB2", seed_genes=["ZEB2", "MECP2"])


def test_validate_expression_dataframe_rejects_negative_log1p_input():
    df = pd.DataFrame(
        {"T1": [1, 2], "T2": [2, 3], "T3": [3, 4], "T4": [4, 5], "T5": [-1, 6]},
        index=["ZEB2", "MECP2"],
    )

    with pytest.raises(ValueError, match="negative expression values"):
        validate_expression_dataframe(
            df,
            gene_of_interest="ZEB2",
            seed_genes=["ZEB2", "MECP2"],
            transform="log1p",
        )


def test_validate_gc_csv_rejects_self_pairs(tmp_path):
    gc_file = tmp_path / "gc.csv"
    gc_file.write_text("gene1,gene2,lag,p-value\nZEB2,ZEB2,1,0.01\n", encoding="utf-8")

    with pytest.raises(ValueError, match="self-pairs"):
        validate_gc_csv(gc_file)


def test_validate_gc_csv_rejects_bad_schema(tmp_path):
    gc_file = tmp_path / "gc.csv"
    gc_file.write_text("gene1,gene2,p\nZEB2,A,0.01\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        validate_gc_csv(gc_file)


def test_validate_frequency_csv_requires_sorted_unit_interval_and_goi(tmp_path):
    frequency_csv = tmp_path / "freq.csv"
    frequency_csv.write_text(
        "Gene,Coassociation Frequency\nA,0.2\nZEB2,0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sorted descending"):
        validate_frequency_csv(frequency_csv, gene_of_interest="ZEB2")

    frequency_csv.write_text("Gene,Coassociation Frequency\nA,1.2\nZEB2,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_frequency_csv(frequency_csv, gene_of_interest="ZEB2")

    frequency_csv.write_text("Gene,Coassociation Frequency\nA,0.9\nB,0.8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gene_of_interest"):
        validate_frequency_csv(frequency_csv, gene_of_interest="ZEB2")
