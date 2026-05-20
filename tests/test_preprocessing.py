import numpy as np
import pandas as pd
import pytest

from gene_analysis.analysis.preprocessing import aggregate_duplicate_genes, apply_expression_preprocessing
from gene_analysis.pipeline.runner import PipelineConfig


def test_log1p_transform_handles_zero_counts():
    df = pd.DataFrame({"T1": [0.0], "T2": [1.0], "T3": [3.0]}, index=["ZEB2"])

    result = apply_expression_preprocessing(df, normalize="none", transform="log1p")

    assert result.loc["ZEB2", "T1"] == pytest.approx(0.0)
    assert result.loc["ZEB2", "T2"] == pytest.approx(np.log1p(1.0))
    assert result.loc["ZEB2", "T3"] == pytest.approx(np.log1p(3.0))


def test_sqrt_transform_is_elementwise():
    df = pd.DataFrame({"T1": [0.0, 4.0], "T2": [9.0, 16.0]}, index=["ZEB2", "MECP2"])

    result = apply_expression_preprocessing(df, normalize="none", transform="sqrt")

    assert result.loc["ZEB2", "T2"] == pytest.approx(3.0)
    assert result.loc["MECP2", "T1"] == pytest.approx(2.0)
    assert result.loc["MECP2", "T2"] == pytest.approx(4.0)


def test_zscore_normalization_is_columnwise():
    df = pd.DataFrame({"T1": [1.0, 2.0, 3.0], "T2": [10.0, 20.0, 30.0]}, index=["A", "B", "C"])

    result = apply_expression_preprocessing(df, normalize="zscore", transform="none")

    assert result["T1"].mean() == pytest.approx(0.0)
    assert result["T2"].mean() == pytest.approx(0.0)
    assert result["T1"].std() == pytest.approx(1.0)
    assert result["T2"].std() == pytest.approx(1.0)


def test_deseq_normalization_uses_median_ratio_size_factors():
    df = pd.DataFrame({"T1": [10.0, 20.0], "T2": [20.0, 40.0]}, index=["A", "B"])

    result = apply_expression_preprocessing(df, normalize="deseq", transform="none")

    assert result.loc["A", "T1"] == pytest.approx(result.loc["A", "T2"])
    assert result.loc["B", "T1"] == pytest.approx(result.loc["B", "T2"])
    assert result.loc["B", "T1"] == pytest.approx(result.loc["A", "T1"] * 2)


def test_aggregate_duplicate_genes_uses_configured_method():
    df = pd.DataFrame(
        {"T1": [1.0, 3.0, 10.0], "T2": [2.0, 6.0, 20.0]},
        index=["GGT1", "GGT1", "MECP2"],
    )

    result = aggregate_duplicate_genes(df, method="mean")

    assert list(result.index) == ["GGT1", "MECP2"]
    assert result.loc["GGT1", "T1"] == pytest.approx(2.0)
    assert result.loc["GGT1", "T2"] == pytest.approx(4.0)


def test_preprocessing_rejects_unknown_normalization():
    df = pd.DataFrame({"T1": [1.0]}, index=["ZEB2"])

    with pytest.raises(ValueError, match="Unknown normalization option"):
        apply_expression_preprocessing(df, normalize="quantile", transform="none")


def test_preprocessing_rejects_nonnumeric_expression_values():
    df = pd.DataFrame({"T1": ["not-a-number"]}, index=["ZEB2"])

    with pytest.raises(ValueError):
        apply_expression_preprocessing(df, normalize="none", transform="none")


def test_benito_default_preprocessing_does_not_force_transform():
    cfg = PipelineConfig.from_dict(
        {
            "run_name": "benito_default",
            "dataset": {
                "name": "benito_human",
                "expression_file": "Data/Benito/Benito_Human",
                "full_gene_file": "Data/Benito/unique_genes.txt",
            },
            "gene_of_interest": "MECP2",
            "seed_gene_file": "Data/Benito/unique_genes.txt",
        }
    )

    assert cfg.preprocessing.normalize == "none"
    assert cfg.preprocessing.transform == "none"
    assert cfg.preprocessing.aggregation == "robust"
