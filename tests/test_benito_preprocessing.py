import pandas as pd
import pytest

from gene_analysis.datasets.benito import (
    aggregate_replicates_by_day,
    load_and_map_benito,
    preprocess_pipeline_benito,
)


pytestmark = pytest.mark.unit


def write_benito_fixture(tmp_path):
    mapping_file = tmp_path / "mapping.tsv"
    metadata_file = tmp_path / "metadata.csv"
    counts_file = tmp_path / "counts.tsv"
    mapping_file.write_text("ENSG1\tZEB2\nENSG2\tMECP2\n", encoding="utf-8")
    metadata_file.write_text(
        "Run,Organism,Time_point\n"
        "SRR1,Human,Day 1\n"
        "SRR2,Human,Day 1\n"
        "SRR3,Human,Day 2\n"
        "SRR4,Human,Day 2\n",
        encoding="utf-8",
    )
    counts_file.write_text(
        "# ignored featureCounts header\n"
        "Geneid\tSRR1_REF\tSRR2_REF\tSRR3_REF\tSRR4_REF\n"
        "ENSG1\t1\t3\t5\t7\n"
        "ENSG2\t2\t4\t6\t8\n",
        encoding="utf-8",
    )
    return counts_file, mapping_file, metadata_file


def test_load_and_map_benito_adds_gene_names_and_timepoint_columns(tmp_path):
    counts_file, mapping_file, metadata_file = write_benito_fixture(tmp_path)

    df = load_and_map_benito(counts_file, mapping_file, metadata_file)

    assert list(df["Gene_Name"]) == ["ZEB2", "MECP2"]
    assert "SRR1_Day 1" in df.columns
    assert "SRR3_Day 2" in df.columns


def test_aggregate_replicates_by_day_supports_mean_and_rejects_unknown_method():
    df = pd.DataFrame(
        {"SRR1_Day 1": [1.0], "SRR2_Day 1": [3.0], "SRR3_Day 2": [5.0]},
        index=["ZEB2"],
    )
    day_map = {"SRR1_Day 1": 1, "SRR2_Day 1": 1, "SRR3_Day 2": 2}

    aggregated = aggregate_replicates_by_day(df, day_map, method="mean")

    assert aggregated.loc["ZEB2", 1] == pytest.approx(2.0)
    assert aggregated.loc["ZEB2", 2] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        aggregate_replicates_by_day(df, day_map, method="mode")


def test_preprocess_pipeline_benito_can_run_without_forced_transform(tmp_path):
    counts_file, mapping_file, metadata_file = write_benito_fixture(tmp_path)

    aggregated, filtered, day_map = preprocess_pipeline_benito(
        counts_file,
        mapping_file,
        metadata_file,
        normalize=None,
        transformed=False,
        aggregation="mean",
    )

    assert day_map == {"SRR1_Day 1": 1, "SRR2_Day 1": 1, "SRR3_Day 2": 2, "SRR4_Day 2": 2}
    assert filtered.loc["ZEB2", "SRR1_Day 1"] == pytest.approx(1.0)
    assert aggregated.loc["ZEB2", 1] == pytest.approx(2.0)
    assert aggregated.loc["ZEB2", 2] == pytest.approx(6.0)
