import json

import pandas as pd
import pytest

import gene_analysis.io.paths as paths
from gene_analysis.dashboard.dataset_explorer import (
    create_dashboard,
    gc_coverage_summary,
    pairwise_comparison,
    quantile_relationship_metrics,
)
from gene_analysis.pipeline.config import DatasetConfig, ExecutionConfig, PreprocessingConfig
from gene_analysis.pipeline.runner import PipelineConfig, PipelineRunner


pytestmark = pytest.mark.unit


def test_preprocessing_stage_can_write_all_four_expression_views(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    expression_file = tmp_path / "expression.csv"
    seed_file = tmp_path / "unique_genes.txt"
    expression_file.write_text(
        "Gene,T1,T2,T3,T4,T5\n"
        "ZEB2,1,2,3,4,5\n"
        "DUP,2,3,4,5,6\n"
        "DUP,4,5,6,7,8\n"
        "OTHER,8,7,6,5,4\n",
        encoding="utf-8",
    )
    seed_file.write_text("ZEB2\nDUP\nMISSING\n", encoding="utf-8")
    cfg = PipelineConfig(
        run_name="preprocessing_exports",
        dataset=DatasetConfig(name="generic_expression", expression_file=expression_file),
        gene_of_interest="ZEB2",
        seed_gene_file=seed_file,
        preprocessing=PreprocessingConfig(
            aggregation="mean",
            export_all_replicates=True,
            export_all_summarized=True,
            export_subset_replicates=True,
            export_subset_summarized=True,
        ),
    )

    artifacts = PipelineRunner(cfg).run(stop_after="00_preprocessing")

    all_replicates = pd.read_csv(artifacts["all_genes_replicates_csv"])
    all_summarized = pd.read_csv(artifacts["all_genes_summarized_csv"])
    subset_replicates = pd.read_csv(artifacts["subset_genes_replicates_csv"])
    subset_summarized = pd.read_csv(artifacts["subset_genes_summarized_csv"])
    assert all_replicates["Gene"].tolist() == ["ZEB2", "DUP", "DUP", "OTHER"]
    assert all_summarized["Gene"].tolist() == ["ZEB2", "DUP", "OTHER"]
    assert subset_replicates["Gene"].tolist() == ["ZEB2", "DUP", "DUP"]
    assert subset_summarized["Gene"].tolist() == ["ZEB2", "DUP"]
    assert subset_summarized.loc[subset_summarized["Gene"] == "DUP", "T1"].item() == pytest.approx(3.0)

    manifest = json.loads((cfg.run_dir / "00_preprocessing" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metrics"]["all_genes_replicate_rows"] == 4
    assert manifest["metrics"]["all_genes_summarized_rows"] == 3
    assert manifest["metrics"]["subset_genes_missing"] == 1
    assert manifest["metrics"]["missing_subset_gene_names"] == ["MISSING"]


def test_preprocessing_exports_are_independently_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    expression_file = tmp_path / "expression.csv"
    seed_file = tmp_path / "seeds.txt"
    expression_file.write_text(
        "Gene,T1,T2,T3,T4,T5\nZEB2,1,2,3,4,5\nA,2,3,4,5,6\n",
        encoding="utf-8",
    )
    seed_file.write_text("ZEB2\n", encoding="utf-8")
    cfg = PipelineConfig(
        run_name="one_export",
        dataset=DatasetConfig(name="generic_expression", expression_file=expression_file),
        gene_of_interest="ZEB2",
        seed_gene_file=seed_file,
        preprocessing=PreprocessingConfig(export_all_summarized=True),
    )

    artifacts = PipelineRunner(cfg).run(stop_after="00_preprocessing")

    assert artifacts["all_genes_summarized_csv"].exists()
    assert "all_genes_replicates_csv" not in artifacts
    assert "subset_genes_replicates_csv" not in artifacts
    assert "subset_genes_summarized_csv" not in artifacts


def test_dataset_explorer_quantile_and_pair_metrics(tmp_path):
    gc = pd.DataFrame(
        {
            "gene1": ["ZEB2", "A", "ZEB2", "B", "A", "B"],
            "gene2": ["A", "ZEB2", "B", "ZEB2", "B", "A"],
            "lag": [1, 1, 1, 1, 1, 1],
            "p-value": [0.001, 0.002, 0.03, 0.04, 0.5, 0.8],
        }
    )

    metrics = quantile_relationship_metrics(gc, "ZEB2", [0.5, 1.0])
    assert metrics.loc[0, "P-value threshold"] == pytest.approx(0.035)
    assert metrics.loc[0, "Outgoing"] == 2
    assert metrics.loc[0, "Incoming"] == 1
    assert metrics.loc[0, "Unique related genes"] == 2
    assert metrics.loc[1, "Directed relationships"] == 4

    pairs = pairwise_comparison(gc, "ZEB2", ["A"])
    assert list(zip(pairs["Source"], pairs["Target"])) == [("ZEB2", "A"), ("A", "ZEB2")]

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"metrics": {"gc_pairs_total": 12}}), encoding="utf-8")
    coverage = gc_coverage_summary(gc, manifest_path)
    assert coverage["rows_written"] == 6
    assert coverage["pairs_attempted"] == 12
    assert coverage["coverage_fraction"] == pytest.approx(0.5)
    assert coverage["appears_complete"] is False


def test_dataset_explorer_app_builds_from_pipeline_artifacts():
    summarized = pd.DataFrame(
        {"T1": [1.0, 2.0], "T2": [2.0, 3.0], "T3": [3.0, 4.0]},
        index=["ZEB2", "A"],
    )
    gc = pd.DataFrame(
        {
            "gene1": ["ZEB2", "A"],
            "gene2": ["A", "ZEB2"],
            "lag": [1, 1],
            "p-value": [0.01, 0.02],
        }
    )

    app = create_dashboard(summarized, gc, replicates=summarized, default_gene="ZEB2")

    assert app.title == "Gene dataset explorer"
    assert app.layout is not None


def test_seed_gc_all_pairs_mode_is_stage_scoped_and_uses_distinct_output(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    cfg = PipelineConfig.from_yaml("configs/test/gene_expansion.sample.yml")
    cfg = PipelineConfig(
        run_name="fixture_all_pairs",
        dataset=cfg.dataset,
        gene_of_interest=cfg.gene_of_interest,
        seed_gene_file=cfg.seed_gene_file,
        preprocessing=cfg.preprocessing,
        network=cfg.network,
        consensus=cfg.consensus,
        probe_selection=cfg.probe_selection,
        execution=ExecutionConfig(
            max_workers=1,
            chunk_size=100,
            resume=False,
            seed_gc_store_all_pairs=True,
        ),
    )

    artifacts = PipelineRunner(cfg).run(stop_after="01_seed_gc")

    assert artifacts["seed_gc_csv"].name == "seed_gc_all_pairs.csv"
    manifest = json.loads((cfg.run_dir / "01_seed_gc" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["p_value_threshold"] == pytest.approx(1.0)
    assert manifest["parameters"]["network_p_value_threshold"] == pytest.approx(cfg.network.p_value_threshold)
    assert manifest["parameters"]["store_all_attempted_pairs"] is True
