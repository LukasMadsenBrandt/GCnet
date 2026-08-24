import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

import gene_analysis.io.paths as paths
from gene_analysis.pipeline.config import (
    DatasetConfig,
    ExecutionConfig,
    ExpansionConfig,
    NetworkConfig,
    PreprocessingConfig,
    ProbeConfig,
    SeedGeneConfig,
)
from gene_analysis.pipeline.dataset_probe import generate_probe_pairs, run as run_probe
from gene_analysis.pipeline.network_expansion import extract_expanded_genes_from_csv, run as run_expansion
from gene_analysis.pipeline.probe_selection import ProbeSelectionConfig, select_probe_genes
from gene_analysis.pipeline.runner import PipelineConfig, PipelineRunner, normalize_stage
from gene_analysis.pipeline.seed_genes import load_seed_genes, validate_seed_genes
from gene_analysis.pipeline.thresholding import compute_lower_quantile_threshold


def test_seed_gene_config_validation_requires_valid_threshold(tmp_path):
    cfg = SeedGeneConfig(seed_gene_file=tmp_path / "seeds.txt", p_threshold=0)

    with pytest.raises(ValueError):
        cfg.validate()


def test_load_and_validate_seed_genes(tmp_path):
    seed_file = tmp_path / "seed_genes.txt"
    seed_file.write_text("# comment\nZEB2\nMECP2\nZEB2\n\n", encoding="utf-8")

    seeds = load_seed_genes(seed_file)
    present, missing = validate_seed_genes(seeds, ["ZEB2", "RPL8"])

    assert seeds == ["ZEB2", "MECP2"]
    assert present == ["ZEB2"]
    assert missing == ["MECP2"]


def test_threshold_calculation_on_tiny_csv(tmp_path):
    csv_file = tmp_path / "gc.csv"
    csv_file.write_text(
        "gene1,gene2,lag,p-value\nA,B,1,0.01\nB,C,1,0.03\nC,D,1,0.05\n",
        encoding="utf-8",
    )

    result = compute_lower_quantile_threshold(csv_file, quantile=0.5)

    assert result.threshold == pytest.approx(0.03)


def test_probe_pair_generation_excludes_self_pairs():
    pairs = generate_probe_pairs(["ZEB2", "MECP2"], ["ZEB2", "A", "MECP2"])

    assert ("ZEB2", "ZEB2") not in pairs
    assert ("MECP2", "MECP2") not in pairs
    assert ("ZEB2", "A") in pairs
    assert ("A", "ZEB2") in pairs
    assert len(pairs) == len(set(pairs))


def test_probe_run_writes_pair_design(tmp_path):
    seed_file = tmp_path / "seeds.txt"
    full_file = tmp_path / "full.txt"
    output_file = tmp_path / "probe_pairs.csv"
    seed_file.write_text("ZEB2\n", encoding="utf-8")
    full_file.write_text("ZEB2\nA\nB\n", encoding="utf-8")

    output = run_probe(
        ProbeConfig(
            seed_gene_file=seed_file,
            full_gene_file=full_file,
            probe_pairs_file=output_file,
        )
    )

    with open(output, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows == [["gene1", "gene2"], ["ZEB2", "A"], ["ZEB2", "B"], ["A", "ZEB2"], ["B", "ZEB2"]]


def test_network_expansion_extracts_candidate_genes(tmp_path):
    gc_file = tmp_path / "probe_gc.csv"
    gc_file.write_text(
        "gene1,gene2,lag,p-value\nZEB2,A,1,0.001\nB,ZEB2,1,0.002\nZEB2,C,1,0.2\n",
        encoding="utf-8",
    )

    genes = extract_expanded_genes_from_csv(gc_file, p_threshold=0.01, gene_of_interest="ZEB2")

    assert genes == ["A", "B", "ZEB2"]


def test_network_expansion_run_includes_seed_genes(tmp_path):
    gc_file = tmp_path / "probe_gc.csv"
    seed_file = tmp_path / "seeds.txt"
    output_file = tmp_path / "expanded.txt"
    gc_file.write_text("gene1,gene2,lag,p-value\nZEB2,A,1,0.001\n", encoding="utf-8")
    seed_file.write_text("MECP2\n", encoding="utf-8")

    output = run_expansion(
        ExpansionConfig(
            candidate_network_csv=gc_file,
            gene_of_interest="ZEB2",
            p_threshold=0.01,
            seed_gene_file=seed_file,
            output_gene_list=output_file,
        )
    )

    assert Path(output).read_text(encoding="utf-8").splitlines() == ["A", "MECP2", "ZEB2"]


def make_pipeline_config(tmp_path, *, run_name="test_run", artifacts=None, selection=None):
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("ZEB2\nMECP2\n", encoding="utf-8")
    expression_file = tmp_path / "expression.tsv"
    expression_file.write_text("Gene\tT1\tT2\nZEB2\t1\t2\n", encoding="utf-8")
    full_gene_file = tmp_path / "all_genes.txt"
    full_gene_file.write_text("ZEB2\nMECP2\nA\n", encoding="utf-8")
    return PipelineConfig(
        run_name=run_name,
        dataset=DatasetConfig(name="kutsche", expression_file=expression_file, full_gene_file=full_gene_file),
        gene_of_interest="ZEB2",
        seed_gene_file=seed_file,
        network=NetworkConfig(p_value_threshold=0.01),
        probe_selection=selection or ProbeSelectionConfig(mode="top_percent", top_percent=50.0),
        artifacts=artifacts or {},
    )


def test_pipeline_config_loads_yaml(tmp_path):
    config_file = tmp_path / "pipeline.yml"
    config_file.write_text(
        """
run_name: zeb2_demo
dataset:
  name: kutsche
  expression_file: expression.tsv
  full_gene_file: all_genes.txt
gene_of_interest: ZEB2
seed_gene_file: seeds.txt
network:
  p_value_threshold: 0.0015
preprocessing:
  normalize: zscore
  transform: log1p
  aggregation: mean
probe_selection:
  mode: min_frequency
  top_percent: null
  min_frequency: 0.75
execution:
  max_workers: 2
  chunk_size: 10
  resume: false
  gc_backend: cpu_statsmodels
  consensus_backend: cpu_louvain
  gpu_device: 0
""",
        encoding="utf-8",
    )

    cfg = PipelineConfig.from_yaml(config_file)

    assert cfg.run_name == "zeb2_demo"
    assert cfg.network.p_value_threshold == pytest.approx(0.0015)
    assert cfg.network.write_svg is True
    assert cfg.network.svg_renderer == "networkx"
    assert cfg.network.svg_layout == "dot"
    assert cfg.preprocessing.normalize == "zscore"
    assert cfg.preprocessing.transform == "log1p"
    assert cfg.preprocessing.aggregation == "mean"
    assert cfg.probe_selection.mode == "min_frequency"
    assert cfg.execution.max_workers == 2
    assert cfg.execution.resume is False
    assert cfg.execution.gc_backend == "cpu_statsmodels"
    assert cfg.execution.consensus_backend == "cpu_louvain"
    assert cfg.execution.gpu_device == 0


def test_preprocessing_config_rejects_unknown_options():
    with pytest.raises(ValueError, match="preprocessing.transform"):
        PreprocessingConfig(transform="rank").validate()


def test_network_config_rejects_non_boolean_svg_flag():
    with pytest.raises(ValueError, match="network.write_svg"):
        NetworkConfig(write_svg="false").validate()


def test_network_config_rejects_unknown_svg_renderer():
    with pytest.raises(ValueError, match="network.svg_renderer"):
        NetworkConfig(svg_renderer="unknown").validate()


def test_network_config_rejects_unknown_svg_layout():
    with pytest.raises(ValueError, match="network.svg_layout"):
        NetworkConfig(svg_layout="unknown").validate()


def test_dataset_pipeline_configs_load():
    for path in (
        "configs/production/gene_expansion.kutsche.yml",
        "configs/production/gene_expansion.benito_human.yml",
        "configs/production/gene_expansion.benito_gorilla.yml",
        "configs/production_like/gene_expansion.kutsche.real_gc_small.yml",
        "configs/production_like/gene_expansion.benito_human.real_gc_small.yml",
        "configs/production_like/gene_expansion.benito_gorilla.real_gc_small.yml",
        "configs/test/gene_expansion.gpu_sample.yml",
    ):
        cfg = PipelineConfig.from_yaml(path)
        assert cfg.dataset.expression_file
        if "production/" not in path:
            assert cfg.dataset.full_gene_file
        assert cfg.seed_gene_file
        assert cfg.network.write_svg is True
        assert cfg.network.svg_renderer == "networkx"
        assert cfg.network.svg_layout == "dot"


def test_generic_expression_dataset_loads_preprocessed_matrix(tmp_path):
    expression_file = tmp_path / "expression.csv"
    full_gene_file = tmp_path / "genes.txt"
    seed_file = tmp_path / "seeds.txt"
    expression_file.write_text(
        "Gene,T1,T2,T3,T4,T5\nZEB2,1,2,3,4,5\nA,2,3,4,5,6\n",
        encoding="utf-8",
    )
    full_gene_file.write_text("ZEB2\nA\n", encoding="utf-8")
    seed_file.write_text("ZEB2\n", encoding="utf-8")
    cfg = PipelineConfig(
        run_name="generic_dataset",
        dataset=DatasetConfig(
            name="generic_expression",
            expression_file=expression_file,
            full_gene_file=full_gene_file,
        ),
        gene_of_interest="ZEB2",
        seed_gene_file=seed_file,
    )

    df = PipelineRunner(cfg).load_expression_dataframe()

    assert list(df.index) == ["ZEB2", "A"]
    assert list(df.columns) == ["T1", "T2", "T3", "T4", "T5"]


def test_generic_expression_applies_log1p_transform(tmp_path):
    expression_file = tmp_path / "expression.csv"
    full_gene_file = tmp_path / "genes.txt"
    seed_file = tmp_path / "seeds.txt"
    expression_file.write_text(
        "Gene,T1,T2,T3,T4,T5\nZEB2,0,1,2,3,4\n",
        encoding="utf-8",
    )
    full_gene_file.write_text("ZEB2\n", encoding="utf-8")
    seed_file.write_text("ZEB2\n", encoding="utf-8")
    cfg = PipelineConfig(
        run_name="generic_dataset",
        dataset=DatasetConfig(
            name="generic_expression",
            expression_file=expression_file,
            full_gene_file=full_gene_file,
        ),
        gene_of_interest="ZEB2",
        seed_gene_file=seed_file,
        preprocessing=PreprocessingConfig(transform="log1p"),
    )

    df = PipelineRunner(cfg).load_expression_dataframe()

    assert df.loc["ZEB2", "T1"] == pytest.approx(0.0)
    assert df.loc["ZEB2", "T2"] == pytest.approx(0.693147, abs=1e-6)


def test_generic_expression_aggregates_duplicate_gene_symbols(tmp_path):
    expression_file = tmp_path / "expression.csv"
    full_gene_file = tmp_path / "genes.txt"
    seed_file = tmp_path / "seeds.txt"
    expression_file.write_text(
        "Gene,T1,T2,T3,T4,T5\n"
        "MECP2,1,2,3,4,5\n"
        "GGT1,1,1,1,1,1\n"
        "GGT1,3,3,3,3,3\n",
        encoding="utf-8",
    )
    full_gene_file.write_text("MECP2\nGGT1\n", encoding="utf-8")
    seed_file.write_text("MECP2\nGGT1\n", encoding="utf-8")
    cfg = PipelineConfig(
        run_name="generic_dataset_duplicates",
        dataset=DatasetConfig(
            name="generic_expression",
            expression_file=expression_file,
            full_gene_file=full_gene_file,
        ),
        gene_of_interest="MECP2",
        seed_gene_file=seed_file,
        preprocessing=PreprocessingConfig(aggregation="mean"),
    )

    df = PipelineRunner(cfg).load_expression_dataframe()

    assert list(df.index) == ["MECP2", "GGT1"]
    assert df.index.has_duplicates is False
    assert df.loc["GGT1", "T1"] == pytest.approx(2.0)


def test_kutsche_expression_applies_shared_log1p_transform(tmp_path):
    expression_file = tmp_path / "kutsche_counts.tsv"
    full_gene_file = tmp_path / "genes.txt"
    seed_file = tmp_path / "seeds.txt"
    expression_file.write_text(
        "\t".join(["Gene", "WT_d1_r1", "WT_d2_r1", "WT_d3_r1", "WT_d4_r1", "WT_d5_r1"])
        + "\n"
        + "\t".join(["ZEB2", "0", "1", "2", "3", "4"])
        + "\n",
        encoding="utf-8",
    )
    full_gene_file.write_text("ZEB2\n", encoding="utf-8")
    seed_file.write_text("ZEB2\n", encoding="utf-8")
    cfg = PipelineConfig(
        run_name="kutsche_dataset",
        dataset=DatasetConfig(
            name="kutsche",
            expression_file=expression_file,
            full_gene_file=full_gene_file,
        ),
        gene_of_interest="ZEB2",
        seed_gene_file=seed_file,
        preprocessing=PreprocessingConfig(normalize="none", transform="log1p", aggregation="mean"),
    )

    df = PipelineRunner(cfg).load_expression_dataframe()

    assert list(df.columns) == [1, 2, 3, 4, 5]
    assert df.loc["ZEB2", 1] == pytest.approx(0.0)
    assert df.loc["ZEB2", 2] == pytest.approx(0.693147, abs=1e-6)


def test_generic_expression_requires_five_timepoints(tmp_path):
    expression_file = tmp_path / "expression.csv"
    full_gene_file = tmp_path / "genes.txt"
    seed_file = tmp_path / "seeds.txt"
    expression_file.write_text("Gene,T1,T2,T3,T4\nZEB2,1,2,3,4\n", encoding="utf-8")
    full_gene_file.write_text("ZEB2\n", encoding="utf-8")
    seed_file.write_text("ZEB2\n", encoding="utf-8")
    cfg = PipelineConfig(
        run_name="generic_dataset",
        dataset=DatasetConfig(
            name="generic_expression",
            expression_file=expression_file,
            full_gene_file=full_gene_file,
        ),
        gene_of_interest="ZEB2",
        seed_gene_file=seed_file,
    )

    with pytest.raises(ValueError, match="at least 5"):
        PipelineRunner(cfg).load_expression_dataframe()


def test_normalize_stage_accepts_numbers_and_prefixes():
    assert normalize_stage("3") == "03_probe_selection"
    assert normalize_stage("03_probe") == "03_probe_selection"

    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        normalize_stage("99")


def test_probe_selection_top_percent_includes_goi(tmp_path):
    frequency_csv = tmp_path / "freq.csv"
    frequency_csv.write_text(
        "Gene,Coassociation Frequency\nA,0.9\nB,0.8\nC,0.1\n",
        encoding="utf-8",
    )

    genes = select_probe_genes(
        frequency_csv,
        gene_of_interest="ZEB2",
        selection=ProbeSelectionConfig(mode="top_percent", top_percent=33.0),
    )

    assert genes == ["ZEB2", "A"]


def test_probe_selection_breaks_frequency_ties_by_gene_name(tmp_path):
    frequency_csv = tmp_path / "freq.csv"
    frequency_csv.write_text(
        "Gene,Coassociation Frequency\nGENE_043,1.0\nGENE_001,1.0\nZEB2,1.0\nGENE_045,1.0\n",
        encoding="utf-8",
    )

    genes = select_probe_genes(
        frequency_csv,
        gene_of_interest="ZEB2",
        selection=ProbeSelectionConfig(mode="top_percent", top_percent=50.0),
    )

    assert genes == ["ZEB2", "GENE_001", "GENE_043"]


def test_probe_selection_min_frequency(tmp_path):
    frequency_csv = tmp_path / "freq.csv"
    frequency_csv.write_text(
        "Gene,Coassociation Frequency\nZEB2,1.0\nA,0.9\nB,0.6\nC,0.1\n",
        encoding="utf-8",
    )

    genes = select_probe_genes(
        frequency_csv,
        gene_of_interest="ZEB2",
        selection=ProbeSelectionConfig(mode="min_frequency", top_percent=None, min_frequency=0.6),
    )

    assert genes == ["ZEB2", "A", "B"]


def test_probe_selection_rejects_dual_mode_config():
    cfg = ProbeSelectionConfig(mode="top_percent", top_percent=5.0, min_frequency=0.5)

    with pytest.raises(ValueError, match="cannot also set min_frequency"):
        cfg.validate()


def test_runner_fails_clearly_when_starting_without_required_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    cfg = make_pipeline_config(tmp_path, run_name="missing_artifact")
    runner = PipelineRunner(cfg)

    with pytest.raises(FileNotFoundError, match="Missing required artifact 'seed_frequency_csv'"):
        runner.run(start_at="03_probe_selection", stop_after="03_probe_selection")


def test_runner_stage_03_from_configured_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    frequency_csv = tmp_path / "seed_frequency.csv"
    frequency_csv.write_text("Gene,Coassociation Frequency\nA,0.9\nB,0.2\n", encoding="utf-8")
    cfg = make_pipeline_config(
        tmp_path,
        run_name="stage_03",
        artifacts={"seed_frequency_csv": frequency_csv},
        selection=ProbeSelectionConfig(mode="top_percent", top_percent=50.0),
    )

    artifacts = PipelineRunner(cfg).run(start_at="03_probe_selection", stop_after="03_probe_selection")

    assert artifacts["probe_genes_file"].read_text(encoding="utf-8").splitlines() == ["ZEB2", "A"]
    manifest = json.loads((cfg.run_dir / "03_probe_selection" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["mode"] == "top_percent"


def test_runner_stage_05_from_configured_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    probe_gc = tmp_path / "probe_gc.csv"
    probe_gc.write_text(
        "gene1,gene2,lag,p-value\nZEB2,A,1,0.001\nB,ZEB2,1,0.002\nZEB2,C,1,0.2\n",
        encoding="utf-8",
    )
    cfg = make_pipeline_config(tmp_path, run_name="stage_05", artifacts={"probe_gc_csv": probe_gc})

    artifacts = PipelineRunner(cfg).run(start_at="05_expanded_genes", stop_after="05_expanded_genes")

    assert artifacts["expanded_genes_file"].read_text(encoding="utf-8").splitlines() == ["A", "B", "MECP2", "ZEB2"]
    manifest = json.loads((cfg.run_dir / "05_expanded_genes" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["p_value_threshold"] == pytest.approx(0.01)
    assert manifest["metrics"]["seed_gene_count"] == 2
    assert manifest["metrics"]["expanded_gene_count"] == 4
    assert manifest["metrics"]["expanded_seed_overlap_count"] == 2
    assert manifest["metrics"]["expanded_new_gene_count"] == 2
    assert manifest["metrics"]["expanded_new_gene_percent"] == pytest.approx(50.0)
    assert manifest["metrics"]["seed_gene_retention_percent"] == pytest.approx(100.0)


def test_run_summary_warns_when_seed_list_is_full_gene_list(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    probe_gc = tmp_path / "probe_gc.csv"
    probe_gc.write_text("gene1,gene2,lag,p-value\nZEB2,MECP2,1,0.001\n", encoding="utf-8")
    cfg = make_pipeline_config(tmp_path, run_name="seed_equals_full", artifacts={"probe_gc_csv": probe_gc})
    cfg.dataset.full_gene_file.write_text("ZEB2\nMECP2\n", encoding="utf-8")

    artifacts = PipelineRunner(cfg).run(start_at="05_expanded_genes", stop_after="05_expanded_genes")

    summary = artifacts["run_summary_md"].read_text(encoding="utf-8")
    manifest = json.loads((cfg.run_dir / "05_expanded_genes" / "manifest.json").read_text(encoding="utf-8"))
    assert "New genes" in summary
    assert "The configured seed gene list overlaps almost completely with the full gene list" in summary
    assert manifest["metrics"]["expanded_new_gene_count"] == 0


def test_dataset_probe_derives_full_gene_landscape_from_expression(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    expression_file = tmp_path / "expression.csv"
    seed_file = tmp_path / "seeds.txt"
    probe_file = tmp_path / "probe_genes.txt"
    expression_file.write_text(
        "Gene,T1,T2,T3,T4,T5\nZEB2,1,2,3,4,5\nA,2,3,4,5,6\nB,5,4,3,2,1\n",
        encoding="utf-8",
    )
    seed_file.write_text("ZEB2\n", encoding="utf-8")
    probe_file.write_text("ZEB2\n", encoding="utf-8")
    cfg = PipelineConfig(
        run_name="derived_landscape",
        dataset=DatasetConfig(name="generic_expression", expression_file=expression_file),
        gene_of_interest="ZEB2",
        seed_gene_file=seed_file,
        artifacts={"probe_genes_file": probe_file},
        execution=ExecutionConfig(max_workers=1, chunk_size=10, resume=False),
    )

    artifacts = PipelineRunner(cfg).run(start_at="04_dataset_probe", stop_after="04_dataset_probe")

    manifest = json.loads((cfg.run_dir / "04_dataset_probe" / "manifest.json").read_text(encoding="utf-8"))
    assert artifacts["dataset_genes_file"].read_text(encoding="utf-8").splitlines() == ["A", "B", "ZEB2"]
    assert manifest["metrics"]["dataset_gene_count"] == 3
    assert manifest["metrics"]["dataset_genes_source"] == "expression_matrix"
    assert manifest["metrics"]["gc_pairs_total"] == 4


def test_expanded_gc_uses_expanded_gene_set_against_derived_expression_landscape(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    expression_file = tmp_path / "expression.csv"
    seed_file = tmp_path / "seeds.txt"
    expanded_file = tmp_path / "expanded_genes.txt"
    expression_file.write_text(
        "Gene,T1,T2,T3,T4,T5\n"
        "ZEB2,1,2,3,4,5\n"
        "A,2,3,4,5,6\n"
        "B,5,4,3,2,1\n"
        "C,1,1,2,3,5\n",
        encoding="utf-8",
    )
    seed_file.write_text("ZEB2\nA\n", encoding="utf-8")
    expanded_file.write_text("ZEB2\nA\nB\n", encoding="utf-8")
    cfg = PipelineConfig(
        run_name="expanded_gc_derived_landscape",
        dataset=DatasetConfig(name="generic_expression", expression_file=expression_file),
        gene_of_interest="ZEB2",
        seed_gene_file=seed_file,
        artifacts={"expanded_genes_file": expanded_file},
        execution=ExecutionConfig(max_workers=1, chunk_size=10, resume=False),
    )

    PipelineRunner(cfg).run(start_at="06_expanded_gc", stop_after="06_expanded_gc")

    manifest = json.loads((cfg.run_dir / "06_expanded_gc" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metrics"]["expanded_gene_count"] == 3
    assert manifest["metrics"]["expanded_genes_present_in_expression"] == 3
    assert manifest["metrics"]["expanded_genes_missing_from_expression"] == 0
    assert manifest["metrics"]["gc_pairs_total"] == 3 * 2
    assert manifest["metrics"]["expected_gc_pairs_total"] == 3 * 2


@pytest.mark.integration
@pytest.mark.visual
def test_sample_fixture_pipeline_runs_all_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    cfg = PipelineConfig.from_yaml("configs/test/gene_expansion.sample.yml")

    artifacts = PipelineRunner(cfg).run()

    expected_stage_dirs = [
        "00_preprocessing",
        "01_seed_gc",
        "02_seed_consensus",
        "03_probe_selection",
        "04_dataset_probe",
        "05_expanded_genes",
        "06_expanded_gc",
        "07_expanded_consensus",
    ]
    for stage in expected_stage_dirs:
        assert (cfg.run_dir / stage / "manifest.json").exists()

    assert artifacts["seed_gc_csv"].exists()
    assert artifacts["seed_network_graphml"].exists()
    assert artifacts["seed_network_svg"].exists()
    assert artifacts["seed_network_edges_csv"].exists()
    assert artifacts["seed_top_consensus_network_graphml"].exists()
    assert artifacts["seed_top_consensus_network_svg"].exists()
    assert artifacts["seed_top_consensus_network_edges_csv"].exists()
    assert artifacts["seed_consensus_history_json"].exists()
    assert artifacts["seed_consensus_progress_jsonl"].exists()
    seed_progress = artifacts["seed_consensus_progress_jsonl"].read_text(encoding="utf-8").splitlines()
    assert seed_progress
    assert "stability_tolerance" in json.loads(seed_progress[-1])
    assert artifacts["probe_genes_file"].exists()
    assert artifacts["probe_gc_csv"].exists()
    assert artifacts["expanded_genes_file"].exists()
    assert artifacts["probe_network_graphml"].exists()
    assert artifacts["probe_network_svg"].exists()
    assert artifacts["probe_network_edges_csv"].exists()
    assert artifacts["expanded_gc_csv"].exists()
    assert artifacts["priority_genes_csv"].exists()
    assert artifacts["expanded_consensus_history_json"].exists()
    assert artifacts["expanded_consensus_progress_jsonl"].exists()
    assert artifacts["expanded_network_graphml"].exists()
    assert artifacts["expanded_network_svg"].exists()
    assert artifacts["expanded_network_edges_csv"].exists()
    assert artifacts["expanded_top_consensus_network_graphml"].exists()
    assert artifacts["expanded_top_consensus_network_svg"].exists()
    assert artifacts["expanded_top_consensus_network_edges_csv"].exists()
    assert artifacts["run_summary_md"].exists()
    assert artifacts["seed_network_figure_svg"].exists()
    assert artifacts["seed_top_consensus_network_figure_svg"].exists()
    assert artifacts["probe_network_figure_svg"].exists()
    assert artifacts["expanded_network_figure_svg"].exists()
    assert artifacts["expanded_top_consensus_network_figure_svg"].exists()
    assert "ZEB2" in artifacts["seed_network_svg"].read_text(encoding="utf-8")
    assert "ZEB2" in artifacts["seed_top_consensus_network_svg"].read_text(encoding="utf-8")
    assert "ZEB2" in artifacts["probe_network_svg"].read_text(encoding="utf-8")
    assert "ZEB2" in artifacts["expanded_network_svg"].read_text(encoding="utf-8")
    assert "ZEB2" in artifacts["expanded_top_consensus_network_svg"].read_text(encoding="utf-8")
    run_summary = artifacts["run_summary_md"].read_text(encoding="utf-8")
    assert "Pipeline Evolution" in run_summary
    assert "Top Priority Genes" in run_summary
    assert "GC backend: `cpu_statsmodels`" in run_summary
    assert "Consensus backend: `cpu_louvain`" in run_summary
    assert "Network SVG renderer: `networkx`" in run_summary
    assert "Network SVG layout: `dot`" in run_summary
    expanded_genes = artifacts["expanded_genes_file"].read_text(encoding="utf-8").splitlines()
    assert len(expanded_genes) == 150
    assert "GENE_149" in expanded_genes
    assert "GENE_999" not in expanded_genes
    assert "ZEB2" in artifacts["priority_genes_csv"].read_text(encoding="utf-8")
    seed_gc_manifest = json.loads((cfg.run_dir / "01_seed_gc" / "manifest.json").read_text(encoding="utf-8"))
    assert seed_gc_manifest["metrics"]["seed_gene_count"] == 50
    assert seed_gc_manifest["metrics"]["gc_pairs_total"] == 50 * 49
    probe_manifest = json.loads((cfg.run_dir / "04_dataset_probe" / "manifest.json").read_text(encoding="utf-8"))
    assert probe_manifest["metrics"]["dataset_gene_count"] == 1000
    expanded_gc_manifest = json.loads((cfg.run_dir / "06_expanded_gc" / "manifest.json").read_text(encoding="utf-8"))
    assert expanded_gc_manifest["metrics"]["expanded_gene_count"] == 150
    assert expanded_gc_manifest["metrics"]["gc_pairs_total"] == 150 * 149
    seed_manifest = json.loads((cfg.run_dir / "02_seed_consensus" / "manifest.json").read_text(encoding="utf-8"))
    assert seed_manifest["metrics"]["genes_total"] == 50
    assert seed_manifest["metrics"]["edges_total"] == 50 * 49
    assert seed_manifest["metrics"]["gene_of_interest_consensus_community_genes"] >= 1
    assert seed_manifest["metrics"]["consensus_total_seconds"] >= 0
    assert seed_manifest["metrics"]["louvain_seconds"] >= 0
    assert seed_manifest["metrics"]["coassociation_seconds"] >= 0
    assert seed_manifest["metrics"]["agglomerative_clustering_seconds"] >= 0
    assert seed_manifest["metrics"]["coassociation_matrix_cells"] == 50 * 50
    assert seed_manifest["metrics"]["top_consensus_genes_total"] >= 1
    assert seed_manifest["metrics"]["top_consensus_subcommunities"] >= 1
    run_manifest = json.loads((cfg.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["settings"]["gene_of_interest"] == "ZEB2"
    assert run_manifest["settings"]["preprocessing"]["transform"] == "none"
    assert run_manifest["settings"]["execution"]["gc_backend"] == "cpu_statsmodels"
    assert run_manifest["settings"]["execution"]["consensus_backend"] == "cpu_louvain"
    assert run_manifest["artifacts"]["run_summary_md"].endswith("RUN_SUMMARY.md")
    assert run_manifest["pipeline_evolution"]
    assert seed_gc_manifest["metrics"]["backend_metadata"]["backend"] == "cpu_statsmodels"
    assert seed_manifest["metrics"]["backend_metadata"]["backend"] == "cpu_louvain"


def test_pipeline_can_skip_network_svg_previews_only(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    cfg = PipelineConfig.from_yaml("configs/test/gene_expansion.sample.yml")
    cfg = replace(cfg, run_name="sample_fixture_no_network_svg", network=NetworkConfig(p_value_threshold=0.01, write_svg=False))

    artifacts = PipelineRunner(cfg).run(stop_after="05_expanded_genes")

    assert artifacts["seed_gc_csv"].exists()
    assert artifacts["seed_frequency_csv"].exists()
    assert artifacts["expanded_genes_file"].exists()
    assert artifacts["seed_network_graphml"].exists()
    assert artifacts["seed_network_edges_csv"].exists()
    assert artifacts["seed_network_nodes_file"].exists()
    assert artifacts["seed_network_summary_json"].exists()
    assert artifacts["seed_top_consensus_network_graphml"].exists()
    assert artifacts["seed_top_consensus_network_edges_csv"].exists()
    assert artifacts["seed_top_consensus_network_nodes_file"].exists()
    assert artifacts["seed_top_consensus_network_summary_json"].exists()
    assert artifacts["probe_network_graphml"].exists()
    assert artifacts["probe_network_edges_csv"].exists()
    assert artifacts["probe_network_nodes_file"].exists()
    assert artifacts["probe_network_summary_json"].exists()
    assert "seed_network_svg" not in artifacts
    assert "seed_top_consensus_network_svg" not in artifacts
    assert "seed_network_figure_svg" not in artifacts
    assert "seed_top_consensus_network_figure_svg" not in artifacts
    assert not (cfg.run_dir / "02_seed_consensus" / "seed_network.svg").exists()
    assert not (cfg.run_dir / "02_seed_consensus" / "seed_top_consensus_network.svg").exists()
    assert not (cfg.run_dir / "05_expanded_genes" / "probe_network.svg").exists()
    assert artifacts["run_summary_md"].exists()

    expanded_genes = artifacts["expanded_genes_file"].read_text(encoding="utf-8").splitlines()
    assert len(expanded_genes) == 150

    seed_manifest = json.loads((cfg.run_dir / "02_seed_consensus" / "manifest.json").read_text(encoding="utf-8"))
    assert seed_manifest["parameters"]["write_network_svg"] is False
    assert "seed_network_svg" not in seed_manifest["outputs"]
    assert "seed_top_consensus_network_svg" not in seed_manifest["outputs"]
    assert "seed_network_graphml" in seed_manifest["outputs"]
    assert "seed_top_consensus_network_graphml" in seed_manifest["outputs"]
    probe_manifest = json.loads((cfg.run_dir / "05_expanded_genes" / "manifest.json").read_text(encoding="utf-8"))
    assert probe_manifest["parameters"]["write_network_svg"] is False
    assert "probe_network_svg" not in probe_manifest["outputs"]
    assert "probe_network_graphml" in probe_manifest["outputs"]


@pytest.mark.real_gc
@pytest.mark.integration
@pytest.mark.visual
def test_sample_real_gc_pipeline_runs_all_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    cfg = PipelineConfig.from_yaml("configs/test/gene_expansion.real_gc_sample.yml")

    artifacts = PipelineRunner(cfg).run()

    assert artifacts["seed_gc_csv"].exists()
    assert artifacts["seed_network_svg"].exists()
    assert artifacts["probe_gc_csv"].exists()
    assert artifacts["probe_network_svg"].exists()
    assert artifacts["expanded_gc_csv"].exists()
    assert artifacts["expanded_network_svg"].exists()
    assert artifacts["priority_genes_csv"].exists()
    assert "ZEB2" in artifacts["priority_genes_csv"].read_text(encoding="utf-8")

    seed_gc_manifest = json.loads((cfg.run_dir / "01_seed_gc" / "manifest.json").read_text(encoding="utf-8"))
    assert seed_gc_manifest["metrics"]["seed_gene_count"] == 6
    assert seed_gc_manifest["metrics"]["gc_pairs_total"] == 6 * 5

    probe_manifest = json.loads((cfg.run_dir / "04_dataset_probe" / "manifest.json").read_text(encoding="utf-8"))
    assert probe_manifest["metrics"]["dataset_gene_count"] == 30
    assert probe_manifest["metrics"]["gc_pairs_total"] > 0

    expanded_gc_manifest = json.loads((cfg.run_dir / "06_expanded_gc" / "manifest.json").read_text(encoding="utf-8"))
    expanded_gene_count = expanded_gc_manifest["metrics"]["expanded_gene_count"]
    assert 2 <= expanded_gene_count < 30
    assert expanded_gc_manifest["metrics"]["gc_pairs_total"] == expanded_gene_count * (expanded_gene_count - 1)
