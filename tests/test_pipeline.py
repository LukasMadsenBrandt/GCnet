import csv
import json
from pathlib import Path

import pytest

import project_paths
from gene_analysis.pipeline.config import DatasetConfig, ExpansionConfig, NetworkConfig, ProbeConfig, SeedGeneConfig
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
probe_selection:
  mode: min_frequency
  top_percent: null
  min_frequency: 0.75
execution:
  max_workers: 2
  chunk_size: 10
  resume: false
""",
        encoding="utf-8",
    )

    cfg = PipelineConfig.from_yaml(config_file)

    assert cfg.run_name == "zeb2_demo"
    assert cfg.network.p_value_threshold == pytest.approx(0.0015)
    assert cfg.probe_selection.mode == "min_frequency"
    assert cfg.execution.max_workers == 2
    assert cfg.execution.resume is False


def test_dataset_pipeline_configs_load():
    for path in (
        "configs/gene_expansion.kutsche.yml",
        "configs/gene_expansion.benito_human.yml",
        "configs/gene_expansion.benito_gorilla.yml",
    ):
        cfg = PipelineConfig.from_yaml(path)
        assert cfg.dataset.expression_file
        assert cfg.dataset.full_gene_file
        assert cfg.seed_gene_file


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
    monkeypatch.setattr(project_paths, "RESULTS_DIR", tmp_path / "results")
    cfg = make_pipeline_config(tmp_path, run_name="missing_artifact")
    runner = PipelineRunner(cfg)

    with pytest.raises(FileNotFoundError, match="Missing required artifact 'seed_frequency_csv'"):
        runner.run(start_at="03_probe_selection", stop_after="03_probe_selection")


def test_runner_stage_03_from_configured_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(project_paths, "RESULTS_DIR", tmp_path / "results")
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
    monkeypatch.setattr(project_paths, "RESULTS_DIR", tmp_path / "results")
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


def test_sample_fixture_pipeline_runs_all_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(project_paths, "RESULTS_DIR", tmp_path / "results")
    cfg = PipelineConfig.from_yaml("configs/gene_expansion.sample.yml")

    artifacts = PipelineRunner(cfg).run()

    expected_stage_dirs = [
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
    assert artifacts["seed_network_edges_csv"].exists()
    assert artifacts["seed_consensus_history_json"].exists()
    assert artifacts["probe_genes_file"].exists()
    assert artifacts["probe_gc_csv"].exists()
    assert artifacts["expanded_genes_file"].exists()
    assert artifacts["probe_network_graphml"].exists()
    assert artifacts["probe_network_edges_csv"].exists()
    assert artifacts["expanded_gc_csv"].exists()
    assert artifacts["priority_genes_csv"].exists()
    assert artifacts["expanded_consensus_history_json"].exists()
    assert artifacts["expanded_network_graphml"].exists()
    assert artifacts["expanded_network_edges_csv"].exists()
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
    run_manifest = json.loads((cfg.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["settings"]["gene_of_interest"] == "ZEB2"
    assert run_manifest["pipeline_evolution"]


def test_sample_real_gc_pipeline_runs_all_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(project_paths, "RESULTS_DIR", tmp_path / "results")
    cfg = PipelineConfig.from_yaml("configs/gene_expansion.real_gc_sample.yml")

    artifacts = PipelineRunner(cfg).run()

    assert artifacts["seed_gc_csv"].exists()
    assert artifacts["probe_gc_csv"].exists()
    assert artifacts["expanded_gc_csv"].exists()
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
