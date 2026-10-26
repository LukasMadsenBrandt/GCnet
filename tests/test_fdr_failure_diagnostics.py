"""Diagnose tiny completed checkpoints without CUDA or modifying saved tests."""

import fcntl
import json
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gene_analysis.pipeline.fdr_diagnostics import run_failure_diagnostics
from gene_analysis.pipeline.runner import PipelineConfig, _config_to_dict, run_pipeline


@pytest.fixture
def completed_failure_run(tmp_path, monkeypatch):
    import gene_analysis.pipeline.runner as runner

    monkeypatch.setattr(runner, "results_path", lambda *parts: tmp_path.joinpath(*parts))
    rng = np.random.default_rng(91)
    times = np.repeat(np.arange(5), 7)
    predictor = rng.normal(size=35) + 10
    values = np.column_stack([predictor, 10 + times * predictor + rng.normal(size=35),
                              np.full(35, 10.0), np.zeros(35)])
    values[0, 3] = 5.0
    genes = ["ZEB2", "A", "CONST", "SPARSE"]
    path = tmp_path / "expression.csv"
    pd.DataFrame([(gene, f"s{sample:02d}", day, values[sample, i])
                  for i, gene in enumerate(genes) for sample, day in enumerate(times)],
                 columns=["Gene", "Sample", "Day", "Expression"]).to_csv(path, index=False)
    seed = tmp_path / "seeds.txt"
    seed.write_text("\n".join(genes), encoding="utf-8")
    config = PipelineConfig.from_dict({
        "run_name": "failure_fixture", "dataset": {"name": "sample_long", "expression_file": str(path)},
        "gene_of_interest": "ZEB2", "seed_gene_file": str(seed),
        "analysis": {"mode": "fdr", "workflow": "all_pairs", "backend": "cpu_numpy", "batch_size": 3},
        "execution": {"resume": True, "progress": False},
    })
    run_pipeline(config)
    return config


def test_diagnostic_classifies_design_failures_and_keeps_checkpoint_unchanged(completed_failure_run):
    config = completed_failure_run
    root = config.run_dir / "fdr"
    database = root / "all_pairs/tests.sqlite"
    before = database.read_bytes()
    directory, archive = run_failure_diagnostics(config, family="all_pairs", fetch_size=2,
                                                samples_per_category=2)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["failed_tests"] == 8
    assert summary["saved_failure_count_verified"] is True
    categories = summary["failed_tests_by_current_input_recheck_category"]
    assert categories["rank_deficient_predictor_design_on_recheck"] == 6
    assert categories["zero_time_adjusted_target_on_recheck"] == 2
    predictors = pd.read_csv(directory / "failed_predictors.csv").set_index("gene")
    assert predictors.loc["CONST", "residual_predictor_design_rank"] == 0
    assert predictors.loc["SPARSE", "residual_predictor_design_rank"] == 1
    assert predictors.loc["SPARSE", "days_with_nonzero_processed_expression"] == 1
    assert len(pd.read_csv(directory / "failed_pair_samples.csv")) <= 4
    assert database.read_bytes() == before
    with tarfile.open(archive) as bundle:
        assert any(member.name.endswith("summary.json") for member in bundle.getmembers())
        assert not any("sqlite" in member.name for member in bundle.getmembers())


def test_diagnostic_marks_valid_cpu_rechecks_as_unresolved(completed_failure_run):
    config = completed_failure_run
    root = config.run_dir / "fdr"
    genes = json.loads((root / "manifest.json").read_text())["genes"]
    with sqlite3.connect(root / "all_pairs/tests.sqlite") as db:
        with db:
            db.execute("UPDATE tests SET p=NULL WHERE source=? AND target=?", (genes.index("ZEB2"), genes.index("A")))
            result = json.loads(db.execute("SELECT result FROM layers WHERE name='overview'").fetchone()[0])
            result["failed_tests"] += 1
            db.execute("UPDATE layers SET result=? WHERE name='overview'", (json.dumps(result),))
    directory, _ = run_failure_diagnostics(config, family="all_pairs")
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["sampled_pair_cpu_recheck_reasons"]["valid_on_cpu_recheck_original_failure_unresolved"] == 1


def test_diagnostic_refuses_changed_input(completed_failure_run):
    config = completed_failure_run
    data = pd.read_csv(config.dataset.expression_file)
    data.loc[0, "Expression"] += 1
    data.to_csv(config.dataset.expression_file, index=False)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_failure_diagnostics(config, family="all_pairs")


def test_diagnostic_refuses_active_writer(completed_failure_run):
    config = completed_failure_run
    with (config.run_dir / "fdr/run.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="still writing"):
            run_failure_diagnostics(config, family="all_pairs")


def test_diagnostic_checks_saved_failure_count(completed_failure_run):
    config = completed_failure_run
    with sqlite3.connect(config.run_dir / "fdr/all_pairs/tests.sqlite") as db:
        with db:
            db.execute("UPDATE tests SET p=1 WHERE id=(SELECT min(id) FROM tests WHERE p IS NULL)")
    with pytest.raises(ValueError, match="saved completed summary reports"):
        run_failure_diagnostics(config, family="all_pairs")


def test_diagnostic_cli_reads_existing_run(completed_failure_run, tmp_path):
    config = completed_failure_run
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(_config_to_dict(config)), encoding="utf-8")
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, str(repository / "scripts/pipeline/diagnose_fdr_failures.py"),
        "--config", str(path), "--family", "all_pairs", "--run-dir", str(config.run_dir),
        "--samples-per-category", "0", "--fetch-size", "2",
    ], cwd=repository, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Failure diagnostics complete" in result.stdout
    summary = json.loads((config.run_dir / "fdr/all_pairs_failure_diagnostics/summary.json").read_text())
    assert summary["sampled_pairs_rechecked"] == 0
