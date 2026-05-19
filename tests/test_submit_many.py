import pytest

import gene_analysis.io.paths as paths
from scripts.pipeline.submit_many import (
    load_configs,
    pipeline_command,
    write_slurm_array_script,
    write_slurm_job_scripts,
)


pytestmark = pytest.mark.unit


def write_config(path, run_name):
    path.write_text(
        f"""
run_name: {run_name}
dataset:
  name: fixture
  expression_file: examples/sample_pipeline/expression.csv
  full_gene_file: examples/sample_pipeline/all_genes.txt
gene_of_interest: ZEB2
seed_gene_file: examples/sample_pipeline/seed_genes.txt
""",
        encoding="utf-8",
    )


def test_load_configs_rejects_duplicate_run_names(tmp_path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    write_config(first, "same_run")
    write_config(second, "same_run")

    with pytest.raises(ValueError, match="Duplicate run_name"):
        load_configs([str(first), str(second)])


def test_load_configs_can_reject_existing_result_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    config = tmp_path / "config.yml"
    write_config(config, "existing_run")
    (paths.RESULTS_DIR / "pipeline" / "existing_run").mkdir(parents=True)

    with pytest.raises(ValueError, match="Existing result folders"):
        load_configs([str(config)], check_existing_results=True)


def test_pipeline_command_uses_canonical_runner():
    assert pipeline_command("config.yml", python="python3") == [
        "python3",
        "scripts/pipeline/run_pipeline.py",
        "--config",
        "config.yml",
    ]


def test_write_slurm_job_scripts_creates_one_script_per_config(tmp_path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    write_config(first, "first_run")
    write_config(second, "second_run")
    configs = load_configs([str(first), str(second)])

    scripts = write_slurm_job_scripts(
        configs,
        tmp_path / "slurm",
        job_name="gene_test",
        cpus_per_task=16,
        mem="64G",
        time_limit="02:00:00",
        python="python3",
        extra_sbatch=["--partition=gpu"],
    )

    assert [script.name for script in scripts] == ["first_run.slurm", "second_run.slurm", "submit_all.sh"]
    text = (tmp_path / "slurm" / "first_run.slurm").read_text(encoding="utf-8")
    assert "#SBATCH --array" not in text
    assert "#SBATCH --job-name=gene_test_first_run" in text
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --partition=gpu" in text
    assert str(first) in text
    assert "python3 scripts/pipeline/run_pipeline.py --config" in text
    submit_all = (tmp_path / "slurm" / "submit_all.sh").read_text(encoding="utf-8")
    assert "sbatch" in submit_all
    assert "first_run.slurm" in submit_all
    assert "second_run.slurm" in submit_all


def test_write_slurm_array_script_remains_available_for_clusters_that_prefer_arrays(tmp_path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    write_config(first, "first_run")
    write_config(second, "second_run")
    configs = load_configs([str(first), str(second)])

    output = write_slurm_array_script(
        configs,
        tmp_path / "submit.sh",
        job_name="gene_test",
        cpus_per_task=16,
        mem="64G",
        time_limit="02:00:00",
        python="python3",
        extra_sbatch=["--partition=gpu"],
    )

    text = output.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1" in text
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --partition=gpu" in text
    assert str(first) in text
    assert str(second) in text
    assert 'python3 scripts/pipeline/run_pipeline.py --config "$CONFIG"' in text
