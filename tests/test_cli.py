import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit


def test_run_pipeline_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/pipeline/run_pipeline.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--start-at" in result.stdout
    assert "--stop-after" in result.stdout


def test_benchmark_cuda_gc_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/pipeline/benchmark_cuda_gc.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--skip-runs" in result.stdout
    assert "--min-speedup" in result.stdout


def test_benchmark_consensus_backend_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/pipeline/benchmark_consensus_backend.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--candidate-backend" in result.stdout
    assert "--min-speedup" in result.stdout
    assert "--top-overlap-threshold-percent" in result.stdout


def test_render_network_figures_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/pipeline/render_network_figures.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--run-dir" in result.stdout
    assert "--renderer" in result.stdout
    assert "--all-layouts" in result.stdout


def test_run_pipeline_cli_fails_for_missing_config():
    result = subprocess.run(
        [sys.executable, "scripts/pipeline/run_pipeline.py", "--config", "missing.yml"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing.yml" in result.stderr
