import json

import pytest

import gene_analysis.io.paths as paths
from scripts.pipeline.benchmark_consensus_backend import benchmark_consensus_backend


pytestmark = pytest.mark.integration


def test_consensus_benchmark_command_compares_cpu_harness(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    report_path = tmp_path / "consensus_benchmark.json"

    output = benchmark_consensus_backend(
        "configs/test/gene_expansion.sample.yml",
        candidate_backend="cpu_louvain",
        output_file=report_path,
        run_prefix="pytest_consensus_sample",
        min_speedup=0.0,
        stop_after="02_seed_consensus",
    )

    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["candidate_backend"] == "cpu_louvain"
    assert report["top_overlap_threshold_percent"] == pytest.approx(95.0)
    assert report["cpu_run_name"] == "pytest_consensus_sample_cpu_consensus_benchmark"
    assert report["candidate_run_name"] == "pytest_consensus_sample_cpu_louvain_consensus_benchmark"
    assert report["frequency_parity"]["seed_frequency_csv"]["top_gene_overlap_percent"] == pytest.approx(100.0)
    assert "02_seed_consensus" in report["benchmarks"]
    assert report["benchmarks"]["02_seed_consensus"]["work_units"] > 0
    assert report["benchmarks"]["02_seed_consensus"]["speedup_passed"] is True
