import json

import pytest

from gene_analysis.analysis.parity import (
    build_benchmark_report,
    compare_frequency_csvs,
    compare_gc_csvs,
    write_report_json,
)


pytestmark = pytest.mark.unit


def test_compare_gc_csvs_reports_p_value_and_significance_disagreements(tmp_path):
    cpu = tmp_path / "cpu_gc.csv"
    gpu = tmp_path / "gpu_gc.csv"
    cpu.write_text(
        "gene1,gene2,lag,p-value\n"
        "A,B,1,0.010000\n"
        "B,C,1,0.040000\n"
        "C,D,1,0.500000\n",
        encoding="utf-8",
    )
    gpu.write_text(
        "gene1,gene2,lag,p-value\n"
        "A,B,1,0.010001\n"
        "B,C,1,0.060000\n"
        "D,E,1,0.001000\n",
        encoding="utf-8",
    )

    report = compare_gc_csvs(cpu, gpu, p_value_tolerance=1e-7, significance_threshold=0.05)

    assert report.reference_rows == 3
    assert report.candidate_rows == 3
    assert report.shared_pairs == 2
    assert report.missing_in_candidate == 1
    assert report.extra_in_candidate == 1
    assert report.p_value_disagreements == 2
    assert report.significant_decision_disagreements == 1


def test_compare_frequency_csvs_reports_top_overlap(tmp_path):
    cpu = tmp_path / "cpu_freq.csv"
    gpu = tmp_path / "gpu_freq.csv"
    cpu.write_text(
        "Gene,Coassociation Frequency\nA,1.0\nB,0.9\nC,0.2\nD,0.1\n",
        encoding="utf-8",
    )
    gpu.write_text(
        "Gene,Coassociation Frequency\nA,1.0\nC,0.8\nB,0.7\nD,0.1\n",
        encoding="utf-8",
    )

    report = compare_frequency_csvs(cpu, gpu, top_k=2)

    assert report.top_gene_overlap_k == 2
    assert report.top_gene_overlap_percent == pytest.approx(50.0)
    assert report.quantile_relative_change >= 0


def test_benchmark_report_and_json_writer(tmp_path):
    report = build_benchmark_report(
        stage="02_seed_consensus",
        cpu_backend="cpu_louvain",
        candidate_backend="gpu_cugraph",
        cpu_seconds=20.0,
        candidate_seconds=5.0,
        work_units=100,
    )

    assert report.speedup == pytest.approx(4.0)
    assert report.work_units_per_second_cpu == pytest.approx(5.0)
    assert report.work_units_per_second_candidate == pytest.approx(20.0)
    assert report.min_expected_speedup is None
    assert report.speedup_passed is None

    output = write_report_json(report, tmp_path / "benchmark.json")
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["speedup"] == pytest.approx(4.0)


def test_benchmark_report_can_enforce_minimum_speedup():
    passing = build_benchmark_report(
        stage="01_seed_gc",
        cpu_backend="cpu_statsmodels",
        candidate_backend="gpu_cuda",
        cpu_seconds=10.0,
        candidate_seconds=2.0,
        min_expected_speedup=4.0,
    )
    failing = build_benchmark_report(
        stage="01_seed_gc",
        cpu_backend="cpu_statsmodels",
        candidate_backend="gpu_cuda",
        cpu_seconds=10.0,
        candidate_seconds=3.0,
        min_expected_speedup=4.0,
    )

    assert passing.speedup == pytest.approx(5.0)
    assert passing.speedup_passed is True
    assert failing.speedup == pytest.approx(10.0 / 3.0)
    assert failing.speedup_passed is False
