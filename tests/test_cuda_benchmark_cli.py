import json

import pytest

from scripts.pipeline.benchmark_cuda_gc import benchmark_cuda_gc


pytestmark = [pytest.mark.cuda, pytest.mark.cuda_benchmark, pytest.mark.slow]


def test_cuda_benchmark_command_runs_sample_and_writes_report(tmp_path):
    report_path = tmp_path / "cuda_benchmark.json"

    output = benchmark_cuda_gc(
        "configs/test/gene_expansion.real_gc_sample.yml",
        output_file=report_path,
        run_prefix="pytest_sample_real_gc",
        min_speedup=0.0,
    )

    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["cpu_run_name"] == "pytest_sample_real_gc_cpu_benchmark"
    assert report["cuda_run_name"] == "pytest_sample_real_gc_cuda_benchmark"
    assert report["min_speedup"] == pytest.approx(0.0)
    assert report["gc_parity"]["seed_gc_csv"]["p_value_disagreements"] == 0
    assert report["gc_parity"]["probe_gc_csv"]["p_value_disagreements"] == 0
    assert report["gc_parity"]["expanded_gc_csv"]["p_value_disagreements"] == 0
    assert report["frequency_parity"]["top_gene_overlap_percent"] == pytest.approx(100.0)
    assert "01_seed_gc" in report["benchmarks"]
    assert report["benchmarks"]["01_seed_gc"]["speedup_passed"] is True
