import csv
import json

import numpy as np
import pandas as pd
import pytest

from gene_analysis.analysis.cuda_environment import collect_cuda_environment_report
from gene_analysis.analysis.parity import build_benchmark_report, compare_gc_csvs
from gene_analysis.analysis.granger_runner import perform_gc
from gene_analysis.analysis.granger_cuda import perform_gc_cuda


pytestmark = pytest.mark.unit


def test_perform_gc_full_mode_writes_header_and_checkpoint_for_constant_data(tmp_path):
    genes_file = tmp_path / "genes.txt"
    output_file = tmp_path / "gc.csv"
    genes_file.write_text("A\nB\nC\n", encoding="utf-8")
    expression = pd.DataFrame(
        {
            "T1": [1, 2, 3],
            "T2": [1, 2, 3],
            "T3": [1, 2, 3],
            "T4": [1, 2, 3],
            "T5": [1, 2, 3],
        },
        index=["A", "B", "C"],
    )

    result = perform_gc(
        expression,
        genes_file=str(genes_file),
        output_file=str(output_file),
        p_threshold=0.01,
        chunk_size=2,
        max_workers=1,
        progress=False,
        resume=False,
        rename_at_end=False,
    )

    assert result["total_pairs_all"] == 3 * 2
    assert result["processed_this_run"] == 6
    assert result["significant_edges"] == 0
    assert result["elapsed_seconds"] >= 0
    assert output_file.exists()
    assert (tmp_path / "gc.seen.bin").exists()
    with output_file.open(newline="", encoding="utf-8") as fh:
        assert list(csv.reader(fh)) == [["gene1", "gene2", "lag", "p-value"]]


def test_perform_gc_resume_skips_seen_pairs(tmp_path):
    genes_file = tmp_path / "genes.txt"
    output_file = tmp_path / "gc.csv"
    genes_file.write_text("A\nB\n", encoding="utf-8")
    expression = pd.DataFrame(
        {
            "T1": [1, 2],
            "T2": [1, 2],
            "T3": [1, 2],
            "T4": [1, 2],
            "T5": [1, 2],
        },
        index=["A", "B"],
    )

    first = perform_gc(
        expression,
        genes_file=str(genes_file),
        output_file=str(output_file),
        p_threshold=0.01,
        chunk_size=1,
        max_workers=1,
        progress=False,
        resume=False,
        rename_at_end=False,
    )
    second = perform_gc(
        expression,
        genes_file=str(genes_file),
        output_file=str(output_file),
        p_threshold=0.01,
        chunk_size=1,
        max_workers=1,
        progress=False,
        resume=True,
        rename_at_end=False,
    )

    assert first["processed_this_run"] == 2
    assert second["total_pairs_all"] == 2
    assert second["processed_this_run"] == 0


def test_perform_gc_probe_mode_counts_bidirectional_probe_pairs(tmp_path):
    genes_file = tmp_path / "probe_genes.txt"
    output_file = tmp_path / "probe_gc.csv"
    genes_file.write_text("A\nB\n", encoding="utf-8")
    expression = pd.DataFrame(
        {
            "T1": [1, 2, 3, 4],
            "T2": [1, 2, 3, 4],
            "T3": [1, 2, 3, 4],
            "T4": [1, 2, 3, 4],
            "T5": [1, 2, 3, 4],
        },
        index=["A", "B", "C", "D"],
    )

    result = perform_gc(
        expression,
        genes_file=str(genes_file),
        output_file=str(output_file),
        p_threshold=0.01,
        chunk_size=3,
        list_to_kutsche=True,
        max_workers=1,
        progress=False,
        resume=False,
        rename_at_end=False,
    )

    # g=2 probe genes, N=4 dataset genes: 2*g*N - g*g - g
    assert result["total_pairs_all"] == 10
    assert result["processed_this_run"] == 10


@pytest.mark.cuda
def test_perform_gc_cuda_matches_cpu_pair_count_and_schema(tmp_path):
    if not collect_cuda_environment_report().cupy_gc_ready:
        pytest.skip("CuPy CUDA environment is not available.")

    genes_file = tmp_path / "genes.txt"
    cpu_output = tmp_path / "cpu_gc.csv"
    gpu_output = tmp_path / "gpu_gc.csv"
    genes_file.write_text("A\nB\nC\n", encoding="utf-8")
    expression = pd.DataFrame(
        {
            "T1": [1.0, 1.0, 0.5],
            "T2": [2.0, 1.2, 1.0],
            "T3": [3.0, 1.8, 1.4],
            "T4": [4.0, 2.8, 1.9],
            "T5": [5.0, 4.0, 2.5],
            "T6": [6.0, 5.5, 3.2],
        },
        index=["A", "B", "C"],
    )

    cpu = perform_gc(
        expression,
        genes_file=str(genes_file),
        output_file=str(cpu_output),
        p_threshold=1.0,
        chunk_size=2,
        max_workers=1,
        progress=False,
        resume=False,
        rename_at_end=False,
    )
    gpu = perform_gc_cuda(
        expression,
        genes_file=str(genes_file),
        output_file=str(gpu_output),
        p_threshold=1.0,
        chunk_size=2,
        progress=False,
        resume=False,
        rename_at_end=False,
        gpu_device=0,
    )

    assert gpu["total_pairs_all"] == cpu["total_pairs_all"] == 6
    assert gpu["processed_this_run"] == cpu["processed_this_run"] == 6
    assert gpu["elapsed_seconds"] >= 0
    with gpu_output.open(newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == ["gene1", "gene2", "lag", "p-value"]


@pytest.mark.cuda
@pytest.mark.cuda_parity
@pytest.mark.cuda_benchmark
@pytest.mark.slow
def test_perform_gc_cuda_benchmark_preserves_cpu_results(tmp_path):
    if not collect_cuda_environment_report().cupy_gc_ready:
        pytest.skip("CuPy CUDA environment is not available.")

    gene_count = 25
    timepoints = 24
    rng = np.random.default_rng(42)
    values = rng.normal(size=(gene_count, timepoints)).cumsum(axis=1)
    genes = [f"G{i:03d}" for i in range(gene_count)]
    expression = pd.DataFrame(values, index=genes, columns=[f"T{i:02d}" for i in range(timepoints)])
    genes_file = tmp_path / "genes.txt"
    cpu_output = tmp_path / "cpu_gc.csv"
    gpu_output = tmp_path / "gpu_gc.csv"
    genes_file.write_text("\n".join(genes) + "\n", encoding="utf-8")

    cpu = perform_gc(
        expression,
        genes_file=str(genes_file),
        output_file=str(cpu_output),
        p_threshold=1.0,
        chunk_size=100,
        max_workers=1,
        progress=False,
        resume=False,
        rename_at_end=False,
    )
    gpu = perform_gc_cuda(
        expression,
        genes_file=str(genes_file),
        output_file=str(gpu_output),
        p_threshold=1.0,
        chunk_size=100,
        progress=False,
        resume=False,
        rename_at_end=False,
        gpu_device=0,
    )

    parity = compare_gc_csvs(cpu_output, gpu_output, p_value_tolerance=1e-5, significance_threshold=0.05)
    benchmark = build_benchmark_report(
        stage="full_gc_benchmark",
        cpu_backend="cpu_statsmodels",
        candidate_backend="gpu_cuda",
        cpu_seconds=cpu["elapsed_seconds"],
        candidate_seconds=gpu["elapsed_seconds"],
        work_units=cpu["total_pairs_all"],
    )
    report = {
        "parity": parity.__dict__,
        "benchmark": benchmark.__dict__,
    }
    (tmp_path / "cuda_gc_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    assert parity.reference_rows == gene_count * (gene_count - 1)
    assert parity.candidate_rows == parity.reference_rows
    assert parity.missing_in_candidate == 0
    assert parity.extra_in_candidate == 0
    assert parity.p_value_disagreements == 0
    assert parity.significant_decision_disagreements == 0
    assert benchmark.work_units_per_second_candidate is not None
