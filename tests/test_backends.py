import pytest

from gene_analysis.analysis.backends import (
    BackendUnavailableError,
    backend_metadata,
    require_available_backend,
    validate_backend_settings,
)
from gene_analysis.analysis.cuda_environment import collect_cuda_environment_report
from gene_analysis.pipeline.runner import PipelineConfig


pytestmark = pytest.mark.unit


def test_cpu_backend_metadata_is_available():
    gc = backend_metadata("gc", "cpu_statsmodels")
    consensus = backend_metadata("consensus", "cpu_louvain")

    assert gc.available is True
    assert gc.package == "statsmodels"
    assert consensus.available is True
    assert consensus.package == "python-louvain"


def test_cuda_gc_backend_availability_matches_environment():
    report = collect_cuda_environment_report()
    metadata = backend_metadata("gc", "gpu_cuda", device=0)

    assert metadata.available is report.cupy_gc_ready
    if not report.cupy_gc_ready:
        with pytest.raises(BackendUnavailableError, match="unavailable"):
            require_available_backend("gc", "gpu_cuda", device=0)


def test_cuda_consensus_backend_availability_matches_environment():
    report = collect_cuda_environment_report()
    metadata = backend_metadata("consensus", "gpu_cugraph", device=0)

    assert metadata.available is report.cugraph_consensus_ready
    if not report.cugraph_consensus_ready:
        with pytest.raises(BackendUnavailableError, match="unavailable"):
            require_available_backend("consensus", "gpu_cugraph", device=0)


def test_backend_validation_rejects_unknown_names():
    with pytest.raises(ValueError, match="execution.gc_backend"):
        validate_backend_settings("gpu_magic", "cpu_louvain")
    with pytest.raises(ValueError, match="execution.consensus_backend"):
        validate_backend_settings("cpu_statsmodels", "gpu_magic")


def test_pipeline_config_parses_backend_settings(tmp_path):
    config_file = tmp_path / "pipeline.yml"
    config_file.write_text(
        """
run_name: backend_demo
dataset:
  name: fixture
  expression_file: examples/sample_expression.csv
  full_gene_file: examples/kutsche_real_gc/all_genes.txt
gene_of_interest: ZEB2
seed_gene_file: examples/kutsche_real_gc/seed_genes.txt
execution:
  gc_backend: cpu_statsmodels
  consensus_backend: cpu_louvain
  gpu_device: 0
""",
        encoding="utf-8",
    )

    cfg = PipelineConfig.from_yaml(config_file)

    assert cfg.execution.gc_backend == "cpu_statsmodels"
    assert cfg.execution.consensus_backend == "cpu_louvain"
    assert cfg.execution.gpu_device == 0
