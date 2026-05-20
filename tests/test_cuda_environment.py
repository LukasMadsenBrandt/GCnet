import json
from pathlib import Path

import pytest
import yaml

from gene_analysis.analysis.cuda_environment import (
    _cupy_nvrtc_note,
    collect_cuda_environment_report,
    write_cuda_environment_report,
)


pytestmark = pytest.mark.unit


def test_collect_cuda_environment_report_is_safe_without_cuda_packages():
    report = collect_cuda_environment_report()

    assert report.python
    assert report.platform
    assert {package.name for package in report.packages} == {"cupy", "cudf", "cugraph", "numba"}
    assert isinstance(report.cupy_gc_ready, bool)
    assert isinstance(report.cugraph_consensus_ready, bool)
    assert report.cugraph_error is None or isinstance(report.cugraph_error, str)
    assert isinstance(report.notes, list)


def test_write_cuda_environment_report_writes_json(tmp_path):
    output = write_cuda_environment_report(tmp_path / "cuda_report.json")

    data = json.loads(output.read_text(encoding="utf-8"))
    assert "python" in data
    assert "packages" in data
    assert "cupy_gc_ready" in data
    assert "cugraph_consensus_ready" in data
    assert "cugraph_error" in data


def test_cupy_nvrtc_loader_error_gets_actionable_note():
    note = _cupy_nvrtc_note(
        "RuntimeError('CuPy failed to load libnvrtc.so.12: "
        "OSError: libnvrtc.so.12: cannot open shared object file')"
    )

    assert note is not None
    assert "nvidia-cuda-nvrtc-cu12" in note
    assert "LD_LIBRARY_PATH" in note


def test_cuda_environment_files_are_parseable_and_separate_from_cpu_requirements():
    root = Path(__file__).resolve().parents[1]
    cpu_requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    cupy_requirements = (root / "requirements-cuda-cupy.txt").read_text(encoding="utf-8")

    assert "cupy" not in cpu_requirements
    assert "cugraph" not in cpu_requirements
    assert "cupy-cuda12x" in cupy_requirements

    cupy_env = yaml.safe_load((root / "envs" / "cuda-cupy.yml").read_text(encoding="utf-8"))
    rapids_env = yaml.safe_load((root / "envs" / "cuda-rapids.yml").read_text(encoding="utf-8"))

    assert cupy_env["name"] == "gene-cuda-cupy"
    assert "conda-forge" in cupy_env["channels"]
    assert any(dep == "python=3.11" for dep in cupy_env["dependencies"])
    assert "pip" in cupy_env["dependencies"]
    assert any(dep == "python=3.11" for dep in rapids_env["dependencies"])
    assert any(dep == "cuda-version=12.3" for dep in rapids_env["dependencies"])
    assert {"cudf", "cugraph", "cupy"}.issubset(set(rapids_env["dependencies"]))
