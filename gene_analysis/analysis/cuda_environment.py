"""CUDA environment inspection for optional GPU pipeline backends."""

from __future__ import annotations

import importlib.util
import ctypes
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PackageStatus:
    """Availability and version metadata for one Python package."""

    name: str
    available: bool
    version: str | None = None


@dataclass(frozen=True)
class CudaDevice:
    """CUDA device metadata when available from CuPy."""

    index: int
    name: str
    compute_capability: str | None
    memory_gb: float | None


@dataclass(frozen=True)
class CudaEnvironmentReport:
    """Hardware, package, and compatibility metadata for CUDA readiness."""

    python: str
    platform: str
    nvidia_smi_available: bool
    nvidia_smi_summary: str | None
    packages: list[PackageStatus]
    cupy_gpu_check_ok: bool
    cupy_error: str | None
    cugraph_error: str | None
    devices: list[CudaDevice]
    cupy_gc_ready: bool
    cugraph_consensus_ready: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable report data."""
        return asdict(self)


def collect_cuda_environment_report() -> CudaEnvironmentReport:
    """Inspect local CUDA readiness without requiring optional GPU packages."""
    packages = [_package_status(name) for name in ("cupy", "cudf", "cugraph", "numba")]
    nvidia_smi_available, nvidia_smi_summary = _nvidia_smi_summary()
    devices: list[CudaDevice] = []
    cupy_error = None
    cugraph_error = None
    cupy_gpu_check_ok = False

    try:
        preload_cuda_wheel_libraries()
        import cupy as cp  # type: ignore

        count = cp.cuda.runtime.getDeviceCount()
        for index in range(count):
            props = cp.cuda.runtime.getDeviceProperties(index)
            name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
            cc = f"{props.get('major')}.{props.get('minor')}"
            memory_gb = float(props["totalGlobalMem"]) / (1024**3)
            devices.append(CudaDevice(index=index, name=name, compute_capability=cc, memory_gb=memory_gb))
        if count > 0:
            # Force a tiny kernel-backed operation. Device discovery alone can
            # succeed even when NVRTC is not loadable, which would fail later
            # during real CuPy calculations.
            cp.asarray([1.0, 2.0]).std().get()
        cupy_gpu_check_ok = count > 0
    except Exception as exc:
        cupy_error = repr(exc)

    package_map = {pkg.name: pkg for pkg in packages}
    max_cc = max((_compute_capability_float(device.compute_capability) for device in devices), default=0.0)
    python_tuple = tuple(int(part) for part in platform.python_version_tuple()[:2])
    cupy_gc_ready = package_map["cupy"].available and cupy_gpu_check_ok and max_cc >= 3.0
    cugraph_smoke_ok = False
    if (
        package_map["cugraph"].available
        and package_map["cudf"].available
        and cupy_gpu_check_ok
        and max_cc >= 7.0
        and python_tuple >= (3, 11)
    ):
        from gene_analysis.analysis.consensus_gpu import cugraph_louvain_smoke_test

        cugraph_smoke_ok, cugraph_error = cugraph_louvain_smoke_test()
    cugraph_consensus_ready = bool(cugraph_smoke_ok)

    notes = []
    if not nvidia_smi_available:
        notes.append("nvidia-smi is not available; on HPC this may require an allocated GPU node.")
    if not package_map["cupy"].available:
        notes.append("CuPy is missing; GPU GC experiments cannot run yet.")
    if not package_map["cugraph"].available:
        notes.append("cuGraph is missing; GPU consensus cannot run yet.")
    if package_map["cugraph"].available and cugraph_error:
        notes.append(f"cuGraph smoke test failed: {cugraph_error}")
    if python_tuple < (3, 11):
        notes.append("Current RAPIDS/cuGraph releases generally require Python 3.11+.")
    if devices and max_cc < 7.0:
        notes.append("Detected GPU compute capability is below the cuGraph/RAPIDS 7.0+ target.")
    if devices and max((device.memory_gb or 0.0 for device in devices), default=0.0) < 12.0:
        notes.append("GPU memory is modest; use this machine for development or small benchmark runs.")

    return CudaEnvironmentReport(
        python=platform.python_version(),
        platform=platform.platform(),
        nvidia_smi_available=nvidia_smi_available,
        nvidia_smi_summary=nvidia_smi_summary,
        packages=packages,
        cupy_gpu_check_ok=cupy_gpu_check_ok,
        cupy_error=cupy_error,
        cugraph_error=cugraph_error,
        devices=devices,
        cupy_gc_ready=cupy_gc_ready,
        cugraph_consensus_ready=cugraph_consensus_ready,
        notes=notes,
    )


def write_cuda_environment_report(output_file: str | Path) -> Path:
    """Write the local CUDA readiness report as JSON."""
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = collect_cuda_environment_report()
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    return output


def preload_cuda_wheel_libraries() -> None:
    """
    Preload CUDA shared libraries shipped by optional pip packages.

    CuPy's CUDA 12 wheel can be paired with NVIDIA's pip-packaged NVRTC runtime.
    On some pyenv/venv installs that library exists in site-packages but is not
    on the dynamic loader path, so preloading it by absolute path makes CuPy's
    later ``dlopen("libnvrtc.so.12")`` resolution work.
    """
    for module_name, patterns in {
        "nvidia.nvjitlink": ("lib/libnvJitLink.so.12", "lib/libnvJitLink.so"),
        "nvidia.cublas": ("lib/libcublas.so.12", "lib/libcublasLt.so.12", "lib/libcublas.so"),
        "nvidia.cusparse": ("lib/libcusparse.so.12", "lib/libcusparse.so"),
        "nvidia.cusolver": ("lib/libcusolver.so.11", "lib/libcusolver.so"),
        "nvidia.cuda_nvrtc": ("lib/libnvrtc.so.12", "lib/libnvrtc.so"),
    }.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        if spec is None or spec.origin is None:
            continue
        root = Path(spec.origin).resolve().parent
        for pattern in patterns:
            candidate = root / pattern
            if candidate.exists():
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass


def _package_status(name: str) -> PackageStatus:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return PackageStatus(name=name, available=False)
    version = None
    try:
        module = __import__(name)
        version = getattr(module, "__version__", None)
    except Exception:
        version = None
    return PackageStatus(name=name, available=True, version=version)


def _nvidia_smi_summary() -> tuple[bool, str | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return False, None
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip() or None
    return True, result.stdout.strip() or None


def _compute_capability_float(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0
