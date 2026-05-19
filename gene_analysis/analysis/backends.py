"""Compute backend selection and metadata for CPU/CUDA execution paths."""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import asdict, dataclass
from typing import Literal


GcBackend = Literal["cpu_statsmodels", "gpu_cuda"]
ConsensusBackend = Literal["cpu_louvain", "gpu_cugraph"]


class BackendUnavailableError(RuntimeError):
    """Raised when an optional compute backend is requested but unavailable."""


@dataclass(frozen=True)
class BackendMetadata:
    """Metadata recorded in manifests so runs document their compute backend."""

    kind: str
    backend: str
    available: bool
    device: int | None = None
    package: str | None = None
    package_available: bool | None = None
    python: str = platform.python_version()
    platform: str = platform.platform()
    note: str | None = None

    def to_dict(self) -> dict[str, str | int | bool | None]:
        """Return JSON-serializable metadata."""
        return asdict(self)


def validate_backend_settings(gc_backend: str, consensus_backend: str) -> None:
    """Validate configured backend names before the runner starts."""
    if gc_backend not in {"cpu_statsmodels", "gpu_cuda"}:
        raise ValueError("execution.gc_backend must be one of: cpu_statsmodels, gpu_cuda.")
    if consensus_backend not in {"cpu_louvain", "gpu_cugraph"}:
        raise ValueError("execution.consensus_backend must be one of: cpu_louvain, gpu_cugraph.")


def backend_metadata(kind: str, backend: str, *, device: int | None = None) -> BackendMetadata:
    """Return metadata for a configured backend without importing heavy optional packages."""
    if backend == "cpu_statsmodels":
        return BackendMetadata(kind=kind, backend=backend, available=True, package="statsmodels", package_available=True)
    if backend == "cpu_louvain":
        return BackendMetadata(kind=kind, backend=backend, available=True, package="python-louvain", package_available=True)
    if backend == "gpu_cuda":
        cupy_available = importlib.util.find_spec("cupy") is not None
        cupy_ready = _cupy_device_available(device) if cupy_available else False
        return BackendMetadata(
            kind=kind,
            backend=backend,
            available=cupy_ready,
            device=device,
            package="cupy",
            package_available=cupy_available,
            note="Experimental CuPy lag-1 Granger backend; compare against CPU before scientific production use.",
        )
    if backend == "gpu_cugraph":
        cugraph_available = importlib.util.find_spec("cugraph") is not None
        cudf_available = importlib.util.find_spec("cudf") is not None
        cupy_available = importlib.util.find_spec("cupy") is not None
        cugraph_ready = _cugraph_device_available(device) if cugraph_available and cudf_available and cupy_available else False
        return BackendMetadata(
            kind=kind,
            backend=backend,
            available=cugraph_ready,
            device=device,
            package="cugraph",
            package_available=cugraph_available,
            note=(
                "Experimental RAPIDS cuGraph Louvain + CuPy coassociation backend; "
                "requires CPU/GPU parity validation before production use."
            ),
        )
    raise ValueError(f"Unknown backend: {backend}")


def require_available_backend(kind: str, backend: str, *, device: int | None = None) -> None:
    """Raise a clear error when an unavailable backend is requested."""
    metadata = backend_metadata(kind, backend, device=device)
    if metadata.available:
        return
    raise BackendUnavailableError(
        f"{kind} backend '{backend}' is unavailable in this environment. "
        f"Install/enable {metadata.package!r}, check the requested GPU device, "
        "or use the CPU backend for validated runs."
    )


def _cupy_device_available(device: int | None = None) -> bool:
    """Return whether CuPy can see at least one usable CUDA device."""
    try:
        from gene_analysis.analysis.cuda_environment import preload_cuda_wheel_libraries

        preload_cuda_wheel_libraries()
        import cupy as cp  # type: ignore

        count = cp.cuda.runtime.getDeviceCount()
        if count < 1:
            return False
        if device is not None and int(device) >= count:
            return False
        if device is not None:
            cp.cuda.Device(int(device)).use()
        cp.asarray([1.0, 2.0]).std().get()
        return True
    except Exception:
        return False


def _cugraph_device_available(device: int | None = None) -> bool:
    """Return whether RAPIDS/cuGraph can run a tiny Louvain smoke test."""
    try:
        from gene_analysis.analysis.consensus_gpu import cugraph_louvain_smoke_test

        ok, _error = cugraph_louvain_smoke_test(gpu_device=device)
        return ok
    except Exception:
        return False
