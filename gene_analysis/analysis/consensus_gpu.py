"""Optional RAPIDS/CuPy consensus helpers for GPU-accelerated clustering."""

from __future__ import annotations

import math
import time
from typing import Any

import networkx as nx
import numpy as np

from gene_analysis.analysis.backends import BackendUnavailableError
from gene_analysis.analysis.cuda_environment import preload_cuda_wheel_libraries


GPU_MEMORY_SAFETY_FRACTION = 0.75
LOUVAIN_SEED_CONTROL = "uncontrolled"


def graph_to_cugraph(G_undirected: nx.Graph):
    """Convert a NetworkX graph to a cuGraph graph using stable integer node IDs."""
    _require_rapids()
    import cudf  # type: ignore
    import cugraph  # type: ignore

    nodes = list(G_undirected.nodes())
    node_to_id = {node: idx for idx, node in enumerate(nodes)}
    edges = [(node_to_id[u], node_to_id[v]) for u, v in G_undirected.edges()]
    if not edges:
        raise ValueError("Cannot run GPU Louvain on a graph with no edges.")
    edge_df = cudf.DataFrame(edges, columns=["source", "destination"])
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_edgelist(edge_df, source="source", destination="destination", renumber=False)
    return graph, nodes


def run_multiple_louvain_cugraph(
    G_undirected: nx.Graph,
    n_runs: int = 100,
    existing_partitions: list[dict] | None = None,
    gpu_device: int | None = None,
) -> list[dict]:
    """Run repeated cuGraph Louvain and return partitions in the CPU backend shape."""
    _require_rapids()
    import cupy as cp  # type: ignore
    import cugraph  # type: ignore

    if gpu_device is not None:
        cp.cuda.Device(int(gpu_device)).use()
    if existing_partitions is None:
        existing_partitions = []
    partitions = list(existing_partitions)
    n_new = int(n_runs) - len(partitions)
    if n_new <= 0:
        return partitions

    graph, nodes = graph_to_cugraph(G_undirected)
    for _seed in range(len(existing_partitions), int(n_runs)):
        partition_df, _modularity = cugraph.louvain(graph)
        pdf = partition_df.to_pandas()
        vertex_col = _first_existing_column(pdf, ("vertex", "vertices"))
        partition_col = _first_existing_column(pdf, ("partition", "labels", "cluster"))
        partitions.append(
            {
                nodes[int(row[vertex_col])]: int(row[partition_col])
                for _, row in pdf.iterrows()
            }
        )
    return partitions


def build_coassociation_matrix_cupy(
    G: nx.Graph,
    partitions: list[dict],
    prev_state: tuple[np.ndarray, int] | None = None,
    gpu_device: int | None = None,
):
    """Build/update the coassociation matrix on GPU and return CPU-compatible outputs."""
    cp = _require_cupy(gpu_device=gpu_device)
    nodes = list(G.nodes())
    n = len(nodes)
    validate_gpu_coassociation_memory(n, cp)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    if prev_state is None:
        coassoc_counts_gpu = cp.zeros((n, n), dtype=cp.int32)
        n_prev = 0
    else:
        coassoc_counts, n_prev = prev_state
        coassoc_counts_gpu = cp.asarray(coassoc_counts, dtype=cp.int32)

    new_partitions = partitions[n_prev:]
    if not new_partitions:
        if n_prev == 0:
            coassoc = np.zeros((n, n), dtype=float)
        else:
            coassoc = cp.asnumpy(coassoc_counts_gpu.astype(cp.float64) / float(n_prev))
        return nodes, coassoc, (cp.asnumpy(coassoc_counts_gpu), n_prev)

    for partition in new_partitions:
        labels_np = np.full(n, -1, dtype=np.int32)
        for node, community in partition.items():
            idx = node_index.get(node)
            if idx is not None:
                labels_np[idx] = int(community)
        labels = cp.asarray(labels_np, dtype=cp.int32)
        valid = labels != -1
        same_community = (labels[:, None] == labels[None, :]) & valid[:, None] & valid[None, :]
        coassoc_counts_gpu += same_community.astype(cp.int32)
        n_prev += 1

    coassoc = cp.asnumpy(coassoc_counts_gpu.astype(cp.float64) / float(n_prev))
    return nodes, coassoc, (cp.asnumpy(coassoc_counts_gpu), n_prev)


def estimate_coassociation_bytes(node_count: int) -> int:
    """Estimate the minimum GPU bytes needed for an int32 coassociation matrix."""
    return int(node_count) * int(node_count) * 4


def validate_gpu_coassociation_memory(
    node_count: int,
    cp: Any,
    *,
    safety_fraction: float = GPU_MEMORY_SAFETY_FRACTION,
) -> None:
    """Raise if the coassociation matrix is too large for available GPU memory."""
    required = estimate_coassociation_bytes(node_count)
    free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    allowed = int(float(free_bytes) * float(safety_fraction))
    if required > allowed:
        required_gb = required / (1024**3)
        allowed_gb = allowed / (1024**3)
        raise MemoryError(
            "GPU coassociation matrix would exceed the configured memory safety limit: "
            f"requires at least {required_gb:.2f} GiB, safety limit is {allowed_gb:.2f} GiB. "
            "Reduce the network size or use consensus_backend: cpu_louvain."
        )


def cugraph_louvain_smoke_test(gpu_device: int | None = None) -> tuple[bool, str | None]:
    """Run a tiny cuGraph Louvain smoke test and return readiness plus error text."""
    try:
        _require_rapids()
        import cupy as cp  # type: ignore
        import cugraph  # type: ignore
        import cudf  # type: ignore

        if gpu_device is not None:
            cp.cuda.Device(int(gpu_device)).use()
        edge_df = cudf.DataFrame({"source": [0, 1], "destination": [1, 2]})
        graph = cugraph.Graph(directed=False)
        graph.from_cudf_edgelist(edge_df, source="source", destination="destination", renumber=False)
        result, _modularity = cugraph.louvain(graph)
        pdf = result.to_pandas()
        if len(pdf) < 2:
            return False, "cuGraph Louvain smoke test returned too few vertices."
        return True, None
    except Exception as exc:
        return False, repr(exc)


def _require_cupy(*, gpu_device: int | None = None):
    preload_cuda_wheel_libraries()
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:
        raise BackendUnavailableError("CuPy is required for GPU coassociation.") from exc
    if gpu_device is not None:
        cp.cuda.Device(int(gpu_device)).use()
    cp.asarray([1.0]).get()
    return cp


def _require_rapids() -> None:
    preload_cuda_wheel_libraries()
    try:
        import cupy  # noqa: F401
        import cudf  # noqa: F401
        import cugraph  # noqa: F401
    except Exception as exc:
        raise BackendUnavailableError(
            "gpu_cugraph requires RAPIDS packages cudf, cugraph, and cupy in a CUDA-capable environment."
        ) from exc


def _first_existing_column(df, names: tuple[str, ...]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"cuGraph Louvain output missing expected columns. Found: {list(df.columns)}")
