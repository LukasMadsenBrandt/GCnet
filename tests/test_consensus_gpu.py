import networkx as nx
import numpy as np
import pytest

from gene_analysis.analysis.consensus_gpu import (
    build_coassociation_matrix_cupy,
    cugraph_louvain_smoke_test,
    estimate_coassociation_bytes,
    graph_to_cugraph,
    validate_gpu_coassociation_memory,
)
from gene_analysis.analysis.cuda_environment import collect_cuda_environment_report
from gene_analysis.analysis.consensus_backend import coassoc_for_chunk


pytestmark = pytest.mark.unit


class _FakeRuntime:
    def __init__(self, free_bytes):
        self._free_bytes = free_bytes

    def memGetInfo(self):
        return self._free_bytes, self._free_bytes


class _FakeCuda:
    def __init__(self, free_bytes):
        self.runtime = _FakeRuntime(free_bytes)


class _FakeCupy:
    def __init__(self, free_bytes):
        self.cuda = _FakeCuda(free_bytes)


def test_estimate_coassociation_bytes_uses_int32_matrix_size():
    assert estimate_coassociation_bytes(10) == 10 * 10 * 4


def test_validate_gpu_coassociation_memory_rejects_oversized_matrix():
    with pytest.raises(MemoryError, match="coassociation matrix"):
        validate_gpu_coassociation_memory(10, _FakeCupy(free_bytes=100), safety_fraction=0.75)


@pytest.mark.cuda
def test_cupy_coassociation_matches_cpu_counts():
    if not collect_cuda_environment_report().cupy_gc_ready:
        pytest.skip("CuPy CUDA environment is not available.")

    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    partitions = [
        {"A": 0, "B": 0, "C": 1, "D": 1},
        {"A": 0, "B": 1, "C": 1, "D": 1},
    ]

    nodes, gpu_coassoc, state = build_coassociation_matrix_cupy(graph, partitions)
    cpu_counts, n_partitions = coassoc_for_chunk(partitions, nodes)
    cpu_coassoc = cpu_counts.astype(float) / float(n_partitions)

    assert state[1] == len(partitions)
    assert np.allclose(gpu_coassoc, cpu_coassoc)


@pytest.mark.cuda
def test_cugraph_graph_conversion_preserves_tiny_topology_when_available():
    report = collect_cuda_environment_report()
    if not report.cugraph_consensus_ready:
        pytest.skip("RAPIDS/cuGraph consensus environment is not available.")

    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("B", "C")])

    cugraph_graph, nodes = graph_to_cugraph(graph)

    assert nodes == ["A", "B", "C"]
    assert cugraph_graph.number_of_vertices() == 3
    assert cugraph_graph.number_of_edges() == 2
    ok, error = cugraph_louvain_smoke_test()
    assert ok is True
    assert error is None
