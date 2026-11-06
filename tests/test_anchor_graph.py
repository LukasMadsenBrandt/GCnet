"""Compare streamed multi-anchor counts with independently built graphs."""
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from gene_analysis.analysis.anchor_graph import ThresholdGraph


@pytest.mark.parametrize("batch", [1, 2, 3, 20])
def test_ties_reciprocal_edges_isolates_and_outside_seed_counts(batch):
    genes = ["A", "B", "C", "D", "E", "F"]
    a = np.array([0, 1, 2, 3, 1, 4, 0], dtype=np.int32)
    b = np.array([1, 0, 3, 4, 2, 2, 4], dtype=np.int32)
    q = np.array([.001, .001, .001, .01, .02, .02, .03])
    pi0 = .5
    report = ThresholdGraph(genes, ["A", "C", "F"], pi0, [0, .0005, .001, .005, .01, .02, 1], [0, 1])
    for begin in range(0, len(q), batch):
        report.consume(a[begin:begin+batch], b[begin:begin+batch], q[begin:begin+batch])
    for row in report.finish():
        selected = q * (1 if row["method"] == "bh" else pi0) <= row["cutoff"]
        x, y = a[selected], b[selected]
        graph = coo_matrix((np.ones(len(x)), (x, y)), shape=(6, 6)).tocsr()
        _, labels = connected_components(graph, directed=False)
        node = genes.index(row["anchor"])
        component = set(np.flatnonzero(labels == labels[node]))
        outgoing, incoming = set(y[x == node]), set(x[y == node])
        neighbors = outgoing | incoming
        assert row["directed_connections"] == selected.sum()
        assert row["undirected_pairs"] == len(set(zip(np.minimum(x,y),np.maximum(x,y))))
        assert row["largest_component_nodes"] == np.bincount(labels).max()
        assert row["anchor_component_nodes"] == len(component)
        assert row["outgoing_neighbors"] == len(outgoing)
        assert row["incoming_neighbors"] == len(incoming)
        assert row["distinct_neighbors"] == len(neighbors)
        assert row["reciprocal_neighbors"] == len(outgoing & incoming)
        assert row["neighbors_outside_seed_list"] == len(neighbors - {0, 1})
        assert row["connected_genes_outside_seed_list"] == len((set(x) | set(y)) - {0,1})
        assert row["anchor_component_genes_outside_seed_list"] == len(component - {0,1})


def test_invalid_anchor_and_unsorted_stream_rejected():
    with pytest.raises(ValueError, match="anchors"):
        ThresholdGraph(["A", "B"], ["missing"], .5, [.01])
    graph = ThresholdGraph(["A", "B"], ["A"], .5, [.01])
    graph.consume([0], [1], [.02])
    with pytest.raises(ValueError, match="nondecreasing"):
        graph.consume([1], [0], [.01])
