import pytest

from gene_analysis_common.network import create_network


def test_create_network_handles_single_dataset_edges():
    graph = create_network([(("A", 1), ("B", 0), 0.01)])

    assert list(graph.edges(data=True)) == [("A", "B", {"lag": 1, "p_value": 0.01})]


def test_create_network_handles_intersection_edges():
    graph = create_network([(("A", 1), ("B", 0), 0.02, 0.04)])

    assert graph["A"]["B"]["lag"] == 1
    assert graph["A"]["B"]["kutsche_p_value"] == 0.02
    assert graph["A"]["B"]["benito_p_value"] == 0.04
    assert graph["A"]["B"]["p_value"] == 0.03


def test_create_network_rejects_invalid_edges():
    with pytest.raises(ValueError):
        create_network([("A", "B")])

