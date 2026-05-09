"""Network construction helpers shared by dashboards, scripts, and pipeline stages."""

from __future__ import annotations

import networkx as nx


def create_network(significant_edges):
    """Create a directed gene network from significant edge tuples."""
    G = nx.DiGraph()
    for edge in significant_edges:
        if len(edge) == 3:
            (source, lag), (target, _), p_value = edge
            G.add_edge(source, target, lag=lag, p_value=p_value)
        elif len(edge) == 4:
            (source, lag), (target, _), kutsche_p_value, benito_p_value = edge
            avg_p_value = (kutsche_p_value + benito_p_value) / 2
            G.add_edge(
                source,
                target,
                lag=lag,
                kutsche_p_value=kutsche_p_value,
                benito_p_value=benito_p_value,
                p_value=avg_p_value,
            )
        else:
            raise ValueError(f"Invalid edge format: {edge}")
    return G
