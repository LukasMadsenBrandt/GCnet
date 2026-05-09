"""Extract expanded candidate genes from significant probe-network edges."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set

from gene_analysis.analysis.granger import collect_significant_edges
from gene_analysis.analysis.network import create_network
from gene_analysis.io.paths import resolve_existing_path
from gene_analysis.pipeline.config import ExpansionConfig
from gene_analysis.pipeline.seed_genes import load_seed_genes


def extract_expanded_genes_from_csv(
    candidate_network_csv: str | Path,
    *,
    p_threshold: float,
    gene_of_interest: str,
) -> List[str]:
    """Return all genes in the significant probe network anchored on the GOI."""
    edges = collect_significant_edges(
        None,
        p_value_threshold=p_threshold,
        file=True,
        filepath=resolve_existing_path(candidate_network_csv),
        starting_genes=[gene_of_interest],
        higher_threshold_for_starting_genes=p_threshold,
    )
    graph = create_network(edges)
    return sorted(map(str, graph.nodes()))


def write_gene_list(genes: Iterable[str], output_file: str | Path) -> Path:
    """Write a sorted, de-duplicated one-gene-per-line list."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for gene in sorted(dict.fromkeys(map(str, genes))):
            fh.write(f"{gene}\n")
    return output_path


def run(config: ExpansionConfig) -> Path:
    """Run network expansion and optionally retain the original seed genes."""
    config.validate()
    expanded: Set[str] = set(
        extract_expanded_genes_from_csv(
            config.candidate_network_csv,
            p_threshold=config.p_threshold,
            gene_of_interest=config.gene_of_interest,
        )
    )
    if config.seed_gene_file:
        expanded.update(load_seed_genes(config.seed_gene_file))
    return write_gene_list(expanded, config.output_gene_list)
