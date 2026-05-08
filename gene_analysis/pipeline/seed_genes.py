"""Load and validate curated seed genes for guided expansion runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from gene_analysis.io.paths import resolve_existing_path


def normalize_gene(gene: str) -> str:
    """Normalize a gene-list line before validation and de-duplication."""
    return gene.strip()


def load_seed_genes(seed_gene_file: str | Path) -> List[str]:
    """Load curated seed genes from a one-gene-per-line text file."""
    path = resolve_existing_path(seed_gene_file)
    genes: list[str] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            gene = normalize_gene(line)
            if not gene or gene.startswith("#"):
                continue
            if gene not in seen:
                genes.append(gene)
                seen.add(gene)
    if not genes:
        raise ValueError(f"No seed genes found in {path}.")
    return genes


def validate_seed_genes(seed_genes: Sequence[str], available_genes: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Return seed genes present in the dataset plus seed genes missing from it."""
    available = set(map(str, available_genes))
    present = [gene for gene in seed_genes if gene in available]
    missing = [gene for gene in seed_genes if gene not in available]
    return present, missing
