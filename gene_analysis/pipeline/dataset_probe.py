"""Design bidirectional probe pairs for the guided full-dataset search."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from gene_analysis.io.paths import resolve_existing_path
from gene_analysis.pipeline.config import ProbeConfig
from gene_analysis.pipeline.seed_genes import load_seed_genes


ProbePair = Tuple[str, str]


def generate_probe_pairs(seed_genes: Sequence[str], dataset_genes: Iterable[str]) -> List[ProbePair]:
    """
    Generate ordered probe pairs between curated seed genes and the full dataset.

    This is the guided-expansion step: test seed -> dataset and dataset -> seed,
    excluding self-pairs and duplicate ordered pairs.
    """
    dataset = list(dict.fromkeys(map(str, dataset_genes)))
    seeds = list(dict.fromkeys(map(str, seed_genes)))
    pairs = [(seed, gene) for seed in seeds for gene in dataset if gene != seed]
    pairs += [(gene, seed) for gene in dataset for seed in seeds if gene != seed]
    return list(dict.fromkeys(pairs))


def write_probe_pairs(pairs: Sequence[ProbePair], output_file: str | Path) -> Path:
    """Write ordered probe pairs as ``gene1,gene2`` CSV rows."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gene1", "gene2"])
        writer.writerows(pairs)
    return output_path


def run(config: ProbeConfig, dataset_genes: Iterable[str] | None = None) -> Path:
    """
    Write the probe-pair design for the guided full-dataset search.

    Heavy GC execution remains delegated to the existing optimized Granger runner.
    """
    config.validate()
    seeds = load_seed_genes(config.seed_gene_file)
    if dataset_genes is None:
        if config.full_gene_file is None:
            raise ValueError("dataset_genes or full_gene_file is required to generate probe pairs.")
        dataset_genes = load_seed_genes(resolve_existing_path(config.full_gene_file))
    pairs = generate_probe_pairs(seeds, dataset_genes)
    return write_probe_pairs(pairs, config.probe_pairs_file)
