"""Shared helpers for comparing coassociation-frequency CSV stability."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import csv
import math


@dataclass(frozen=True)
class GeneStabilityConfig:
    """Column names and thresholds for frequency-CSV stability checks."""

    gene_col: str = "Gene"
    freq_col: str = "Coassociation Frequency"
    case_insensitive: bool = True

    rowset_mode: str = "inter"          # "inter" | "union"
    denominator: str = "previous"       # "previous" | "max" | "mean"
    eps: float = 1e-9

    quantile_p: float = 0.90            # pick one and keep consistent
    top_fraction: float = 0.05
    top_k: Optional[int] = None


def _norm_gene(g: str, case_insensitive: bool) -> str:
    return g.lower() if case_insensitive else g


def read_gene_freq_csv(path: str, *, gene_col: str, freq_col: str, case_insensitive: bool) -> Dict[str, float]:
    """Read gene frequencies from a CSV into a normalized mapping."""
    out: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"{path}: Missing CSV header.")

        headers = {h.lower(): h for h in r.fieldnames}
        gcol = headers.get(gene_col.lower(), gene_col)
        fcol = headers.get(freq_col.lower(), freq_col)

        if gcol not in r.fieldnames:
            raise ValueError(f"{path}: missing gene column '{gene_col}'. Have: {r.fieldnames}")
        if fcol not in r.fieldnames:
            raise ValueError(f"{path}: missing freq column '{freq_col}'. Have: {r.fieldnames}")

        for row in r:
            g = (row.get(gcol) or "").strip()
            v = (row.get(fcol) or "").strip()
            if not g:
                continue
            try:
                freq = float(v)
            except ValueError:
                continue
            out[_norm_gene(g, case_insensitive)] = freq
    return out


def _rel_diff(prev: float, curr: float, *, denominator: str, eps: float) -> float:
    num = abs(prev - curr)
    if denominator == "previous":
        den = max(prev, eps)
    elif denominator == "max":
        den = max(prev, curr, eps)
    elif denominator == "mean":
        den = max((prev + curr) / 2.0, eps)
    else:
        raise ValueError(f"Invalid denominator: {denominator}")
    return num / den


def _quantile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 1:
        return max(values)

    xs = sorted(values)
    pos = p * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _top_k_from_freqs(freqs: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    return sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:k]


def compute_csv_stability_metrics(
    previous_csv: str,
    current_csv: str,
    *,
    cfg: GeneStabilityConfig = GeneStabilityConfig(),
) -> Tuple[float, float, int]:
    """
    Returns:
      q_rel: quantile(cfg.quantile_p) of per-gene relative diffs
      overlap_pct: top-K overlap percent
      k: chosen top-K
    """
    prev = read_gene_freq_csv(previous_csv, gene_col=cfg.gene_col, freq_col=cfg.freq_col, case_insensitive=cfg.case_insensitive)
    curr = read_gene_freq_csv(current_csv,  gene_col=cfg.gene_col, freq_col=cfg.freq_col, case_insensitive=cfg.case_insensitive)

    prev_genes = set(prev)
    curr_genes = set(curr)
    union_genes = prev_genes | curr_genes
    inter_genes = prev_genes & curr_genes

    if cfg.rowset_mode.lower() == "union":
        genes = union_genes
    elif cfg.rowset_mode.lower() == "inter":
        genes = inter_genes
    else:
        raise ValueError("rowset_mode must be 'union' or 'inter'")

    rels: List[float] = []
    for g in genes:
        p = prev.get(g, 0.0)
        c = curr.get(g, 0.0)
        # stats typically computed on intersection only; but since you chose rowset_mode,
        # we’ll honor it.
        rels.append(_rel_diff(p, c, denominator=cfg.denominator, eps=cfg.eps))

    q_rel = _quantile(rels, cfg.quantile_p)

    # top-K overlap uses each file independently
    n_prev = len(prev)
    n_curr = len(curr)
    if cfg.top_k is not None:
        k = int(cfg.top_k)
    else:
        k = max(1, int(round(cfg.top_fraction * min(n_prev, n_curr))))

    top_prev = {g for g, _ in _top_k_from_freqs(prev, k)}
    top_curr = {g for g, _ in _top_k_from_freqs(curr, k)}
    overlap_pct = (len(top_prev & top_curr) / k) * 100.0 if k > 0 else float("nan")

    return q_rel, overlap_pct, k
