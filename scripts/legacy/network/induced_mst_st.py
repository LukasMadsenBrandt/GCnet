"""Network extraction, MST, Steiner, and GOI-path utilities for figure workflows."""

from itertools import chain
import os
import time
from matplotlib import pyplot as plt
from networkx.algorithms.shortest_paths.weighted import multi_source_dijkstra_path_length

from decimal import Decimal

import pandas as pd
import networkx as nx
from typing import List, Optional, Set, Tuple

from gene_analysis.analysis.consensus_backend import (
    consensus_partition
)
import math
from networkx.algorithms.tree import minimum_spanning_tree
from networkx.algorithms.approximation import steiner_tree

# --- add these imports near your other imports ---
import html as html_escape
import graphviz
import numpy as np
from typing import Iterable, Optional, Dict, Tuple

# --- logging setup ---
import logging, json, os, time
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from datetime import datetime

class DefaultExtrasFilter(logging.Filter):
    """Ensure every record has .run and .event so format strings never break."""
    def __init__(self, run_default="-", event_default="misc"):
        super().__init__()
        self.run_default = run_default
        self.event_default = event_default
    def filter(self, record):
        """Inject default contextual fields into a log record."""
        if not hasattr(record, "run"):
            record.run = self.run_default
        if not hasattr(record, "event"):
            record.event = self.event_default
        return True

class ConsoleEventFilter(logging.Filter):
    """Show only high-level events on console."""
    def __init__(self, allowed=("dataset","run_start","summary","run_end","warning","error")):
        super().__init__()
        self.allowed = set(allowed)
    def filter(self, record):
        """Return whether a log record should be shown on the console."""
        return getattr(record, "event", None) in self.allowed or record.levelno >= logging.ERROR

def get_logger(name="gene_net", level=logging.DEBUG, logs_dir="logs") -> logging.Logger:
    """Create or return the shared rotating-file logger."""
    lg = logging.getLogger(name)
    if lg.handlers:  # already configured
        return lg

    lg.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(run)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    # Console: only key events
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    ch.addFilter(DefaultExtrasFilter())   # <-- inject defaults
    ch.addFilter(ConsoleEventFilter())    # <-- limit noise
    lg.addHandler(ch)

    # Global rotating file: everything
    os.makedirs(logs_dir, exist_ok=True)
    fh = RotatingFileHandler(os.path.join(logs_dir, "gene_net.log"), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    fh.addFilter(DefaultExtrasFilter())   # <-- inject defaults
    lg.addHandler(fh)

    lg.propagate = False
    return lg

@contextmanager
def run_logger(base: logging.Logger, run_id: str, run_log_path: str):
    """Attach a run-specific file handler for the duration of a workflow."""
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(run)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(run_log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.addFilter(DefaultExtrasFilter(run_default=run_id))
    base.addHandler(fh)
    adapter = logging.LoggerAdapter(base, {"run": run_id})
    try:
        adapter.info("==== run start ====", extra={"event":"run_start"})
        yield adapter
    finally:
        adapter.info("==== run end ====", extra={"event":"run_end"})
        base.removeHandler(fh); fh.close()

import math
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Any

def to_jsonable(x: Any):
    """
    Recursively convert common non-JSON-serializable types to plain Python types.
    - numpy / pandas scalars -> int/float/bool/str
    - numpy arrays -> lists
    - sets/tuples -> lists
    - Decimal -> float
    - NaN/Inf floats -> None
    - pd.Timestamp -> ISO string
    - dict/list -> recurse
    - fallback -> str(x)
    """
    # Primitives
    if x is None or isinstance(x, (str, bool, int, float)):
        # normalize non-finite floats
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    # NumPy scalars
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)

    # Pandas scalars
    if isinstance(x, (pd.Int64Dtype,)):  # rarely hit; defensive
        return int(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()

    # Decimal
    if isinstance(x, Decimal):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return str(x)

    # Collections
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(i) for i in list(x)]
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return [to_jsonable(i) for i in x.tolist()]

    # Fallback
    return str(x)


def write_manifest(out_dir: str, payload: dict):
    """Write a JSON manifest with a timestamp into an output directory."""
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(payload)
    payload["timestamp"] = datetime.utcnow().isoformat() + "Z"

    # sanitize before dumping
    safe_payload = to_jsonable(payload)

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(safe_payload, f, indent=2, ensure_ascii=False)


def append_summary(out_dir: str, row: dict, header_order=None):
    """
    Append one CSV row into summary.csv placed at out_dir's parent (per frequency bucket).
    """
    import csv
    parent = out_dir  # or use os.path.dirname(out_dir) if you prefer one level up
    os.makedirs(parent, exist_ok=True)
    path = os.path.join(parent, "summary.csv")

    # Decide header
    if header_order is None:
        header_order = [
            "csv_file","p_threshold","freq_csv","freq_threshold","layout","mode",
            "input_genes","nodes","edges","steiner_added",
            "steiner_hops","steiner_connectors_pmax",          # requested knobs (optional)
            "steiner_hops_used","steiner_connectors_pmax_used",# actual used
            "steiner_kept_nodes","steiner_total_nodes","steiner_connected_across_terminals",
            "cost_p_sum","score_neglog10_sum","tree_edges_undirected"
            "selection_mode","selection_value",

        ]



    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if write_header:
            w.writeheader()
        # massage types for csv
        row2 = dict(row)
        if isinstance(row2.get("steiner_added"), (list, set, tuple)):
            row2["steiner_added"] = ";".join(map(str, row2["steiner_added"]))
        w.writerow({k: row2.get(k, "") for k in header_order})


# -------------------------------
# Normalize values for visualization
# -------------------------------
def normalize(values: Iterable[float],
              min_size: float = 0.1,
              max_size: float = 2.0,
              *,
              clamp: bool = True) -> list[float]:
    """
    Linearly map 'values' to [min_size, max_size].
    Handles empty, constant, NaN/inf inputs gracefully.

    If clamp=True, any NaN/inf becomes min_size.
    """
    vals = np.asarray(list(values), dtype=float)
    n = vals.size
    if n == 0:
        return []

    # Replace NaN/inf if desired
    if clamp:
        bad = ~np.isfinite(vals)
        if bad.any():
            vals[bad] = np.nan
        # if all are bad, return all min_size
        if np.isnan(vals).all():
            return [float(min_size)] * n
        # fill remaining NaNs with finite min
        finite_vals = vals[np.isfinite(vals)]
        vals[np.isnan(vals)] = np.nanmin(finite_vals)

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    rng = vmax - vmin

    if rng == 0.0:
        return [float(min_size)] * n

    scale = (max_size - min_size) / rng
    return [float(min_size + (x - vmin) * scale) for x in vals]

# -------------------------------
# Create Graphviz DOT representation
# -------------------------------
def create_graphviz_dot(
    G,
    partition: Dict[str, int],
    community_colors: Dict[int, Tuple[str, str]],
    highlight_node: Optional[str] = None,
    layout: str = "dot",
    graph_attr: Optional[dict] = None,
    weighted_edges: bool = False,
    simple_layout: bool = False,
    highlight_new_genes: Optional[tuple[bool, list[str]]] = None,
    logger: Optional[logging.Logger] = None
):
    """
    - Node size ~ out-degree if weighted_edges=True, else fixed.
    - Edge width ~ -log10(p_value) if weighted_edges=True, else fixed.
    - If highlight_new_genes[0] is True, nodes in highlight_new_genes[1] are GREY (Steiner-added).
    - If highlight_node is set, its border is thicker and its outgoing edges get extra width.
    """
    
    if not highlight_new_genes:
        new_nodes_flag, new_nodes_list = False, []
    else:
        new_nodes_flag, new_nodes_list = highlight_new_genes
    new_nodes = set(new_nodes_list) if new_nodes_flag else set()
    logger = logger or get_logger()
    if graph_attr is None:
        graph_attr = {}

    dot = graphviz.Digraph(engine=layout, format="svg", graph_attr=graph_attr)
    dot.attr(tooltip="")

    # Empty graph
    if G.number_of_nodes() == 0:
        logger.warning("Empty graph passed to Graphviz renderer.")
        return dot

    # ---------- NODE SIZING ----------
    outdegrees = [G.out_degree(node) for node in G.nodes()]
    # size in inches for Graphviz 'width'/'height'
    node_sizes = normalize(outdegrees, min_size=1.0, max_size=2.0) if weighted_edges else [1.0] * len(outdegrees)

    # ---------- NODE RENDER ----------
    new_nodes_flag, new_nodes_list = highlight_new_genes
    new_nodes = set(new_nodes_list) if new_nodes_flag else set()

    for idx, node in enumerate(G.nodes()):
        community = partition.get(node, None)
        fill_hex, _ = community_colors.get(community, ("#d3d3d3", "gray"))
        penwidth = "3" if node == highlight_node else "1"
        size = f"{node_sizes[idx]:.3f}"

        # Steiner-added nodes: white fill
        if node in new_nodes:
            fill_hex = "#FFFFFF"  # White

        # Build node tooltip
        out_edges = list(G.out_edges(node, data=True))
        outdeg = len(out_edges)
        parts = []
        for _, tgt, data in out_edges:
            p = data.get("p_value", None)
            kp = data.get("kutsche_p_value", None)
            bp = data.get("benito_p_value", None)
            if kp is not None and bp is not None:
                parts.append(f"{tgt}: ({kp:.6f})({bp:.6f})")
            elif p is not None:
                parts.append(f"{tgt}: ({p:.6f})")
            else:
                parts.append(f"{tgt}: (NA)")
        out_edges_info = ", ".join(parts)

        if any(("kutsche_p_value" in d and "benito_p_value" in d) for _, _, d in out_edges):
            tip = f"{node} may granger cause {outdeg} gene(s), formatted: Gene: (Kutsche)(Benito-Kwiecinski) : \n{out_edges_info}"
        else:
            tip = f"{node} may granger cause {outdeg} gene(s), formatted: Gene: (p-value) \n{out_edges_info}"
        hover_text = html_escape.escape(tip)

        # Simple layout: Graphviz 'plaintext' ignores width/height, so set a label only.
        if simple_layout:
            dot.node(
                node,
                label=node,
                shape="plaintext",
                style="filled",
                fillcolor="white",
                color="black",
                penwidth="1",
                tooltip=hover_text,
            )
        else:
            label = wrap_label(node, max_chars=7)
            margin = "0.1,0.05"
            dot.node(
                node,
                label=label,
                shape="circle",
                style="filled",
                fillcolor=fill_hex,
                color="black",
                penwidth=penwidth,
                tooltip=hover_text,
                width=size,
                height=size,
                fixedsize="true",
                margin=margin,
            )

    # No edges
    if G.number_of_edges() == 0:
        return dot

    # ---------- EDGE WIDTHS ----------
    # Safe -log10(p): use epsilon to avoid log(0)
    eps = 1e-300
    p_values = [G[u][v].get("p_value", 1.0) for u, v in G.edges()]
    if weighted_edges:
        w_vals = [-math.log10(max(p, eps)) for p in p_values]
        edge_widths = normalize(w_vals, min_size=1.0, max_size=10.0)
    else:
        edge_widths = [3.0] * len(p_values)

    # ---------- EDGE RENDER ----------
    for (idx, (source, target)) in enumerate(G.edges()):
        data = G[source][target]
        lag = data.get("lag", "NA")
        p = data.get("p_value", None)
        kp = data.get("kutsche_p_value", None)
        bp = data.get("benito_p_value", None)

        # Color from source community
        src_comm = partition.get(source, None)
        edge_color, _ = community_colors.get(src_comm, ("#d3d3d3", "gray"))

        # If source node is Steiner-added, maybe use a neutral color (optional)
        if source in new_nodes:
            edge_color = "#000000"  # DarkGray

        # Hover text
        if kp is not None and bp is not None:
            tip = f"{source} may Granger Cause {target} at lag {lag} with p={kp:.6f} (Kutsche) and p={bp:.6f} (Benito-Kwiecinski)"
        elif p is not None:
            tip = f"{source} may Granger Cause {target} at lag {lag} with p={p:.6f}"
        else:
            tip = f"{source} → {target} (lag {lag}, p=NA)"
        hover_text = html_escape.escape(tip)

        width = edge_widths[idx]
        # Extra emphasis from highlighted node
        if highlight_node and source == highlight_node:
            width = width + 3.0
            edge_color = "black"

        if simple_layout:
            dot.edge(source, target, color="black", penwidth="1", tooltip=hover_text)
        else:
            dot.edge(source, target, color=edge_color, penwidth=f"{width:.3f}", tooltip=hover_text)

    return dot

def wrap_label(name: str, max_chars: int = 10) -> str:
    """
    Wrap a label string for Graphviz:
     - Preferentially break at hyphens (“-”).
     - If a piece is still ≥ max_chars, hard-wrap it in (max_chars-2)-sized chunks.
     - Then greedily pack pieces into lines of ≤ max_chars (no overflow).
    """
    hard_wrap = max(1, max_chars - 2)
    parts = name.split("-")
    tokens = []
    for i, part in enumerate(parts):
        tokens.append(part)
        if i < len(parts) - 1:
            tokens.append("-")

    expanded = []
    for tok in tokens:
        if tok == "-" or len(tok) < max_chars:
            expanded.append(tok)
        else:
            for i in range(0, len(tok), hard_wrap):
                expanded.append(tok[i : i + hard_wrap])

    lines, current = [], ""
    for tok in expanded:
        cand = current + tok
        if len(cand) > max_chars:
            if current:
                lines.append(current)
            current = tok
        else:
            current = cand
    if current:
        lines.append(current)

    return "\\n".join(lines)


# --- NEW: read genes by hard frequency threshold ---
def load_genes_by_frequency(freq_csv_path: str, min_frequency: float, top_n: Optional[int] = None) -> list[str]:
    """
    Read 'Gene,Coassociation Frequency' CSV and return genes with frequency >= min_frequency.
    Optionally keep only top_n by frequency (descending).
    """
    df = pd.read_csv(freq_csv_path)
    # Accept a few common header variants:
    gene_col = next(c for c in df.columns if c.strip().lower() in {"gene", "symbol", "gene_symbol"})
    freq_col = next(c for c in df.columns if "freq" in c.strip().lower())

    df = df.rename(columns={gene_col: "gene", freq_col: "frequency"})
    df = df[df["frequency"] >= float(min_frequency)].sort_values("frequency", ascending=False)

    if top_n is not None:
        df = df.head(int(top_n))

    return df["gene"].tolist()

def to_undirected_weighted(
    G: nx.DiGraph,
    weight_attr: str = "p_value",
    hop_penalty: float = 0.0,
    terminals: Optional[Iterable[str]] = None,
    penalize_mode: str = "any_nonterminal",
) -> nx.Graph:
    """
    Create an undirected weighted graph where the weight is based on p-values and an
    optional penalty on edges that go through Steiner nodes.

    If `terminals` is provided:

      - terminal–terminal edges:
            w_eff = p_value

      - edges with Steiner involvement (depending on `penalize_mode`):

        * penalize_mode == "any_nonterminal"  (recommended default)
            penalize if (u not terminal) OR (v not terminal)

        * penalize_mode == "both_nonterminal"
            penalize only if (u not terminal) AND (v not terminal)

    If `terminals` is None or hop_penalty == 0, this reduces to plain p_value weights.

    When an undirected edge already exists, we keep the smaller weight.
    """
    U = nx.Graph()
    term_set = set(terminals) if terminals is not None else None

    for u, v, data in G.edges(data=True):
        w = data.get(weight_attr, None)
        if w is None:
            continue
        base = float(w)

        # No terminals or no penalty -> just use p-values
        if term_set is None or hop_penalty == 0.0:
            w_eff = base
        else:
            u_term = (u in term_set)
            v_term = (v in term_set)

            if penalize_mode == "any_nonterminal":
                penalize = (not u_term) or (not v_term)
            elif penalize_mode == "both_nonterminal":
                penalize = (not u_term) and (not v_term)
            else:
                # fallback: penalize everything non-terminal-ish
                penalize = (not u_term) or (not v_term)

            w_eff = base + (hop_penalty if penalize else 0.0)

        if U.has_edge(u, v):
            U[u][v]["weight"] = min(U[u][v]["weight"], w_eff)
        else:
            U.add_edge(u, v, weight=w_eff)

    return U



def tree_cost_from_undirected(T: nx.Graph, weight_key: str = "weight") -> Tuple[float, float, int]:
    """
    Compute costs for an (undirected) tree/forest T:
      - cost_p_sum: sum of edge weights (here: p-values)
      - score_neglog10_sum: sum of -log10(weight)
      - m: number of undirected edges
    """
    eps = 1e-300
    cost_p_sum = 0.0
    score_neglog10_sum = 0.0
    m = 0
    for _, _, d in T.edges(data=True):
        w = float(d.get(weight_key, 1.0))
        cost_p_sum += w
        score_neglog10_sum += (-math.log10(max(w, eps)))
        m += 1
    return cost_p_sum, score_neglog10_sum, m


# --- NEW: extract directed edges that correspond to an undirected edge set ---
def directed_edges_from_undirected_pairs(G_dir: nx.DiGraph, undirected_edges: Set[Tuple[str, str]]) -> nx.DiGraph:
    """
    Given a directed graph and a set of undirected pairs (u,v), include whichever
    directed edges exist in G_dir for each pair.
    """
    H = nx.DiGraph()
    for u, v in undirected_edges:
        if G_dir.has_edge(u, v):
            H.add_edge(u, v, **G_dir[u][v])
        if G_dir.has_edge(v, u):
            H.add_edge(v, u, **G_dir[v][u])
    return H


# --- NEW: mode builders ---
def build_induced(G_dir: nx.DiGraph, genes: list[str]) -> nx.DiGraph:
    """Return the induced directed subgraph for selected genes."""
    return G_dir.subgraph(genes).copy()

def build_mst(G_dir: nx.DiGraph, genes: list[str]) -> Tuple[nx.DiGraph, float, float, int]:
    """Build a directed minimum spanning forest for selected genes."""
    U = to_undirected_weighted(G_dir, weight_attr="p_value")
    U_sub = U.subgraph([g for g in genes if g in U]).copy()
    if U_sub.number_of_nodes() == 0:
        return nx.DiGraph(), 0.0, 0.0, 0

    undirected_mst_edges = set()
    cost_p_total = 0.0
    score_total = 0.0
    m_total = 0

    for comp_nodes in nx.connected_components(U_sub):
        comp = U_sub.subgraph(comp_nodes)
        if comp.number_of_edges() == 0:
            continue
        T = minimum_spanning_tree(comp, weight="weight")
        c, s, m = tree_cost_from_undirected(T, weight_key="weight")
        cost_p_total += c
        score_total += s
        m_total += m
        undirected_mst_edges.update(tuple(sorted(e)) for e in T.edges())

    H = directed_edges_from_undirected_pairs(G_dir, undirected_mst_edges)
    # NEW: ensure all selected genes are present as nodes in the MST view
    H.add_nodes_from(genes)
    return H, cost_p_total, score_total, m_total


def build_steiner(G_candidate: nx.DiGraph, terminals: List[str], *, hop_penalty: float = 0.0) -> Tuple[nx.DiGraph, Set[str], float, float, int]:
    """
    Runs Mehlhorn Steiner on an UNDIRECTED weighted view where:
      weight = min p across directions + hop_penalty.
    Returns: directed projection, added (connector) nodes, sum(weight), sum(-log10 weight), edge count.
    """
    U = to_undirected_weighted(
        G_candidate,
        weight_attr="p_value",
        hop_penalty=hop_penalty,
        terminals=terminals,
        penalize_mode="any_nonterminal",
    )

    # # ---- DEBUG STEINER-U EDGES + PURE CLASSIFICATION ----
    # import hashlib
    # term_set = set(terminals)

    # # MD5 of Steiner-U edge set (undirected pairs)
    # undirected_pairs = sorted(tuple(sorted((u, v))) for u, v in U.edges())
    # edges_repr = "\n".join(f"{u},{v}" for (u, v) in undirected_pairs)
    # edges_hash = hashlib.md5(edges_repr.encode()).hexdigest()

    # # Pure definition: only terminal–terminal are unpenalized; everything else penalized
    # pure_penalized = 0
    # pure_unpenalized = 0
    # for u, v in U.edges():
    #     if (u in term_set) and (v in term_set):
    #         pure_unpenalized += 1
    #     else:
    #         pure_penalized += 1

    # logger = get_logger()
    # logger.info(
    #     "[DEBUG] SteinerU_edges_n=%d SteinerU_edges_md5=%s | PURE penalized=%d unpenalized=%d",
    #     len(undirected_pairs),
    #     edges_hash,
    #     pure_penalized,
    #     pure_unpenalized,
    # )
    # -----------------------------------------------

    terms = [g for g in terminals if g in U]
    logger.info("Steiner: %d terminals", (len(terms)))
    if len(terms) <= 1:
        return nx.DiGraph(), set(), 0.0, 0.0, 0

    T = steiner_tree(U, terms, weight="weight")
    undirected_edges = set(tuple(sorted(e)) for e in T.edges())
    H = directed_edges_from_undirected_pairs(G_candidate, undirected_edges)

    added_nodes = set(T.nodes()) - set(terms)
    cost_p_sum, score_neglog10_sum, m = tree_cost_from_undirected(T, weight_key="weight")

    # NEW: keep all terminals as nodes in the directed view
    H.add_nodes_from(terms)

    return H, added_nodes, cost_p_sum, score_neglog10_sum, m


def build_steiner_forest(G_candidate: nx.DiGraph, terminals: List[str], *, hop_penalty: float = 0.0) -> Tuple[nx.DiGraph, set, float, float, int, int]:
    """
    Steiner per connected component (≥2 terminals) with hop penalty.
    """
    U = to_undirected_weighted(
    G_candidate,
    weight_attr="p_value",
    hop_penalty=hop_penalty,
    terminals=terminals,                 # <--- NEW: tell it who the terminals are
    penalize_mode="any_nonterminal",     # edges touching non-terminals get penalized
    )
    term_set = set(t for t in terminals if t in U)
    H_union = nx.DiGraph()
    added_all = set()
    cost_sum = score_sum = 0.0
    m_sum = 0
    used = 0

    for comp_nodes in nx.connected_components(U):
        comp_terms = list(term_set.intersection(comp_nodes))
        if len(comp_terms) < 2:
            continue
        G_sub = G_candidate.subgraph(comp_nodes).copy()
        H, added, c_p, s_neglog, m = build_steiner(G_sub, comp_terms, hop_penalty=hop_penalty)
        H_union.add_nodes_from(H.nodes(data=True))
        H_union.add_edges_from(H.edges(data=True))
        added_all.update(added)
        cost_sum += c_p
        score_sum += s_neglog
        m_sum += m
        used += 1

    # NEW: keep all terminals as nodes in the union view
    H_union.add_nodes_from(term_set)
    return H_union, added_all, cost_sum, score_sum, m_sum, used


def build_candidate_graph(
    csv_file: str,
    p_threshold: float,
    connectors_pmax: Optional[float] = None,
    *,
    terminals: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> nx.DiGraph:
    """
    Build the candidate graph for Steiner.

    Two modes:

    1) Legacy mode (terminals is None):
       - Equivalent to the old behaviour:
         keep all edges with p <= min(p_threshold, connectors_pmax) (or p_threshold if connectors_pmax is None).

    2) Terminals-aware mode (recommended for Steiner):
       - Always include ALL terminal–terminal edges with p <= p_threshold.
       - Additionally include "connector" edges with p <= connectors_pmax
         (these can be terminal–terminal, terminal–nonterminal, or nonterminal–nonterminal).
       - The penalty is applied *later* in to_undirected_weighted based on which nodes are terminals.

    This guarantees:
       - The Steiner solver always has access to the same terminal–terminal edges as MST.
       - Extra ultra-strong edges (≤ connectors_pmax) are available to create alternative
         paths through Steiner nodes, but they are penalized via hop_penalty.
    """
    logger = logger or get_logger()

    df = pd.read_csv(csv_file)
    required = {"gene1", "gene2", "lag", "p-value"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns in {csv_file}: {missing}"

    # -----------------------------
    # MODE 1: legacy behaviour
    # -----------------------------
    if terminals is None or connectors_pmax is None:
        # Exactly what you had before:
        #   p_cut = p_threshold if connectors_pmax is None else min(p_threshold, connectors_pmax)
        #   df_f = df[df["p-value"] <= p_cut]
        if connectors_pmax is None:
            p_cut = float(p_threshold)
            logger.info(
                "build_candidate_graph (legacy): using p_cut=%.6g (no connectors_pmax), terminals=None",
                p_cut,
            )
        else:
            p_cut = min(float(p_threshold), float(connectors_pmax))
            logger.info(
                "build_candidate_graph (legacy): using p_cut=min(%.6g, %.6g)=%.6g, terminals=None",
                float(p_threshold), float(connectors_pmax), p_cut,
            )

        df_f = df[df["p-value"] <= p_cut].copy()
        logger.info(
            "build_candidate_graph (legacy): kept %d edges (p<=%.6g)",
            len(df_f), p_cut,
        )

    # -----------------------------
    # MODE 2: terminals-aware
    # -----------------------------
    else:
        term_set = set(terminals)

        p_thr = float(p_threshold)
        p_conn = float(connectors_pmax)

        # 1) All terminal–terminal edges up to p_threshold (MST-compatible backbone)
        mask_term_term = (
            df["gene1"].isin(term_set)
            & df["gene2"].isin(term_set)
            & (df["p-value"] <= p_thr)
        )

        # 2) Extra ultra-strong edges (connectors) up to connectors_pmax
        #    We allow any endpoints here; the "Steiner-ness" comes from later
        #    when we penalize edges touching non-terminals.
        mask_conn = df["p-value"] <= p_conn

        df_term_term = df[mask_term_term].copy()
        df_conn = df[mask_conn].copy()

        # Union of both sets
        df_f = pd.concat([df_term_term, df_conn], axis=0).drop_duplicates(
            subset=["gene1", "gene2", "lag", "p-value"]
        )

        logger.info(
            "build_candidate_graph (terminals-aware): "
            "terminals=%d | term-term edges<=%.6g: %d | conn-edges<=%.6g: %d | union: %d",
            len(term_set),
            p_thr, len(df_term_term),
            p_conn, len(df_conn),
            len(df_f),
        )

    # -----------------------------
    # Build the directed graph
    # -----------------------------
    G = nx.DiGraph()
    for _, row in df_f.iterrows():
        u = row["gene1"]
        v = row["gene2"]
        lag = row["lag"]
        p = float(row["p-value"])
        G.add_edge(u, v, lag=lag, p_value=p)

    logger.info(
        "build_candidate_graph: final candidate graph nodes=%d, edges=%d",
        G.number_of_nodes(), G.number_of_edges(),
    )
    return G



def adaptive_choose_by_hops_then_p(
    csv_file: str,
    p_threshold: float,
    terminals: List[str],
    *,
    init_hops: int = 2,
    steiner_max_nodes: Optional[int] = 5000,
    p_factors: Optional[Tuple[float, ...]] = None,
    min_terminal_coverage: float = 0.9,
    logger: Optional[logging.Logger] = None,
) -> Tuple[nx.DiGraph, dict]:
    """
    Strategy:
      For hops h = init_hops, init_hops-1, ..., 0:
        For each connectors_pmax in p_factors (absolute p-value thresholds):
          - build candidate graph with p <= min(p_threshold, connectors_pmax)
          - keep nodes within h hops (unweighted) of ANY terminal
          - respect steiner_max_nodes (if given)
          - compute terminal coverage in the largest connected component
          - if coverage >= min_terminal_coverage: accept and return that component

    If no configuration achieves the desired coverage, fall back to the smallest
    under-cap candidate seen (may be disconnected).

    Returns:
      (G_trim, meta)

    meta keys:
      - steiner_hops_used
      - steiner_connectors_pmax_used
      - kept_nodes
      - total_nodes
      - connected_across_terminals   (True if we returned a single main component)
      - terminal_coverage_fraction   (0–1, for the chosen candidate if available)
      - terminals_total
      - terminals_in_largest_component
    """
    logger = logger or get_logger()

    # If no p_factors supplied, just use the base p_threshold as a single cutoff
    if p_factors is None:
        p_factors = (p_threshold,)

    # Track best under-cap candidate for fallback
    best_under_cap = None  # dict with keys: keep, kept_nodes, h, pmax, total, etc.

    for h in range(int(init_hops), -1, -1):
        for pmax in p_factors:
            # candidate graph under this connectors cutoff
            G_cand = build_candidate_graph(
                csv_file,
                p_threshold,
                connectors_pmax=pmax,
                terminals=terminals,
                logger=logger,
            )

            # # ---- DEBUG CANDIDATE EDGES (before hops trimming) ----
            # import hashlib
            # U_full = G_cand.to_undirected()
            # cand_edges = sorted(tuple(sorted((u, v))) for u, v in U_full.edges())
            # edges_repr = "\n".join(f"{u},{v}" for (u, v) in cand_edges)
            # edges_hash = hashlib.md5(edges_repr.encode()).hexdigest()
            # logger.info(
            #     "[DEBUG] cand_edges_n=%d cand_edges_md5=%s (h=%d, pmax=%.6g)",
            #     len(cand_edges),
            #     edges_hash,
            #     h,
            #     pmax,
            # )
            # # ------------------------------------------------------

            U = G_cand.to_undirected()

            terms = [t for t in terminals if t in U]
            if len(terms) < 2:
                logger.info(
                    "Adaptive(h=%d, pmax=%.6g): <2 terminals present in candidate; skip",
                    h, pmax
                )
                continue

            # nodes within h hops (unweighted) of any terminal
            keep = set(multi_source_dijkstra_path_length(
                U, sources=terms, cutoff=h, weight=lambda u, v, d: 1
            ).keys())
            kept = len(keep)
            total = U.number_of_nodes()

            logger.info(
                "Adaptive(h=%d, pmax=%.6g): kept=%d (from %d)",
                h, pmax, kept, total
            )

            # skip empty
            if kept == 0:
                continue

            # respect node cap
            if steiner_max_nodes is not None and kept > steiner_max_nodes:
                logger.info(
                    "Adaptive(h=%d, pmax=%.6g): kept=%d exceeds steiner_max_nodes=%d; skipping",
                    h, pmax, kept, steiner_max_nodes
                )
                continue

            # Subgraph restricted to kept nodes (for coverage analysis)
            U_keep = U.subgraph(keep).copy()
            term_set = set(terms)

            # components that contain at least one terminal
            components_with_terms = []
            for comp in nx.connected_components(U_keep):
                comp = set(comp)
                comp_terms = term_set.intersection(comp)
                if comp_terms:
                    components_with_terms.append((len(comp_terms), comp, comp_terms))

            if not components_with_terms:
                # no terminals in kept region (should be rare)
                logger.info(
                    "Adaptive(h=%d, pmax=%.6g): no terminals in kept region; continuing",
                    h, pmax
                )
                continue

            # largest component by number of terminals
            components_with_terms.sort(key=lambda x: x[0], reverse=True)
            largest_k, largest_comp_nodes, largest_comp_terms = components_with_terms[0]
            total_terms = len(terms)
            coverage = largest_k / max(1, total_terms)

            logger.info(
                "Adaptive(h=%d, pmax=%.6g): terminal coverage in largest component = %.3f (%d/%d)",
                h, pmax, coverage, largest_k, total_terms
            )

            # Update best-under-cap candidate (prefer smallest kept; you could also
            # refine this to prioritize higher coverage if you want later)
            if (best_under_cap is None) or (kept < best_under_cap["kept_nodes"]):
                best_under_cap = {
                    "keep": keep,
                    "kept_nodes": kept,
                    "h": h,
                    "pmax": pmax,
                    "total": total,
                    "terminal_coverage_fraction": coverage,
                    "terminals_total": total_terms,
                    "terminals_in_largest_component": largest_k,
                }

            # Accept if coverage is "good enough"
            if coverage >= float(min_terminal_coverage):
                # Restrict candidate to the largest terminal-containing component
                G_trim = G_cand.subgraph(largest_comp_nodes).copy()
                kept_in_main = G_trim.number_of_nodes()

                meta = {
                    "steiner_hops_used": h,
                    "steiner_connectors_pmax_used": pmax,
                    "kept_nodes": kept_in_main,
                    "total_nodes": total,
                    # In this trimmed view, all included terminals are in one component
                    "connected_across_terminals": True,
                    "terminal_coverage_fraction": coverage,
                    "terminals_total": total_terms,
                    "terminals_in_largest_component": largest_k,
                }
                return G_trim, meta

            # otherwise: coverage too low -> continue searching (other pmax / hops)

        # try next smaller h

    # --- Fallback: best under-cap candidate (may be disconnected / low coverage) ---
    if best_under_cap is not None:
        logger.info(
            "Adaptive fallback: using best under-cap candidate with kept=%d (coverage=%.3f, h=%d, pmax=%.6g)",
            best_under_cap["kept_nodes"],
            best_under_cap.get("terminal_coverage_fraction", 0.0),
            best_under_cap["h"],
            best_under_cap["pmax"],
        )
        G_cand = build_candidate_graph(csv_file, p_threshold, terminals=terminals, connectors_pmax=best_under_cap["pmax"])
        G_trim = G_cand.subgraph(best_under_cap["keep"]).copy()
        meta = {
            "steiner_hops_used": best_under_cap["h"],
            "steiner_connectors_pmax_used": best_under_cap["pmax"],
            "kept_nodes": best_under_cap["kept_nodes"],
            "total_nodes": best_under_cap["total"],
            "connected_across_terminals": False,
            "terminal_coverage_fraction": best_under_cap.get("terminal_coverage_fraction", None),
            "terminals_total": best_under_cap.get("terminals_total", None),
            "terminals_in_largest_component": best_under_cap.get("terminals_in_largest_component", None),
        }
        return G_trim, meta

    # --- Nothing workable at all ---
    logger.info(
        "Adaptive: no workable candidate; returning empty graph.",
    )
    return nx.DiGraph(), {
        "steiner_hops_used": None,
        "steiner_connectors_pmax_used": None,
        "kept_nodes": 0,
        "total_nodes": 0,
        "connected_across_terminals": False,
        "terminal_coverage_fraction": 0.0,
        "terminals_total": len(terminals),
        "terminals_in_largest_component": 0,
    }

def assign_colors(partition, primary_gene=None, logger: Optional[logging.Logger] = None):
    """Assign display colors to community ids, prioritizing the GOI community."""
    logger = logger or get_logger()
    logger.debug(f"Assigning colors to communities: {set(partition.values())}")
    cmaps = ['tab20', 'tab20b']
    all_colors = list(chain.from_iterable(plt.get_cmap(c).colors for c in cmaps))
    community_colors = {}

    # Ensure the community of the primary_gene is first in order
    if primary_gene and primary_gene in partition:
        primary_community = partition[primary_gene]
        unique_communities = [primary_community] + [c for c in sorted(set(partition.values())) if c != primary_community]
    else:
        unique_communities = sorted(set(partition.values()))  # Sort to maintain consistency

    for idx, community in enumerate(unique_communities):
        color = all_colors[idx % len(all_colors)]
        color_hex = '#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3])
        community_colors[community] = (color_hex, str(idx + 1))  # Label communities as 1, 2, 3, ...

    logger.debug(f"Assigned community colors: {community_colors}")
    return community_colors

def build_gene_network(gene_list: List[str], csv_file: str, p_threshold: float = 0.05,
                       logger: Optional[logging.Logger] = None) -> nx.DiGraph:

    """
    Build a gene network from a CSV file given a list of starting genes and a p-value threshold.

    Parameters:
        gene_list (List[str]): Genes to start the search from (source).
        csv_file (str): Path to CSV file with columns: Gene1, Gene2, Lag, P_Value.
        p_threshold (float): Maximum p-value for edge inclusion.

    Returns:
        networkx.DiGraph: Directed graph of genes.
    """
    logger = logger or get_logger()
    df = pd.read_csv(csv_file)
    required_columns = {'gene1', 'gene2', 'lag', 'p-value'}
    assert required_columns.issubset(df.columns), f"Missing required columns: {required_columns - set(df.columns)}"

    df_filtered = df[(df['gene1'].isin(gene_list) & df['gene2'].isin(gene_list)) & (df['p-value'] <= p_threshold)]
    
    logger.debug("Filtered edges: %d", len(df_filtered))
    G = nx.DiGraph()
    for _, row in df_filtered.iterrows():
        source, target, lag, p = row['gene1'], row['gene2'], row['lag'], row['p-value']
        G.add_edge(source, target, lag=lag, p_value=p)

    # NEW: ensure all selected genes are present as nodes,
    # even if they have no edges under this p-threshold.
    G.add_nodes_from(gene_list)

    logger.debug("Full graph built: nodes=%d, edges=%d", G.number_of_nodes(), G.number_of_edges())
    return G

def visualize_gene_network(
    csv_file,
    p_threshold=0.05,
    layout="dot",
    graph_attr={},
    highlight_node=None,
    gene_of_interest=None,
    output_path="output_graph",
    number_of_runs=None,
    simple_layout=True,
    *,
    freq_csv: Optional[str] = None,
    freq_threshold: Optional[float] = None,
    top_n: Optional[int] = None,
    top_percent: Optional[float] = None,   # <--- NEW
    mode: str = "induced",
    # --- Steiner knobs ---
    steiner_hops: Optional[int] = 2,
    steiner_hop_penalty: Optional[float] = None,  # per-edge penalty in Steiner weights

    steiner_p_factors: Optional[float] = None,
    steiner_max_nodes: Optional[int] = 5000,
    logger: Optional[logging.Logger] = None,
):
    """Build and render a configured induced, MST, or Steiner gene network."""

    # --- validate inputs early ---
    # --- validate inputs early ---
    if freq_csv is None:
        raise ValueError("freq_csv is required.")
    if all(x is None for x in (freq_threshold, top_n, top_percent)):
        raise ValueError("Provide one of: freq_threshold, top_n, or top_percent (optionally combine with freq_threshold).")


    logger = logger or get_logger()
    
    gene_list = select_genes_from_frequency(
        freq_csv,
        min_frequency=freq_threshold,
        top_n=top_n,
        top_percent=top_percent,
    )

    sel_desc = (
        f"freq>={freq_threshold}" if freq_threshold is not None else
        f"top_n={top_n}" if top_n is not None else
        f"top_percent={top_percent:.3f}"
    )
    logger.info("Gene selection from %s: %s → %d genes", freq_csv, sel_desc, len(gene_list))


    G_full = build_gene_network(gene_list, csv_file, p_threshold, logger=logger)
    all_inputs = set(gene_list)
    graph_nodes = set(G_full.nodes())
    missing = sorted(all_inputs - graph_nodes)

    if missing:
        logger.info(
            "Inputs without any edges under p<=%.4g: %d genes (example: %s)",
            p_threshold,
            len(missing),
            ", ".join(missing)
        )

    logger.info("Initial graph: inputs=%d, nodes=%d, edges=%d",
                len(gene_list), G_full.number_of_nodes(), G_full.number_of_edges())
    
    if G_full.number_of_nodes() == 0:
        logger.warning("No edges found. Empty graph.")
        return
    


    # 2) Choose build mode
    if mode == "induced":
        G_view = build_induced(G_full, gene_list)
        steiner_added = set()
        tree_cost_p_sum = None
        tree_score_neglog10_sum = None
        m_tree = None

    elif mode == "mst":
        G_view, tree_cost_p_sum, tree_score_neglog10_sum, m_tree = build_mst(G_full, gene_list)
        steiner_added = set()
        avg_p = tree_cost_p_sum / max(1, m_tree)
        logger.info(
            "MST cost: edges=%d | cost_p=%.6g (avg %.3g) | score=%.3f",
            m_tree, tree_cost_p_sum, avg_p, tree_score_neglog10_sum
        )


    elif mode == "steiner":
        # All freq-selected genes are terminals
        terminals = list(gene_list)

        # # ---- DEBUG TERMINALS ----
        # import hashlib
        # sorted_terms = sorted(terminals)
        # term_repr = "\n".join(sorted_terms)
        # term_hash = hashlib.md5(term_repr.encode()).hexdigest()
        # logger.info(
        #     "[DEBUG] terminals_n=%d terminals_md5=%s",
        #     len(sorted_terms),
        #     term_hash,
        # )
        # # -------------------------


        # Pick a workable candidate by trying p-factors at init_hops, then decreasing hops
        G_cand_trim, meta = adaptive_choose_by_hops_then_p(
            csv_file=csv_file,
            p_threshold=p_threshold,
            terminals=terminals,
            init_hops=steiner_hops if steiner_hops is not None else 2,
            steiner_max_nodes=steiner_max_nodes if steiner_max_nodes is not None else 5000,
            p_factors=steiner_p_factors,
            min_terminal_coverage=0.99,
            logger=logger,
        )


        logger.info(
            "Steiner adaptive chosen: hops=%s, pmax=%s | kept=%d (from %s) | connected=%s",
            str(meta.get("steiner_hops_used")),
            (f"{meta.get('steiner_connectors_pmax_used'):.6g}"
            if meta.get("steiner_connectors_pmax_used") else "NA"),
            int(meta.get("kept_nodes", 0)),
            str(meta.get("total_nodes", "NA")),
            str(meta.get("connected_across_terminals", False)),
        )

        if G_cand_trim.number_of_nodes() == 0:
            logger.info("No candidates kept; Steiner will be empty.")
            G_view, steiner_added = nx.DiGraph(), set()
            cost_p_sum = score_neglog10_sum = 0.0
            m_tree = 0

        elif meta.get("connected_across_terminals", False):
            G_view, steiner_added, tree_cost_p_sum, tree_score_neglog10_sum, m_tree = build_steiner(G_cand_trim, terminals, hop_penalty=float(steiner_hop_penalty or 0.0))
            avg_p = tree_cost_p_sum / max(1, m_tree)
            logger.info(
                "Steiner cost: edges=%d | cost_p=%.6g (avg %.3g) | score=%.3f",
                m_tree, tree_cost_p_sum, avg_p, tree_score_neglog10_sum
            )

        else:
            G_view, steiner_added, tree_cost_p_sum, tree_score_neglog10_sum, m_tree, n_used = build_steiner_forest(G_cand_trim, terminals, hop_penalty=float(steiner_hop_penalty or 0.0))
            avg_p = tree_cost_p_sum / max(1, m_tree)
            logger.info(
                "Steiner forest: comps=%d | edges=%d | cost_p=%.6g (avg %.3g) | score=%.3f",
                n_used, m_tree, tree_cost_p_sum, avg_p, tree_score_neglog10_sum
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    logger.debug("Final graph (%s): inputs=%d, nodes=%d, edges=%d", mode, len(gene_list), G_view.number_of_nodes(), G_view.number_of_edges())
    # 3) Community partitioning (as before)
    if number_of_runs is not None:
        G_view_und = G_view.to_undirected()

        consensus, coassoc, partitions, n_genes_same_comm, nodes, coassoc_state = consensus_partition(
            G_view,
            G_view_und,
            n_runs=number_of_runs,
            gene_of_interest=gene_of_interest,
            plot=False,
            n_threads=os.cpu_count()-1,          # coassociation workers
            n_louvain_workers=os.cpu_count()-1,  # Louvain workers
        )

        partition = consensus
    else:
        partition = {node: 0 for node in G_view.nodes()}

    community_colors = assign_colors(partition, primary_gene=gene_of_interest, logger=logger)

    # --- Community-level stats ---
    from collections import Counter

    comm_sizes = Counter(partition.values())          # community_id -> size
    n_communities = len(comm_sizes)
    if n_communities > 0:
        avg_community_size = sum(comm_sizes.values()) / n_communities
    else:
        avg_community_size = 0.0

    # Size of the community that the gene_of_interest belongs to (in the current view)
    gene_of_interest_comm_size = None
    if gene_of_interest is not None and gene_of_interest in partition:
        goi_comm_id = partition[gene_of_interest]
        gene_of_interest_comm_size = comm_sizes.get(goi_comm_id, 0)

    logger.info(
        "Communities: n=%d | avg_size=%.2f | GOI_comm_size=%s",
        n_communities,
        avg_community_size,
        str(gene_of_interest_comm_size),
    )


    # 4) Highlight "newly added" nodes in STEINER mode as grey
    if mode == "steiner" and steiner_added:
        highlight_tuple = (True, sorted(steiner_added))
    else:
        highlight_tuple = (False, [])  # MST and induced

    if mode != "steiner":
        # Sanity check: MST/induced must not highlight anything
        assert highlight_tuple == (False, []), "Non-Steiner run should not highlight nodes."

    # 5) Render via Graphviz (unchanged except we pass G_view)
    dot = create_graphviz_dot(
        G_view,
        partition=partition,
        community_colors=community_colors,
        highlight_node=highlight_node,
        layout=layout,
        graph_attr=graph_attr,
        simple_layout=simple_layout,
        highlight_new_genes=highlight_tuple
    )
    logger.debug("Figure created: nodes=%d, edges=%d", G_view.number_of_nodes(), G_view.number_of_edges())

    # AFTER — handles both “directory” and “file-stem” cases
    if os.path.isdir(output_path) or output_path.endswith(os.sep):
        out_dir = output_path
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{mode}_{layout}" 
        dot.render(filename=filename, directory=out_dir, format="svg", cleanup=True)
        svg_path = os.path.join(out_dir, f"{filename}.svg")
    else:
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        # treat output_path as a stem (no extension)
        dot.render(output_path, format="svg", cleanup=True)
        svg_path = f"{output_path}.svg"

    terminals_for_paths = list(gene_list)

    (minsum_summary_csv,
    minsum_full_csv,
    minmax_summary_csv,
    minmax_full_csv,
    path_metrics) = compute_and_save_paths_to_gene_of_interest_split(
        G_view,
        terminals=terminals_for_paths,
        output_dir=out_dir,
        gene_of_interest=gene_of_interest,
        logger=logger,
        steiner_added=(steiner_added if mode == "steiner" else set()),
    )

    path_cost_p_sum = path_metrics["path_cost_p_sum"]
    path_score_neglog10_sum = path_metrics["path_score_neglog10_sum"]
    path_n = path_metrics["path_n"]
    path_cost_p_mean = path_cost_p_sum / max(1, path_n)
    path_score_neglog10_mean = path_score_neglog10_sum / max(1, path_n)




    logger.debug("Saved SVG: %s", svg_path)

    layouts = ["dot", "neato", "fdp", "sfdp", "circo", "twopi"]
    for l in layouts:
        gene_of_interest_meta = save_gene_of_interest_community_bundle(
            G_view,
            partition=partition,
            community_colors=community_colors,
            gene_of_interest=gene_of_interest,
            output_path=out_dir if os.path.isdir(output_path) or output_path.endswith(os.sep) else os.path.dirname(svg_path) or ".",
            layout=l,
            graph_attr=graph_attr,
            logger=logger,
            steiner_added=(steiner_added if mode == "steiner" else set()),
        )


    ret = {
        "csv_file": csv_file,
        "p_threshold": p_threshold,
        "freq_csv": freq_csv,
        "hop_penalty": steiner_hop_penalty if mode == "steiner" else None,

        # Always present (may be None)
        "freq_threshold": freq_threshold,
        "top_n": top_n,
        "top_percent": top_percent,

        "layout": layout,
        "mode": mode,
        "input_genes": len(gene_list),
        "nodes": G_view.number_of_nodes(),
        "edges": G_view.number_of_edges(),

        # --- path-based metrics ---
        "cost_p_sum": path_cost_p_sum,
        "score_neglog10_sum": path_score_neglog10_sum,
        "paths_n": path_n,
        "cost_p_mean_per_path": path_cost_p_mean,
        "score_neglog10_mean_per_path": path_score_neglog10_mean,

        # --- legacy tree metrics ---
        "tree_cost_p_sum": tree_cost_p_sum,
        "tree_score_neglog10_sum": tree_score_neglog10_sum,
        "tree_edges_undirected": m_tree,

        # --- NEW: community stats ---
        "n_communities": n_communities,
        "avg_community_size": avg_community_size,
        "gene_of_interest_comm_size": gene_of_interest_comm_size,
        f"paths_to_{gene_of_interest}_minsum_summary_csv": minsum_summary_csv,
        f"paths_to_{gene_of_interest}_minsum_full_csv": minsum_full_csv,
        f"paths_to_{gene_of_interest}_minmax_summary_csv": minmax_summary_csv,
        f"paths_to_{gene_of_interest}_minmax_full_csv": minmax_full_csv,
        #--- Steiner metadata ---
        "steiner_hops_used": meta.get("steiner_hops_used") if mode == "steiner" else None,
        "steiner_connectors_pmax_used": meta.get("steiner_connectors_pmax_used") if mode == "steiner" else None,
        "steiner_kept_nodes": meta.get("kept_nodes") if mode == "steiner" else None,
        "steiner_total_nodes": meta.get("total_nodes") if mode == "steiner" else None,
        "steiner_connected_across_terminals": meta.get("connected_across_terminals") if mode == "steiner" else None,
        "steiner_added": sorted(steiner_added) if mode == "steiner" else [],

    }



    # append gene_of_interest community metadata (may be empty if gene_of_interest not present)
    ret.update(gene_of_interest_meta)
    return ret


def extract_community_nodes(partition: Dict[str, int], anchor: str) -> Optional[Tuple[int, list[str]]]:
    """
    Given a node 'anchor' (e.g., 'gene_of_interest'), return (community_id, sorted node list)
    for the community containing that node. Returns None if not found.
    """
    if anchor not in partition:
        return None
    cid = partition[anchor]
    members = sorted([n for n, c in partition.items() if c == cid])
    return cid, members

import heapq
import math
import os
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def minimax_single_source_paths(
    G: nx.Graph,
    source: str,
    weight_fn,
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Single-source minimax (bottleneck) paths.

    dist[v] = minimal possible maximum-edge-weight along any path from source to v.
    Also returns reconstructed node paths.
    """
    dist: Dict[str, float] = {source: 0.0}
    prev: Dict[str, str] = {}
    # (current best bottleneck value, node)
    pq = [(0.0, source)]

    while pq:
        d_u, u = heapq.heappop(pq)
        # stale entry
        if d_u != dist.get(u, None):
            continue

        for v, attrs in G[u].items():
            w_uv = float(weight_fn(u, v, attrs))
            cand = max(d_u, w_uv)
            if cand < dist.get(v, float("inf")):
                dist[v] = cand
                prev[v] = u
                heapq.heappush(pq, (cand, v))

    # reconstruct paths
    paths: Dict[str, List[str]] = {}
    for v in dist.keys():
        if v == source:
            paths[v] = [source]
            continue
        # follow prev pointers back to source
        cur = v
        path_rev = [cur]
        while cur != source:
            cur = prev[cur]  # will exist if reachable
            path_rev.append(cur)
        paths[v] = list(reversed(path_rev))

    return dist, paths


def compute_and_save_paths_to_gene_of_interest_split(
    G_view: nx.DiGraph,
    terminals: List[str],
    output_dir: str,
    gene_of_interest: str,
    *,
    logger=None,
    steiner_added: Optional[Set[str]] = None,
    # filenames
    minsum_summary_filename: Optional[str] = None,
    minsum_full_filename: Optional[str] = None,
    minmax_summary_filename: Optional[str] = None,
    minmax_full_filename: Optional[str] = None,
) -> Tuple[str, str, str, str, str, dict]:
    """
    Compute TWO path notions from each terminal to gene_of_interest on an UNDIRECTED view:

      (A) Min-sum shortest path (Dijkstra, weight = p_value)
      (B) Min-max (bottleneck/minimax) path (minimize max p_value along path)

    Writes FIVE CSVs:
      - minsum summary
      - minsum full
      - minmax summary
      - minmax full
      - combined summary

    Returns:
      (
        minsum_summary_path,
        minsum_full_path,
        minmax_summary_path,
        minmax_full_path,
        combined_summary_path,
        metrics
      )
    """
    logger = logger or (lambda *a, **k: None)
    os.makedirs(output_dir, exist_ok=True)

    G_u = G_view.to_undirected()
    steiner_added = set(steiner_added or [])
    eps = 1e-300

    # Default filenames
    if minsum_summary_filename is None:
        minsum_summary_filename = f"paths_to_{gene_of_interest}.minsum.summary.csv"
    if minsum_full_filename is None:
        minsum_full_filename = f"paths_to_{gene_of_interest}.minsum.full.csv"
    if minmax_summary_filename is None:
        minmax_summary_filename = f"paths_to_{gene_of_interest}.minmax.summary.csv"
    if minmax_full_filename is None:
        minmax_full_filename = f"paths_to_{gene_of_interest}.minmax.full.csv"

    minsum_summary_path = os.path.join(output_dir, minsum_summary_filename)
    minsum_full_path = os.path.join(output_dir, minsum_full_filename)
    minmax_summary_path = os.path.join(output_dir, minmax_summary_filename)
    minmax_full_path = os.path.join(output_dir, minmax_full_filename)

    # If GOI is absent, emit NA rows for all terminals
    if gene_of_interest not in G_u:
        unique_terminals = list(dict.fromkeys(terminals))

        df_minsum_summary = pd.DataFrame({
            "terminal": unique_terminals,
            "length_edges": [np.nan] * len(unique_terminals),
            "total_weight": [np.nan] * len(unique_terminals),
        })
        df_minsum_full = pd.DataFrame({
            "terminal": unique_terminals,
            "length_edges": [np.nan] * len(unique_terminals),
            "total_weight": [np.nan] * len(unique_terminals),
            "steiner_hops_used": [np.nan] * len(unique_terminals),
            "score_neglog10_path": [np.nan] * len(unique_terminals),
            "path": ["No Path Available"] * len(unique_terminals),
        })
        df_minmax_summary = pd.DataFrame({
            "terminal": unique_terminals,
            "length_edges": [np.nan] * len(unique_terminals),
            "bottleneck_p": [np.nan] * len(unique_terminals),
        })
        df_minmax_full = pd.DataFrame({
            "terminal": unique_terminals,
            "length_edges": [np.nan] * len(unique_terminals),
            "bottleneck_p": [np.nan] * len(unique_terminals),
            "steiner_hops_used": [np.nan] * len(unique_terminals),
            "path": ["No Path Available"] * len(unique_terminals),
        })

        df_minsum_summary.to_csv(minsum_summary_path, index=False)
        df_minsum_full.to_csv(minsum_full_path, index=False)
        df_minmax_summary.to_csv(minmax_summary_path, index=False)
        df_minmax_full.to_csv(minmax_full_path, index=False)

        combined_summary_path = merge_summary_only(
            minsum_summary_path=minsum_summary_path,
            minmax_summary_path=minmax_summary_path,
            terminals=unique_terminals,
            output_dir=output_dir,
            gene_of_interest=gene_of_interest,
        )

        metrics = {
            "path_n": 0,
            "minsum_cost_sum": 0.0,
            "minsum_cost_mean": 0.0,
            "minsum_neglog10_sum": 0.0,
            "minsum_neglog10_mean": 0.0,
            "minmax_bottleneck_sum": 0.0,
            "minmax_bottleneck_mean": 0.0,
        }

        return (
            minsum_summary_path,
            minsum_full_path,
            minmax_summary_path,
            minmax_full_path,
            combined_summary_path,
            metrics,
        )

    def w(u, v, d):
        """Return p-value edge weight for path scoring."""
        return float(d.get("p_value", 1.0))

    # --- (A) Min-sum (Dijkstra) ---
    try:
        minsum_lengths, minsum_paths = nx.single_source_dijkstra(
            G_u,
            source=gene_of_interest,
            weight=w,
        )
    except nx.NetworkXNoPath:
        minsum_lengths, minsum_paths = {}, {}

    # --- (B) Min-max (bottleneck/minimax) ---
    minmax_lengths, minmax_paths = minimax_single_source_paths(
        G_u,
        source=gene_of_interest,
        weight_fn=w,
    )

    rows_minsum_full = []
    rows_minmax_full = []

    agg_n = 0
    agg_minsum_cost_sum = 0.0
    agg_minsum_neglog10_sum = 0.0
    agg_minmax_bottleneck_sum = 0.0

    seen = set()
    for t in terminals:
        if t in seen:
            continue
        seen.add(t)

        reachable_minsum = (t in G_u and t in minsum_paths)
        reachable_minmax = (t in G_u and t in minmax_paths)

        # ---------- MIN-SUM ----------
        if not reachable_minsum:
            rows_minsum_full.append({
                "terminal": t,
                "length_edges": np.nan,
                "total_weight": np.nan,
                "steiner_hops_used": np.nan,
                "score_neglog10_path": np.nan,
                "path": "No Path Available",
            })
        else:
            p_nodes_gr = minsum_paths[t]          # GOI -> ... -> t
            p_nodes = list(reversed(p_nodes_gr))  # t -> ... -> GOI
            length_edges = max(0, len(p_nodes) - 1)

            total_p = 0.0
            total_neglog10 = 0.0
            for i in range(len(p_nodes) - 1):
                u, v = p_nodes[i], p_nodes[i + 1]
                p_val = float(G_u[u][v].get("p_value", 1.0))
                total_p += p_val
                total_neglog10 += -math.log10(max(p_val, eps))

            internal_nodes = p_nodes[1:-1] if len(p_nodes) >= 3 else []
            steiner_hops_used = sum(1 for n in internal_nodes if n in steiner_added)

            rows_minsum_full.append({
                "terminal": t,
                "length_edges": length_edges,
                "total_weight": round(total_p, 6),
                "steiner_hops_used": steiner_hops_used,
                "score_neglog10_path": round(total_neglog10, 6),
                "path": ";".join(p_nodes),
            })

            agg_minsum_cost_sum += total_p
            agg_minsum_neglog10_sum += total_neglog10

        # ---------- MIN-MAX ----------
        if not reachable_minmax:
            rows_minmax_full.append({
                "terminal": t,
                "length_edges": np.nan,
                "bottleneck_p": np.nan,
                "steiner_hops_used": np.nan,
                "path": "No Path Available",
            })
        else:
            p_nodes_gr = minmax_paths[t]          # GOI -> ... -> t
            p_nodes = list(reversed(p_nodes_gr))  # t -> ... -> GOI
            length_edges = max(0, len(p_nodes) - 1)

            bneck = 0.0
            for i in range(len(p_nodes) - 1):
                u, v = p_nodes[i], p_nodes[i + 1]
                p_val = float(G_u[u][v].get("p_value", 1.0))
                bneck = max(bneck, p_val)

            internal_nodes = p_nodes[1:-1] if len(p_nodes) >= 3 else []
            steiner_hops_used = sum(1 for n in internal_nodes if n in steiner_added)

            rows_minmax_full.append({
                "terminal": t,
                "length_edges": length_edges,
                "bottleneck_p": round(bneck, 6),
                "steiner_hops_used": steiner_hops_used,
                "path": ";".join(p_nodes),
            })

            agg_minmax_bottleneck_sum += bneck

        if reachable_minsum or reachable_minmax:
            agg_n += 1

    # Build full DFs
    df_minsum_full = pd.DataFrame(rows_minsum_full).replace([np.inf, -np.inf], np.nan)
    df_minmax_full = pd.DataFrame(rows_minmax_full).replace([np.inf, -np.inf], np.nan)

    # Sort full outputs independently
    df_minsum_full = df_minsum_full.sort_values(
        by=["total_weight", "length_edges", "terminal"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    df_minmax_full = df_minmax_full.sort_values(
        by=["bottleneck_p", "length_edges", "terminal"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    # Summary outputs
    df_minsum_summary = df_minsum_full[["terminal", "length_edges", "total_weight"]].copy()
    df_minmax_summary = df_minmax_full[["terminal", "length_edges", "bottleneck_p"]].copy()

    # Save original 4 files
    df_minsum_summary.to_csv(minsum_summary_path, index=False)
    df_minsum_full.to_csv(minsum_full_path, index=False)
    df_minmax_summary.to_csv(minmax_summary_path, index=False)
    df_minmax_full.to_csv(minmax_full_path, index=False)

    # Save merged summary
    combined_summary_path = merge_summary_only(
        minsum_summary_path=minsum_summary_path,
        minmax_summary_path=minmax_summary_path,
        terminals=list(seen),
        output_dir=output_dir,
        gene_of_interest=gene_of_interest,
    )

    metrics = {
        "path_n": int(agg_n),
        "minsum_cost_sum": float(agg_minsum_cost_sum),
        "minsum_cost_mean": float(agg_minsum_cost_sum / max(1, agg_n)),
        "minsum_neglog10_sum": float(agg_minsum_neglog10_sum),
        "minsum_neglog10_mean": float(agg_minsum_neglog10_sum / max(1, agg_n)),
        "minmax_bottleneck_sum": float(agg_minmax_bottleneck_sum),
        "minmax_bottleneck_mean": float(agg_minmax_bottleneck_sum / max(1, agg_n)),
    }

    return (
        minsum_summary_path,
        minsum_full_path,
        minmax_summary_path,
        minmax_full_path,
        combined_summary_path,
        metrics,
    )

import os
import pandas as pd
import numpy as np


def merge_summary_only(
    *,
    minsum_summary_path: str,
    minmax_summary_path: str,
    terminals: list[str],
    output_dir: str,
    gene_of_interest: str,
) -> str:
    """
    Merge minsum + minmax summary files into one combined summary.

    Ensures:
    - All terminals appear in output
    - Terminals missing from the CSVs get NA metrics
    - Missing terminals are placed at the bottom

    Output columns:
        terminal,
        minsum_length_edges,
        minsum_totalweight,
        minmax_length_edges,
        bottleneck_p
    """

    df_minsum = pd.read_csv(minsum_summary_path)
    df_minmax = pd.read_csv(minmax_summary_path)

    # Rename columns for clarity
    df_minsum = df_minsum.rename(columns={
        "length_edges": "minsum_length_edges",
        "total_weight": "minsum_totalweight",
    })

    df_minmax = df_minmax.rename(columns={
        "length_edges": "minmax_length_edges",
    })

    # Merge summaries
    df = pd.merge(
        df_minsum,
        df_minmax,
        on="terminal",
        how="outer",
    )

    # Ensure ALL terminals exist in output
    df_all = pd.DataFrame({"terminal": terminals})
    df = df_all.merge(df, on="terminal", how="left")

    # helper column to push NA rows to bottom
    df["_all_na"] = (
        df["minsum_totalweight"].isna() &
        df["bottleneck_p"].isna()
    )

    df = df.sort_values(
        by=["_all_na", "bottleneck_p", "minsum_totalweight", "terminal"],
        ascending=[True, True, True, True],
        na_position="last",
    )

    df = df.drop(columns="_all_na")

    # enforce column order
    df = df.reindex(columns=[
        "terminal",
        "minsum_length_edges",
        "minsum_totalweight",
        "minmax_length_edges",
        "bottleneck_p",
    ])

    output_path = os.path.join(
        output_dir,
        f"paths_to_{gene_of_interest}.combined.summary.csv"
    )

    df.to_csv(output_path, index=False)

    return output_path
def add_quantile_flags(df_full: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    """
    Adds quantile-based inclusion flags for min-sum and min-max scores.
    q=0.05 means "keep best 5%" (lowest scores).
    """
    out = df_full.copy()

    # Min-sum quantile threshold (lower is better)
    if "total_weight_minsum" in out.columns:
        thr_minsum = out["total_weight_minsum"].dropna().quantile(q)
        out["minsum_q"] = q
        out["minsum_thr"] = thr_minsum
        out["keep_minsum_q"] = out["total_weight_minsum"] <= thr_minsum
    else:
        out["minsum_q"] = q
        out["minsum_thr"] = np.nan
        out["keep_minsum_q"] = False

    # Min-max (bottleneck) quantile threshold (lower is better)
    if "bottleneck_p_minmax" in out.columns:
        thr_minmax = out["bottleneck_p_minmax"].dropna().quantile(q)
        out["minmax_q"] = q
        out["minmax_thr"] = thr_minmax
        out["keep_minmax_q"] = out["bottleneck_p_minmax"] <= thr_minmax
    else:
        out["minmax_q"] = q
        out["minmax_thr"] = np.nan
        out["keep_minmax_q"] = False

    return out


def save_gene_of_interest_community_bundle(
    G_full_view: nx.DiGraph,
    partition: Dict[str, int],
    community_colors: Dict[int, Tuple[str, str]],
    output_path: str,
    gene_of_interest: Optional[str] = None,
    *,
    layout: str = "dot",
    graph_attr: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
    steiner_added: Optional[Set[str]] = None,
) -> dict:
    """
    Build & persist the gene_of_interest community-only outputs:

      Folder: <output_path>/<gene_of_interest>_community/

      - graph_<layout>.svg      (one per layout)
      - genes.csv               (basic node stats + is_steiner)
      - edges.csv               (internal edges)
      - steiner_genes.txt       (optional convenience list)

    Returns small metadata dict (counts & paths). If gene_of_interest missing, returns {}.
    """
    logger = logger or get_logger()

    found = extract_community_nodes(partition, gene_of_interest)
    if not found:
        logger.info("%s not present in partition/graph; skipping community export.", gene_of_interest)
        return {}

    cid, members = found
    keep = [n for n in members if n in G_full_view]
    if not keep:
        logger.info("%s community has no nodes in current view; skipping export.", gene_of_interest)
        return {}

    Gc = G_full_view.subgraph(keep).copy()

    # Prepare output dir: ONE folder per GOI community, independent of layout
    out_dir = os.path.join(output_path, f"{gene_of_interest}_community")
    os.makedirs(out_dir, exist_ok=True)

    layout_tag = (layout or "default").lower().replace(" ", "_")

    # Steiner nodes only within this community
    steiner_added = set(steiner_added or set())
    steiner_in_comm = sorted(set(keep).intersection(steiner_added))

    # Slim partition & colors to this community
    part_c = {n: cid for n in Gc.nodes()}  # single-community map
    colors_c = {cid: community_colors.get(cid, ("#d3d3d3", "1"))}

    # Render SVG with Steiner highlights (grey fill like main graph)
    dot_c = create_graphviz_dot(
        Gc,
        partition=part_c,
        community_colors=colors_c,
        highlight_node=None,
        layout=layout,
        graph_attr=graph_attr or {},
        weighted_edges=False,
        simple_layout=False,
        highlight_new_genes=(bool(steiner_in_comm), steiner_in_comm),
    )

    # One SVG per layout: graph_<layout>.svg
    svg_stem = os.path.join(out_dir, f"graph_{layout_tag}")
    dot_c.render(svg_stem, format="svg", cleanup=True)
    svg_path = f"{svg_stem}.svg"

    # genes.csv (basic stats + is_steiner) – shared across layouts
    genes_path = os.path.join(out_dir, "genes.csv")
    df_genes = pd.DataFrame({
        "gene": list(Gc.nodes()),
        "out_degree": [Gc.out_degree(n) for n in Gc.nodes()],
        "in_degree": [Gc.in_degree(n) for n in Gc.nodes()],
        "is_steiner": [int(n in steiner_in_comm) for n in Gc.nodes()],
    }).sort_values(
        ["is_steiner", "out_degree", "in_degree", "gene"],
        ascending=[False, False, False, True],
    )
    df_genes.to_csv(genes_path, index=False)

    # Optional convenience file: list of Steiner genes in this community
    steiner_list_path = os.path.join(out_dir, "steiner_genes.txt")
    with open(steiner_list_path, "w", encoding="utf-8") as fh:
        for g in steiner_in_comm:
            fh.write(f"{g}\n")

    # edges.csv (internal edges) – shared across layouts
    edges_path = os.path.join(out_dir, "edges.csv")
    rows = []
    for u, v, d in Gc.edges(data=True):
        rows.append({
            "gene1": u,
            "gene2": v,
            "lag": d.get("lag"),
            "p-value": d.get("p_value"),
        })
    pd.DataFrame(rows).to_csv(edges_path, index=False)

    logger.info(
        "Saved %s community | layout=%s | nodes=%d | edges=%d | steiner_in_comm=%d | %s",
        gene_of_interest, layout_tag,
        Gc.number_of_nodes(), Gc.number_of_edges(), len(steiner_in_comm),
        svg_path,
    )

    # Metadata: keep dir + "latest" svg for this layout, plus layout-specific key
    return {
        f"{gene_of_interest}_comm_id": cid,
        f"{gene_of_interest}_comm_nodes": Gc.number_of_nodes(),
        f"{gene_of_interest}_comm_edges": Gc.number_of_edges(),
        f"{gene_of_interest}_comm_dir": out_dir,
        # generic "current layout" svg
        f"{gene_of_interest}_comm_svg": svg_path,
        # layout-specific svg path (so you can inspect later if needed)
        f"{gene_of_interest}_comm_svg_{layout_tag}": svg_path,
        f"{gene_of_interest}_comm_genes_csv": genes_path,
        f"{gene_of_interest}_comm_edges_csv": edges_path,
        f"{gene_of_interest}_comm_steiner_genes_txt": steiner_list_path,
        f"{gene_of_interest}_comm_steiner_nodes": len(steiner_in_comm),
    }



# --- helpers ---

from typing import Optional

def select_genes_from_frequency(
    freq_csv_path: str,
    *,
    min_frequency: Optional[float] = None,
    top_n: Optional[int] = None,
    top_percent: Optional[float] = None,
) -> list[str]:
    """
    Read 'Gene,Coassociation Frequency' CSV and select genes by one of:
      - min_frequency       (keep frequency >= threshold)
      - top_n               (keep N highest frequency)
      - top_percent (0-1)   (keep ceil(percent * total) highest)
    You may combine min_frequency with top_n/top_percent to clip AFTER filtering.
    """
    df = pd.read_csv(freq_csv_path)
    gene_col = next(c for c in df.columns if c.strip().lower() in {"gene", "symbol", "gene_symbol"})
    freq_col = next(c for c in df.columns if "freq" in c.strip().lower())

    df = df.rename(columns={gene_col: "gene", freq_col: "frequency"})

    # 1) optional threshold first
    if min_frequency is not None:
        df = df[df["frequency"] >= float(min_frequency)]

    # 2) ranking selection
    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)

    if top_percent is not None:
        assert 0 < float(top_percent) <= 1.0, "top_percent must be in (0, 1]."
        k = max(1, int(math.floor(float(top_percent) * len(df))))
        df = df.head(k)
    elif top_n is not None:
        df = df.head(int(top_n))
    else:
        if min_frequency is None:
            raise ValueError("Provide at least one of: min_frequency, top_n, or top_percent.")

    return df["gene"].tolist()



# Backward-compatible wrapper (keeps your current callers working)
def load_genes_by_frequency(freq_csv_path: str, min_frequency: float, top_n: Optional[int] = None) -> list[str]:
    """Backward-compatible wrapper for frequency-based gene selection."""
    return select_genes_from_frequency(freq_csv_path, min_frequency=min_frequency, top_n=top_n)


def p_slug(p: float) -> str:
    """Return a compact p-value slug for folder names."""
    # e.g. 0.0015 -> "p00015"
    return "p" + str(p).replace(".", "")

def freq_slug(x: float) -> str:
    """Return a compact frequency slug for folder names."""
    # e.g. 0.95 -> "freq095"; 1.0 -> "freq100"
    return f"freq{int(round(x * 100)):03d}"

def file_slug(path: str) -> str:
    """Return a filesystem-friendly stem for a path."""
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def selection_desc(summary: dict) -> str:
    """Return a human-readable description of a gene-selection summary."""
    fthr = summary.get("freq_threshold", None)
    tn   = summary.get("top_n", None)
    tp   = summary.get("top_percent", None)
    if fthr is not None:
        try:
            return f"freq>={float(fthr):.2f}"
        except Exception:
            return f"freq>={fthr}"
    if tn is not None:
        try:
            return f"top_n={int(tn)}"
        except Exception:
            return f"top_n={tn}"
    if tp is not None:
        try:
            pct = float(tp) * 100.0
            return f"top_percent={int(pct)}%" if pct.is_integer() else f"top_percent={pct:.2f}%"
        except Exception:
            return f"top_percent={tp}"
    return "selection=?"

def selection_tag(*, freq_threshold=None, top_n=None, top_percent=None) -> str:
    """Folder-friendly tag for the selection."""
    if freq_threshold is not None:
        return freq_slug(float(freq_threshold))  # e.g., freq095
    if top_n is not None:
        return f"topn_{int(top_n):05d}"
    if top_percent is not None:
        return f"toppct_{int(round(float(top_percent)*100)):03d}"
    return "selection_unknown"

def selection_kwargs_desc(kwargs: dict) -> str:
    """Human-friendly selection string for logs."""
    if kwargs.get("freq_threshold") is not None:
        return f"freq>={float(kwargs['freq_threshold']):.2f}"
    if kwargs.get("top_n") is not None:
        return f"top_n={int(kwargs['top_n'])}"
    if kwargs.get("top_percent") is not None:
        pct = float(kwargs["top_percent"]) * 100.0
        return f"top_percent={int(pct)}%" if pct.is_integer() else f"top_percent={pct:.2f}%"
    return "selection=?"


if __name__ == "__main__":
    t0 = time.perf_counter()
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # ---------------------------------------
    # CONFIG: link each CSV to its p-threshold and a LIST of frequency files
    # You can also override freq_thresholds/layouts per CSV.
    # ---------------------------------------
    datasets = {
        
        # "gc_r_p00015_503811of190674672.csv": {
        #    "p_threshold": 0.0015,
        #    "freq_files": [
        #        "SM7_00015_7200_runs.csv",
        #   ],
        #    #"number_of_runs": 1000
        # },
        
        "granger_causality_results_truncated_benito_human.csv": {
            "p_threshold": 0.0015,
            "gene_of_interest": "ZEB2",
            "freq_files": [
                "ZEB2_coassoc_00015_110906_runs.csv",
            ],
            #"number_of_runs": 1000     
        },
        "granger_causality_results_truncated_benito_gorilla.csv": {
            "p_threshold": 0.002,
            "gene_of_interest": "ZEB2",
            "freq_files": [
                "ZEB2_coassoc_0002_110906_runs.csv",
            ],
            #"number_of_runs": 1000     
        },
        "granger_causality_results_truncated.csv": {
            "p_threshold": 0.0015,
            "gene_of_interest": "MECP2",
            "freq_files": [
                "MECP2_coassoc_00015_6000_runs.csv",
            ],
            #"number_of_runs": 1000     
        } 
    }

    # GLOBAL defaults (used if a dataset doesn't override)
    #global_freq_thresholds = [0.95, 0.90]
    global_freq_thresholds = None
    global_layouts = ["fdp"]
    steiner_hop_penalty = 0.00015
    modes = [
        #"induced", 
        "mst", 
        #"steiner"
    ]

    graph_attr = {
        #"rotate": "90",
        "nodesep": "0.2",
        "ranksep": "0.2",
        "pad": "0.2",
        "outputorder": "edgesfirst",
    }

    consensus_step = "2nd_explore_45"





    # ---------------------------------------
    # RUN
    # ---------------------------------------

    logger = get_logger()

    for csv_path, cfg in datasets.items():
        p_threshold     = float(cfg["p_threshold"])
        freq_files      = cfg.get("freq_files", [])
        gene_of_interest= cfg.get("gene_of_interest", None)
        freq_thresholds = cfg.get("freq_thresholds", global_freq_thresholds)
        layouts         = cfg.get("layouts", global_layouts)
        number_of_runs  = int(cfg.get("number_of_runs", 5000))
        top_n           = cfg.get("top_n", None)

        # (optional) make sure this prints on console
        logger.info("=== Dataset ===\nCSV: %s\nP: %.6f\nFreq files: %s\n",
                    csv_path, p_threshold, freq_files,
                    extra={"event":"dataset","run":"-"})

        for layout in layouts:
            base_static = dict(
                csv_file=csv_path,
                p_threshold=p_threshold,
                layout=layout,
                graph_attr=graph_attr,
                highlight_node=None,
                number_of_runs=number_of_runs,
                simple_layout=False,
                gene_of_interest=gene_of_interest,
            )
            if top_n is not None:
                base_static["top_n"] = int(top_n)

            start = Decimal(f"{p_threshold}")
            step = Decimal("0.0001")
            start = Decimal("0.0002")
            pvalues_to_check = []
            x = start
            while x > 0:
                pvalues_to_check.append(float(x))  # cast back to float if needed
                x -= step
            #pvalues_to_check = [start] # to only do starting p-value
            for freq_file in freq_files:
                freq_file_tag      = file_slug(freq_file)
                base_for_freq_file = dict(base_static, freq_csv=freq_file)

                # Choose selections to iterate: either the provided freq_thresholds, or a single top-%/top-N selection
                # Example below: we run only top_percent=0.05. Adjust as needed (top_n or multiple values).
                selection_list = []
                if freq_thresholds:  # hard frequency(s)
                    for fthr in freq_thresholds:
                        selection_list.append({"freq_threshold": float(fthr)})
                else:
                    # No hard frequencies configured → run a top-% selection here (change as you like)
                    selection_list.append({"top_percent": 0.05})
                    # Or: selection_list.append({"top_n": 1000})

                for sel in selection_list:
                    # Build output base using a clean, folder-friendly tag
                    sel_tag = selection_tag(**sel)
                    out_base = os.path.join(
                        "network", consensus_step, p_slug(p_threshold),
                        freq_file_tag, sel_tag,
                    )

                    for mode in modes:
                        output_path = os.path.join(out_base, mode)
                        run_log     = os.path.join(output_path, "run.log")
                        os.makedirs(output_path, exist_ok=True)

                        run_id = f"{file_slug(csv_path)}|{file_slug(freq_file)}|{layout}|{p_slug(p_threshold)}|{sel_tag}|{mode}"

                        with run_logger(logger, run_id, run_log_path=run_log) as rlog:
                            rlog.info(
                                "Layout=%s | Coassoc=%s | %s | Output=%s",
                                layout, freq_file_tag, selection_kwargs_desc(sel), output_path
                            )

                            kwargs = dict(
                                output_path=output_path,
                                mode=mode,
                                logger=rlog,
                                **base_for_freq_file,
                                **sel,  # inject selection (freq_threshold or top_n/top_percent)
                            )

                            if mode == "steiner":
                                kwargs.update(
                                    steiner_hops=1,
                                    steiner_p_factors=pvalues_to_check,
                                    steiner_max_nodes=20000,
                                    steiner_hop_penalty=steiner_hop_penalty,
                                )

                            try:
                                summary = visualize_gene_network(**kwargs)
                                if not summary:
                                    rlog.info(
                                        "EMPTY | %s | layout=%s | %s | saved=%s",
                                        mode, layout, selection_kwargs_desc(sel), output_path,
                                        extra={"event":"summary"}
                                    )
                                    continue

                                # Persist metadata
                                write_manifest(output_path, summary)
                                append_summary(output_path, summary)

                                # Compose cost string
                                cost_str = ""
                                if summary.get("cost_p_sum") is not None:
                                    denom = max(1, summary.get("paths_n", 1) or 1)
                                    avg_p = summary["cost_p_sum"] / denom
                                    cost_str = (
                                        " | path_cost_p_sum={:.6g} (mean {:.3g})"
                                        " | path_score_neglog10_sum={:.2f} (mean {:.2f})"
                                    ).format(
                                        summary["cost_p_sum"],
                                        avg_p,
                                        summary.get("score_neglog10_sum", 0.0),
                                        summary.get("score_neglog10_mean_per_path", 0.0),
                                    )

                                rlog.info(
                                    "OK | %s | layout=%s | %s | input=%d | nodes=%d | edges=%d%s | saved=%s",
                                    summary.get("mode"), summary.get("layout"), selection_kwargs_desc(sel),
                                    int(summary.get("input_genes", 0) or 0),
                                    int(summary.get("nodes", 0) or 0),
                                    int(summary.get("edges", 0) or 0),
                                    cost_str,
                                    os.path.join(output_path, "graph.svg"),
                                    extra={"event": "summary"}
                                )

                            except Exception as e:
                                rlog.exception("Run failed: %s", e)
                            finally:
                                highlight_new_genes = (False, [])
                                
    logger.info("Total run time: %.2fs", time.perf_counter() - t0)
