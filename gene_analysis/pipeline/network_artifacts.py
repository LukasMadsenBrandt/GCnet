"""Write inspectable network artifacts from significant Granger edges."""

from __future__ import annotations

import csv
import html
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import networkx as nx

from gene_analysis.analysis.granger import collect_significant_edges
from gene_analysis.analysis.network import create_network
from gene_analysis.io.paths import resolve_existing_path

try:
    import community as community_louvain
except Exception:  # pragma: no cover - optional import is validated through fallbacks
    community_louvain = None

GRAPHVIZ_LAYOUTS = ("dot", "neato", "fdp", "sfdp", "circo", "twopi")


def build_significant_network(
    gc_csv: str | Path,
    *,
    p_threshold: float,
    gene_of_interest: str,
) -> nx.DiGraph:
    """Build a directed significant-edge network from a Granger CSV."""
    edges = collect_significant_edges(
        None,
        p_value_threshold=p_threshold,
        file=True,
        filepath=resolve_existing_path(gc_csv),
        starting_genes=[gene_of_interest],
        higher_threshold_for_starting_genes=p_threshold,
    )
    return create_network(edges)


def summarize_network(graph: nx.DiGraph, *, gene_of_interest: str) -> dict[str, int | float | str | bool]:
    """Compute compact network metrics for pipeline audit manifests."""
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    weak_components = list(nx.weakly_connected_components(graph)) if node_count else []
    strong_components = list(nx.strongly_connected_components(graph)) if node_count else []
    goi_present = gene_of_interest in graph
    density = nx.density(graph) if node_count > 1 else 0.0

    return {
        "genes_total": node_count,
        "edges_total": edge_count,
        "density": density,
        "weak_components": len(weak_components),
        "strong_components": len(strong_components),
        "largest_weak_component_genes": max((len(component) for component in weak_components), default=0),
        "largest_strong_component_genes": max((len(component) for component in strong_components), default=0),
        "average_in_degree": edge_count / node_count if node_count else 0.0,
        "average_out_degree": edge_count / node_count if node_count else 0.0,
        "gene_of_interest": gene_of_interest,
        "gene_of_interest_present": goi_present,
        "gene_of_interest_in_degree": graph.in_degree(gene_of_interest) if goi_present else 0,
        "gene_of_interest_out_degree": graph.out_degree(gene_of_interest) if goi_present else 0,
        "gene_of_interest_total_degree": graph.degree(gene_of_interest) if goi_present else 0,
    }


def summarize_subcommunities(
    graph: nx.DiGraph,
    partition: dict[str, int],
    *,
    gene_of_interest: str,
) -> dict[str, int | str | bool | list[str]]:
    """Summarize the visual subcommunities detected within a network artifact."""
    communities: dict[int, list[str]] = defaultdict(list)
    for node in graph.nodes():
        community_id = partition.get(str(node))
        if community_id is not None:
            communities[int(community_id)].append(str(node))

    for genes in communities.values():
        genes.sort()

    goi_node = _matching_node(graph, gene_of_interest)
    goi_community = partition.get(goi_node) if goi_node is not None else None
    goi_genes = communities.get(int(goi_community), []) if goi_community is not None else []
    largest = max((len(genes) for genes in communities.values()), default=0)

    return {
        "subcommunity_count": len(communities),
        "largest_subcommunity_genes": largest,
        "gene_of_interest_subcommunity": int(goi_community) if goi_community is not None else "",
        "gene_of_interest_subcommunity_genes": len(goi_genes),
        "gene_of_interest_subcommunity_gene_list": goi_genes,
    }


def detect_subcommunities(graph: nx.DiGraph) -> dict[str, int]:
    """Detect deterministic subcommunities for visual coloring within a network."""
    if graph.number_of_nodes() == 0:
        return {}
    undirected = graph.to_undirected()
    if graph.number_of_edges() == 0:
        return {str(node): idx for idx, node in enumerate(sorted(map(str, graph.nodes())))}
    if community_louvain is not None:
        try:
            return {
                str(node): int(community)
                for node, community in community_louvain.best_partition(undirected, random_state=0).items()
            }
        except Exception:
            pass
    return _component_partition(graph)


def select_top_consensus_genes(
    frequency_csv: str | Path,
    *,
    gene_of_interest: str,
    top_fraction: float = 0.05,
) -> list[str]:
    """Select the top coassociation-frequency genes, always including the GOI."""
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be greater than 0 and less than or equal to 1.")

    rows: list[tuple[str, float]] = []
    with open(resolve_existing_path(frequency_csv), "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gene = (row.get("Gene") or row.get("gene") or "").strip()
            if not gene:
                continue
            raw_frequency = row.get("Coassociation Frequency") or row.get("coassociation frequency") or ""
            try:
                frequency = float(raw_frequency)
            except ValueError:
                continue
            rows.append((gene, frequency))

    if not rows:
        return [gene_of_interest]

    rows.sort(key=lambda item: (-item[1], item[0].casefold()))
    selected_count = max(1, math.ceil(len(rows) * top_fraction))
    selected = [gene for gene, _ in rows[:selected_count]]
    if not any(gene.upper() == gene_of_interest.upper() for gene in selected):
        selected.insert(0, gene_of_interest)
    return _dedupe_preserve_order(selected)


def write_network_svg(
    graph: nx.DiGraph,
    output_file: str | Path,
    *,
    gene_of_interest: str,
    renderer: str = "networkx",
    layout: str = "dot",
    partition: dict[str, int] | None = None,
) -> Path:
    """Write an SVG network preview using the selected renderer."""
    renderer = str(renderer).lower()
    if renderer == "networkx":
        return write_networkx_svg(graph, output_file, gene_of_interest=gene_of_interest, partition=partition)
    if renderer == "graphviz":
        return write_graphviz_svg(graph, output_file, gene_of_interest=gene_of_interest, layout=layout, partition=partition)
    raise ValueError("SVG renderer must be 'networkx' or 'graphviz'.")


def write_networkx_svg(
    graph: nx.DiGraph,
    output_file: str | Path,
    *,
    gene_of_interest: str,
    partition: dict[str, int] | None = None,
) -> Path:
    """Write a deterministic lightweight SVG preview for quick network inspection."""
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    height = 700
    margin = 70
    nodes = sorted(map(str, graph.nodes()))

    if not nodes:
        output.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="700" viewBox="0 0 900 700">'
            '<rect width="100%" height="100%" fill="white"/>'
            '<text x="450" y="350" text-anchor="middle" font-family="Arial" font-size="18">'
            "No significant network edges</text></svg>\n",
            encoding="utf-8",
        )
        return output

    undirected = graph.to_undirected()
    if len(nodes) == 1:
        layout = {nodes[0]: (0.5, 0.5)}
    else:
        layout = nx.spring_layout(undirected, seed=42, scale=1.0)

    xs = [float(layout[node][0]) for node in nodes]
    ys = [float(layout[node][1]) for node in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    positions: dict[str, tuple[float, float]] = {}
    for node in nodes:
        x, y = layout[node]
        px = margin + ((float(x) - min_x) / span_x) * (width - 2 * margin)
        py = margin + ((float(y) - min_y) / span_y) * (height - 2 * margin)
        positions[node] = (px, py)

    max_degree = max((graph.degree(node) for node in nodes), default=1)
    partition = partition or _component_partition(graph)
    community_colors = _community_colors(partition)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="700" viewBox="0 0 900 700">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<g class="edges" stroke="#6b7280" stroke-opacity="0.65" fill="none">',
    ]
    for source, target, data in graph.edges(data=True):
        x1, y1 = positions[str(source)]
        x2, y2 = positions[str(target)]
        p_value = data.get("p_value", "")
        title = html.escape(f"{source} -> {target} p={p_value}")
        lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke-width="1.4" marker-end="url(#arrow)"><title>{title}</title></line>'
        )
    lines.append("</g>")
    lines.append('<g class="nodes" font-family="Arial, sans-serif" font-size="12">')
    for node in nodes:
        x, y = positions[node]
        radius = 8 + 10 * math.sqrt(graph.degree(node) / max_degree) if max_degree else 8
        is_goi = node.upper() == gene_of_interest.upper()
        fill = community_colors.get(partition.get(node), "#2563eb")
        stroke = "#7f1d1d" if is_goi else "#1e3a8a"
        stroke_width = "3" if is_goi else "1.5"
        label = html.escape(node)
        lines.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{y + radius + 14:.2f}" text-anchor="middle" fill="#111827">{label}</text>'
        )
    lines.append("</g>")
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_graphviz_svg(
    graph: nx.DiGraph,
    output_file: str | Path,
    *,
    gene_of_interest: str,
    layout: str = "dot",
    weighted_edges: bool = False,
    partition: dict[str, int] | None = None,
) -> Path:
    """Write a Graphviz SVG using the legacy dashboard visual style."""
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "digraph G {",
        '  graph [rankdir="LR", overlap="false", splines="true", bgcolor="white", tooltip=""];',
        '  node [fontname="Arial"];',
        '  edge [fontname="Arial"];',
    ]
    if graph.number_of_nodes() == 0:
        lines.append('  "empty" [label="No significant network edges", shape="plaintext", fontsize="18"];')
        lines.append("}")
        return _render_graphviz_dot("\n".join(lines), output, layout=layout)

    partition = partition or _component_partition(graph)
    community_colors = _community_colors(partition)
    outdegrees = [graph.out_degree(node) for node in graph.nodes()]
    node_sizes = _normalize(outdegrees, min_size=1.0, max_size=2.0) if weighted_edges else [1.0] * len(outdegrees)

    for idx, node in enumerate(graph.nodes()):
        community = partition.get(node)
        fill_hex = community_colors.get(community, "#d3d3d3")
        penwidth = "3" if str(node).upper() == gene_of_interest.upper() else "1"
        out_edges = list(graph.out_edges(node, data=True))
        out_edges_info = ", ".join(
            f"{target}: ({_format_p_value(data.get('p_value'))})" for _, target, data in out_edges
        )
        hover_text = html.escape(
            f"{node} may granger cause {len(out_edges)} gene(s), formatted: Gene: (p-value)\n{out_edges_info}"
        )
        lines.append(
            f"  {_dot_id(str(node))} "
            f'[label={_dot_value(_wrap_label(str(node), max_chars=7))}, shape="circle", style="filled", '
            f'fillcolor="{fill_hex}", color="black", penwidth="{penwidth}", tooltip={_dot_value(hover_text)}, '
            f'width="{node_sizes[idx]:.3f}", height="{node_sizes[idx]:.3f}", fixedsize="true", margin="0.1,0.05"];'
        )

    p_values = [float(graph[source][target].get("p_value", 1.0) or 1.0) for source, target in graph.edges()]
    if weighted_edges:
        edge_widths = _normalize([-math.log10(max(p, 1e-300)) for p in p_values], min_size=1.0, max_size=10.0)
    else:
        edge_widths = [3.0] * len(p_values)

    for idx, (source, target) in enumerate(graph.edges()):
        data = graph[source][target]
        lag = data.get("lag", "NA")
        p_value = data.get("p_value")
        edge_color = community_colors.get(partition.get(source), "#d3d3d3")
        if str(source).upper() == gene_of_interest.upper():
            edge_color = "black"
            width = edge_widths[idx] + 3.0
        else:
            width = edge_widths[idx]
        hover_text = html.escape(
            f"{source} may Granger Cause {target} at lag {lag} with p={_format_p_value(p_value)}"
        )
        lines.append(
            f"  {_dot_id(str(source))} -> {_dot_id(str(target))} "
            f'[color="{edge_color}", penwidth="{width:.3f}", tooltip={_dot_value(hover_text)}];'
        )

    lines.append("}")
    return _render_graphviz_dot("\n".join(lines), output, layout=layout)


def render_network_artifacts_with_renderer(
    *,
    run_dir: str | Path,
    renderer: str = "graphviz",
    stages: tuple[str, ...] = ("seed", "seed_top_consensus", "probe", "expanded", "expanded_top_consensus"),
    layouts: tuple[str, ...] = ("dot",),
    skip_failed_layouts: bool = False,
) -> dict[str, Path]:
    """Render SVGs later from existing GraphML artifacts in a completed run."""
    run = Path(run_dir)
    stage_specs = {
        "seed": ("02_seed_consensus", "seed_network.graphml", "seed_network.svg"),
        "seed_top_consensus": (
            "02_seed_consensus",
            "seed_top_consensus_network.graphml",
            "seed_top_consensus_network.svg",
        ),
        "probe": ("05_expanded_genes", "probe_network.graphml", "probe_network.svg"),
        "expanded": ("07_expanded_consensus", "expanded_network.graphml", "expanded_network.svg"),
        "expanded_top_consensus": (
            "07_expanded_consensus",
            "expanded_top_consensus_network.graphml",
            "expanded_top_consensus_network.svg",
        ),
    }
    outputs: dict[str, Path] = {}
    gene_of_interest = _gene_of_interest_from_run(run)
    layouts = tuple(layout.lower() for layout in layouts)
    if renderer == "graphviz":
        unknown = sorted(set(layouts) - set(GRAPHVIZ_LAYOUTS))
        if unknown:
            raise ValueError(f"Unknown Graphviz layout(s): {', '.join(unknown)}")
    for stage in stages:
        stage_dir, graphml_name, svg_name = stage_specs[stage]
        graphml = run / stage_dir / graphml_name
        if not graphml.exists():
            continue
        graph = nx.read_graphml(graphml)
        for index, layout in enumerate(layouts):
            suffix = "" if len(layouts) == 1 else f"_{layout}"
            svg = run / stage_dir / svg_name.replace(".svg", f"{suffix}.svg")
            try:
                outputs[f"{stage}_network{suffix}_svg"] = write_network_svg(
                    graph,
                    svg,
                    gene_of_interest=gene_of_interest,
                    renderer=renderer,
                    layout=layout,
                )
            except RuntimeError as exc:
                if skip_failed_layouts:
                    print(f"Skipping {stage} layout '{layout}': {exc}", file=sys.stderr)
                    continue
                raise
            figures_dir = run / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            figure_svg = figures_dir / svg.name
            shutil.copyfile(svg, figure_svg)
            outputs[f"{stage}_network{suffix}_figure_svg"] = figure_svg
            if index == 0 and len(layouts) > 1:
                canonical_svg = run / stage_dir / svg_name
                canonical_figure_svg = figures_dir / svg_name
                shutil.copyfile(svg, canonical_svg)
                shutil.copyfile(svg, canonical_figure_svg)
                outputs[f"{stage}_network_svg"] = canonical_svg
                outputs[f"{stage}_network_figure_svg"] = canonical_figure_svg
    return outputs


def write_network_artifacts(
    gc_csv: str | Path,
    *,
    p_threshold: float,
    gene_of_interest: str,
    output_dir: str | Path,
    prefix: str,
    write_svg_preview: bool = True,
    svg_renderer: str = "networkx",
    svg_layout: str = "dot",
) -> dict[str, Path]:
    """Write machine-readable network files and optionally an SVG preview."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = build_significant_network(
        gc_csv,
        p_threshold=p_threshold,
        gene_of_interest=gene_of_interest,
    )

    return write_graph_network_artifacts(
        graph,
        source_gc_csv=gc_csv,
        p_threshold=p_threshold,
        gene_of_interest=gene_of_interest,
        output_dir=out_dir,
        prefix=prefix,
        write_svg_preview=write_svg_preview,
        svg_renderer=svg_renderer,
        svg_layout=svg_layout,
    )


def write_top_consensus_network_artifacts(
    gc_csv: str | Path,
    frequency_csv: str | Path,
    *,
    p_threshold: float,
    gene_of_interest: str,
    output_dir: str | Path,
    prefix: str,
    top_fraction: float = 0.05,
    write_svg_preview: bool = True,
    svg_renderer: str = "networkx",
    svg_layout: str = "dot",
) -> dict[str, Path]:
    """Write a top-consensus induced subnetwork with subcommunity coloring."""
    graph = build_significant_network(
        gc_csv,
        p_threshold=p_threshold,
        gene_of_interest=gene_of_interest,
    )
    selected_genes = select_top_consensus_genes(
        frequency_csv,
        gene_of_interest=gene_of_interest,
        top_fraction=top_fraction,
    )
    nodes_by_upper = {str(node).upper(): str(node) for node in graph.nodes()}
    selected_nodes = [nodes_by_upper.get(gene.upper(), gene) for gene in selected_genes]
    subgraph = graph.subgraph([node for node in selected_nodes if node in graph]).copy()
    for node in selected_nodes:
        if node not in subgraph:
            subgraph.add_node(node)

    return write_graph_network_artifacts(
        subgraph,
        source_gc_csv=gc_csv,
        source_frequency_csv=frequency_csv,
        p_threshold=p_threshold,
        gene_of_interest=gene_of_interest,
        output_dir=output_dir,
        prefix=prefix,
        write_svg_preview=write_svg_preview,
        svg_renderer=svg_renderer,
        svg_layout=svg_layout,
        top_consensus_fraction=top_fraction,
        selected_genes=selected_genes,
    )


def write_graph_network_artifacts(
    graph: nx.DiGraph,
    *,
    source_gc_csv: str | Path,
    p_threshold: float,
    gene_of_interest: str,
    output_dir: str | Path,
    prefix: str,
    write_svg_preview: bool,
    svg_renderer: str,
    svg_layout: str,
    source_frequency_csv: str | Path | None = None,
    top_consensus_fraction: float | None = None,
    selected_genes: list[str] | None = None,
) -> dict[str, Path]:
    """Write a complete machine-readable and visual artifact bundle for a graph."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_csv = out_dir / f"{prefix}_network_edges.csv"
    node_txt = out_dir / f"{prefix}_network_nodes.txt"
    graphml = out_dir / f"{prefix}_network.graphml"
    svg = out_dir / f"{prefix}_network.svg"
    summary_json = out_dir / f"{prefix}_network_summary.json"
    partition = detect_subcommunities(graph)

    with open(edge_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gene1", "gene2", "lag", "p-value"])
        for source, target, data in graph.edges(data=True):
            writer.writerow([source, target, data.get("lag", 1), data.get("p_value", "")])

    with open(node_txt, "w", encoding="utf-8") as fh:
        for node in sorted(map(str, graph.nodes())):
            fh.write(f"{node}\n")

    nx.write_graphml(graph, graphml)
    if write_svg_preview:
        write_network_svg(
            graph,
            svg,
            gene_of_interest=gene_of_interest,
            renderer=svg_renderer,
            layout=svg_layout,
            partition=partition,
        )

    metrics = {
        **summarize_network(graph, gene_of_interest=gene_of_interest),
        **summarize_subcommunities(graph, partition, gene_of_interest=gene_of_interest),
    }
    if top_consensus_fraction is not None:
        metrics["top_consensus_fraction"] = top_consensus_fraction
        metrics["top_consensus_gene_count"] = len(selected_genes or [])
        metrics["top_consensus_network_gene_count"] = graph.number_of_nodes()

    summary = {
        "source_gc_csv": str(source_gc_csv),
        "gene_of_interest": gene_of_interest,
        "p_value_threshold": p_threshold,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "metrics": metrics,
        "graphml": str(graphml),
        "edge_csv": str(edge_csv),
        "node_txt": str(node_txt),
    }
    if source_frequency_csv is not None:
        summary["source_frequency_csv"] = str(source_frequency_csv)
    if selected_genes is not None:
        summary["selected_genes"] = selected_genes
    if write_svg_preview:
        summary["svg"] = str(svg)
        summary["svg_renderer"] = svg_renderer
        summary["svg_layout"] = svg_layout
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    artifacts = {
        "edge_csv": edge_csv,
        "node_txt": node_txt,
        "graphml": graphml,
        "summary_json": summary_json,
    }
    if write_svg_preview:
        artifacts["svg"] = svg
    return artifacts


def _render_graphviz_dot(dot_source: str, output: Path, *, layout: str) -> Path:
    with tempfile.TemporaryDirectory() as tmp_dir:
        dot_file = Path(tmp_dir) / f"{output.stem}.dot"
        dot_file.write_text(dot_source, encoding="utf-8")
        result = subprocess.run(
            [layout, "-Tsvg", str(dot_file), "-o", str(output)],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Graphviz renderer '{layout}' failed with exit code {result.returncode}: {result.stderr.strip()}"
            )
    return output


def _dot_id(value: str) -> str:
    return _dot_value(value)


def _dot_value(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _component_partition(graph: nx.DiGraph) -> dict[str, int]:
    undirected = graph.to_undirected()
    components = sorted((sorted(map(str, component)) for component in nx.connected_components(undirected)), key=lambda c: c[0])
    return {node: idx for idx, component in enumerate(components) for node in component}


def _matching_node(graph: nx.DiGraph, gene: str) -> str | None:
    for node in graph.nodes():
        if str(node).upper() == gene.upper():
            return str(node)
    return None


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.upper()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _community_colors(partition: dict[str, int]) -> dict[int, str]:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return {community: palette[community % len(palette)] for community in sorted(set(partition.values()))}


def _normalize(values, *, min_size: float, max_size: float) -> list[float]:
    values = list(values)
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [min_size for _ in values]
    return [min_size + (max_size - min_size) * (value - min_val) / (max_val - min_val) for value in values]


def _wrap_label(name: str, max_chars: int = 10) -> str:
    hard_wrap = max(1, max_chars - 2)
    parts = name.split("-")
    tokens = []
    for idx, part in enumerate(parts):
        tokens.append(part)
        if idx < len(parts) - 1:
            tokens.append("-")
    expanded = []
    for token in tokens:
        if token == "-" or len(token) < max_chars:
            expanded.append(token)
        else:
            expanded.extend(token[idx : idx + hard_wrap] for idx in range(0, len(token), hard_wrap))
    lines: list[str] = []
    current = ""
    for token in expanded:
        candidate = current + token
        if len(candidate) > max_chars:
            if current:
                lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\\n".join(lines)


def _format_p_value(value) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return "NA"


def _gene_of_interest_from_run(run_dir: Path) -> str:
    config = run_dir / "pipeline_config.yml"
    if config.exists():
        try:
            import yaml

            raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
            gene = raw.get("gene_of_interest")
            if gene:
                return str(gene)
        except Exception:
            pass
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        gene = (raw.get("settings") or {}).get("gene_of_interest")
        if gene:
            return str(gene)
    return ""
