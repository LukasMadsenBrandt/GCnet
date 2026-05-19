import csv
import json
import shutil
import xml.etree.ElementTree as ET

import networkx as nx
import pytest

from gene_analysis.pipeline.network_artifacts import (
    render_network_artifacts_with_renderer,
    write_network_artifacts,
    write_network_svg,
    write_top_consensus_network_artifacts,
)


pytestmark = [pytest.mark.unit, pytest.mark.visual]


def test_write_network_artifacts_creates_loadable_machine_and_visual_files(tmp_path):
    gc_file = tmp_path / "gc.csv"
    gc_file.write_text(
        "gene1,gene2,lag,p-value\n"
        "ZEB2,A,1,0.001\n"
        "A,B,1,0.002\n"
        "B,ZEB2,1,0.003\n"
        "ZEB2,C,1,0.2\n",
        encoding="utf-8",
    )

    artifacts = write_network_artifacts(
        gc_file,
        p_threshold=0.01,
        gene_of_interest="ZEB2",
        output_dir=tmp_path / "network",
        prefix="seed",
    )

    graph = nx.read_graphml(artifacts["graphml"])
    assert set(graph.nodes()) == {"ZEB2", "A", "B"}
    assert graph.number_of_edges() == 3

    with artifacts["edge_csv"].open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["gene1", "gene2", "lag", "p-value"]
    assert len(rows) - 1 == graph.number_of_edges()

    nodes = artifacts["node_txt"].read_text(encoding="utf-8").splitlines()
    assert nodes == sorted(graph.nodes())

    summary = json.loads(artifacts["summary_json"].read_text(encoding="utf-8"))
    assert summary["nodes"] == graph.number_of_nodes()
    assert summary["edges"] == graph.number_of_edges()
    assert summary["metrics"]["gene_of_interest"] == "ZEB2"
    assert summary["metrics"]["gene_of_interest_present"] is True
    assert summary["graphml"] == str(artifacts["graphml"])
    assert summary["svg"] == str(artifacts["svg"])

    svg_text = artifacts["svg"].read_text(encoding="utf-8")
    ET.parse(artifacts["svg"])
    assert "<svg" in svg_text
    assert "ZEB2" in svg_text
    assert "A" in svg_text


def test_write_top_consensus_network_artifacts_creates_subcommunity_bundle(tmp_path):
    gc_file = tmp_path / "gc.csv"
    gc_file.write_text(
        "gene1,gene2,lag,p-value\n"
        "ZEB2,A,1,0.001\n"
        "A,B,1,0.002\n"
        "B,ZEB2,1,0.003\n"
        "C,D,1,0.004\n"
        "D,C,1,0.004\n",
        encoding="utf-8",
    )
    frequency_file = tmp_path / "priority_genes.csv"
    frequency_file.write_text(
        "Gene,Coassociation Frequency\n"
        "ZEB2,1.0\n"
        "A,0.9\n"
        "B,0.8\n"
        "C,0.7\n"
        "D,0.6\n",
        encoding="utf-8",
    )

    artifacts = write_top_consensus_network_artifacts(
        gc_file,
        frequency_file,
        p_threshold=0.01,
        gene_of_interest="ZEB2",
        output_dir=tmp_path / "network",
        prefix="seed_top_consensus",
        top_fraction=0.4,
    )

    graph = nx.read_graphml(artifacts["graphml"])
    assert set(graph.nodes()) == {"ZEB2", "A"}
    assert graph.number_of_edges() == 1

    nodes = artifacts["node_txt"].read_text(encoding="utf-8").splitlines()
    assert nodes == ["A", "ZEB2"]

    summary = json.loads(artifacts["summary_json"].read_text(encoding="utf-8"))
    assert summary["source_frequency_csv"] == str(frequency_file)
    assert summary["selected_genes"] == ["ZEB2", "A"]
    assert summary["metrics"]["top_consensus_fraction"] == pytest.approx(0.4)
    assert summary["metrics"]["top_consensus_gene_count"] == 2
    assert summary["metrics"]["subcommunity_count"] >= 1
    assert summary["metrics"]["gene_of_interest_subcommunity_genes"] >= 1

    svg_text = artifacts["svg"].read_text(encoding="utf-8")
    ET.parse(artifacts["svg"])
    assert "<svg" in svg_text
    assert "ZEB2" in svg_text
    assert "A" in svg_text


def test_write_network_svg_handles_empty_network(tmp_path):
    output = write_network_svg(nx.DiGraph(), tmp_path / "empty.svg", gene_of_interest="ZEB2")

    svg_text = output.read_text(encoding="utf-8")
    ET.parse(output)
    assert "<svg" in svg_text
    assert "No significant network edges" in svg_text


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz dot executable is not installed")
def test_write_network_svg_can_use_graphviz_renderer(tmp_path):
    graph = nx.DiGraph()
    graph.add_edge("ZEB2", "A", lag=1, p_value=0.001)

    output = write_network_svg(
        graph,
        tmp_path / "graphviz.svg",
        gene_of_interest="ZEB2",
        renderer="graphviz",
    )

    svg_text = output.read_text(encoding="utf-8")
    ET.parse(output)
    assert "<svg" in svg_text
    assert "ZEB2" in svg_text
    assert "A" in svg_text


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz dot executable is not installed")
def test_render_network_artifacts_with_renderer_renders_existing_graphml(tmp_path):
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "02_seed_consensus"
    stage_dir.mkdir(parents=True)
    (run_dir / "pipeline_config.yml").write_text("gene_of_interest: ZEB2\n", encoding="utf-8")
    graph = nx.DiGraph()
    graph.add_edge("ZEB2", "A", lag=1, p_value=0.001)
    nx.write_graphml(graph, stage_dir / "seed_network.graphml")

    outputs = render_network_artifacts_with_renderer(
        run_dir=run_dir,
        renderer="graphviz",
        stages=("seed",),
    )

    assert outputs["seed_network_svg"].exists()
    assert outputs["seed_network_figure_svg"].exists()
    assert "ZEB2" in outputs["seed_network_figure_svg"].read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz dot executable is not installed")
def test_render_network_artifacts_with_renderer_renders_top_consensus_graphml(tmp_path):
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "02_seed_consensus"
    stage_dir.mkdir(parents=True)
    (run_dir / "pipeline_config.yml").write_text("gene_of_interest: ZEB2\n", encoding="utf-8")
    graph = nx.DiGraph()
    graph.add_edge("ZEB2", "A", lag=1, p_value=0.001)
    nx.write_graphml(graph, stage_dir / "seed_top_consensus_network.graphml")

    outputs = render_network_artifacts_with_renderer(
        run_dir=run_dir,
        renderer="graphviz",
        stages=("seed_top_consensus",),
    )

    assert outputs["seed_top_consensus_network_svg"].exists()
    assert outputs["seed_top_consensus_network_figure_svg"].exists()
    assert "ZEB2" in outputs["seed_top_consensus_network_figure_svg"].read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz dot executable is not installed")
def test_render_network_artifacts_can_render_multiple_graphviz_layouts(tmp_path):
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "02_seed_consensus"
    stage_dir.mkdir(parents=True)
    (run_dir / "pipeline_config.yml").write_text("gene_of_interest: ZEB2\n", encoding="utf-8")
    graph = nx.DiGraph()
    graph.add_edge("ZEB2", "A", lag=1, p_value=0.001)
    nx.write_graphml(graph, stage_dir / "seed_network.graphml")

    outputs = render_network_artifacts_with_renderer(
        run_dir=run_dir,
        renderer="graphviz",
        stages=("seed",),
        layouts=("dot", "neato"),
        skip_failed_layouts=True,
    )

    assert outputs["seed_network_dot_svg"].exists()
    assert outputs["seed_network_svg"].exists()
    assert outputs["seed_network_dot_figure_svg"].exists()
