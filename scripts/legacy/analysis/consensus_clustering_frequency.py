"""Explore how consensus-frequency cutoffs change network size."""

import csv
import os
from typing import List, Tuple, Dict
import networkx as nx
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def filter_genes_by_threshold(file_path: str, threshold: float) -> list[str]:
    """Return genes with coassociation frequency above the threshold."""
    genes = []
    with open(file_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                freq = float(row["Coassociation Frequency"])
                if freq >= threshold:
                    genes.append(row["Gene"])
            except (ValueError, KeyError):
                continue
    return genes

def build_gene_network(
    gene_list: List[str],
    csv_file: str,
    p_threshold: float = 0.05,
    connections: str = "both"
) -> nx.DiGraph:
    """Build a directed significant-edge network for a selected gene set."""
    df = pd.read_csv(csv_file)
    required_columns = {'gene1', 'gene2', 'lag', 'p-value'}
    missing = required_columns - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"

    if connections == "both_genes_in_list":
        df_filtered = df[
            (df['gene1'].isin(gene_list) & df['gene2'].isin(gene_list)) &
            (df['p-value'] <= p_threshold)
        ]
    elif connections == "atleast_one_gene_in_list":
        df_filtered = df[
            ((df['gene1'].isin(gene_list)) | (df['gene2'].isin(gene_list))) &
            (df['p-value'] <= p_threshold)
        ]
    else:
        df_filtered = df[
            (
                ((df['gene1'] == "ZEB2") & (df['gene2'].isin(gene_list))) |
                ((df['gene1'].isin(gene_list)) & (df['gene2'] == "ZEB2"))
            ) & (df['p-value'] <= p_threshold)
        ]

    G = nx.DiGraph()
    for _, row in df_filtered.iterrows():
        source, target, lag, p = row['gene1'], row['gene2'], row['lag'], row['p-value']
        G.add_edge(source, target, lag=lag, p_value=p)

    print(f"Graph built, Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    return G

def safe_slug(s: str) -> str:
    """Return a filesystem-safe slug based on a path or label."""
    return str(Path(s).stem).replace(" ", "_").replace("/", "_").replace("\\", "_")

def ensure_dirs(base_out: Path, datafile: str, coassoc_file: str, connection: str) -> Path:
    """Create and return the output folder for one experiment condition."""
    exp_dir = base_out / f"exp_{safe_slug(datafile)}__{safe_slug(coassoc_file)}"
    conn_dir = exp_dir / connection
    conn_dir.mkdir(parents=True, exist_ok=True)
    return conn_dir

def plot_and_save(group_df: pd.DataFrame, out_png: Path, title: str) -> None:
    """Plot node/edge counts over frequency thresholds and save the figure."""
    # Frequency high -> low on X
    g = group_df.sort_values("frequency", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(g["frequency"], g["nodes"], marker="o", label="Nodes")
    ax.plot(g["frequency"], g["edges"], marker="o", label="Edges")
    ax.set_xlabel("Consensus clustering frequency threshold (≥)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    datafiles = [
        #["granger_results_p00005_26295of26806506.csv", "SM4_exploration_00005_ZEB2_coassoc_6000_runs.csv", 0.0005],
        #["granger_results_p0001_134186of88877756.csv", "SM7_exploration_0001_ZEB2_coassoc_7200_runs.csv", 0.001],
        ["granger_results_p00015_503811of190674672.csv", "SM8_exploration_00015_ZEB2_coassoc_5000_runs.csv", 0.0015],
    ]
    connections = ["both_genes_in_list", "atleast_one_gene_in_list"]
    frequencies = [1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0]

    base_out = Path("outputs")
    base_out.mkdir(exist_ok=True)

    # Collect in-memory, then write per subfolder
    rows = []  # will hold: (datafile, coassoc_file, connection, frequency, seed_genes, nodes, edges, p_threshold)

    for frequency in frequencies:
        for datafile, coassociation_file, p_threshold in datafiles:
            filtered_genes = filter_genes_by_threshold(coassociation_file, frequency)
            for connection in connections:
                print(f"\nBuilding network for frequency >= {frequency}, datafile: {datafile}, connections: {connection}")
                G = build_gene_network(
                    gene_list=filtered_genes,
                    csv_file=datafile,
                    p_threshold=p_threshold,
                    connections=connection
                )
                rows.append({
                    "datafile": datafile,
                    "coassoc_file": coassociation_file,
                    "p_threshold": float(p_threshold),
                    "connection": connection,
                    "frequency": float(frequency),
                    "seed_genes": len(filtered_genes),
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                })

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Write per (datafile, coassoc_file, connection) into the requested subfolders
    for (datafile, coassoc_file, connection), g in df.groupby(["datafile", "coassoc_file", "connection"], dropna=False):
        conn_dir = ensure_dirs(base_out, datafile, coassoc_file, connection)
        # 1) CSV inside subfolder
        out_csv = conn_dir / "results.csv"
        g.sort_values("frequency", ascending=False).to_csv(out_csv, index=False)
        # 2) Figure inside subfolder
        title = f"{Path(datafile).name}\nCoassoc: {Path(coassoc_file).name} | Conn: {connection}"
        plot_and_save(g, conn_dir / "growth.png", title)

    print(f"\nDone. Organized outputs under: {base_out.resolve()}")
