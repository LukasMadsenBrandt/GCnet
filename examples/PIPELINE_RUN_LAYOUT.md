# Example Pipeline Run Layout

This file shows what a completed run folder is expected to look like and what a
researcher should inspect at each stage. It is a tracked example only; real
pipeline outputs live under `results/pipeline/<run_name>/` and are not committed
to Git.

The best first file after any run is:

```text
results/pipeline/<run_name>/RUN_SUMMARY.md
```

It lists the settings, key metrics, important artifact links, and top final
priority genes.

## Folder Layout

```text
results/pipeline/<run_name>/
├── pipeline_config.yml
├── run_manifest.json
├── RUN_SUMMARY.md
├── figures/
│   ├── seed_network.svg
│   ├── seed_top_consensus_network.svg
│   ├── probe_network.svg
│   ├── expanded_network.svg
│   └── expanded_top_consensus_network.svg
├── 01_seed_gc/
│   ├── manifest.json
│   └── seed_gc.csv
├── 02_seed_consensus/
│   ├── manifest.json
│   ├── priority_genes.csv
│   ├── consensus_history.json
│   ├── seed_network.graphml
│   ├── seed_network_edges.csv
│   ├── seed_network_nodes.txt
│   ├── seed_network_summary.json
│   ├── seed_network.svg
│   ├── seed_top_consensus_network.graphml
│   ├── seed_top_consensus_network_edges.csv
│   ├── seed_top_consensus_network_nodes.txt
│   ├── seed_top_consensus_network_summary.json
│   └── seed_top_consensus_network.svg
├── 03_probe_selection/
│   ├── manifest.json
│   └── probe_genes.txt
├── 04_dataset_probe/
│   ├── manifest.json
│   └── probe_gc.csv
├── 05_expanded_genes/
│   ├── manifest.json
│   ├── expanded_genes.txt
│   ├── probe_network.graphml
│   ├── probe_network_edges.csv
│   ├── probe_network_nodes.txt
│   ├── probe_network_summary.json
│   └── probe_network.svg
├── 06_expanded_gc/
│   ├── manifest.json
│   └── expanded_gc.csv
└── 07_expanded_consensus/
    ├── manifest.json
    ├── priority_genes.csv
    ├── consensus_history.json
    ├── expanded_network.graphml
    ├── expanded_network_edges.csv
    ├── expanded_network_nodes.txt
    ├── expanded_network_summary.json
    ├── expanded_network.svg
    ├── expanded_top_consensus_network.graphml
    ├── expanded_top_consensus_network_edges.csv
    ├── expanded_top_consensus_network_nodes.txt
    ├── expanded_top_consensus_network_summary.json
    └── expanded_top_consensus_network.svg
```

If `network.write_svg: false`, the `.svg` files and `figures/` copies are
skipped. The GraphML, CSV, TXT, JSON, and manifest files are still written.

## What Each Stage Means

| Stage | What It Does | Main Files | What To Derive |
| --- | --- | --- | --- |
| `01_seed_gc` | Runs directed GC among the curated seed genes. | `seed_gc.csv` | Confirm the expected `x * (x - 1)` directed seed-pair tests were generated. |
| `02_seed_consensus` | Builds the significant seed network and runs stability-controlled GOI consensus. | `priority_genes.csv`, `consensus_history.json`, `seed_network.*`, `seed_top_consensus_network.*` | Inspect seed-network structure, consensus stability, and the top seed genes that repeatedly cluster with the GOI. |
| `03_probe_selection` | Selects genes from the seed consensus ranking for full-dataset probing. | `probe_genes.txt` | Confirm the probe set matches the configured top percent or frequency cutoff and includes the GOI. |
| `04_dataset_probe` | Tests probe genes against the full dataset in both directions. | `probe_gc.csv` | Identify new dataset genes connected to the high-confidence seed-community genes. |
| `05_expanded_genes` | Builds the significant probe network and extracts all genes in that network. | `expanded_genes.txt`, `probe_network.*` | Confirm the expanded set `n` is smaller than the full dataset and biologically focused. |
| `06_expanded_gc` | Runs directed GC among the expanded genes only. | `expanded_gc.csv` | Confirm GC was run on `n * (n - 1)` expanded candidate pairs, not all dataset genes. |
| `07_expanded_consensus` | Runs final consensus on the expanded network. | `priority_genes.csv`, `expanded_network.*`, `expanded_top_consensus_network.*` | Use the sorted priority list as the main biological output; inspect the top-consensus subnetwork for subcommunities among the strongest candidates. |

## Files To Open First

1. `RUN_SUMMARY.md`: quick run overview, settings, metrics, figures, and top genes.
2. `figures/expanded_top_consensus_network.svg`: strongest final candidates and their subcommunities.
3. `07_expanded_consensus/priority_genes.csv`: final sorted gene-priority list.
4. `07_expanded_consensus/consensus_history.json`: how the final consensus stabilized.
5. `05_expanded_genes/expanded_genes.txt`: the discovered expanded set used for final GC.

## Network File Meanings

Each network bundle has the same shape:

| File | Meaning |
| --- | --- |
| `*_network.graphml` | Machine-readable network for Cytoscape, Gephi, NetworkX, or later rendering. |
| `*_network_edges.csv` | Directed significant GC edges used in the network. |
| `*_network_nodes.txt` | Genes present in the network. |
| `*_network_summary.json` | Network metrics such as node count, edge count, density, components, GOI degree, and top-consensus subcommunity metrics when relevant. |
| `*_network.svg` | Quick visual preview for human inspection. |

The `top_consensus` networks are induced subnetworks containing the top 5% of
genes from the corresponding consensus frequency CSV, always including the GOI.
They are meant for inspecting subcommunities within the strongest GOI-associated
genes, not for replacing the full seed or expanded network files.
