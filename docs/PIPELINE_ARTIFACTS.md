# Pipeline Artifacts

The YAML runner can start from any stage with `--start-at`. If earlier stages
have not already been run in the same `results/pipeline/<run_name>/` folder,
provide the required input artifact paths in the config under `artifacts:`.

Example:

```yaml
artifacts:
  seed_gc_csv: results/pipeline/my_run/01_seed_gc/seed_gc.csv
```

Network export files are enabled by default:

```yaml
network:
  p_value_threshold: 0.0015
  write_svg: true
  svg_renderer: networkx  # networkx | graphviz
  svg_layout: dot          # dot | neato | fdp | sfdp | circo | twopi
```

Set `network.write_svg: false` when you want the pipeline to skip the
lightweight SVG previews. GraphML, edge CSV, node TXT, and summary JSON files
are still written because they are part of the machine-readable pipeline
contract. This does not skip the network-dependent analysis itself: stage 5
still builds the significant probe network internally to extract the expanded
gene set `n`, and the consensus stages still use significant-edge networks for
clustering.

`network.svg_renderer` controls only the SVG preview style. The default
`networkx` renderer is fast and has no system Graphviz dependency. Set it to
`graphviz` when you want the legacy Graphviz-style network figures during the
pipeline run. `network.svg_layout` controls the Graphviz layout used during the
pipeline run.

## Stage Contracts

| Stage | Purpose | Required Config Inputs | Required Resume Artifacts | Main Outputs |
| --- | --- | --- | --- | --- |
| `00_preprocessing` | Validate and optionally export expression views | `dataset.expression_file`, `seed_gene_file`, `preprocessing` | none | Up to four all-gene/subset, replicate/summarized CSVs |
| `01_seed_gc` | Directed GC among seed genes | `seed_gene_file`, `dataset.expression_file`, `network.p_value_threshold` | none | `seed_gc_csv` |
| `02_seed_consensus` | Significant seed network + stable GOI Louvain consensus | `gene_of_interest`, `network`, `consensus` | `seed_gc_csv` | `seed_frequency_csv`, seed network files, top-5%-consensus network files |
| `03_probe_selection` | Select high-frequency GOI-community genes | `probe_selection`, `gene_of_interest` | `seed_frequency_csv` | `probe_genes_file` |
| `04_dataset_probe` | Bidirectional probe GC against the full dataset | `dataset.expression_file`, `network` | `probe_genes_file` | `probe_gc_csv`, derived `dataset_genes.txt` |
| `05_expanded_genes` | Build probe network and extract expanded genes | `gene_of_interest`, `network`, `seed_gene_file` | `probe_gc_csv` | `expanded_genes_file`, probe network files |
| `06_expanded_gc` | Directed GC among expanded genes | `dataset.expression_file`, `network` | `expanded_genes_file` | `expanded_gc_csv` |
| `07_expanded_consensus` | Expanded significant network + final priority list | `gene_of_interest`, `network`, `consensus` | `expanded_gc_csv` | `priority_genes_csv`, expanded network files, top-5%-consensus network files |

Every stage also writes `manifest.json` with:

- `inputs`: exact files consumed by the stage.
- `outputs`: exact files written by the stage.
- `parameters`: p-value thresholds, GOI, probe-selection settings, chunk size, workers, and consensus stability settings used by that stage.
- `metrics`: stage-specific values for tracking how the analysis evolves.

The run folder writes:

- `RUN_SUMMARY.md`: human-readable settings, pipeline evolution, key outputs,
  figure links, and the top final priority genes.
- `run_manifest.json`: machine-readable settings snapshot, all known artifacts,
  compacted stage metrics, and a compact `pipeline_evolution` list. Long gene
  lists are represented as `{count, preview, truncated}` objects; the complete
  gene lists live in the stage files such as `*_network_nodes.txt`,
  `expanded_genes.txt`, and `priority_genes.csv`.
- `figures/`: copied SVG previews for quick inspection when
  `network.write_svg: true`.

A concrete tracked example of the expected run tree and what to inspect at each
stage is available in
[`examples/PIPELINE_RUN_LAYOUT.md`](../examples/PIPELINE_RUN_LAYOUT.md).

## Stage Interpretation Guide

| Stage | Primary Question | First Files To Inspect | Main Scientific Use |
| --- | --- | --- | --- |
| `00_preprocessing` | What expression values and replicates enter the analysis? | `all_genes_summarized.csv`, `all_genes_replicates.csv`, `manifest.json` | Audit the exact summarized matrix passed to GC and inspect replicate-level variation. |
| `01_seed_gc` | Which directed GC relationships exist inside the curated seed set? | `seed_gc.csv`, `manifest.json` | Verify the seed-pair search space and retain the directed GC evidence used to build the seed network. |
| `02_seed_consensus` | Which seed-network genes repeatedly cluster with the GOI? | `priority_genes.csv`, `seed_network.svg`, `seed_top_consensus_network.svg`, `consensus_progress.jsonl`, `consensus_history.json` | Choose high-confidence GOI-community genes for probing and inspect their subcommunities. |
| `03_probe_selection` | Which consensus genes will probe the full dataset? | `probe_genes.txt`, `manifest.json` | Confirm the configured top-percent or frequency-cutoff selection before the broad probe stage. |
| `04_dataset_probe` | Which full-dataset genes connect to the selected probe genes? | `probe_gc.csv`, `manifest.json` | Generate directed evidence for newly suggested candidate genes without brute-forcing all dataset pairs. |
| `05_expanded_genes` | Which genes are in the significant probe network? | `expanded_genes.txt`, `probe_network.svg`, `probe_network_summary.json` | Define the expanded candidate set `n` for the final focused GC run. |
| `06_expanded_gc` | What are the directed GC relationships inside `n`? | `expanded_gc.csv`, `manifest.json` | Confirm final GC was run on `n * (n - 1)` candidate pairs, not all dataset genes. |
| `07_expanded_consensus` | Which expanded genes are the strongest biological priorities? | `priority_genes.csv`, `expanded_top_consensus_network.svg`, `expanded_network_summary.json`, `consensus_progress.jsonl`, `consensus_history.json` | Use the sorted priority list for biological follow-up and inspect top-candidate subcommunities. |

## Artifact Keys

Use these keys under `artifacts:` when resuming from a later stage.

| Artifact Key | Produced By | Needed By | Default Path |
| --- | --- | --- | --- |
| `all_genes_replicates_csv` | `00_preprocessing` | Dataset explorer | `results/pipeline/<run>/00_preprocessing/all_genes_replicates.csv` |
| `all_genes_summarized_csv` | `00_preprocessing` | Dataset explorer | `results/pipeline/<run>/00_preprocessing/all_genes_summarized.csv` |
| `subset_genes_replicates_csv` | `00_preprocessing` | Audit/export | `results/pipeline/<run>/00_preprocessing/subset_genes_replicates.csv` |
| `subset_genes_summarized_csv` | `00_preprocessing` | Audit/export | `results/pipeline/<run>/00_preprocessing/subset_genes_summarized.csv` |
| `seed_gc_csv` | `01_seed_gc` | `02_seed_consensus` | `results/pipeline/<run>/01_seed_gc/seed_gc.csv` |
| `seed_frequency_csv` | `02_seed_consensus` | `03_probe_selection` | `results/pipeline/<run>/02_seed_consensus/priority_genes.csv` |
| `probe_genes_file` | `03_probe_selection` | `04_dataset_probe` | `results/pipeline/<run>/03_probe_selection/probe_genes.txt` |
| `probe_gc_csv` | `04_dataset_probe` | `05_expanded_genes` | `results/pipeline/<run>/04_dataset_probe/probe_gc.csv` |
| `expanded_genes_file` | `05_expanded_genes` | `06_expanded_gc` | `results/pipeline/<run>/05_expanded_genes/expanded_genes.txt` |
| `expanded_gc_csv` | `06_expanded_gc` | `07_expanded_consensus` | `results/pipeline/<run>/06_expanded_gc/expanded_gc.csv` |

Network output keys are produced for inspection and are not required for
resuming. GraphML is intended for downstream tools, while SVG previews are
lightweight visual checks for the seed, probe, expanded, and top-consensus
subnetworks. SVG keys are produced only when `network.write_svg: true`:

| Artifact Key | Default Path |
| --- | --- |
| `seed_network_edges_csv` | `results/pipeline/<run>/02_seed_consensus/seed_network_edges.csv` |
| `seed_network_nodes_file` | `results/pipeline/<run>/02_seed_consensus/seed_network_nodes.txt` |
| `seed_network_graphml` | `results/pipeline/<run>/02_seed_consensus/seed_network.graphml` |
| `seed_network_svg` | `results/pipeline/<run>/02_seed_consensus/seed_network.svg` |
| `seed_network_summary_json` | `results/pipeline/<run>/02_seed_consensus/seed_network_summary.json` |
| `seed_top_consensus_network_edges_csv` | `results/pipeline/<run>/02_seed_consensus/seed_top_consensus_network_edges.csv` |
| `seed_top_consensus_network_nodes_file` | `results/pipeline/<run>/02_seed_consensus/seed_top_consensus_network_nodes.txt` |
| `seed_top_consensus_network_graphml` | `results/pipeline/<run>/02_seed_consensus/seed_top_consensus_network.graphml` |
| `seed_top_consensus_network_svg` | `results/pipeline/<run>/02_seed_consensus/seed_top_consensus_network.svg` |
| `seed_top_consensus_network_summary_json` | `results/pipeline/<run>/02_seed_consensus/seed_top_consensus_network_summary.json` |
| `seed_consensus_progress_jsonl` | `results/pipeline/<run>/02_seed_consensus/consensus_progress.jsonl` |
| `seed_consensus_history_json` | `results/pipeline/<run>/02_seed_consensus/consensus_history.json` |
| `probe_network_edges_csv` | `results/pipeline/<run>/05_expanded_genes/probe_network_edges.csv` |
| `probe_network_nodes_file` | `results/pipeline/<run>/05_expanded_genes/probe_network_nodes.txt` |
| `probe_network_graphml` | `results/pipeline/<run>/05_expanded_genes/probe_network.graphml` |
| `probe_network_svg` | `results/pipeline/<run>/05_expanded_genes/probe_network.svg` |
| `probe_network_summary_json` | `results/pipeline/<run>/05_expanded_genes/probe_network_summary.json` |
| `expanded_network_edges_csv` | `results/pipeline/<run>/07_expanded_consensus/expanded_network_edges.csv` |
| `expanded_network_nodes_file` | `results/pipeline/<run>/07_expanded_consensus/expanded_network_nodes.txt` |
| `expanded_network_graphml` | `results/pipeline/<run>/07_expanded_consensus/expanded_network.graphml` |
| `expanded_network_svg` | `results/pipeline/<run>/07_expanded_consensus/expanded_network.svg` |
| `expanded_network_summary_json` | `results/pipeline/<run>/07_expanded_consensus/expanded_network_summary.json` |
| `expanded_top_consensus_network_edges_csv` | `results/pipeline/<run>/07_expanded_consensus/expanded_top_consensus_network_edges.csv` |
| `expanded_top_consensus_network_nodes_file` | `results/pipeline/<run>/07_expanded_consensus/expanded_top_consensus_network_nodes.txt` |
| `expanded_top_consensus_network_graphml` | `results/pipeline/<run>/07_expanded_consensus/expanded_top_consensus_network.graphml` |
| `expanded_top_consensus_network_svg` | `results/pipeline/<run>/07_expanded_consensus/expanded_top_consensus_network.svg` |
| `expanded_top_consensus_network_summary_json` | `results/pipeline/<run>/07_expanded_consensus/expanded_top_consensus_network_summary.json` |
| `expanded_consensus_progress_jsonl` | `results/pipeline/<run>/07_expanded_consensus/consensus_progress.jsonl` |
| `expanded_consensus_history_json` | `results/pipeline/<run>/07_expanded_consensus/consensus_history.json` |

Run-level figure copies are also produced when SVG previews are enabled:

| Artifact Key | Default Path |
| --- | --- |
| `seed_network_figure_svg` | `results/pipeline/<run>/figures/seed_network.svg` |
| `seed_top_consensus_network_figure_svg` | `results/pipeline/<run>/figures/seed_top_consensus_network.svg` |
| `probe_network_figure_svg` | `results/pipeline/<run>/figures/probe_network.svg` |
| `expanded_network_figure_svg` | `results/pipeline/<run>/figures/expanded_network.svg` |
| `expanded_top_consensus_network_figure_svg` | `results/pipeline/<run>/figures/expanded_top_consensus_network.svg` |

You can also render or replace SVG previews after a pipeline has already run:

```sh
python3 scripts/pipeline/render_network_figures.py \
  --run-dir results/pipeline/<run> \
  --renderer graphviz
```

This reads the existing GraphML files and writes stage SVGs plus copies under
`results/pipeline/<run>/figures/`.

To let a researcher choose the best-looking figure later, render all common
Graphviz layouts:

```sh
python3 scripts/pipeline/render_network_figures.py \
  --run-dir results/pipeline/<run> \
  --renderer graphviz \
  --all-layouts
```

`--all-layouts` skips only the individual Graphviz engines that fail and keeps
any successful SVGs. A single explicit `--layout <name>` fails if that specific
layout cannot be rendered.

## Metrics

Network stages report metrics such as:

- `genes_total`
- `edges_total`
- `density`
- `weak_components`
- `strong_components`
- `largest_weak_component_genes`
- `largest_strong_component_genes`
- GOI in-degree, out-degree, and total degree

The expanded-gene stage also reports how much the probe network grew beyond the
original seed list:

- `seed_gene_count`
- `expanded_gene_count`
- `expanded_seed_overlap_count`
- `expanded_new_gene_count`
- `expanded_seed_overlap_percent`
- `expanded_new_gene_percent`
- `seed_gene_retention_percent`

Consensus stages additionally report:

- `final_runs`
- `stable`
- `stability_quantile_relative_change`
- `top_gene_overlap_percent`
- `top_gene_overlap_k`
- `consensus_total_seconds`
- `louvain_seconds`
- `coassociation_seconds`
- `agglomerative_clustering_seconds`
- `goi_frequency_seconds`
- `coassociation_matrix_cells`
- `consensus_communities`
- `gene_of_interest_consensus_community`
- `gene_of_interest_consensus_community_genes`
- average/min/max Louvain community counts across partitions
- average/min/max Louvain GOI-community size across partitions

During a running consensus stage, `consensus.log` prints the stopping criteria
after each increment and `consensus_progress.jsonl` appends one JSON object per
increment. Use those live files to see whether the quantile relative-change and
top-gene-overlap checks are moving toward their thresholds. After completion,
`consensus_history.json` keeps the same per-increment values in a formatted JSON
array for audit and reporting.

## Validation

The canonical runner validates inputs and outputs at stage boundaries:

- gene-list files are non-empty, de-duplicated, and include the GOI where required
- expression matrices have numeric values, at least 5 timepoint/condition columns, no duplicate gene names, and enough configured genes for the stage
- `log1p` and sqrt preprocessing reject negative expression values
- GC CSVs must contain `gene1`, `gene2`, `lag`, and `p-value`, with no self-pairs
- consensus frequency CSVs must contain `Gene` and `Coassociation Frequency`, use values in `[0, 1]`, be sorted descending, and include the GOI
- network GraphML, edge CSV, node TXT, summary JSON, and SVG previews are checked for consistency

## Compute Backends

The YAML config records compute backends under `execution:`. CPU backends are
the validated defaults:

- `gc_backend: cpu_statsmodels`
- `consensus_backend: cpu_louvain`

The recommended accelerated production combination is:

- `gc_backend: gpu_cuda`
- `consensus_backend: cpu_louvain`

This accelerates GC while preserving the validated Python Louvain consensus
behavior across CPU cores. `gpu_cugraph` is available for consensus experiments
only. It uses RAPIDS/cuGraph for Louvain, CuPy for coassociation matrix
construction, and the existing CPU agglomerative consensus step. If selected on
an unsupported machine, it fails clearly rather than silently falling back to
CPU. It should not be used for scientific production unless a dataset-specific
parity report passes against `cpu_louvain`.

Use `python3 scripts/pipeline/check_cuda.py` on local or HPC machines to record
Python, NVIDIA, CuPy, cuDF, cuGraph, and device readiness. GPU production work
should write parity and benchmark JSON reports comparing candidate GPU outputs
against the validated CPU outputs. See [GPU acceleration](GPU_ACCELERATION.md).

## Resume Examples

Start at stage 2:

```yaml
artifacts:
  seed_gc_csv: results/pipeline/my_run/01_seed_gc/seed_gc.csv
```

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/my_run.yml --start-at 02_seed_consensus
```

Start at stage 5:

```yaml
artifacts:
  probe_gc_csv: results/pipeline/my_run/04_dataset_probe/probe_gc.csv
```

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/my_run.yml --start-at 05_expanded_genes
```

Start at stage 7:

```yaml
artifacts:
  expanded_gc_csv: results/pipeline/my_run/06_expanded_gc/expanded_gc.csv
```

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/my_run.yml --start-at 07_expanded_consensus
```
