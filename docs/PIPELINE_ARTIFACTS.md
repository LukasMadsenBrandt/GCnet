# Pipeline Artifacts

The YAML runner can start from any stage with `--start-at`. If earlier stages
have not already been run in the same `results/pipeline/<run_name>/` folder,
provide the required input artifact paths in the config under `artifacts:`.

Example:

```yaml
artifacts:
  seed_gc_csv: results/pipeline/my_run/01_seed_gc/seed_gc.csv
```

## Stage Contracts

| Stage | Purpose | Required Config Inputs | Required Resume Artifacts | Main Outputs |
| --- | --- | --- | --- | --- |
| `01_seed_gc` | Directed GC among seed genes | `seed_gene_file`, `dataset.expression_file`, `network.p_value_threshold` | none | `seed_gc_csv` |
| `02_seed_consensus` | Significant seed network + stable GOI Louvain consensus | `gene_of_interest`, `network`, `consensus` | `seed_gc_csv` | `seed_frequency_csv`, seed network files |
| `03_probe_selection` | Select high-frequency GOI-community genes | `probe_selection`, `gene_of_interest` | `seed_frequency_csv` | `probe_genes_file` |
| `04_dataset_probe` | Bidirectional probe GC against the full dataset | `dataset.expression_file`, `dataset.full_gene_file`, `network` | `probe_genes_file` | `probe_gc_csv` |
| `05_expanded_genes` | Build probe network and extract expanded genes | `gene_of_interest`, `network`, `seed_gene_file` | `probe_gc_csv` | `expanded_genes_file`, probe network files |
| `06_expanded_gc` | Directed GC among expanded genes | `dataset.expression_file`, `network` | `expanded_genes_file` | `expanded_gc_csv` |
| `07_expanded_consensus` | Expanded significant network + final priority list | `gene_of_interest`, `network`, `consensus` | `expanded_gc_csv` | `priority_genes_csv`, expanded network files |

Every stage also writes `manifest.json` with:

- `inputs`: exact files consumed by the stage.
- `outputs`: exact files written by the stage.
- `parameters`: p-value thresholds, GOI, probe-selection settings, chunk size, workers, and consensus stability settings used by that stage.
- `metrics`: stage-specific values for tracking how the analysis evolves.

The run folder writes `run_manifest.json` with the full settings snapshot, all known artifacts, all stage metrics, and a compact `pipeline_evolution` list for quick inspection.

## Artifact Keys

Use these keys under `artifacts:` when resuming from a later stage.

| Artifact Key | Produced By | Needed By | Default Path |
| --- | --- | --- | --- |
| `seed_gc_csv` | `01_seed_gc` | `02_seed_consensus` | `results/pipeline/<run>/01_seed_gc/seed_gc.csv` |
| `seed_frequency_csv` | `02_seed_consensus` | `03_probe_selection` | `results/pipeline/<run>/02_seed_consensus/priority_genes.csv` |
| `probe_genes_file` | `03_probe_selection` | `04_dataset_probe` | `results/pipeline/<run>/03_probe_selection/probe_genes.txt` |
| `probe_gc_csv` | `04_dataset_probe` | `05_expanded_genes` | `results/pipeline/<run>/04_dataset_probe/probe_gc.csv` |
| `expanded_genes_file` | `05_expanded_genes` | `06_expanded_gc` | `results/pipeline/<run>/05_expanded_genes/expanded_genes.txt` |
| `expanded_gc_csv` | `06_expanded_gc` | `07_expanded_consensus` | `results/pipeline/<run>/06_expanded_gc/expanded_gc.csv` |

Network output keys are produced for inspection, not required for resuming:

| Artifact Key | Default Path |
| --- | --- |
| `seed_network_edges_csv` | `results/pipeline/<run>/02_seed_consensus/seed_network_edges.csv` |
| `seed_network_nodes_file` | `results/pipeline/<run>/02_seed_consensus/seed_network_nodes.txt` |
| `seed_network_graphml` | `results/pipeline/<run>/02_seed_consensus/seed_network.graphml` |
| `seed_network_summary_json` | `results/pipeline/<run>/02_seed_consensus/seed_network_summary.json` |
| `seed_consensus_history_json` | `results/pipeline/<run>/02_seed_consensus/consensus_history.json` |
| `probe_network_edges_csv` | `results/pipeline/<run>/05_expanded_genes/probe_network_edges.csv` |
| `probe_network_nodes_file` | `results/pipeline/<run>/05_expanded_genes/probe_network_nodes.txt` |
| `probe_network_graphml` | `results/pipeline/<run>/05_expanded_genes/probe_network.graphml` |
| `probe_network_summary_json` | `results/pipeline/<run>/05_expanded_genes/probe_network_summary.json` |
| `expanded_network_edges_csv` | `results/pipeline/<run>/07_expanded_consensus/expanded_network_edges.csv` |
| `expanded_network_nodes_file` | `results/pipeline/<run>/07_expanded_consensus/expanded_network_nodes.txt` |
| `expanded_network_graphml` | `results/pipeline/<run>/07_expanded_consensus/expanded_network.graphml` |
| `expanded_network_summary_json` | `results/pipeline/<run>/07_expanded_consensus/expanded_network_summary.json` |
| `expanded_consensus_history_json` | `results/pipeline/<run>/07_expanded_consensus/consensus_history.json` |

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

Consensus stages additionally report:

- `final_runs`
- `stable`
- `stability_quantile_relative_change`
- `top_gene_overlap_percent`
- `top_gene_overlap_k`
- `consensus_communities`
- `gene_of_interest_consensus_community`
- `gene_of_interest_consensus_community_genes`
- average/min/max Louvain community counts across partitions
- average/min/max Louvain GOI-community size across partitions

The `consensus_history.json` files keep these values for every stability
increment, so the run documents how many partitions were needed before the
frequency ranking stabilized.

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
