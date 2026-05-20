# Workflows

These commands show the new canonical entrypoints. Full Granger and consensus jobs can be very large; run them on suitable hardware.

## Configs And Examples

Pipeline configs are grouped by intent:

| Folder | Purpose |
| --- | --- |
| `configs/test/` | Fast fixtures that should run before code changes. |
| `configs/production_like/` | Small real-loader, real-GC runs for Kutsche, Benito human, and Benito gorilla. |
| `configs/production/` | Full production runs for suitable local/HPC hardware. |
| `configs/templates/` | Starting points for new datasets or new analyses. |

Example fixtures live in `examples/`:

- `sample_pipeline/`: larger deterministic fixture with 50 seed genes and 1000 total genes.
- `sample_real_gc/`: smaller synthetic fixture that runs statsmodels GC and keeps some genes out of the expanded set.
- `kutsche_real_gc/`: small gene lists for a Kutsche production-like real-GC run.
- `benito_real_gc/`: small gene lists for Benito human/gorilla production-like real-GC runs.
- `seed_genes_neurodeficiency.txt`: tiny demo of the one-gene-per-line seed format.

## Canonical Resumable Pipeline

Copy `configs/templates/gene_expansion.example.yml`, edit the dataset, gene of interest, seed gene list, p-value threshold, and probe-selection rule, then run:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/templates/gene_expansion.example.yml
```

For a completely new already-preprocessed dataset, copy
`configs/templates/gene_expansion.generic_example.yml`, set `dataset.name:
generic_expression`, and point it at a matrix with `Gene` plus at least 5
ordered numeric timepoint/condition columns.

Use the `preprocessing:` block to choose normalization and transformation:

```yaml
preprocessing:
  normalize: none      # none | deseq | zscore
  transform: log1p     # none | log1p | sqrt
  aggregation: robust  # robust | mean | median
```

Dataset-specific starting configs are also provided:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/production/gene_expansion.kutsche.yml
python3 scripts/pipeline/run_pipeline.py --config configs/production/gene_expansion.benito_human.yml
python3 scripts/pipeline/run_pipeline.py --config configs/production/gene_expansion.benito_gorilla.yml
```

Before launching those, use the production-like configs to check the real
loaders and real GC path on a deliberately small gene set:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/production_like/gene_expansion.kutsche.real_gc_small.yml
python3 scripts/pipeline/run_pipeline.py --config configs/production_like/gene_expansion.benito_human.real_gc_small.yml
python3 scripts/pipeline/run_pipeline.py --config configs/production_like/gene_expansion.benito_gorilla.real_gc_small.yml
```

These production-style configs require the real datasets to be present locally:

- `Data/Kutsche/Kutsche_Counts.txt`
- `Data/Benito/Benito_Human`
- `Data/Benito/Benito_Gorilla`
- `Data/Benito/gene_id_to_gene_name.txt`

See [Data preparation](DATA_PREPARATION.md) before adapting the pipeline to a
new dataset or replacing the local production matrices.

The repository tracks the small gene lists and metadata, but not the large
expression matrices. Use the example configs below when working from a fresh
clone without local data.

All three produce intermediate seed, probe, and expanded network artifacts
under `results/pipeline/<run_name>/`. The seed and expanded consensus stages
also produce top-5%-consensus subnetworks, which are useful for inspecting the
subcommunities among the strongest GOI-associated genes.

Network inspection files are controlled by the network block:

```yaml
network:
  p_value_threshold: 0.0015
  write_svg: true
  svg_renderer: networkx
  svg_layout: dot
```

Leave `write_svg: true` for research runs where intermediate SVG previews are
useful for quick review. Set it to `false` only for lean computational reruns.
GraphML, edge CSV, node TXT, and network-summary JSON files are still exported,
and the internal network logic is still used for expansion and consensus.
The default `svg_renderer: networkx` creates fast previews. Use
`svg_renderer: graphviz` when you want the legacy Graphviz-style figures during
the run. `svg_layout` is used for Graphviz figures and can be `dot`, `neato`,
`fdp`, `sfdp`, `circo`, or `twopi`.

You can also render Graphviz figures later from a completed run without
rerunning GC or consensus:

```sh
python3 scripts/pipeline/render_network_figures.py \
  --run-dir results/pipeline/<run_name> \
  --renderer graphviz
```

Render all common Graphviz layouts after a run:

```sh
python3 scripts/pipeline/render_network_figures.py \
  --run-dir results/pipeline/<run_name> \
  --renderer graphviz \
  --all-layouts
```

If one Graphviz engine fails on a dense or awkward network, `--all-layouts`
continues with the remaining layouts and reports the skipped engine. Running a
single `--layout <name>` fails loudly so you know that exact requested figure
was not produced.

## Run Multiple Pipelines

Independent configs can run in parallel as long as every config has a unique
`run_name`. The helper validates that first, so separate jobs do not write into
the same `results/pipeline/<run_name>/` folder.

Preview commands without running anything:

```sh
python3 scripts/pipeline/submit_many.py \
  configs/production/gene_expansion.kutsche.yml \
  configs/production/gene_expansion.benito_human.yml \
  configs/production/gene_expansion.benito_gorilla.yml \
  --dry-run
```

Write one SLURM job script per config:

```sh
python3 scripts/pipeline/submit_many.py \
  configs/production/gene_expansion.kutsche.yml \
  configs/production/gene_expansion.benito_human.yml \
  configs/production/gene_expansion.benito_gorilla.yml \
  --write-slurm scripts/pipeline/slurm_jobs \
  --cpus-per-task 32 \
  --mem 128G \
  --time 48:00:00
```

This writes `scripts/pipeline/slurm_jobs/<run_name>.slurm` for each config plus
`scripts/pipeline/slurm_jobs/submit_all.sh`. Each submitted SLURM job runs
exactly one pipeline.

Extra SLURM directives can be added with `--sbatch=...`; use the equals form
when the directive itself starts with dashes:

```sh
python3 scripts/pipeline/submit_many.py configs/production/*.yml \
  --write-slurm scripts/pipeline/slurm_jobs \
  --sbatch=--partition=gpu
```

Submit all generated jobs on a SLURM system:

```sh
bash scripts/pipeline/slurm_jobs/submit_all.sh
```

If your cluster prefers job arrays, `--write-slurm-array path/to/file.slurm`
is still available, but the recommended mode is one job script per pipeline.

To avoid accidentally reusing an existing run folder, add:

```sh
--check-existing-results
```

For local debugging only, run sequentially in the current shell:

```sh
python3 scripts/pipeline/submit_many.py configs/test/gene_expansion.sample.yml --run-local
```

Do not launch multiple local pipelines that each request all CPU cores. On HPC,
prefer one pipeline per job and set each config's `execution.max_workers` to
match the allocated `--cpus-per-task`.

For GPU nodes, first create the CUDA environment described in
[GPU acceleration](GPU_ACCELERATION.md) and save a compatibility report
from the allocated node:

```sh
python scripts/pipeline/check_cuda.py --output-file results/cuda_compatibility.json
```

The baseline production configs intentionally keep `gc_backend: cpu_statsmodels`
and `consensus_backend: cpu_louvain`. B200/HPC GPU-GC variants live next to
them in `configs/production/*.gpu_b200.yml`; they use `gc_backend: gpu_cuda`
with `consensus_backend: cpu_louvain` after a CPU/GPU GC parity benchmark
passes on the target hardware.

To validate the experimental CuPy Granger backend against the CPU reference,
run the same config through the benchmark/parity helper:

```sh
python3 scripts/pipeline/benchmark_cuda_gc.py \
  --config configs/test/gene_expansion.real_gc_sample.yml
```

Add `--min-speedup 2.0` when you want the report to fail unless every
benchmarked GC stage is at least twice as fast as the CPU reference. Leave it
unset for pure reproducibility checks.

This creates `<run>_cpu_benchmark` and `<run>_cuda_benchmark` pipeline runs,
compares seed/probe/expanded GC CSVs, compares the final priority list, and
writes a JSON report under `results/pipeline/`. Use this on production-like
configs before considering CUDA for full production runs.

The pytest benchmark/parity checks are opt-in:

```sh
python3 -m pytest tests -m cuda_benchmark
```

To profile and compare consensus backends experimentally, use the consensus
benchmark helper:

```sh
python3 scripts/pipeline/benchmark_consensus_backend.py \
  --config configs/test/gene_expansion.sample.yml \
  --candidate-backend gpu_cugraph
```

The candidate `gpu_cugraph` uses RAPIDS/cuGraph for repeated Louvain and CuPy
for coassociation matrix construction, while preserving the existing CPU
agglomerative consensus step. It is not currently the recommended scientific
backend because cuGraph Louvain can produce different community structure from
the validated Python Louvain method on expanded networks. Use it only when a
dataset-specific parity report passes.

To run the GPU-GC sample pipeline in a CUDA/CuPy environment:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/test/gene_expansion.gpu_sample.yml
```

For a medium-sized end-to-end smoke test, use the sample fixture instead of real data:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/test/gene_expansion.sample.yml
```

For a smaller end-to-end smoke test that runs the actual statsmodels Granger
causality calculations, use:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/test/gene_expansion.real_gc_sample.yml
```

The pytest version of that sample is the recommended prerequisite check before code changes:

```sh
python3 -m pytest tests/test_pipeline.py::test_sample_fixture_pipeline_runs_all_stages
```

The runner writes one folder per run:

```text
results/pipeline/<run_name>/
├── pipeline_config.yml
├── run_manifest.json
├── RUN_SUMMARY.md
├── figures/
├── 01_seed_gc/
├── 02_seed_consensus/
├── 03_probe_selection/
├── 04_dataset_probe/
├── 05_expanded_genes/
├── 06_expanded_gc/
└── 07_expanded_consensus/
```

Each stage writes a `manifest.json` with its inputs, outputs, parameters,
status, and metrics. The run-level `RUN_SUMMARY.md` is the best first file to
open after a run: it lists the settings, pipeline evolution, key outputs,
figure links, and top priority genes. The run-level `run_manifest.json` also
includes the full settings snapshot and a compact `pipeline_evolution` table
for comparing seed, probe, and expanded-network results.

## Resume From A Stage

Use `--start-at` and `--stop-after` to run any contiguous section:

```sh
python3 scripts/pipeline/run_pipeline.py \
  --config configs/templates/gene_expansion.example.yml \
  --start-at 03_probe_selection \
  --stop-after 05_expanded_genes
```

If you start after stage 1, the config must provide the required artifact in `artifacts:` or the expected previous-stage output must already exist in the run folder. For example, starting at stage 3 needs `seed_frequency_csv`; starting at stage 5 needs `probe_gc_csv`.

See [Pipeline artifacts](PIPELINE_ARTIFACTS.md) for the full artifact contract and intermediate network outputs.

## Pipeline Stages

1. `01_seed_gc`: run directed GC among the `x` curated seed genes, excluding self-pairs.
2. `02_seed_consensus`: threshold the seed GC network and run stability-controlled Louvain consensus around the gene of interest.
3. `03_probe_selection`: select high coassociation-frequency genes by top percent or minimum frequency; always include the gene of interest.
4. `04_dataset_probe`: probe selected genes against the full dataset in both directions.
5. `05_expanded_genes`: build a significant-edge network from probe results and write all network genes as the expanded set `y`.
6. `06_expanded_gc`: run directed GC among the `y` expanded genes.
7. `07_expanded_consensus`: rerun consensus and write the final sorted biological priority list.

## Lightweight Utilities

These helpers remain useful for one-off checks and small fixtures:

```sh
python3 scripts/pipeline/thresholding.py \
  results/pipeline/sample_real_gc/01_seed_gc/seed_gc.csv \
  --quantile 0.0005
```

```sh
python3 scripts/pipeline/dataset_probe.py \
  --seed-gene-file examples/seed_genes_neurodeficiency.txt \
  --full-gene-file Data/Kutsche/genes_all.txt
```

```sh
python3 scripts/pipeline/network_expansion.py \
  --candidate-network-csv results/pipeline/sample_real_gc/04_dataset_probe/probe_gc.csv \
  --gene-of-interest ZEB2 \
  --p-threshold 0.001 \
  --seed-gene-file examples/seed_genes_neurodeficiency.txt
```

```sh
python3 scripts/pipeline/consensus.py \
  --gc-result-file results/pipeline/sample_real_gc/06_expanded_gc/expanded_gc.csv \
  --gene-of-interest ZEB2 \
  --n-runs 100 \
  --p-threshold 0.001
```

## Existing Commands

The `scripts/` folder is organized by support level:

| Folder | Purpose |
| --- | --- |
| `scripts/pipeline/` | Supported command-line entrypoints for the canonical pipeline and small stage helpers. |
| `scripts/reporting/` | Post-pipeline reporting utilities for gene-list and stability comparisons. |
| `scripts/legacy/` | Historical dashboards, UCloud runs, exploratory network scripts, and old entrypoints kept for traceability. |

Use canonical `scripts/...` paths for pipeline and reporting helpers:

```sh
python3 scripts/pipeline/thresholding.py <gc-results.csv>
python3 scripts/reporting/gene_list_compare.py
python3 scripts/reporting/relative_difference.py
```

The old Dash network exploration app is preserved for reference:

```sh
python3 scripts/legacy/dashboard_apps/app.py
```

Old root-level commands were moved to `scripts/legacy/entrypoints/` for
traceability.
