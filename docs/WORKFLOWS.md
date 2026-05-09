# Workflows

These commands show the new canonical entrypoints. Full Granger and consensus jobs can be very large; run them on suitable hardware.

## Configs And Examples

Pipeline configs live in `configs/`:

- `gene_expansion.kutsche.yml`: Kutsche production-style run.
- `gene_expansion.benito_human.yml`: Benito human production-style run.
- `gene_expansion.benito_gorilla.yml`: Benito gorilla production-style run.
- `gene_expansion.sample.yml`: deterministic 1000-gene guided-flow fixture.
- `gene_expansion.real_gc_sample.yml`: small fixture that runs real GC.
- `gene_expansion.example.yml`: editable template.

Example fixtures live in `examples/`:

- `sample_pipeline/`: larger deterministic fixture with 50 seed genes and 1000 total genes.
- `sample_real_gc/`: smaller synthetic fixture that runs statsmodels GC and keeps some genes out of the expanded set.
- `seed_genes_neurodeficiency.txt`: tiny demo of the one-gene-per-line seed format.

## Canonical Resumable Pipeline

Copy `configs/gene_expansion.example.yml`, edit the dataset, gene of interest, seed gene list, p-value threshold, and probe-selection rule, then run:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.example.yml
```

Dataset-specific starting configs are also provided:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.kutsche.yml
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.benito_human.yml
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.benito_gorilla.yml
```

These production-style configs require the real datasets to be present locally:

- `Data/Kutsche/Kutsche_Counts.txt`
- `Data/Benito/Benito_Human`
- `Data/Benito/Benito_Gorilla`
- `Data/Benito/gene_id_to_gene_name.txt`

The repository tracks the small gene lists and metadata, but not the large
expression matrices. Use the example configs below when working from a fresh
clone without local data.

All three produce intermediate seed, probe, and expanded network artifacts
under `results/pipeline/<run_name>/`.

For a medium-sized end-to-end smoke test, use the sample fixture instead of real data:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.sample.yml
```

For a smaller end-to-end smoke test that runs the actual statsmodels Granger
causality calculations, use:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.real_gc_sample.yml
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
├── 01_seed_gc/
├── 02_seed_consensus/
├── 03_probe_selection/
├── 04_dataset_probe/
├── 05_expanded_genes/
├── 06_expanded_gc/
└── 07_expanded_consensus/
```

Each stage writes a `manifest.json` with its inputs, outputs, parameters,
status, and metrics. The run-level `run_manifest.json` also includes the full
settings snapshot and a compact `pipeline_evolution` table for comparing seed,
probe, and expanded-network results.

## Resume From A Stage

Use `--start-at` and `--stop-after` to run any contiguous section:

```sh
python3 scripts/pipeline/run_pipeline.py \
  --config configs/gene_expansion.example.yml \
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
