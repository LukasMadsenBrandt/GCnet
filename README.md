# Gene Analysis Dashboard

Guided neurodeficiency gene-expansion pipeline using Granger causality,
network construction, stability-controlled Louvain consensus, and ranked
candidate gene lists. The repository also includes a Dash dashboard for
interactive network exploration.

The pipeline is intentionally not brute-force all-pairs discovery first. It
starts from a curated seed gene set, finds directed GC relationships inside that
trusted set, uses the strongest seed-network communities to probe the full
dataset, extracts the discovered network gene set `n`, and only then runs
expanded GC on `n * (n - 1)` ordered pairs.

## Quick Start

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the deterministic 1000-gene guided-flow sample:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.sample.yml
```

Run the smaller sample that performs real statsmodels GC calculations:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.real_gc_sample.yml
```

Run production-style configs:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.kutsche.yml
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.benito_human.yml
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.benito_gorilla.yml
```

Run verification:

```sh
python3 -m pytest tests
```

## What Gets Produced

| Stage | Main outputs |
| --- | --- |
| Seed GC | Directed GC CSV for seed genes |
| Seed consensus | Seed network files, GOI coassociation CSV, consensus history |
| Probe selection | Probe gene list chosen from seed coassociation frequencies |
| Dataset probe | Bidirectional GC CSV between probe genes and full dataset |
| Expanded genes | Probe network files and discovered candidate gene set `n` |
| Expanded GC | Directed GC CSV for `n * (n - 1)` candidate pairs |
| Expanded consensus | Final priority list, expanded network files, consensus history |

Each run writes a folder under `results/pipeline/<run_name>/` with stage
manifests, parameters, metrics, and network artifacts.

## Documentation

| Need | Read |
| --- | --- |
| Scientific idea and pipeline flow | [Pipeline concept](docs/PIPELINE_CONCEPT.md) |
| Exact commands and resume examples | [Workflows](docs/WORKFLOWS.md) |
| Stage inputs, outputs, metrics, networks | [Pipeline artifacts](docs/PIPELINE_ARTIFACTS.md) |
| What belongs in Git vs local data/results | [Data and results](docs/DATA_AND_RESULTS.md) |
| Developer checks and contribution workflow | [Development](docs/DEVELOPMENT.md) |
| Historical exploratory scripts | [Legacy index](docs/LEGACY_INDEX.md) |

## Repository Layout

```text
configs/                  YAML configs for sample and production-style runs
docs/                     Concise project documentation
examples/                 Small tracked fixtures
gene_analysis/            Canonical pipeline and analysis package
gene_analysis_benito/     Benito dataset helpers
gene_analysis_kutsche/    Kutsche dataset helpers
scripts/pipeline/         Supported command-line pipeline tools
scripts/reporting/        Supported reporting utilities
scripts/legacy/           Historical exploratory scripts
tests/                    Fast verification suite
```

Large expression matrices, full GC outputs, generated figures, logs, zip
bundles, and pipeline results should stay local or externally linked rather
than committed to GitHub.
