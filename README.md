# GCnet: Inferring Dynamic Causal Interactions among Genes in Neuronal Development via Granger Causality with Implications for Intellectual Disability

We model gene expression data collected across multiple time points during in vitro brain development as time series and apply Granger causality to infer dynamic causal dependencies among genes. Based on this approach, we implement GCnet, a framework for constructing directed gene co-expression networks. 

Carrying out the entire Granger causality analysis and the follow-up community detection for networks of larger size (e.g. 45,904 in Kutsche’s data set (Kutsche, et al., 2018)) is possible but rather time-consuming, at least for moderate computational resources. Therefore, we introduce the following heuristic routine, which is applicable with two additional inputs 

a) a predefined list of 2,310 genes from the Human Phenotype Ontology database (Gargano, et al., 2024), and <br>
b) a guide gene selected from known syndrome-associated genes.

More details of our method can be found in the arXiv https://arxiv.org/abs/2508.05136

## Quick Start

Python 3.11 is the supported project baseline.

```sh
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Conda users can create the CPU environment with:

```sh
conda env create -f envs/cpu.yml
conda activate gene-analysis
```

Run the deterministic 1000-gene guided-flow sample:

```sh
python scripts/pipeline/run_pipeline.py --config configs/test/gene_expansion.sample.yml
```

Run the smaller sample that performs real statsmodels GC calculations:

```sh
python scripts/pipeline/run_pipeline.py --config configs/test/gene_expansion.real_gc_sample.yml
```

Run production-like real-GC checks on small gene sets:

```sh
python scripts/pipeline/run_pipeline.py --config configs/production_like/gene_expansion.kutsche.real_gc_small.yml
python scripts/pipeline/run_pipeline.py --config configs/production_like/gene_expansion.benito_human.real_gc_small.yml
python scripts/pipeline/run_pipeline.py --config configs/production_like/gene_expansion.benito_gorilla.real_gc_small.yml
```

Run production-style configs:

```sh
python scripts/pipeline/run_pipeline.py --config configs/production/gene_expansion.kutsche.yml
python scripts/pipeline/run_pipeline.py --config configs/production/gene_expansion.benito_human.yml
python scripts/pipeline/run_pipeline.py --config configs/production/gene_expansion.benito_gorilla.yml
```

Preview or generate one HPC job script per independent config:

```sh
python scripts/pipeline/submit_many.py configs/production/*.yml --dry-run
python scripts/pipeline/submit_many.py configs/production/*.yml --write-slurm scripts/pipeline/slurm_jobs
```

Check CUDA readiness on a local machine or GPU node:

```sh
python scripts/pipeline/check_cuda.py
```

Run the GPU-accelerated GC sample inside a CUDA/CuPy-ready environment:

```sh
python scripts/pipeline/run_pipeline.py --config configs/test/gene_expansion.gpu_sample.yml
```

Optional CUDA work uses separate environments from the validated CPU pipeline;
see [GPU acceleration](docs/GPU_ACCELERATION.md) before installing CuPy.

For a new preprocessed dataset, copy
`configs/templates/gene_expansion.generic_example.yml` and use `dataset.name:
generic_expression`. Generic datasets should provide `Gene` plus at least 5
ordered numeric timepoint/condition columns, with optional `preprocessing:`
settings for normalization and `log1p`/sqrt transforms.

Run verification and the fast pre-commit-equivalent suite:

```sh
python -m pytest tests
python -m pytest tests -m "not slow and not real_gc and not cuda"
python -m pytest tests --cov --cov-fail-under=75
pre-commit install
```

## What Gets Produced

| Stage | Main outputs |
| --- | --- |
| Seed GC | Directed GC CSV for seed genes |
| Seed consensus | Seed network files, top-5%-consensus subnetwork, SVG previews, GOI coassociation CSV, consensus history |
| Probe selection | Probe gene list chosen from seed coassociation frequencies |
| Dataset probe | Bidirectional GC CSV between probe genes and full dataset |
| Expanded genes | Probe network files, SVG preview, and discovered candidate gene set `n` |
| Expanded GC | Directed GC CSV for `n * (n - 1)` candidate pairs |
| Expanded consensus | Final priority list, expanded network files, top-5%-consensus subnetwork, SVG previews, consensus history |

Each run writes a folder under `results/pipeline/<run_name>/` with stage
manifests, parameters, metrics, validated output schemas, and network artifacts.
The easiest entrypoint after a run is `results/pipeline/<run_name>/RUN_SUMMARY.md`;
it summarizes settings, stage metrics, key artifacts, figure links, and the top
priority genes. SVG previews are also collected under
`results/pipeline/<run_name>/figures/`.
Network SVG previews are enabled by default with `network.write_svg: true`;
disabling this only skips the visual SVG files, not the GraphML/CSV/TXT/JSON
network files or the internal network analysis used for expansion and
consensus. The default SVG renderer is `networkx` for fast previews. Set
`network.svg_renderer: graphviz` and choose `network.svg_layout` for legacy
Graphviz-style figures, or render one/all Graphviz layouts after a run with
`scripts/pipeline/render_network_figures.py`.
Consensus stages also write a top-5%-consensus subnetwork so the strongest GOI
coassociation genes can be inspected with their internal subcommunities.
For a concrete stage-by-stage example of the folder layout and what to inspect,
see [Example pipeline run layout](examples/PIPELINE_RUN_LAYOUT.md).

## Documentation

| Need | Read |
| --- | --- |
| Scientific idea and pipeline flow | [Pipeline concept](docs/PIPELINE_CONCEPT.md) |
| Exact commands and resume examples | [Workflows](docs/WORKFLOWS.md) |
| Preparing new datasets | [Data preparation](docs/DATA_PREPARATION.md) |
| Setting up GPU-accelerated GC | [GPU acceleration](docs/GPU_ACCELERATION.md) |
| Stage inputs, outputs, metrics, networks | [Pipeline artifacts](docs/PIPELINE_ARTIFACTS.md) |
| What belongs in Git vs local data/results | [Data and results](docs/DATA_AND_RESULTS.md) |
| Citation, reuse, and co-authorship expectations | [Citation and reuse](docs/CITATION_AND_REUSE.md) |
| Developer checks and contribution workflow | [Development](docs/DEVELOPMENT.md) |
| Historical exploratory scripts | [Legacy index](docs/LEGACY_INDEX.md) |

## Repository Layout

```text
configs/                  YAML configs for sample and production-style runs
docs/                     Concise project documentation
examples/                 Small tracked fixtures
gene_analysis/            Canonical pipeline and analysis package
scripts/pipeline/         Supported command-line pipeline tools
scripts/reporting/        Supported reporting utilities
scripts/legacy/           Historical dashboards and exploratory scripts
tests/                    Fast verification suite
```

Large expression matrices, full GC outputs, generated figures, logs, zip
bundles, and pipeline results should stay local or externally linked rather
than committed to GitHub.

If you use this pipeline in research, please cite the repository. The code is
MIT-licensed for open reuse; see [Citation and reuse](docs/CITATION_AND_REUSE.md)
for citation and collaboration expectations.
