# Development

## Setup

```sh
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Python 3.11 is the supported baseline. Python 3.9 is no longer supported.
Conda users can instead run:

```sh
conda env create -f envs/cpu.yml
conda activate gene-analysis
```

The canonical pipeline defaults to lightweight NetworkX SVG previews without
requiring system Graphviz. The optional `network.svg_renderer: graphviz` mode
and `scripts/pipeline/render_network_figures.py --renderer graphviz` require
the Graphviz `dot` executable in addition to the Python `graphviz` package.

## CUDA Environments

CPU remains the validated default. Keep CUDA experiments in a separate
environment so normal development and GitHub-ready tests stay reproducible on
machines without a GPU.

For local CuPy-only backend development, use the pip-based CUDA 12 wheel stack:

```sh
conda env create -f envs/cuda-cupy.yml
conda activate gene-cuda-cupy
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda-cupy.txt
python scripts/pipeline/check_cuda.py
```

If you prefer `venv` and already have Python 3.11 available:

```sh
python3.11 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda-cupy.txt
python scripts/pipeline/check_cuda.py
```

For RAPIDS/cuGraph consensus experiments, use the conda environment template:

```sh
conda env create -f envs/cuda-rapids.yml
conda activate gene-cuda-rapids
python scripts/pipeline/check_cuda.py
```

RAPIDS releases are strict about Python, CUDA, and driver compatibility. If the
template fails on a cluster, use the official RAPIDS install selector for that
cluster's CUDA module and Python version, then run `check_cuda.py` and save the
report with:

```sh
python scripts/pipeline/check_cuda.py --output-file results/cuda_compatibility.json
```

The recommended accelerated mode is `gc_backend: gpu_cuda` with
`consensus_backend: cpu_louvain`. This accelerates lag-1 GC with CuPy while
preserving the validated CPU Louvain community detection. `gpu_cugraph` exists
for consensus experiments only and should not be used for scientific production
until CPU/GPU consensus parity passes on the target dataset and hardware.

## Verification

Run fast checks before committing. The sample pipeline test is the prerequisite
guard for cleanup changes because it exercises all seven pipeline stages on a
medium-sized deterministic fixture, including machine-readable and SVG network
artifacts, without launching the full supercomputer-scale GC job:

```sh
pre-commit install
python -m py_compile $(find . -name '*.py' -not -path './venv/*')
python -m pytest tests/test_pipeline.py::test_sample_fixture_pipeline_runs_all_stages
python -m pytest tests -m "not slow and not real_gc and not cuda"
python -m pytest tests --cov --cov-fail-under=75
pre-commit run --all-files
python scripts/reporting/gene_list_compare.py
python scripts/reporting/relative_difference.py
```

If the shell cannot find the `pre-commit` executable after installation, use
`python -m pre_commit run --all-files`.

Run the real-GC sample before major changes to Granger, consensus,
preprocessing, or runner behavior:

```sh
python -m pytest tests -m real_gc
```

CUDA backends are represented in the config and manifests, but are not treated
as production-ready until parity and benchmark tests exist for the target
hardware. CPU remains the validated default:

```yaml
execution:
  gc_backend: cpu_statsmodels
  consensus_backend: cpu_louvain
  gpu_device: null
```

Future CUDA checks should use dedicated markers so they never run accidentally
in pre-commit:

```sh
python -m pytest tests -m cuda
python -m pytest tests -m cuda_parity
python -m pytest tests -m cuda_benchmark
```

Check a machine's CUDA readiness before enabling GPU work:

```sh
python scripts/pipeline/check_cuda.py
python scripts/pipeline/check_cuda.py --json
python scripts/pipeline/check_cuda.py --output-file results/cuda_compatibility.json
```

CPU/GPU parity reports should compare GC CSVs, coassociation frequency CSVs,
and benchmark speedups before any CUDA backend is accepted for production runs.
Use `scripts/pipeline/benchmark_cuda_gc.py --min-speedup <factor>` when a
hardware-specific speedup should be enforced in addition to parity.
Use `scripts/pipeline/benchmark_consensus_backend.py` to profile consensus
timings and compare future consensus backends against `cpu_louvain`.

Avoid running full supercomputer-scale Granger or consensus jobs as routine cleanup checks.

## Documentation

Keep documentation centralized. The intended doc set is:

- `README.md`
- `docs/PIPELINE_CONCEPT.md`
- `docs/WORKFLOWS.md`
- `docs/PIPELINE_ARTIFACTS.md`
- `docs/DATA_AND_RESULTS.md`
- `docs/DATA_PREPARATION.md`
- `docs/GPU_ACCELERATION.md`
- `docs/DEVELOPMENT.md`
- `docs/CITATION_AND_REUSE.md`
- `docs/LEGACY_INDEX.md`

Supported modules and scripts should have module docstrings plus docstrings for
public functions/classes. Legacy scripts are documented once in
`docs/LEGACY_INDEX.md`.
