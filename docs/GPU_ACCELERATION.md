# GPU Acceleration

GPU support is intended to accelerate the Granger causality stages while keeping
the validated CPU Louvain consensus workflow unchanged.

## Recommended Mode

Use this backend combination for production-style accelerated runs:

```yaml
execution:
  gc_backend: gpu_cuda
  consensus_backend: cpu_louvain
  gpu_device: 0
```

`gpu_cuda` uses CuPy for lag-1 Granger calculations. `cpu_louvain` keeps the
existing Python Louvain consensus implementation and can still use CPU cores via
`execution.max_workers`.

The `gpu_cugraph` consensus backend is available for experiments, but it is not
the recommended scientific backend. Benchmarks showed that cuGraph Louvain can
produce different community structure from the validated Python Louvain method
on expanded networks. Treat it as exploratory until a CPU/GPU consensus parity
report passes for the target dataset and hardware.

## Requirements

- Python 3.11
- NVIDIA GPU with a working driver
- CUDA 12-compatible CuPy wheel or Conda environment
- Enough GPU memory for the selected GC chunk size

Check the machine from the same environment that will run the pipeline:

```sh
python scripts/pipeline/check_cuda.py
python scripts/pipeline/check_cuda.py --output-file results/cuda_compatibility.json
```

For the recommended `gpu_cuda` backend, `GPU GC ready` must be `True`.
`GPU consensus ready` is only needed for experimental `gpu_cugraph` work.

## Setup

Conda users can create the CuPy environment with:

```sh
conda env create -f envs/cuda-cupy.yml
conda activate gene-cuda-cupy
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda-cupy.txt
python scripts/pipeline/check_cuda.py
```

If the cluster uses modules, load the CUDA module first, then create or activate
the Python 3.11 environment and run `check_cuda.py` from an allocated GPU node.

For RAPIDS/cuGraph experiments only:

```sh
conda env create -f envs/cuda-rapids.yml
conda activate gene-cuda-rapids
python scripts/pipeline/check_cuda.py
```

RAPIDS versions are strict about Python, CUDA, and driver compatibility. Prefer
the official RAPIDS install selector when a supercomputer provides a specific
CUDA module.

## Running A GPU-GC Pipeline

Start with the tracked sample:

```sh
python scripts/pipeline/run_pipeline.py --config configs/test/gene_expansion.gpu_sample.yml
```

For a new or production-like config, change only the execution block first:

```yaml
execution:
  max_workers: 32
  chunk_size: 5000
  resume: true
  gc_backend: gpu_cuda
  consensus_backend: cpu_louvain
  gpu_device: 0
```

Tune `chunk_size` to fit GPU memory. Larger chunks reduce overhead but require
more memory.

## Validation Before Production

Before trusting GPU GC on new hardware or a new dataset shape, run a CPU/GPU
parity benchmark:

```sh
python scripts/pipeline/benchmark_cuda_gc.py \
  --config configs/test/gene_expansion.real_gc_sample.yml
```

For production-like configs:

```sh
python scripts/pipeline/benchmark_cuda_gc.py \
  --config configs/production_like/gene_expansion.kutsche.real_gc_small.yml
```

The report should show:

- no missing or extra GC rows;
- no p-value disagreements above tolerance;
- no significant-decision disagreements;
- acceptable speedup for the machine.

Only after that should you run full production GC with `gc_backend: gpu_cuda`.

## What Stays On CPU

Consensus clustering should normally stay on:

```yaml
consensus_backend: cpu_louvain
```

This preserves the validated Louvain community-detection behavior while still
using available CPU cores for repeated partitions. This is the safer scientific
choice until an alternative consensus backend passes parity on the target
analysis.
