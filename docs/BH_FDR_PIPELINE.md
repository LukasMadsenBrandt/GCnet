# BH interaction pipeline

The existing entrypoint now accepts `analysis.mode: gc` (the default) or `fdr`.
FDR mode implements Jing's exhaustive overview and cumulative anchor heuristic.
It uses the quadratic interaction model from the existing experiment:

```
target = a0 + a1*time + a2*time² + a3*predictor + a4*time*predictor + error
H0: a4 = 0
```

All individual WT samples are retained. The model uses a two-sided classical
OLS t test with `samples - 5` residual degrees of freedom. Centering the time
and predictor columns preserves the interaction coefficient and test. This is
not the categorical-day/HC3 experiment. No permutations or bootstrap are used
for selection. Regression direction does not establish causal regulation.

## Small local example

This command uses five synthetic genes and 35 samples only:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/test/gene_expansion.fdr_sample.yml
```

Rerun the same command to resume. To test CUDA on these same small inputs,
copy the sample config, choose a new `run_name`, set `analysis.backend: gpu_cuda`
and set `execution.gpu_device: 0`. The CPU reference is `cpu_numpy`.

`configs/production/gene_expansion.kutsche.fdr_gpu.yml` prepares the real-data
GPU run. It has not been executed as part of implementation. Use an allocated
GPU node and the existing CUDA environment instructions before running it.
The implementation raises a CUDA error when no device is available.
GPU startup also checks kernel compilation before loading expression data.
For a `libnvrtc.so.12` loader error after installing CUDA requirements, pull
the latest code (including NVIDIA namespace-package discovery), then run
`python3 scripts/pipeline/check_cuda.py`. See
[GPU acceleration](GPU_ACCELERATION.md) for package and library-path checks.

## The two searches

`analysis.workflow` selects `all_pairs`, `anchor`, or `both`.

1. **Overview:** test all ordered non-self pairs within the retained seed list.
   BH uses one complete family, not separate per-gene corrections. For G genes,
   `N = G*(G-1)`: 2,311 retained genes give 5,338,410 tests. The manifest records
   actual retained genes, omitted zero/missing genes, and planned test count.
   Primary alpha defaults to 0.05; comparison alpha defaults to 0.10.
2. **Local search:** begin at `gene_of_interest`. Test its outgoing, incoming,
   or both directions against the retained seed set. After each complete layer,
   combine previously tested and newly tested unique ordered pairs and rerun BH
   across that cumulative family. Old edges can lose significance as N changes.
   Never count a pair twice. Never reuse overview q-values for this family.
3. **Stopping:** stop after `max_depth` complete layers, when the discovered
   node count reaches `max_nodes`, or when no new nodes can be expanded. A
   complete layer can exceed the node target; significant edges are not truncated
   to force exactly 50 nodes. Discovery is cumulative: once discovered, a node
   stays eligible for expansion even if its original edge later loses significance.
   Current selected edges are always recalculated.
4. **Genome probe:** if `genome_expansion: true`, use the anchor plus discovered
   local nodes as the core and test them against every retained gene in the input
   expression matrix, once. Add new pairs to the same local family and rerun BH.
   A seed-only input cannot explore genes outside that input: use the complete
   expression matrix to enable this final step.

For sorted p-values, BH finds the largest rank k satisfying `p(k) <= alpha*k/N`
and selects every rank through k. It is a step-up procedure, not independent
pass/fail classification of each rank. Exported BH-adjusted p-values implement
the reverse cumulative minimum. Failed fits are recorded, remain in N as p=1
during correction, and cannot be selected.

The adaptive search reproduces the proposed cumulative-BH heuristic. Selecting
future tests based on earlier results does not itself establish global adaptive
FDR control. Its summaries and result email should describe it as exploratory.

## Progress, checkpoints, and interruption

Each run writes `results/pipeline/<run_name>/fdr/`:

- `progress.json`: current phase, cumulative test count, elapsed time and batch
  progress. Console updates are flushed after each committed batch. BH and
  export phases are also announced.
- `manifest.json`: input/settings fingerprint, model, sample count and test family.
- `all_pairs/tests.sqlite` and `anchor/tests.sqlite`: independent cumulative
  families, including failures, persisted layer plans and completed results.
- `all_pairs/overview/`, `anchor/depth_01/`, etc., and `anchor/genome_probe/`:
  all tests, selected edge tables for each alpha, summaries and SVG overviews.
- `anchor/core_genes.txt` and `core_summary.json`: core and local stop reason.
- `RUN_SUMMARY.md`: results and notes for the later email to Jing.

For example, monitor from another terminal:

```sh
watch -n 2 cat results/pipeline/fdr_sample/fdr/progress.json
```

Each batch writes pair identities and numerical results in one SQLite
transaction. Ctrl-C preserves completed batches; the interrupted batch is
recomputed on resume. A lock prevents concurrent writers to the same run.
Exports use temporary files and are regenerated if interrupted. BH needs the
complete family; it runs again if interrupted before completing a layer.

Resume refuses changed data, gene lists, model or semantic search settings.
Use a new `run_name` for such changes. Batch size and `max_seconds` may change
on resume. `max_seconds` is an optional per-invocation budget checked between
batches; it pauses the run rather than silently completing a partial BH family.
Increase that budget or remove it to continue. Existing runs with
`execution.resume: false` are protected from overwrite.

### Checkpoint write performance

The checkpoint keeps the unique `(source, target)` index that prevents duplicate
tests. It no longer maintains a second target-only index: when results arrive
in predictor order, that reverse index spreads updates over increasing numbers
of pages and can make saving progressively slower. Both outgoing and incoming
pair generation now group writes by predictor. Incoming membership checks can
use a small in-memory table joined against the unique pair index.

SQLite retains WAL journaling and `synchronous=FULL`. A connection-local page
cache can use up to approximately 256 MiB as needed; only one family is open at
a time. These changes do not change model results or disable transactional
checkpoint durability.

On resume, the input fingerprint is validated before an old `tests_target`
index is removed in a transaction. No saved test rows, IDs, layer plans or
completed summaries are deleted or rewritten. Pair membership determines resume
state, so an interrupted run can continue even if some incoming tests were saved
in the older order. Removing the index may take a moment and frees pages for
reuse; it does not necessarily shrink the database file immediately.

Test checkpoint throughput on the same filesystem as the pipeline:

```sh
python3 scripts/pipeline/benchmark_fdr_checkpoints.py --directory results/pipeline
```

The benchmark compares the previous index/cache/insertion pattern with the
optimized path using about one million synthetic rows in each direction. It
checks saved counts and pair sums, and removes only its own temporary directory.
It performs no regression fitting and never reads existing pipeline results.
Use `--pairs 262144` for a shorter check. Reported timings include database
saves and membership lookups, but exclude row generation and database close;
the comparison does not predict a full run's speed on another filesystem.

## GPU and memory

The predictor design is identical across its targets. Each predictor is
decomposed once using CPU double-precision SVD, then its orthonormal basis and
interaction contrast are cached on the GPU. Pair fitting uses batched GPU
projections and reductions rather than repeating SVD per pair. This avoids
CuPy 13's loop over individual SVD calls for matrices larger than 32 in either
dimension (these designs have 35 rows). `analysis.batch_size` controls temporary
device memory. Batches span predictor rows so a larger batch size is actually
used even when the seed set has fewer genes. The prepared B200 config uses
32,768 pairs per batch. An implementation update can resume existing checkpoints;
the model and hypothesis family are unchanged, with floating-point differences
possible within the tested numerical tolerance.
SciPy calculates t-distribution tails on the CPU. BH sorts the complete family
on the CPU; it must not be applied independently to GPU batches. Statistics
are checkpointed on disk, and CSV export streams rows. Global BH still needs
host arrays proportional to the number of tests. Allow adequate CPU RAM,
disk space, and GPU memory for the intended universe.

Fitting progress now reports cumulative `fit_seconds`, `checkpoint_seconds`,
`lookup_seconds` and `new_tests_per_second` for the current layer and invocation.
Fitting time includes GPU transfers and CPU p-values. Checkpoint and lookup time
can dominate after numerical optimization, especially on network filesystems.

Compare the old and cached implementations on a small synthetic workload:

```sh
python3 scripts/pipeline/benchmark_fdr.py --pairs 4096 --repeats 3
```

This checks numerical agreement and reports CPU/GPU timings without a real-data
analysis. Use `--cpu-only` on machines without CUDA. It excludes checkpoints and
preparation from timings; the live pipeline counters measure checkpoint costs.
GPU speedup depends on workload and host performance and should be measured.

Kutsche preprocessing reuses the existing WT filter, DESeq-style normalization
and log1p transformation. Normalization factors are estimated over the supplied
input matrix before restricting to the seed list, so using the full matrix
can differ from historical runs normalized on a seed-only matrix. The prepared
production config explicitly averages duplicate gene-symbol rows per sample
before normalization; it never averages biological replicates. Default duplicate
handling is an error. Generic inputs use a complete long CSV with columns
`Gene,Sample,Day,Expression`; sample days must agree across genes.

## Result interpretation and downstream use

Report selected directed edges, undirected pairs, gene coverage, isolated genes,
components, largest component and edge density at 5% and 10%. SVGs compare degree
distributions and selected edge counts without drawing millions of lines.

**Concern to include in the later email:** BH may include nearly every gene
and produce a dense network, making a focused gene set difficult to identify.
Gene coverage and edge density are different quantities; report both.

`network_edges.csv` is compatible with existing graph readers: its `p-value`
column contains BH-adjusted values, `raw_p_value` preserves the model evidence,
and `lag` is NaN. This workflow uses the shared pipeline entrypoint and config,
but has its own overview/traversal stages. It does not run the GC workflow's
Louvain consensus or probe-ranking stages. GC remains the default workflow.
FDR uses `analysis.workflow` rather than GC `--start-at`/`--stop-after` stages.

## Validation

```sh
python3 -m pytest tests/test_fdr_pipeline.py -m 'not cuda'
python3 -m pytest tests/test_fdr_pipeline.py -m cuda
```

The CPU tests use only tiny synthetic samples. They compare regression to
statsmodels, BH to its reference implementation, verify cumulative pair counts,
and exercise interruption, resume, input fingerprint checks and genome probing.
The CUDA test skips when hardware is unavailable; no speedup or GPU numerical
parity should be claimed until it has run on the target node.
