# Exhaustive genome-wide interaction pipeline

This workflow tests **every ordered non-self pair of retained genes**, then
calculates global BH-adjusted p-values and Storey q-values. It uses the shared
runner with `analysis.mode: fdr` and `analysis.workflow: genome_all_pairs`.

The same disk-backed runner also supports `analysis.workflow: seed_genome`:
all retained seed genes against the genome, in both directions, with each
ordered pair included once. See [the seed-to-genome and multi-anchor workflow](SEED_GENOME_ANCHOR_REPORTS.md)
for its configuration and the saved seed-network exports.

The production input check retained **45,902 genes and 35 WT samples**, giving
**2,106,947,702 tests**. The earlier seed overview and adaptive core-to-genome
probe are smaller, different families. Their adjusted values cannot substitute
for a complete genome-family correction. This workflow uses a separate output
directory and does not combine those earlier results.

## Statistical definition

```text
target ~ 1 + time + time² + predictor + time:predictor
H0: coefficient of time:predictor = 0
family size = G × (G − 1)
```

Production preprocessing retains the existing WT filter, mean aggregation of
duplicate symbols, median-ratio normalization and log1p transformation.
Replicates remain separate. The seed-file field is required by the shared
configuration schema but is **not read or used to restrict this workflow**.

All finite raw p-values enter one complete-family adjustment. Failed tests
remain in the denominator as p=1 bookkeeping placeholders, but their raw
statistics remain NaN and they do not appear in the adjusted export. Storey
uses the same closed-form bootstrap estimator as the seed comparison, with
lambda=0.05,0.10,...,0.95 and `pfdr=False`. Additive tail counts reproduce the
estimator without loading all p-values. A sensitivity estimate excludes failed
placeholders. Neither method is calculated independently per shard or anchor.

A pilot produces fitting checkpoints and timings, **not preliminary genome
q-values**. Adjustment requires the complete family. The interpretation limits
in the [study guide](BH_STOREY_STUDY_GUIDE.md) still apply: model arrows do not
establish causal regulation, and small q-values do not establish effect size.

## Install, plan and pilot

Install in the same Python environment:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-cuda.txt
python3 -m pip install -r requirements-genome.txt
python3 scripts/pipeline/check_cuda.py
```

Use [the production configuration](../configs/production/genome_all_pairs.kutsche.fdr_gpu.yml).
Outputs go to `results/pipeline/kutsche_genome_fdr_gpu/genome_fdr/`. Place the
checkout/results on a sufficiently large volume. Sorting scratch files live
in that run's `scratch/` directory.

Validate the input and inspect storage without fitting, writing output or
allocating a GPU:

```bash
python3 scripts/pipeline/run_genome_fdr.py \
  --config configs/production/genome_all_pairs.kutsche.fdr_gpu.yml --plan
```

Run a short pilot **on the GPU machine**:

```bash
python3 scripts/pipeline/run_genome_fdr.py \
  --config configs/production/genome_all_pairs.kutsche.fdr_gpu.yml \
  --pilot-shards 4
```

This commits at most four new fitting shards, approximately 4.2 million tests
at the default shard size. Existing work is reused. Inspect `fit_metrics.json`
for measured pairs/second, raw size and remaining fitting time. The measured
rate includes fitting and checkpoint writing during that invocation, but
excludes preprocessing, sorting, adjustment and reporting. A short pilot is
only an initial estimate, especially on a shared or network filesystem.

Run or resume the complete workflow through the standard entry point:

```bash
python3 scripts/pipeline/run_pipeline.py \
  --config configs/production/genome_all_pairs.kutsche.fdr_gpu.yml
```

## Separate GPU fitting and CPU postprocessing

Fitting evaluates parallel pair batches on **one configured GPU**. CPU code
calculates t-distribution p-values and manages files. This is not multi-GPU
scheduling; an exclusive lock rejects multiple writers to one run directory.

To release the GPU before sorting and reporting, first run:

```bash
python3 scripts/pipeline/run_genome_fdr.py \
  --config configs/production/genome_all_pairs.kutsche.fdr_gpu.yml --stage fit
```

After `fit_complete`, mount the same complete run directory and input on a
CPU machine and continue:

```bash
python3 scripts/pipeline/run_genome_fdr.py \
  --config configs/production/genome_all_pairs.kutsche.fdr_gpu.yml \
  --backend cpu_numpy --stage report
```

`--stage adjust` stops after adjustment. Each stage includes missing
prerequisites: if fitting is incomplete, the CPU invocation will finish it
on the CPU. Check `fit_complete` before transferring the job. Changing backend,
batch/shard size, sort resources, stage or time budget preserves genome
checkpoints; changing prepared inputs or statistical settings is rejected.

## Capacity and runtime

Integer pair IDs and compressed Parquet avoid billions of SQLite index entries
and repeated gene names in CSV. The numeric payload before compression is:

| Data | Bytes per test |
| --- | ---: |
| Raw: pair ID, coefficient, standard error, t statistic and p-value | 40 |
| Sorted checkpoint: pair ID and p-value | 16 per estimable test |
| Adjusted: pair ID, source/target IDs, p-value, BH and Storey | 40 per estimable test |

Assuming every test is estimable gives about **202 GB of column payload**,
before compression and Parquet overhead. Sorting also needs spill space.
Provision several hundred GB of fast writable storage; **500 GB free is an
initial planning allowance, not a guarantee**. Refine estimates using the
pilot and monitor free space. Raw and sorted checkpoints are retained; the
workflow does not automatically delete them to reclaim space.

Defaults are an 8 GB sort memory setting and four CPU threads. The setting is
not a hard process RSS limit: allow additional memory for Arrow, NumPy and
sorting overhead. A 32 GB host allocation is a practical starting point to
validate. The graph pair bitmap needs about 263 MB for 45,902 genes. Fitting
holds one host shard (1,048,576 pairs) and sends 32,768-pair GPU batches.
Adjustment reads one Parquet row group at a time. No stage intentionally
materializes all pair statistics in RAM.

Disk sorting uses DuckDB. Its [memory-management description](https://www.duckdb.org/2024/07/09/memory-management)
and [workload tuning guide](https://duckdb.org/docs/stable/guides/performance/how_to_tune_workloads)
explain spilling and why process memory needs headroom beyond the engine limit.

Estimate feasibility from the real pilot, not a small in-memory benchmark:

```text
remaining fitting hours = remaining pairs / measured pairs per second / 3600
```

CPU postprocessing adds wall time. For one GPU, GPU-hours approximately equal
the allocation's elapsed hours. An allocation of thousands of GPU-hours alone
does not establish runtime or storage feasibility.

## Progress and recovery

Console updates and `progress.json` expose the stage. Fitting shows checked
and committed tests, percentage, throughput and fitting-only ETA. Sorting
emits a heartbeat, scratch usage and engine progress when available (`-1`
means unavailable). Adjustment reports committed groups; reporting shows the
current shard and completed thresholds.

| Stage | Resume boundary |
| --- | --- |
| Fitting | Each committed raw Parquet shard |
| External sort | Entire sorted file; an interrupted sort restarts |
| BH/Storey adjustment | Each committed adjusted row group |
| Graph/anchor report | Entire report stage; an interrupted scan restarts |

Temporary files are flushed and atomically renamed, then the directory is
synced. Incomplete temporary files are ignored. Resume checks the prepared-data
fingerprint, contiguous raw ranges and row counts, and adjustment carry values.
These structural checks do not replace filesystem integrity or backups.

`analysis.max_seconds` pauses an invocation. Fitting checks between shards,
so allow one additional shard to finish. Sorting checks about every five
seconds and interrupts the query; that stage restarts later. Adjustment and
reporting check between chunks. Scheduler allocations need enough time for
the entire external sort plus a shutdown margin. Re-run the same command to
continue committed work. `resume: false` rejects an existing run.

## Outputs

- `manifest.json`: ordered genes, model, family and input fingerprint.
- `genes.csv`: integer IDs and symbols.
- `raw/part-*.parquet`: every test, including non-estimable fits.
- `sorted_p.parquet`: globally sorted finite raw p-values, retained for recovery.
- `adjusted/part-*.parquet`: finite tests with `pair_id`, `source_id`, `target_id`,
  `raw_p_value`, `bh_adjusted_p_value`, and `storey_q_value`. Filename order is
  increasing raw p and nondecreasing adjusted value. Failed tests stay in raw
  output rather than receiving fabricated adjusted results.
- `anchor_connections.csv`: estimable anchor pairs with symbols and both
  global adjustments; no separate anchor-family correction.
- `anchor_connections_all.csv` and `anchor_cutoff_comparison.csv`: corresponding
  exports for every requested anchor. Set `analysis.report_anchors` to add genes;
  the production configuration includes ZEB2, ZBTB18 and MECP2. Changing these report
  anchors does not invalidate existing fits. Each row identifies its anchor.
- `cutoff_comparison.csv`: both methods at powers 10^-14 through 10^-5,
  the earlier fixed thresholds, and configured alpha/comparison levels.
  It reports directed edges, undirected pairs, active genes, largest weak
  component, anchor component and distinct anchor neighbors. Legacy `zeb2_*`
  column names refer to the configured anchor.

When a reference seed file is available, reports also count connected genes,
anchor-component genes and neighbors outside that retained seed list. In the
full-genome workflow the reference list affects reporting only; it does not
restrict the test family. In `seed_genome` it defines the test family itself.
- `storey.json`: pi0 diagnostics and failure-policy sensitivity.
- `fit_metrics.json`, `progress.json`, `summary.json`, `README.md`: performance,
  status and a compact overview.

Comparisons use **adjusted value <= cutoff**, including complete ties across
chunk boundaries. Components ignore direction and their coverage denominator
is the full retained genome. Reciprocal edges count twice as directed edges
but once as neighbors. The report uses fixed thresholds; it does not search
for the exact first 95%-connectivity cutoff or constrain neighbor counts.

Decode raw pair IDs for G genes as follows:

```text
source = pair_id // (G − 1)
offset = pair_id % (G − 1)
target = offset + (offset >= source)
```

Join raw and adjusted data on `pair_id` for coefficients, and use `genes.csv`
for symbols. Keep complete data in Parquet and export only selected networks
to CSV. Community detection can use a chosen thresholded graph; a closed
neighborhood consists of the anchor and its direct neighbors. Both are
downstream choices and retain the original global correction. Neither
community detection nor biological candidate prioritization runs automatically.

## Validation scope

Small synthetic tests compare global adjustments with the existing in-memory
implementations, including failures and reverse adjustment across row groups.
They independently check graph counts against SciPy, enumerate 64-bit pair
IDs, interrupt fitting and adjustment, verify checkpoint reuse, and reject
changed inputs. Existing seed/q-value tests are also retained.

```bash
python3 -m pytest tests/test_genome_fdr.py tests/test_fdr_pipeline.py \
  tests/test_qvalue_connectivity.py -q
```

The production input has been inspected without fitting. No full genome run
or full-scale external-sort benchmark was performed locally. GPU execution
must be validated with the pilot on the target machine.
