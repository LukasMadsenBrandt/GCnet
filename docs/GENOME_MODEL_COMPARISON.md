# Genome-wide model comparison

The workflow compares three directed time-interaction models over the same retained gene universe and the same ordered non-self pairs:

- Model 0: `target ~ 1 + time + time^2 + predictor + time:predictor`
- Model 1: `target ~ 1 + time + time^2 + predictor + time:predictor + predictor^2`
- Model 2: `target ~ 1 + time + predictor + time:predictor`

The tested coefficient is `time:predictor` in every model. Each model has its own complete global BH and Storey adjustment. Failed planned tests remain in the correction denominator as p=1 placeholders.

## Automatic selection

For each adjustment method, the pipeline selects the largest complete tied adjusted-value block satisfying:

```text
adjusted value * selected directed connections < 1
```

The resulting cutoffs and connection counts are stored in `selection.json`. Tied adjusted values are never split to force the product below one.

## Per-model outputs

Each run writes the usual raw, adjusted, connectivity, and marker reports. `model_comparison/` additionally contains:

- `significant_pairs_bh.parquet` and `significant_pairs_storey.parquet`: globally selected directed pairs with raw statistics, both adjusted values, AIC, BIC, and adjusted R-squared.
- `reference_seed_pairs_all.parquet`: all estimable and failed directed reference-seed pairs. Failed rows have blank adjusted values and diagnostics.
- `marker_*_all_pairs.parquet`: all directed pairs involving each configured marker.
- `README.json`: model equation, parameter count, diagnostic definitions, and automatic cutoffs.

Residual sum of squares is checkpointed once per pair. AIC, BIC, and adjusted R-squared are derived during export, avoiding three additional genome-scale raw columns.

## Runtime profiling

`timing.json` accumulates active runtime across resumable invocations and also records calendar span. `timing.csv` is the readable table. Timed stages are preprocessing, fitting, external sorting, global adjustment, automatic selection, reporting, and comparison export.

Detailed operations include input preparation, fingerprinting, model preparation, GPU index transfer, model computation, result transfer, CPU p-value calculation, shard statistics, Parquet checkpoints, external sorting, q-value computation, adjustment writes, graph updates, marker-row serialization, and every comparison export.

## UCloud run sequence

The production configurations assume one NVIDIA B200, 48 vCPUs, and 288 GB host RAM. DuckDB may use up to 192 GB and 32 threads for sorting and comparison exports. The remaining resources are reserved for the operating system, filesystem cache, preprocessing process, and GPU runtime.

Run the warm-kernel microbenchmark for all three models on the allocated GPU:

```bash
python3 scripts/pipeline/benchmark_fdr.py --model model0 --pairs 32768 --repeats 5
python3 scripts/pipeline/benchmark_fdr.py --model model1 --pairs 32768 --repeats 5
python3 scripts/pipeline/benchmark_fdr.py --model model2 --pairs 32768 --repeats 5
```

Plan and pilot each model before the full jobs:

```bash
python3 scripts/pipeline/run_genome_fdr.py --config configs/production/genome_all_pairs.kutsche.model0.fdr_gpu.yml --plan
python3 scripts/pipeline/run_genome_fdr.py --config configs/production/genome_all_pairs.kutsche.model0.fdr_gpu.yml --pilot-shards 4

python3 scripts/pipeline/run_genome_fdr.py --config configs/production/genome_all_pairs.kutsche.model1.fdr_gpu.yml --plan
python3 scripts/pipeline/run_genome_fdr.py --config configs/production/genome_all_pairs.kutsche.model1.fdr_gpu.yml --pilot-shards 4

python3 scripts/pipeline/run_genome_fdr.py --config configs/production/genome_all_pairs.kutsche.model2.fdr_gpu.yml --plan
python3 scripts/pipeline/run_genome_fdr.py --config configs/production/genome_all_pairs.kutsche.model2.fdr_gpu.yml --pilot-shards 4
```

Resume each pilot as a complete run by omitting `--pilot-shards`:

```bash
python3 scripts/pipeline/run_genome_fdr.py --config configs/production/genome_all_pairs.kutsche.model0.fdr_gpu.yml
python3 scripts/pipeline/run_genome_fdr.py --config configs/production/genome_all_pairs.kutsche.model1.fdr_gpu.yml
python3 scripts/pipeline/run_genome_fdr.py --config configs/production/genome_all_pairs.kutsche.model2.fdr_gpu.yml
```

After all three runs complete, create the comparison using Storey q-values:

```bash
python3 scripts/pipeline/compare_genome_models.py \
  --model0 results/pipeline/kutsche_genome_model0_fdr_gpu \
  --model1 results/pipeline/kutsche_genome_model1_fdr_gpu \
  --model2 results/pipeline/kutsche_genome_model2_fdr_gpu \
  --method storey \
  --output results/reports/genome_model_comparison
```

The comparison writes model-level network and runtime results, common significant directed pairs, a strict reference-seed subset, and counts of which model has the best AIC, BIC, and adjusted R-squared. Lower AIC/BIC is better; higher adjusted R-squared is better; exact ties are reported separately.
