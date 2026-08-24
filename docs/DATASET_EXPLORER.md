# Dataset Explorer

The responsive Dash explorer is intended for choosing and auditing a scientific
p-value threshold before interpreting the later pipeline stages. It does not
change a pipeline run.

Enable the relevant artifacts in the YAML config:

```yaml
preprocessing:
  normalize: none
  transform: none
  aggregation: robust
  export_all_replicates: true
  export_all_summarized: true
  export_subset_replicates: true
  export_subset_summarized: true

network:
  p_value_threshold: 0.0015

execution:
  seed_gc_store_all_pairs: true
```

`seed_gc_store_all_pairs` changes only the stage-1 CSV storage threshold to
`1.0`. Stage 2 and every later network operation still filter that CSV using
`network.p_value_threshold`. The all-pairs file is named
`01_seed_gc/seed_gc_all_pairs.csv`, so it has a separate resume checkpoint from
the ordinary thresholded `seed_gc.csv`.

Every attempted pair receives a row. Successful tests retain a full-precision
p-value and leave `error` empty; constant series, failed tests, and malformed
results use `p-value=NaN` and preserve a short description in `error`. The
dashboard compares CSV rows with the stage manifest's attempted-pair count and
shows a warning when coverage is incomplete, for example after an interrupted
or mismatched resume.

Launch the explorer after stage 1 or a complete run:

```sh
python3 scripts/pipeline/explore_dataset.py \
  --run-dir results/pipeline/<run_name> \
  --gene ZEB2
```

The interface provides:

- a searchable gene-of-interest selector;
- multi-gene summarized-expression comparison;
- pre-aggregation replicate traces when exported;
- lower-tail p-value thresholds for several selectable quantiles;
- overlaid `-log10(p)` histograms for all pairs and pairs involving the GOI;
- incoming, outgoing, directed, and unique-related-gene counts per quantile;
- the GOI's share of all relationships retained at each quantile;
- both directed GC results for explicitly selected comparison genes;
- result-coverage, gene-count, timepoint-count, and valid-p-value summaries.

Override any auto-discovered artifact when comparing files from another run:

```sh
python3 scripts/pipeline/explore_dataset.py \
  --run-dir results/pipeline/<run_name> \
  --summarized-csv path/to/expression.csv \
  --replicates-csv path/to/replicates.csv \
  --gc-csv path/to/all_pairs.csv
```
