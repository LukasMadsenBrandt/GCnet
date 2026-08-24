# Data Preparation

This guide describes what a new dataset must look like before it can be used by
the guided gene-expansion pipeline.

## Pipeline Input Contract

Every run needs these inputs in the YAML config:

- `dataset.name`: use `generic_expression` for a new preprocessed dataset, or
  one of the built-in dataset loaders: `kutsche`, `benito_human`,
  `benito_gorilla`. The older `sample_real_gc` name is kept as a sample alias.
- `dataset.expression_file`: expression matrix or raw count file for that
  dataset type.
- `dataset.full_gene_file`: optional legacy/search-space override. Production
  runs normally omit it so the searchable full gene landscape is derived from
  the expression matrix after dataset-specific preprocessing and gene-name
  mapping.
- `gene_of_interest`: the gene used for coassociation ranking, such as `ZEB2`
  or `MECP2`.
- `seed_gene_file`: one curated seed gene symbol per line.
- `preprocessing`: optional normalization, transformation, and replicate
  aggregation settings.

The gene of interest should appear in both the seed gene list and the expression
matrix. Seed genes must use the same identifiers as the expression matrix after
preprocessing.

## Generic Preprocessed Matrix

For a new dataset, the simplest route is to prepare a preprocessed expression
matrix in the same shape as `examples/sample_real_gc/expression.csv` and use
`dataset.name: generic_expression`.

Expected format:

```text
Gene,T1,T2,T3,T4,T5,T6
ZEB2,12.1,12.4,13.0,14.2,15.1,15.5
GENE_A,3.0,3.4,4.1,4.8,5.0,5.3
```

Requirements:

- first column must be named `Gene`;
- one row per gene;
- remaining columns must be ordered timepoints or ordered conditions;
- all expression values must be numeric;
- no duplicate gene names;
- gene symbols must match `seed_gene_file`;
- at least 5 ordered timepoint/condition columns are required by the generic
  loader; more timepoints are preferable for more robust Granger causality.

This generic route does not run Kutsche- or Benito-specific aggregation. It is
best for already-cleaned matrices and for running a completely new dataset
without adding a dataset-specific loader.

Start from `configs/templates/gene_expansion.generic_example.yml`:

```yaml
dataset:
  name: generic_expression
  expression_file: Data/MyDataset/expression.csv

gene_of_interest: ZEB2
seed_gene_file: Data/MyDataset/seed_genes.txt

preprocessing:
  normalize: none
  transform: log1p
  aggregation: robust
```

`aggregation` is ignored for generic preprocessed matrices because the matrix is
already expected to contain one value per gene per ordered timepoint/condition.

## Preprocessing Options

The pipeline supports the same YAML preprocessing block for all dataset types:

```yaml
preprocessing:
  normalize: none      # none | deseq | zscore
  transform: log1p     # none | log1p | sqrt
  aggregation: robust  # robust | mean | median
  export_all_replicates: true
  export_all_summarized: true
  export_subset_replicates: true
  export_subset_summarized: true
```

`normalize` is applied before transformation. For Kutsche and Benito raw-count
loaders, `deseq` is applied before replicate aggregation. `zscore` is applied to
the final matrix after replicate aggregation. For generic preprocessed matrices,
normalization is applied directly to the provided matrix.

`transform` supports `log1p` and `sqrt`. Plain natural log is intentionally not
available in the canonical pipeline because count-like expression matrices
often contain zeros. Use `log1p` for log-style scaling.

`aggregation` controls how Kutsche and Benito replicate columns are reduced to
one value per day. It can be `robust`, `mean`, or `median`.

The four export switches are independent. `all` includes every gene remaining
after dataset-specific filtering. `subset` includes every available gene from
`seed_gene_file` (for example `unique_genes.txt`), not merely the single
`gene_of_interest`. `replicates` is the pre-aggregation matrix; `summarized` is
the post-aggregation, duplicate-symbol-collapsed matrix passed to Granger
causality. Generic preprocessed matrices contain no separate replicate metadata,
so their replicate export mirrors their provided timepoint matrix.

The files and an audit manifest are written under
`results/pipeline/<run>/00_preprocessing/`. All switches default to `false` to
avoid unexpectedly duplicating very large production matrices.

## Kutsche Dataset

Production Kutsche configs expect:

- `Data/Kutsche/Kutsche_Counts.txt`
- `Data/Kutsche/unique_genes.txt`

`Kutsche_Counts.txt` is local-only and not committed to GitHub. It should be a
tab-separated count matrix where the first column is `Gene` and the remaining
columns are numeric expression columns. The current loader extracts columns
containing `WT`, orders them by day from names containing `d<day>`, removes
all-zero genes, and aggregates replicates by robust weighted mean.

`unique_genes.txt` is the curated seed set, one gene per line. The searchable
gene universe is derived from `Kutsche_Counts.txt` after WT filtering,
preprocessing, and replicate aggregation.

## Benito Datasets

Production Benito configs expect:

- `Data/Benito/Benito_Human`
- `Data/Benito/Benito_Gorilla`
- `Data/Benito/unique_genes.txt`
- `Data/Benito/gene_id_to_gene_name.txt`
- `Data/Benito/map_speciment_to_gene.csv`

`unique_genes.txt` is the curated seed set, one gene per line. The searchable
gene universe is derived from `Benito_Human` or `Benito_Gorilla` after gene-name
mapping, replicate filtering, preprocessing, and duplicate-symbol aggregation.

`Benito_Human` and `Benito_Gorilla` are local-only and not committed to GitHub.
They should be featureCounts-style tab-separated files with metadata on the
first line and the real header on the second line. The loader reads them with
`header=1`, so the second line should contain columns such as:

```text
Geneid  Chr  Start  End  Strand  Length  SRR...
```

`gene_id_to_gene_name.txt` is tracked because it is project-specific reference
metadata. It must be tab-separated with one mapping per line:

```text
ENSEMBL_GENE_ID    GENE_SYMBOL
```

`map_speciment_to_gene.csv` is also tracked metadata. It must contain:

```text
Run,Organism,Time_point
```

The Benito loader maps `Geneid` to `Gene_Name`, renames SRR columns with their
timepoint, applies the configured preprocessing transform, and aggregates
replicates per day using robust weighted means.

The production Benito configs use `transform: log1p` because Benito count files
contain zeros.

## Pre-Run Checklist

Before launching a production run:

- confirm the large expression matrix exists at the path in the YAML config;
- confirm `gene_of_interest` appears in the seed list and processed expression
  matrix;
- confirm all seed genes are present after preprocessing;
- remove duplicate gene names from gene lists;
- choose preprocessing settings before comparing runs;
- choose a p-value threshold and probe-selection rule deliberately;
- run one sample pipeline first.

Useful first checks:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/test/gene_expansion.sample.yml
python3 scripts/pipeline/run_pipeline.py --config configs/test/gene_expansion.real_gc_sample.yml
```

For a new production config, start with an early stop to validate loading and
seed GC before launching the whole workflow:

```sh
python3 scripts/pipeline/run_pipeline.py \
  --config configs/my_dataset.yml \
  --stop-after 01_seed_gc
```

Outputs are written under `results/pipeline/<run_name>/`. See
`docs/PIPELINE_ARTIFACTS.md` for every stage artifact and manifest.
