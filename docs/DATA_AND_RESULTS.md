# Data And Results Policy

The GitHub repository should stay lean and clone-friendly.

## Tracked

- Source code
- Tests
- Documentation
- Tiny example fixtures
- `results/MIGRATION.md`

## Not Tracked

- Large expression matrices
- Full Granger result CSVs
- Generated coassociation outputs
- Generated figures
- Zip bundles
- Logs and caches

## Local Layout

Place real datasets under `Data/` using the existing Kutsche and Benito folder conventions. Generated outputs should go under `results/`.

Production configs currently expect these local resources:

- `Data/Kutsche/Kutsche_Counts.txt`
- `Data/Kutsche/genes_all.txt`
- `Data/Kutsche/unique_genes.txt`
- `Data/Benito/Benito_Human`
- `Data/Benito/Benito_Gorilla`
- `Data/Benito/unique_genes.txt`
- `Data/Benito/gene_id_to_gene_name.txt`
- `Data/Benito/map_speciment_to_gene.csv`

Small fixtures under `examples/` are intended to be tracked and used in tests.

Pipeline output conventions use one run folder per YAML execution:

- `results/pipeline/<run_name>/01_seed_gc/`
- `results/pipeline/<run_name>/02_seed_consensus/`
- `results/pipeline/<run_name>/03_probe_selection/`
- `results/pipeline/<run_name>/04_dataset_probe/`
- `results/pipeline/<run_name>/05_expanded_genes/`
- `results/pipeline/<run_name>/06_expanded_gc/`
- `results/pipeline/<run_name>/07_expanded_consensus/`

The path resolver searches the categorized `results/` folders, so filename-only references can still work during the migration.
