# Results Migration

Generated root-level outputs were moved under `results/` so the repository root can stay focused on code, documentation, and source data.

## New locations

- `results/granger/`: Granger causality result CSVs, including `granger_*`, `gc_r_*`, `*_GC_results.csv`, and Benito/Kutsche Granger exports.
- `results/coassociation/`: Coassociation run CSVs and related summary table inputs.
- `results/gene_lists/`: Generated gene-list files, top-N annotated CSVs, and summary text outputs.
- `results/comparisons/`: Relative-difference comparison CSVs and summaries.
- `results/degree_outputs/`: Degree quantile outputs.
- `results/figures/`: Newly generated figures from scripts.
- `results/bundles/`: Generated zip bundles.
- `results/logs/`: Generated log files.
- `results/pipeline/<run_name>/01_seed_gc/`: GC outputs among curated neurodeficiency seed genes.
- `results/pipeline/<run_name>/02_seed_consensus/`: Significant seed network and stable GOI coassociation frequencies.
- `results/pipeline/<run_name>/03_probe_selection/`: Probe gene lists selected by top percent or minimum frequency.
- `results/pipeline/<run_name>/04_dataset_probe/`: Full-dataset probe GC outputs.
- `results/pipeline/<run_name>/05_expanded_genes/`: Expanded candidate gene lists extracted from probe networks.
- `results/pipeline/<run_name>/06_expanded_gc/`: GC outputs among expanded candidate genes.
- `results/pipeline/<run_name>/07_expanded_consensus/`: Final stable priority lists for biological follow-up.

Code should use `project_paths.resolve_existing_path()` for configured inputs during the migration period. That resolver keeps old filename-only references working by searching the categorized `results/` folders.
