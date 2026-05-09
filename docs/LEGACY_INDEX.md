# Legacy Index

Historical exploratory scripts are preserved under `scripts/legacy/`, but they
are not the supported pipeline interface. Many still contain hardcoded filenames
from earlier analyses. New work should use:

```sh
python3 scripts/pipeline/run_pipeline.py --config configs/gene_expansion.kutsche.yml
```

## Network Exploration

`scripts/legacy/network/`

- historical network construction and inspection scripts
- path/neighborhood exploration around genes of interest
- old induced-network and Steiner-style helpers

The canonical pipeline now writes seed, probe, and expanded network artifacts
automatically during stages `02_seed_consensus`, `05_expanded_genes`, and
`07_expanded_consensus`.

## Old Analysis And UCloud Runs

`scripts/legacy/analysis/`

- historical UCloud consensus scripts
- partition-comparison experiments
- noise-baseline analysis
- older standalone coassociation-frequency scripts

The canonical consensus flow now lives in the YAML pipeline runner.

## Figures And Dash Experiments

`scripts/legacy/figures/` and `scripts/legacy/dashboard_apps/`

- historical figure and degree-distribution helpers
- standalone Dash experiments
- exploratory plotting scripts

The old network exploration dashboard is preserved as
`scripts/legacy/dashboard_apps/app.py`. It is not the primary project
entrypoint; new analysis runs should use the YAML pipeline runner.

## Old Entrypoints And Maintenance

`scripts/legacy/entrypoints/` and `scripts/legacy/maintenance/`

- old root-level commands such as `main.py` and threshold wrappers
- historical GC gathering and maintenance helpers

Production-style runs should now use the dataset YAML configs in `configs/`.
