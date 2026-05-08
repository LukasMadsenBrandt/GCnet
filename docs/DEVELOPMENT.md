# Development

## Setup

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Graphviz must also be installed on the system for graph rendering.

## Verification

Run fast checks before committing. The sample pipeline test is the prerequisite
guard for cleanup changes because it exercises all seven pipeline stages on a
medium-sized deterministic fixture without launching the full supercomputer-scale
GC job:

```sh
python3 -m py_compile $(find . -name '*.py' -not -path './venv/*')
python3 -m pytest tests/test_pipeline.py::test_sample_fixture_pipeline_runs_all_stages
python3 -m pytest tests/test_pipeline.py::test_sample_real_gc_pipeline_runs_all_stages
python3 -m pytest tests
python3 scripts/reporting/gene_list_compare.py
python3 scripts/reporting/relative_difference.py
```

Avoid running full supercomputer-scale Granger or consensus jobs as routine cleanup checks.

## Documentation

Keep documentation centralized. The intended doc set is:

- `README.md`
- `docs/PIPELINE_CONCEPT.md`
- `docs/WORKFLOWS.md`
- `docs/PIPELINE_ARTIFACTS.md`
- `docs/DATA_AND_RESULTS.md`
- `docs/DEVELOPMENT.md`
- `docs/LEGACY_INDEX.md`

Supported modules and scripts should have module docstrings plus docstrings for
public functions/classes. Legacy scripts are documented once in
`docs/LEGACY_INDEX.md`.
