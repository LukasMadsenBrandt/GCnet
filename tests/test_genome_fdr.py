"""Small exhaustive reference checks; no real datasets or GPU jobs."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

pytest.importorskip("duckdb")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("numba")

from gene_analysis.analysis.fdr import benjamini_hochberg
from gene_analysis.analysis.storey import bootstrap_pi0, storey_qvalues
from gene_analysis.pipeline.genome_fdr import GenomeFDRRunner, decode_pairs, pi0_from_counts, GRID
from gene_analysis.pipeline.runner import PipelineConfig, run_pipeline


@pytest.fixture
def genome_config(tmp_path, monkeypatch):
    import gene_analysis.pipeline.runner as runner
    monkeypatch.setattr(runner, "results_path", lambda *parts: tmp_path.joinpath(*parts))
    rng = np.random.default_rng(311)
    times = np.repeat(np.arange(5), 7)
    values = rng.normal(size=(35, 12)) + 10
    values[:, 1] += times * values[:, 0]
    values[:, -1] = (times == 2).astype(float)  # Non-estimable predictor.
    genes = ["ZEB2", *[f"G{i}" for i in range(1, 12)]]
    path = tmp_path / "expression.csv"
    pd.DataFrame([(gene, f"s{s:02d}", times[s], values[s, g])
                  for g, gene in enumerate(genes) for s in range(35)],
                 columns=["Gene", "Sample", "Day", "Expression"]).to_csv(path, index=False)
    return PipelineConfig.from_dict(dict(run_name="genome_test",
        dataset=dict(name="sample_long", expression_file=str(path)),
        seed_gene_file=str(tmp_path / "unused_seeds.txt"), gene_of_interest="ZEB2",
        analysis=dict(mode="fdr", workflow="genome_all_pairs", backend="cpu_numpy", batch_size=7,
                      genome_shard_pairs=23, genome_sort_memory="64MB", genome_sort_threads=1),
        execution=dict(resume=True, progress=False)))


def test_pair_ids_and_streamed_pi0():
    a, b = decode_pairs(np.arange(20, dtype=np.int64), 5)
    assert list(zip(a, b)) == [(x, y) for x in range(5) for y in range(5) if x != y]
    # Beyond signed int32: genome sizes must not silently wrap pair IDs.
    a, b = decode_pairs(np.array([2_500_000_000], dtype=np.int64), 60_000)
    assert int(a[0]) != int(b[0]) and 0 <= b[0] < 60_000
    rng = np.random.default_rng(3)
    p = np.r_[rng.uniform(size=1000), rng.beta(.2, 1, 2000), 1., 1.]
    tails = sum(np.array([(part >= level).sum() for level in GRID]) for part in np.array_split(p, 13))
    actual = pi0_from_counts(tails, len(p), p.max())
    expected = bootstrap_pi0(p)
    assert actual["pi0"] == pytest.approx(expected["pi0"], rel=1e-14)
    np.testing.assert_allclose(actual["bootstrap_mse"], expected["bootstrap_mse"], rtol=1e-14)


def _read_parts(path):
    return pd.concat([pq.read_table(p).to_pandas() for p in sorted(path.glob("*.parquet"))], ignore_index=True)


def test_global_reference_resume_and_graphs(genome_config, monkeypatch):
    config = genome_config
    pilot = replace(config, analysis=replace(config.analysis, genome_shards_per_invocation=2))
    run_pipeline(pilot)
    root = config.run_dir / "genome_fdr"
    assert json.loads((root / "progress.json").read_text())["status"] == "paused"
    saved = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in (root / "raw").glob("*.parquet")}
    assert sum(pq.read_metadata(p).num_rows for p in saved) == 46
    assert not (root / "sorted_p.parquet").exists()  # No subset-family correction.

    original_sort = GenomeFDRRunner._sort
    def small_groups(self):
        result = original_sort(self)
        if result and not list((self.directory / "adjusted").glob("*.parquet")):
            path = self.directory / "sorted_p.parquet"
            table = pq.read_table(path)
            pq.write_table(table, path, row_group_size=7)
        return result
    monkeypatch.setattr(GenomeFDRRunner, "_sort", small_groups)
    runner = GenomeFDRRunner(config)
    monkeypatch.setattr(runner, "expired", lambda: bool(list((root / "adjusted").glob("*.parquet"))))
    runner.run()  # Interrupt after one committed reverse adjustment group.
    assert len(list((root / "adjusted").glob("*.parquet"))) == 1
    adjusted_first = next((root / "adjusted").glob("*.parquet"))
    adjusted_time = adjusted_first.stat().st_mtime_ns
    run_pipeline(config)
    assert adjusted_first.stat().st_mtime_ns == adjusted_time
    for path, (mtime, digest) in saved.items():
        assert path.stat().st_mtime_ns == mtime
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert json.loads((root / "progress.json").read_text())["status"] == "complete"
    raw = _read_parts(root / "raw").sort_values("pair_id")
    assert raw.pair_id.tolist() == list(range(12*11))
    adjusted = _read_parts(root / "adjusted")
    p = raw.raw_p_value.to_numpy()
    assert np.isnan(p).sum() >= 11
    expected_bh = benjamini_hochberg(p)
    expected_storey, details = storey_qvalues(p)
    ids = adjusted.pair_id.to_numpy()
    np.testing.assert_allclose(adjusted.bh_adjusted_p_value, expected_bh[ids], rtol=1e-14)
    np.testing.assert_allclose(adjusted.storey_q_value, expected_storey[ids], rtol=1e-14)
    assert np.all(np.diff(adjusted.bh_adjusted_p_value) >= 0)
    assert json.loads((root / "storey.json").read_text())["pi0"] == pytest.approx(details["pi0"])
    for row in pd.read_csv(root / "cutoff_comparison.csv").itertuples():
        q = adjusted.bh_adjusted_p_value if row.method == "bh" else adjusted.storey_q_value
        selected = adjusted[q <= row.cutoff]
        a, b = selected.source_id.to_numpy(), selected.target_id.to_numpy()
        graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(12, 12)).tocsr()
        _, labels = connected_components(graph, directed=False)
        largest = np.bincount(labels).max()
        assert row.directed_connections == len(a)
        assert row.largest_component_nodes == largest
        assert row.zeb2_component_nodes == (labels == labels[11]).sum()  # Lexicographic gene order: ZEB2 last.
        assert row.zeb2_neighbors == len(set(b[a == 11]) | set(a[b == 11]))
        assert row.undirected_pairs == len(set(zip(np.minimum(a,b), np.maximum(a,b))))
    zeb = pd.read_csv(root / "anchor_connections.csv")
    assert len(zeb) == ((adjusted.source_id == 11) | (adjusted.target_id == 11)).sum()
    assert ((zeb.Genefrom == "ZEB2") | (zeb.Geneto == "ZEB2")).all()


def test_changed_input_and_resume_false_refused(genome_config):
    config = replace(genome_config, analysis=replace(genome_config.analysis, genome_shards_per_invocation=1))
    run_pipeline(config)
    with pytest.raises(ValueError, match="resume=false"):
        run_pipeline(replace(config, execution=replace(config.execution, resume=False)))
    frame = pd.read_csv(config.dataset.expression_file)
    frame.loc[0, "Expression"] += 1
    frame.to_csv(config.dataset.expression_file, index=False)
    with pytest.raises(ValueError, match="Resume refused"):
        run_pipeline(config)


def test_ties_cross_groups_and_cpu_postprocess_without_gpu(genome_config, monkeypatch):
    import gene_analysis.pipeline.genome_fdr as module
    config = replace(genome_config, analysis=replace(genome_config.analysis, genome_stage="fit"))
    # Deterministic regression outputs isolate exact ties and threshold boundaries.
    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, source, target):
            ids = source * 11 + target - (target > source)
            p = np.take([1e-15, 1e-5, .001, .2, .99, 1., np.nan], ids % 7)
            return np.zeros(len(p)), np.ones(len(p)), np.zeros(len(p)), p

    monkeypatch.setattr(module, "QuadraticInteraction", Model)
    run_pipeline(config)
    root = config.run_dir / "genome_fdr"
    assert json.loads((root / "progress.json").read_text())["status"] == "fit_complete"
    # Existing fits must not require CUDA, even if backend is configured as GPU.
    def forbidden(*args, **kwargs):
        pytest.fail("Postprocessing tried to instantiate a fitting backend")
    monkeypatch.setattr(module, "QuadraticInteraction", forbidden)
    original_sort = GenomeFDRRunner._sort
    def tiny_groups(self):
        result = original_sort(self)
        path = self.directory / "sorted_p.parquet"
        table = pq.read_table(path)
        pq.write_table(table, path, row_group_size=3)
        return result
    monkeypatch.setattr(GenomeFDRRunner, "_sort", tiny_groups)
    # An abandoned temporary file is not a completed sort/checkpoint.
    (root / "sorted_p.parquet.tmp").write_bytes(b"incomplete")
    config = replace(config, analysis=replace(config.analysis, backend="gpu_cuda", genome_stage="report"))
    run_pipeline(config)
    raw = _read_parts(root / "raw").sort_values("pair_id")
    adjusted = _read_parts(root / "adjusted")
    ids = adjusted.pair_id.to_numpy()
    bh = benjamini_hochberg(raw.raw_p_value.to_numpy())
    storey, _ = storey_qvalues(raw.raw_p_value.to_numpy())
    np.testing.assert_allclose(adjusted.bh_adjusted_p_value, bh[ids], rtol=1e-14)
    np.testing.assert_allclose(adjusted.storey_q_value, storey[ids], rtol=1e-14)
    assert adjusted.groupby("raw_p_value").bh_adjusted_p_value.nunique().max() == 1
    for row in pd.read_csv(root / "cutoff_comparison.csv").itertuples():
        q = bh if row.method == "bh" else storey
        assert row.directed_connections == (q <= row.cutoff).sum()


@pytest.mark.parametrize("change, message", [
    ({"fingerprint": "different"}, "fingerprint mismatch"),
    ({"rows": 999}, "row count mismatch"),
    ({"start": 1}, "Gap/overlap"),
    ({"stop": 999}, "Gap/overlap"),
])
def test_corrupt_raw_checkpoint_is_refused(genome_config, change, message):
    pilot = replace(genome_config, analysis=replace(genome_config.analysis, genome_shards_per_invocation=1))
    run_pipeline(pilot)
    path = next((pilot.run_dir / "genome_fdr/raw").glob("*.parquet"))
    table = pq.read_table(path)
    info = json.loads(table.schema.metadata[b"genome_fdr"])
    info.update(change)
    table = table.replace_schema_metadata({b"genome_fdr": json.dumps(info).encode()})
    pq.write_table(table, path)
    with pytest.raises(ValueError, match=message):
        run_pipeline(pilot)


@pytest.mark.parametrize("settings", [
    {"workflow": "unknown"}, {"genome_stage": "unknown"},
    {"genome_shard_pairs": 0}, {"genome_sort_threads": 0},
    {"genome_shards_per_invocation": 0}, {"backend": "unknown"},
    {"direction": "unknown"}, {"duplicate_gene_method": "unknown"},
    {"genome_expansion": "yes"}, {"alpha": 0}, {"batch_size": 0},
    {"max_seconds": -1},
])
def test_invalid_resource_and_analysis_settings_rejected(genome_config, settings):
    with pytest.raises(ValueError):
        replace(genome_config.analysis, **settings).validate()


def test_storey_rejects_unidentifiable_null_fraction():
    with pytest.raises(ValueError, match="nonempty"):
        pi0_from_counts(np.zeros(19), 0, 1)
    with pytest.raises(ValueError, match="untruncated"):
        pi0_from_counts(np.zeros(19), 10, .1)
    with pytest.raises(ValueError, match="Nonpositive"):
        pi0_from_counts(np.zeros(19), 10, 1)


@pytest.mark.parametrize("damage", ["missing_rows", "unsorted", "adjusted_carry"])
def test_corrupt_sort_or_adjustment_is_refused(genome_config, damage):
    config = replace(genome_config, analysis=replace(genome_config.analysis, genome_stage="adjust"))
    run_pipeline(config)
    root = config.run_dir / "genome_fdr"
    if damage == "adjusted_carry":
        path = next((root / "adjusted").glob("*.parquet"))
        table = pq.read_table(path)
        info = json.loads(table.schema.metadata[b"genome_fdr"])
        info["incoming_min"] = -1
        pq.write_table(table.replace_schema_metadata({b"genome_fdr": json.dumps(info).encode()}), path)
        match = "Adjusted checkpoint"
    else:
        path = root / "sorted_p.parquet"
        table = pq.read_table(path)
        if damage == "missing_rows":
            table = table.slice(0, table.num_rows-1)
            match = "incorrect row count"
        else:
            table = table.take(np.arange(table.num_rows-1, -1, -1))
            # Remove derived checkpoints in this disposable test fixture only.
            for part in (root / "adjusted").glob("*.parquet"):
                part.unlink()
            match = "not globally ordered"
        pq.write_table(table, path)
    with pytest.raises(ValueError, match=match):
        run_pipeline(config)
