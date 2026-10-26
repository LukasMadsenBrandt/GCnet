"""Small scientific-reference and interruption tests; never load real datasets."""

from dataclasses import replace
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from gene_analysis.analysis.fdr import QuadraticInteraction, benjamini_hochberg
from gene_analysis.pipeline.fdr import FDRRunner, FDRSettings, Family
from gene_analysis.pipeline.runner import PipelineConfig, run_pipeline


def test_bh_step_up_ties_and_failed_tests():
    p = np.array([0.01, 0.04, 0.03, 0.04, np.nan, 0.8])
    expected = multipletests(np.nan_to_num(p, nan=1.0), method="fdr_bh")[1]
    actual = benjamini_hochberg(p)
    np.testing.assert_allclose(actual[np.isfinite(p)], expected[np.isfinite(p)])
    assert np.isnan(actual[4])
    # The first rank fails its own inequality; BH still selects both ranks.
    np.testing.assert_allclose(benjamini_hochberg(np.array([0.03, 0.04])), [0.04, 0.04])
    assert benjamini_hochberg(np.array([])).size == 0


def test_quadratic_batches_match_independent_statsmodels():
    rng = np.random.default_rng(83)
    time = np.repeat(np.arange(5), 7).astype(float)
    values = rng.normal(size=(35, 4)) + 2
    values[:, 1] += 0.3 * time * values[:, 0]
    model = QuadraticInteraction(values, time)
    coefficient, se, statistic, p = model.fit([0, 1, 2], [1, 0, 3])
    for i, (source, target) in enumerate([(0, 1), (1, 0), (2, 3)]):
        design = np.column_stack([np.ones(35), time, time**2, values[:, source], time * values[:, source]])
        reference = sm.OLS(values[:, target], design).fit()
        np.testing.assert_allclose([coefficient[i], se[i], statistic[i], p[i]],
                                   [reference.params[-1], reference.bse[-1], reference.tvalues[-1], reference.pvalues[-1]],
                                   rtol=1e-10, atol=1e-12)
    assert p[0] != pytest.approx(p[1])


def test_rank_deficiency_and_self_pairs_are_failed():
    time = np.repeat(np.arange(5), 7).astype(float)
    values = np.column_stack([np.ones(35), time, np.random.default_rng(1).normal(size=35)])
    model = QuadraticInteraction(values, time)
    *_, p = model.fit([0, 1, 2], [2, 2, 2])
    assert np.isnan(p).all()


def test_repeated_predictor_batches_reuse_prepared_factorization(monkeypatch):
    rng = np.random.default_rng(84)
    time = np.repeat(np.arange(5), 7).astype(float)
    values = rng.normal(size=(35, 6))
    model = QuadraticInteraction(values, time)

    def unexpected_svd(*args, **kwargs):
        pytest.fail("Pair fitting must reuse the predictor decomposition.")

    sources, targets = [0, 0, 0, 2, 2], [1, 2, 3, 1, 4]
    with monkeypatch.context() as context:
        context.setattr(np.linalg, "svd", unexpected_svd)
        actual = model.fit(sources, targets)
    for i, (source, target) in enumerate(zip(sources, targets)):
        design = np.column_stack([np.ones(35), time, time**2, values[:, source], time * values[:, source]])
        reference = sm.OLS(values[:, target], design).fit(method="qr")
        expected = [reference.params[-1], reference.bse[-1], reference.tvalues[-1], reference.pvalues[-1]]
        np.testing.assert_allclose([column[i] for column in actual], expected, rtol=1e-10, atol=1e-12)


@pytest.fixture
def sample_config(tmp_path, monkeypatch):
    import gene_analysis.pipeline.runner as runner

    monkeypatch.setattr(runner, "results_path", lambda *parts: tmp_path.joinpath(*parts))
    rng = np.random.default_rng(832)
    time = np.repeat(np.arange(5), 7)
    values = rng.normal(size=(35, 5)) + 10
    values[:, 1] = 10 + (time + 1) * values[:, 0] + rng.normal(scale=0.1, size=35)
    genes = ["ZEB2", "A", "B", "C", "D"]
    records = [(gene, f"s{sample:02d}", day, values[sample, i])
               for i, gene in enumerate(genes) for sample, day in enumerate(time)]
    path = tmp_path / "samples.csv"
    pd.DataFrame(records, columns=["Gene", "Sample", "Day", "Expression"]).to_csv(path, index=False)
    seeds = tmp_path / "seeds.txt"
    seeds.write_text("ZEB2\nA\nB\nC\n")
    return PipelineConfig.from_dict({
        "run_name": "small_fdr", "dataset": {"name": "sample_long", "expression_file": str(path)},
        "gene_of_interest": "ZEB2", "seed_gene_file": str(seeds),
        "analysis": {"mode": "fdr", "backend": "cpu_numpy", "batch_size": 3,
                     "max_depth": 2, "max_nodes": 50, "genome_expansion": True},
        "execution": {"resume": True, "progress": False},
    })


def test_entrypoint_overview_local_cumulative_and_idempotent_resume(sample_config):
    artifacts = run_pipeline(sample_config)
    root = sample_config.run_dir / "fdr"
    assert json.loads((root / "progress.json").read_text())["status"] == "complete"
    overview = json.loads((root / "all_pairs/overview/summary.json").read_text())
    assert overview["bh_denominator"] == 4 * 3
    assert overview["failed_tests"] == 0
    depth = json.loads((root / "anchor/depth_01/summary.json").read_text())
    assert depth["family_tests"] == 2 * 3
    previous = depth["family_tests"]
    for path in sorted((root / "anchor").glob("*/summary.json")):
        result = json.loads(path.read_text())
        assert result["bh_denominator"] == result["family_tests"]
        if path.parent.name.startswith("depth"):
            assert result["family_tests"] >= previous
            previous = result["family_tests"]
        frame = pd.read_csv(path.parent / "all_tests.csv")
        assert not frame.duplicated(["predictor_gene", "target_gene"]).any()
        reference = multipletests(frame.raw_p_value, method="fdr_bh")[1]
        np.testing.assert_allclose(frame.bh_adjusted_p_value, reference, rtol=1e-10)
    final = pd.read_csv(root / "anchor/genome_probe/all_tests.csv")
    assert "D" in set(final.predictor_gene) | set(final.target_gene)
    before = {p: p.read_bytes() for p in root.glob("*/*/all_tests.csv")}
    repeated = run_pipeline(sample_config)
    assert set(artifacts) == set(repeated)
    for path, content in before.items():
        assert path.read_bytes() == content
    assert "Lukas's concern" in artifacts["summary"].read_text(encoding="utf-8")


def test_resume_after_interruption_keeps_committed_batch(sample_config, monkeypatch):
    sample_config = replace(sample_config, analysis=replace(sample_config.analysis, workflow="all_pairs"))
    original = QuadraticInteraction.fit
    calls = []

    def interrupted(self, source, target):
        calls.append(len(source))
        if len(calls) == 2:
            raise KeyboardInterrupt("simulated stop")
        return original(self, source, target)

    monkeypatch.setattr(QuadraticInteraction, "fit", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(sample_config)
    path = sample_config.run_dir / "fdr/all_pairs/tests.sqlite"
    with sqlite3.connect(path) as db:
        committed = db.execute("SELECT source,target,p FROM tests ORDER BY id").fetchall()
    assert len(committed) == calls[0]
    resumed_pairs = []

    def resumed(self, source, target):
        resumed_pairs.extend(zip(source.tolist(), target.tolist()))
        return original(self, source, target)

    monkeypatch.setattr(QuadraticInteraction, "fit", resumed)
    run_pipeline(sample_config)
    assert not set(resumed_pairs).intersection((a, b) for a, b, _ in committed)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT source,target,p FROM tests ORDER BY id LIMIT ?", (len(committed),)).fetchall() == committed
        assert db.execute("SELECT count(*) FROM tests").fetchone()[0] == 12


def test_resume_rejects_changed_inputs(sample_config):
    run_pipeline(sample_config)
    raw = pd.read_csv(sample_config.dataset.expression_file)
    raw.loc[0, "Expression"] += 0.1
    raw.to_csv(sample_config.dataset.expression_file, index=False)
    with pytest.raises(ValueError, match="Resume refused"):
        run_pipeline(sample_config)


def test_pause_then_resume(sample_config, monkeypatch):
    original = FDRRunner.expired
    monkeypatch.setattr(FDRRunner, "expired", lambda self: True)
    run_pipeline(sample_config)
    progress = sample_config.run_dir / "fdr/progress.json"
    assert json.loads(progress.read_text())["status"] == "paused"
    monkeypatch.setattr(FDRRunner, "expired", original)
    run_pipeline(sample_config)
    assert json.loads(progress.read_text())["status"] == "complete"


def test_pair_generation_has_unique_union(tmp_path):
    family = Family(tmp_path / "family", "test")
    try:
        pairs = [pair for block in family.blocks([0, 1], [0, 1, 2, 3], "both", 2) for pair in block]
        assert len(pairs) == len(set(pairs)) == 10
        assert all(a != b for a, b in pairs)
    finally:
        family.db.close()


def test_batches_span_predictors_and_resume_mixed_batches(tmp_path):
    family = Family(tmp_path / "family", "test")
    try:
        blocks = list(family.blocks([0, 1, 2], [0, 1, 2], "outgoing", 4))
        assert [len(block) for block in blocks] == [4, 2]
        assert len({source for source, target in blocks[0]}) == 2
        saved = [blocks[0][0], blocks[0][2]]
        family.save(saved, (np.zeros(2), np.ones(2), np.zeros(2), np.ones(2)))
        assert family.pending(blocks[0]) == [pair for pair in blocks[0] if pair not in saved]
    finally:
        family.db.close()


def test_incoming_resume_uses_unique_pair_index_and_retains_failures(tmp_path):
    family = Family(tmp_path / "family", "test")
    try:
        block = [(source, 5) for source in range(5)]
        # A failed fit is still a completed test and must not be recomputed.
        family.save([block[1], block[3]], (np.zeros(2), np.ones(2), np.zeros(2), np.array([np.nan, 0.5])))
        assert family.pending(block) == [block[0], block[2], block[4]]
        indexes = {row[1] for row in family.db.execute("PRAGMA index_list(tests)")}
        assert "tests_target" not in indexes
        plan = family.db.execute("""EXPLAIN QUERY PLAN
            SELECT wanted.source,wanted.target FROM pending_pairs AS wanted
            WHERE EXISTS (SELECT 1 FROM tests
                WHERE tests.source=wanted.source AND tests.target=wanted.target)
        """).fetchall()
        assert any("SEARCH" in row[-1] and "tests" in row[-1]
                   and "USING COVERING INDEX" in row[-1] for row in plan)
    finally:
        family.db.close()


def test_incoming_order_groups_predictors_without_changing_pair_family(tmp_path):
    family = Family(tmp_path / "family", "test")
    try:
        sources, universe = [1, 3], [0, 1, 2, 3, 4]
        pairs = [pair for block in family.blocks(sources, universe, "incoming", 3) for pair in block]
        expected = {(source, target) for target in sources for source in universe if source != target}
        assert set(pairs) == expected
        assert [source for source, target in pairs] == sorted(source for source, target in pairs)
        # Resume recognizes previously committed pairs in the old target order.
        old_order = [(source, target) for target in sources for source in universe if source != target]
        family.save(old_order[:3], (np.zeros(3), np.ones(3), np.zeros(3), np.ones(3)))
        assert family.pending(pairs) == [pair for pair in pairs if pair not in old_order[:3]]
    finally:
        family.db.close()


def test_legacy_checkpoint_index_migration_preserves_results_and_plans(tmp_path):
    directory = tmp_path / "family"
    family = Family(directory, "test")
    family.save([(0, 1), (2, 1)], (np.zeros(2), np.ones(2), np.zeros(2), np.array([np.nan, 0.5])))
    family.plan("depth_01", [0], [0, 1, 2], "both")
    with family.db:
        family.db.execute("CREATE INDEX tests_target ON tests(target)")
        family.db.execute("UPDATE layers SET result=?", ('{"family_tests": 2}',))
    before = family.db.execute("SELECT * FROM tests ORDER BY id").fetchall()
    family.db.close()
    resumed = Family(directory, "test")
    try:
        assert resumed.db.execute("SELECT * FROM tests ORDER BY id").fetchall() == before
        assert resumed.plan("depth_01", [0], [0, 1, 2], "both") == {"family_tests": 2}
        assert resumed.db.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert resumed.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert "tests_target" not in {row[1] for row in resumed.db.execute("PRAGMA index_list(tests)")}
    finally:
        resumed.db.close()


def test_fingerprint_rejection_does_not_drop_legacy_index(tmp_path):
    directory = tmp_path / "family"
    family = Family(directory, "original")
    with family.db:
        family.db.execute("CREATE INDEX tests_target ON tests(target)")
    family.db.close()
    with pytest.raises(ValueError, match="Resume refused"):
        Family(directory, "changed")
    with sqlite3.connect(directory / "tests.sqlite") as db:
        assert "tests_target" in {row[1] for row in db.execute("PRAGMA index_list(tests)")}


def test_failed_checkpoint_transaction_rolls_back_all_rows(tmp_path):
    family = Family(tmp_path / "family", "test")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            family.save([(0, 1), (0, 1)], (np.zeros(2), np.ones(2), np.zeros(2), np.ones(2)))
        assert family.count() == 0
        assert family.pending([(0, 1)]) == [(0, 1)]
    finally:
        family.db.close()


def test_cuda_preflight_fails_before_reading_expression(sample_config, monkeypatch):
    import gene_analysis.pipeline.fdr as pipeline

    def broken_cuda(device):
        raise RuntimeError("libnvrtc.so.12 unavailable")

    def unexpected_load(config):
        pytest.fail("Expression must not be loaded before the CUDA check.")

    monkeypatch.setattr(pipeline, "require_cuda_kernels", broken_cuda)
    monkeypatch.setattr(pipeline, "load_samples", unexpected_load)
    config = replace(sample_config, analysis=replace(sample_config.analysis, backend="gpu_cuda"))
    with pytest.raises(RuntimeError, match="libnvrtc"):
        run_pipeline(config)


def test_no_new_neighbors_still_probes_genome_once(sample_config, monkeypatch):
    def no_signal(self, source, target):
        return np.zeros(len(source)), np.ones(len(source)), np.zeros(len(source)), np.ones(len(source))

    monkeypatch.setattr(QuadraticInteraction, "fit", no_signal)
    config = replace(sample_config, analysis=replace(sample_config.analysis, workflow="anchor"))
    run_pipeline(config)
    root = config.run_dir / "fdr/anchor"
    assert json.loads((root / "core_summary.json").read_text())["stop_reason"] == "no_new_neighbors"
    assert not (root / "depth_02").exists()
    assert json.loads((root / "genome_probe/summary.json").read_text())["family_tests"] == 8


@pytest.mark.cuda
def test_cuda_small_reference_parity():
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1).sum().item()
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    rng = np.random.default_rng(12)
    times = np.repeat(np.arange(5), 7).astype(float)
    values = rng.normal(size=(35, 5))
    expected = QuadraticInteraction(values, times).fit([0, 1, 2], [1, 3, 4])
    actual = QuadraticInteraction(values, times, backend="gpu_cuda").fit([0, 1, 2], [1, 3, 4])
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
