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
