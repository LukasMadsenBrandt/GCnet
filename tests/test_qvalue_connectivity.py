"""Scientific references for Storey/BH and tied-cutoff graph summaries."""

import fcntl
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from gene_analysis.analysis.fdr import benjamini_hochberg
from gene_analysis.analysis.storey import bootstrap_pi0, storey_qvalues
from gene_analysis.experiments.qvalue_connectivity import (
    HEADER, connectivity_events, run_qvalue_comparison, summarize_cutoffs,
)


def test_storey_bootstrap_against_independent_r_reference(tmp_path):
    if not shutil.which("Rscript"):
        pytest.skip("R is optional; this reference check needs base R.")
    rng = np.random.default_rng(71)
    p = np.r_[rng.uniform(size=1500), rng.beta(0.3, 1, 2500), 1.0, 1.0]
    path = tmp_path / "p.txt"
    np.savetxt(path, p)
    script = tmp_path / "reference.R"
    script.write_text('''args <- commandArgs(trailingOnly=TRUE)
p <- scan(args[1], quiet=TRUE)
grid <- seq(0.05, 0.95, 0.05)
n <- length(p)
tail <- vapply(grid, function(x) sum(p >= x), numeric(1))
estimates <- tail / n / (1-grid)
loss <- tail * (1-tail/n) / (n*n*(1-grid)^2) + (estimates-quantile(estimates, 0.1))^2
pi0 <- min(1, estimates[loss == min(loss)])
bh <- p.adjust(p, method="BH")
write.csv(data.frame(bh=bh, storey=pi0*bh, pi0=pi0), args[2], row.names=FALSE)
''')
    reference = tmp_path / "reference.csv"
    subprocess.run(["Rscript", str(script), str(path), str(reference)], check=True, capture_output=True)
    expected = pd.read_csv(reference)
    q, details = storey_qvalues(p)
    np.testing.assert_allclose(benjamini_hochberg(p), expected.bh, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(q, expected.storey, rtol=1e-12, atol=1e-15)
    assert details["pi0"] == pytest.approx(expected.pi0.iloc[0], rel=1e-12)


def test_failed_tests_kept_in_denominator_and_pi0_but_not_given_q():
    p = np.r_[np.linspace(0.001, 0.999, 100), np.nan]
    q, details = storey_qvalues(p)
    expected_pi0 = bootstrap_pi0(np.nan_to_num(p, nan=1.0))["pi0"]
    np.testing.assert_allclose(q[:-1], benjamini_hochberg(p)[:-1] * expected_pi0)
    assert np.isnan(q[-1])
    assert details["failed_tests_as_p_one"] == 1
    assert details["p_values_used"] == 101
    assert details["pi0_sensitivity_excluding_failed_tests"] is not None


def test_pi0_boundary_uses_greater_than_or_equal():
    result = bootstrap_pi0([0.1, 0.5, 0.5, 1.0], lambdas=[0.5])
    assert result["pi0_lambda"] == [1.5]
    assert result["pi0"] == 1


@pytest.mark.parametrize("p", [[], [np.nan], [-0.1, 1], [0.1, np.inf]])
def test_invalid_pi0_inputs(p):
    with pytest.raises(ValueError):
        bootstrap_pi0(p)


def test_truncated_p_values_are_not_accepted_as_complete_family():
    with pytest.raises(ValueError, match="truncated"):
        bootstrap_pi0([0.001, 0.02, 0.04])


def test_connectivity_matches_scipy_at_every_tied_cutoff():
    a = np.array([0, 1, 2, 3, 1, 4, 0, 5], dtype=np.int32)
    b = np.array([1, 0, 3, 4, 2, 5, 5, 0], dtype=np.int32)
    q = np.array([0.001, 0.001, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005])
    values, ends, counts = connectivity_events(a, b, q, 6, 0)
    for threshold, end, row in zip(values, ends, counts):
        selected = q <= threshold
        graph = coo_matrix((np.ones(selected.sum()), (a[selected], b[selected])), shape=(6, 6)).tocsr()
        _, labels = connected_components(graph, directed=False)
        largest = np.bincount(labels).max()
        neighbors = np.union1d(b[selected & (a == 0)], a[selected & (b == 0)])
        undirected = len(set(zip(np.minimum(a[selected], b[selected]), np.maximum(a[selected], b[selected]))))
        expected = [selected.sum(), undirected, len(np.union1d(a[selected], b[selected])),
                    largest, int((labels == labels[0]).sum()), int((labels == labels[0]).sum() == largest),
                    len(neighbors), int((selected & (a == 0)).sum()), int((selected & (b == 0)).sum())]
        np.testing.assert_array_equal(row, expected)
        assert end == selected.sum()
    assert counts[0, 0] == 3  # All three tied links are included together.


def test_connectivity_and_five_neighbors_can_be_incompatible():
    a = np.array([0, 0, 0, 0, 0, 0, 6], dtype=np.int32)
    b = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    q = np.arange(1, 8) / 1000
    values, _, counts = connectivity_events(a, b, q, 8, 0)
    _, result = summarize_cutoffs("bh", values, counts, 8)
    reached = result["milestones"]["minimum_95pct_including_zeb2"]
    assert reached["cutoff"] == 0.007
    assert reached["largest_component_nodes"] == 8
    assert reached["zeb2_neighbors"] == 6
    assert not result["at_most_five_neighbors_compatible_with_95pct"]
    assert result["milestones"]["last_at_most_5_zeb2_neighbors"]["cutoff"] == 0.005


@pytest.fixture
def saved_seed_run(tmp_path):
    root = tmp_path / "run/fdr"
    directory = root / "all_pairs/overview"
    directory.mkdir(parents=True)
    (root / "run.lock").touch()
    names = ["ZEB2", "A", "B", "C", "D", "NA"]
    (root / "manifest.json").write_text(json.dumps({
        "schema": 1, "anchor": "ZEB2", "genes": names, "subset": list(range(6)), "fingerprint": "fixture",
    }))
    # A later incomplete stage must not invalidate a completed seed export.
    (root / "progress.json").write_text('{"status": "fitting", "stage": "genome_probe"}')
    pairs = [(a, b) for a in names for b in names if a != b]
    p = np.linspace(0.00001, 0.999, len(pairs))
    p[1] = np.nan
    frame = pd.DataFrame(pairs, columns=["predictor_gene", "target_gene"])
    frame["raw_p_value"] = p
    frame["bh_adjusted_p_value"] = benjamini_hochberg(p)
    frame["status"] = np.where(np.isfinite(p), "ok", "failed")
    frame.to_csv(directory / "all_tests.csv", index=False)
    (directory / "summary.json").write_text(json.dumps({"family_tests": len(p), "failed_tests": 1}))
    return root


def test_complete_reports_preserve_global_values_and_failed_rows(saved_seed_run):
    source = saved_seed_run / "all_pairs/overview/all_tests.csv"
    before = source.read_bytes()
    output = run_qvalue_comparison(saved_seed_run, chunk_size=7)
    report = json.loads((output / "summary.json").read_text())
    original = pd.read_csv(source, keep_default_na=False)
    pi0 = report["storey"]["pi0"]
    for method, multiplier in [("bh", 1), ("storey", pi0)]:
        table = pd.read_csv(output / method / "connections_sorted.csv.gz", keep_default_na=False)
        assert list(table) == HEADER
        assert len(table) == 30
        assert table["q-value"].iloc[-1] == ""
        valid = table[table["q-value"] != ""].copy()
        valid["q-value"] = pd.to_numeric(valid["q-value"])
        assert valid["q-value"].is_monotonic_increasing
        for _, group in valid.groupby("q-value"):
            cutoff = group["q-value"].iloc[0]
            assert (group.total_connections_at_or_below_q.astype(int) == (valid["q-value"] <= cutoff).sum()).all()
            assert group.largest_component_nodes.nunique() == 1
        subset = pd.read_csv(output / method / "zeb2_connections.csv", keep_default_na=False)
        assert len(subset) == 10
        assert ((subset.Genefrom == "ZEB2") | (subset.Geneto == "ZEB2")).all()
        for _, row in valid.iterrows():
            match = original[(original.predictor_gene == row.Genefrom) & (original.target_gene == row.Geneto)].iloc[0]
            assert float(row["q-value"]) == pytest.approx(float(match.bh_adjusted_p_value) * multiplier)
    assert report["saved_bh_verified"]
    milestone_neighbors = pd.read_csv(output / "zeb2_neighbors_at_milestones.csv", keep_default_na=False)
    for method, info in report["methods"].items():
        for criterion, row in info["milestones"].items():
            if row is not None:
                selected = milestone_neighbors[(milestone_neighbors.method == method)
                                               & (milestone_neighbors.criterion == criterion)]
                assert len(selected) == row["zeb2_neighbors"]
    assert source.read_bytes() == before
    assert (output / "cutoff_comparison.png").is_file()


@pytest.mark.parametrize("problem", ["truncated", "duplicate", "wrong_bh", "failed_count"])
def test_refuses_bad_saved_data(saved_seed_run, problem):
    path = saved_seed_run / "all_pairs/overview/all_tests.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    if problem == "truncated":
        frame = frame.iloc[:-1]
    elif problem == "duplicate":
        frame.iloc[2] = frame.iloc[0]
    elif problem == "wrong_bh":
        frame.loc[0, "bh_adjusted_p_value"] = 0.9
    else:
        summary = path.with_name("summary.json")
        summary.write_text('{"family_tests": 30, "failed_tests": 0}')
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        run_qvalue_comparison(saved_seed_run, chunk_size=7)
    assert not (saved_seed_run / "seed_qvalue_comparison/summary.json").exists()


def test_refuses_source_writer(saved_seed_run):
    with (saved_seed_run / "run.lock").open("r") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="still writing"):
            run_qvalue_comparison(saved_seed_run)
