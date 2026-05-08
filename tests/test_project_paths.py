from pathlib import Path

import project_paths


def test_resolve_existing_path_prefers_current_working_directory(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    root = tmp_path / "repo"
    results = root / "results"
    cwd.mkdir()
    results.mkdir(parents=True)
    file_path = cwd / "input.csv"
    file_path.write_text("x\n", encoding="utf-8")

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(project_paths, "PROJECT_ROOT", root)
    monkeypatch.setattr(project_paths, "RESULTS_DIR", results)

    assert project_paths.resolve_existing_path("input.csv") == file_path


def test_resolve_existing_path_finds_result_category(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    results = root / "results"
    granger_file = results / "granger" / "edges.csv"
    granger_file.parent.mkdir(parents=True)
    granger_file.write_text("gene1,gene2\n", encoding="utf-8")

    monkeypatch.setattr(project_paths, "PROJECT_ROOT", root)
    monkeypatch.setattr(project_paths, "RESULTS_DIR", results)

    assert project_paths.resolve_existing_path("edges.csv") == granger_file


def test_results_path_creates_parent(tmp_path, monkeypatch):
    results = tmp_path / "results"
    monkeypatch.setattr(project_paths, "RESULTS_DIR", results)

    path = project_paths.results_path("comparisons", "diff.csv")

    assert path == results / "comparisons" / "diff.csv"
    assert path.parent.exists()

