"""Shared project paths for inputs and generated results."""

from __future__ import annotations

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULT_CATEGORIES = (
    "granger",
    "coassociation",
    "gene_lists",
    "comparisons",
    "degree_outputs",
    "figures",
    "bundles",
    "logs",
)


def project_path(*parts: PathLike) -> Path:
    """Return a path rooted at the repository directory."""
    return PROJECT_ROOT.joinpath(*(Path(part) for part in parts))


def data_path(*parts: PathLike) -> Path:
    """Return a path rooted at the Data directory."""
    return DATA_DIR.joinpath(*(Path(part) for part in parts))


def results_path(*parts: PathLike, create_parent: bool = True) -> Path:
    """Return a path rooted at results/, creating its parent by default."""
    path = RESULTS_DIR.joinpath(*(Path(part) for part in parts))
    if create_parent:
        parent = path if path.suffix == "" else path.parent
        parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_existing_path(path: PathLike) -> Path:
    """
    Resolve a configured path.

    Relative paths are looked up from the current working directory first, then
    from results/, then from the project root. This lets scripts read either the
    old root-level result files or newly bundled files without per-script logic.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    for base in (Path.cwd(), RESULTS_DIR, PROJECT_ROOT):
        resolved = base / candidate
        if resolved.exists():
            return resolved

    for category in RESULT_CATEGORIES:
        resolved = RESULTS_DIR / category / candidate
        if resolved.exists():
            return resolved

    matches = sorted(RESULTS_DIR.glob(f"**/{candidate.name}")) if RESULTS_DIR.exists() else []
    if len(matches) == 1:
        return matches[0]
    return PROJECT_ROOT / candidate


__all__ = [
    "DATA_DIR",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "data_path",
    "project_path",
    "resolve_existing_path",
    "results_path",
]
