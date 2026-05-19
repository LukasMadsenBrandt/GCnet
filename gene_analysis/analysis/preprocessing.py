"""Shared expression preprocessing utilities for pipeline datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_gene_expression(df: pd.DataFrame, method: str | None) -> pd.DataFrame:
    """Normalize expression values using a supported column-wise method."""
    method = _normalize_option(method)
    if method == "none":
        return df
    if method == "deseq":
        return _normalize_deseq_size_factors(df)
    if method == "zscore":
        std = df.std(axis=0).replace(0, np.nan)
        return df.subtract(df.mean(axis=0), axis=1).div(std, axis=1).fillna(0.0)
    raise ValueError(f"Unknown normalization option: {method}")


def transform_gene_expression(df: pd.DataFrame, method: str | None) -> pd.DataFrame:
    """Transform expression values using a supported element-wise transform."""
    method = _transform_option(method)
    if method == "none":
        return df
    if method == "log1p":
        return np.log1p(df)
    if method == "sqrt":
        return np.sqrt(df)
    raise ValueError(f"Unknown transform option: {method}")


def apply_expression_preprocessing(df: pd.DataFrame, *, normalize: str | None, transform: str | None) -> pd.DataFrame:
    """Apply normalization followed by transformation to an expression matrix."""
    numeric = df.apply(pd.to_numeric, errors="raise")
    normalized = normalize_gene_expression(numeric, normalize)
    return transform_gene_expression(normalized, transform)


def _normalize_deseq_size_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the DESeq2 median-ratio size-factor normalization."""
    counts_no_zeros = df.replace(0, np.nan)
    log_counts = np.log(counts_no_zeros)
    geo_means = np.exp(log_counts.mean(axis=1, skipna=True))
    ratios = df.div(geo_means, axis=0)
    size_factors = ratios.median(axis=0, skipna=True)
    size_factors = size_factors.replace([0, np.nan, np.inf, -np.inf], np.nan).fillna(1.0)
    return df.div(size_factors, axis=1)


def _normalize_option(method: str | None) -> str:
    """Return a canonical normalization option."""
    if method in (None, False):
        return "none"
    method = str(method).lower()
    aliases = {
        "false": "none",
        "null": "none",
        "none": "none",
        "no": "none",
        "deseq": "deseq",
        "deseq2": "deseq",
        "size_factors": "deseq",
        "zscore": "zscore",
        "z-score": "zscore",
    }
    return aliases.get(method, method)


def _transform_option(method: str | None) -> str:
    """Return a canonical transformation option."""
    if method in (None, False):
        return "none"
    method = str(method).lower()
    aliases = {
        "false": "none",
        "null": "none",
        "none": "none",
        "no": "none",
        "log+1": "log1p",
        "log1p": "log1p",
        "sqrt": "sqrt",
    }
    return aliases.get(method, method)
