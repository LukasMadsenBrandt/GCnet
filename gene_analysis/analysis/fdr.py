"""Batched quadratic interaction tests and Benjamini–Hochberg adjustment.

The model is target ~ 1 + time + time**2 + predictor + time*predictor.
The two-sided OLS t test concerns the interaction coefficient, not causality.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values; failed tests remain in N as p=1.

    Retaining failed planned tests is conservative relative to dropping them.
    The reverse cumulative minimum implements the step-up rule, including ties.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
        raise ValueError("Finite p-values must be in [0, 1].")
    valid = np.isfinite(p)
    safe = np.where(valid, p, 1.0)
    order = np.argsort(safe, kind="stable")
    result = np.empty_like(p)
    ranked = safe[order] * len(p) / np.arange(1, len(p) + 1)
    result[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    result[~valid] = np.nan
    return result


class QuadraticInteraction:
    """Cache time-residualized sample data on CPU or GPU and fit pair batches."""

    def __init__(self, expression, time, *, backend="cpu_numpy", gpu_device=None):
        values = np.asarray(expression, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)
        if values.ndim != 2 or time.shape != (values.shape[0],):
            raise ValueError("Expected expression (samples, genes) and time (samples,).")
        if values.shape[0] <= 5 or not np.isfinite(values).all() or not np.isfinite(time).all():
            raise ValueError("Need more than five samples and finite expression/time.")
        # Centering preserves this model's interaction coefficient and p-value.
        time = time - time.mean()
        baseline = np.column_stack([np.ones(len(time)), time, time**2])
        if np.linalg.matrix_rank(baseline) != 3:
            raise ValueError("Quadratic time baseline is rank deficient.")
        if backend == "gpu_cuda":
            from gene_analysis.analysis.cuda_environment import require_cuda_kernels

            xp = require_cuda_kernels(gpu_device)
        elif backend == "cpu_numpy":
            xp = np
        else:
            raise ValueError("FDR backend must be cpu_numpy or gpu_cuda.")
        self.xp = xp
        self.df = values.shape[0] - 5
        values = values - values.mean(axis=0)
        baseline_q = np.linalg.qr(baseline, mode="reduced")[0]
        interaction = time[:, None] * values
        rank_scale = np.maximum(
            np.linalg.norm(baseline),
            np.sqrt(np.sum(values**2 + interaction**2, axis=0)),
        )
        residual = values - baseline_q @ (baseline_q.T @ values)
        interaction = interaction - baseline_q @ (baseline_q.T @ interaction)
        # A predictor's design is identical for every target. Decompose it ONCE,
        # on the CPU: CuPy 13 loops over individual SVD calls for 35x2 matrices
        # because its fast batched solver only accepts dimensions <=32.
        design = np.stack([residual.T, interaction.T], axis=2)
        u, singular, vh = np.linalg.svd(design, full_matrices=False)
        tolerance = np.finfo(float).eps * max(values.shape[0], 5) * rank_scale[:, None]
        inverse = np.where(singular > tolerance, 1 / np.maximum(singular, np.finfo(float).tiny), 0)
        self.predictor_valid = xp.asarray(np.all(singular > tolerance, axis=1))
        self.residual = xp.asarray(residual.T.copy())
        self.basis = xp.asarray(u)
        # Row of the pseudoinverse that estimates the interaction coefficient.
        self.contrast = xp.asarray(np.einsum("gnk,gk->gn", u, vh[:, :, 1] * inverse))
        self.se_scale = xp.asarray(np.sqrt(np.sum((vh[:, :, 1] * inverse)**2, axis=1)))

    def fit(self, predictors, targets):
        """Return coefficient, standard error, t and p arrays; invalid fits are NaN."""
        xp = self.xp
        source = xp.asarray(predictors, dtype=xp.int64)
        target = xp.asarray(targets, dtype=xp.int64)
        u = self.basis[source]
        response = self.residual[target]
        valid = self.predictor_valid[source].copy()
        projection = xp.einsum("bni,bn->bi", u, response)
        error = response - xp.einsum("bni,bi->bn", u, projection)
        variance = xp.sum(error**2, axis=1) / self.df
        coefficient = xp.einsum("bn,bn->b", self.contrast[source], response)
        standard_error = xp.sqrt(variance) * self.se_scale[source]
        valid &= (standard_error > 0) & xp.isfinite(standard_error)
        valid &= source != target
        statistic = coefficient / xp.where(standard_error > 0, standard_error, 1)
        arrays = [xp.where(valid, value, xp.nan) for value in (coefficient, standard_error, statistic)]
        if xp is not np:
            arrays = [xp.asnumpy(value) for value in arrays]
        p_value = 2 * student_t.sf(np.abs(arrays[2]), self.df)
        return (*arrays, p_value)
