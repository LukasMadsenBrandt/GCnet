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
            from gene_analysis.analysis.cuda_environment import preload_cuda_wheel_libraries

            preload_cuda_wheel_libraries()
            import cupy as xp

            if gpu_device is not None:
                xp.cuda.Device(int(gpu_device)).use()
            # Fail immediately if CUDA cannot execute; never silently fall back.
            xp.zeros(1).sum().item()
        elif backend == "cpu_numpy":
            xp = np
        else:
            raise ValueError("FDR backend must be cpu_numpy or gpu_cuda.")
        self.xp = xp
        self.df = values.shape[0] - 5
        values = xp.asarray(values - values.mean(axis=0))
        baseline_q = xp.asarray(np.linalg.qr(baseline, mode="reduced")[0])
        interaction = xp.asarray(time)[:, None] * values
        self.rank_scale = xp.maximum(
            xp.linalg.norm(xp.asarray(baseline)),
            xp.sqrt(xp.sum(values**2 + interaction**2, axis=0)),
        )
        self.residual = values - baseline_q @ (baseline_q.T @ values)
        self.interaction = interaction - baseline_q @ (baseline_q.T @ interaction)

    def fit(self, predictors, targets):
        """Return coefficient, standard error, t and p arrays; invalid fits are NaN."""
        xp = self.xp
        source = xp.asarray(predictors, dtype=xp.int64)
        target = xp.asarray(targets, dtype=xp.int64)
        design = xp.stack([self.residual[:, source].T, self.interaction[:, source].T], axis=2)
        response = self.residual[:, target].T
        # Batched SVD avoids squaring the condition number via normal equations.
        u, singular, vh = xp.linalg.svd(design, full_matrices=False)
        tolerance = (np.finfo(float).eps * max(design.shape[1], 5)
                     * self.rank_scale[source, None])
        valid = xp.all(singular > tolerance, axis=1)
        inverse = xp.where(singular > tolerance, 1 / xp.maximum(singular, np.finfo(float).tiny), 0)
        projection = xp.einsum("bni,bn->bi", u, response)
        beta = xp.einsum("bij,bj->bi", vh.transpose(0, 2, 1), inverse * projection)
        error = response - xp.einsum("bni,bi->bn", design, beta)
        variance = xp.sum(error**2, axis=1) / self.df
        coefficient = beta[:, 1]
        standard_error = xp.sqrt(variance * xp.sum((vh[:, :, 1] * inverse)**2, axis=1))
        valid &= (standard_error > 0) & xp.isfinite(standard_error)
        valid &= source != target
        statistic = coefficient / xp.where(standard_error > 0, standard_error, 1)
        arrays = [xp.where(valid, value, xp.nan) for value in (coefficient, standard_error, statistic)]
        if xp is not np:
            arrays = [xp.asnumpy(value) for value in arrays]
        p_value = 2 * student_t.sf(np.abs(arrays[2]), self.df)
        return (*arrays, p_value)
