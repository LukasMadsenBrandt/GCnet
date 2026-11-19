"""Batched time-interaction tests and Benjamini–Hochberg adjustment.

The default model is target ~ 1 + time + time**2 + predictor + time*predictor.
The two-sided OLS t test concerns the interaction coefficient, not causality.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t
import time as clock


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


INTERACTION_MODELS = {
    "model0": dict(quadratic_time=True, quadratic_predictor=False, parameters=5),
    "model1": dict(quadratic_time=True, quadratic_predictor=True, parameters=6),
    "model2": dict(quadratic_time=False, quadratic_predictor=False, parameters=4),
}


class QuadraticInteraction:
    """Cache residualized sample data on CPU or GPU and fit pair batches."""

    def __init__(self, expression, time, *, backend="cpu_numpy", gpu_device=None,
                 model="model0"):
        values = np.asarray(expression, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)
        if values.ndim != 2 or time.shape != (values.shape[0],):
            raise ValueError("Expected expression (samples, genes) and time (samples,).")
        if model not in INTERACTION_MODELS:
            raise ValueError(f"Unknown interaction model: {model}")
        specification = INTERACTION_MODELS[model]
        if (values.shape[0] <= specification["parameters"] or not np.isfinite(values).all()
                or not np.isfinite(time).all()):
            raise ValueError("Need more samples than model parameters and finite expression/time.")
        # Centering preserves this model's interaction coefficient and p-value.
        time = time - time.mean()
        baseline_columns = [np.ones(len(time)), time]
        if specification["quadratic_time"]:
            baseline_columns.append(time**2)
        baseline = np.column_stack(baseline_columns)
        if np.linalg.matrix_rank(baseline) != baseline.shape[1]:
            raise ValueError("Time baseline is rank deficient.")
        if backend == "gpu_cuda":
            from gene_analysis.analysis.cuda_environment import require_cuda_kernels

            xp = require_cuda_kernels(gpu_device)
        elif backend == "cpu_numpy":
            xp = np
        else:
            raise ValueError("FDR backend must be cpu_numpy or gpu_cuda.")
        self.xp = xp
        self.model = model
        self.parameters = specification["parameters"]
        self.samples = values.shape[0]
        self.df = self.samples - self.parameters
        values = values - values.mean(axis=0)
        self.target_sum_squares = np.sum(values**2, axis=0)
        # An exact fit has no estimable error variance. Roundoff in QR/SVD can
        # leave a tiny positive RSS and turn floating-point noise into enormous
        # t statistics. Use the squared backward-error scale for a dot-product
        # residual to distinguish numerical zero from a genuinely small RSS.
        residual_scale = (np.finfo(float).eps * max(self.samples, self.parameters))**2
        self.residual_numerical_floor = xp.asarray(self.target_sum_squares * residual_scale)
        baseline_q = np.linalg.qr(baseline, mode="reduced")[0]
        interaction = time[:, None] * values
        design_norm = values**2 + interaction**2
        if specification["quadratic_predictor"]:
            design_norm = design_norm + values**4
        rank_scale = np.maximum(np.linalg.norm(baseline), np.sqrt(np.sum(design_norm, axis=0)))
        residual = values - baseline_q @ (baseline_q.T @ values)
        interaction = interaction - baseline_q @ (baseline_q.T @ interaction)
        # A predictor's design is identical for every target. Decompose it ONCE,
        # on the CPU: CuPy 13 loops over individual SVD calls for these small matrices
        # because its fast batched solver only accepts dimensions <=32.
        design_columns = [residual.T, interaction.T]
        if specification["quadratic_predictor"]:
            predictor_squared = values**2
            predictor_squared -= baseline_q @ (baseline_q.T @ predictor_squared)
            design_columns.append(predictor_squared.T)
        design = np.stack(design_columns, axis=2)
        u, singular, vh = np.linalg.svd(design, full_matrices=False)
        tolerance = np.finfo(float).eps * max(values.shape[0], 5) * rank_scale[:, None]
        self.predictor_singular_values = singular
        self.predictor_rank_tolerance = tolerance[:, 0]
        inverse = np.where(singular > tolerance, 1 / np.maximum(singular, np.finfo(float).tiny), 0)
        self.predictor_valid = xp.asarray(np.all(singular > tolerance, axis=1))
        self.residual = xp.asarray(residual.T.copy())
        self.basis = xp.asarray(u)
        # Row of the pseudoinverse that estimates the interaction coefficient.
        interaction_inverse = vh[:, :, 1] * inverse
        self.contrast = xp.asarray(np.einsum("gnk,gk->gn", u, interaction_inverse))
        self.se_scale = xp.asarray(np.sqrt(np.sum(interaction_inverse**2, axis=1)))

    def _fit(self, predictors, targets, diagnostics):
        xp = self.xp
        started = clock.perf_counter()
        source = xp.asarray(predictors, dtype=xp.int64)
        target = xp.asarray(targets, dtype=xp.int64)
        if xp is not np:
            xp.cuda.get_current_stream().synchronize()
        index_seconds = clock.perf_counter() - started
        started = clock.perf_counter()
        u = self.basis[source]
        response = self.residual[target]
        valid = self.predictor_valid[source].copy()
        projection = xp.einsum("bni,bn->bi", u, response)
        error = response - xp.einsum("bni,bi->bn", u, projection)
        residual_sum_squares = xp.sum(error**2, axis=1)
        variance = residual_sum_squares / self.df
        coefficient = xp.einsum("bn,bn->b", self.contrast[source], response)
        standard_error = xp.sqrt(variance) * self.se_scale[source]
        valid &= (standard_error > 0) & xp.isfinite(standard_error)
        valid &= residual_sum_squares > self.residual_numerical_floor[target]
        valid &= source != target
        statistic = coefficient / xp.where(standard_error > 0, standard_error, 1)
        values = (coefficient, standard_error, statistic, residual_sum_squares)
        arrays = [xp.where(valid, value, xp.nan) for value in values]
        if xp is not np:
            xp.cuda.get_current_stream().synchronize()
        compute_seconds = clock.perf_counter() - started
        if xp is np:
            transfer_seconds = 0.0
            started = clock.perf_counter()
            p_value = 2 * student_t.sf(np.abs(arrays[2]), self.df)
            pvalue_seconds = clock.perf_counter() - started
            pvalue_timing = {"cpu_pvalue_seconds": pvalue_seconds}
        else:
            # For a two-sided t test, p = I_x(df/2, 1/2), where
            # x = df / (df + t**2). Computing the equivalent incomplete-beta
            # expression on the device avoids making the CPU the batch bottleneck.
            from cupyx.scipy.special import betainc

            started = clock.perf_counter()
            statistic = arrays[2]
            statistic_squared = statistic**2
            x = self.df / (self.df + statistic_squared)
            p_value = betainc(self.df / 2.0, 0.5, x)
            xp.cuda.get_current_stream().synchronize()
            pvalue_seconds = clock.perf_counter() - started
            started = clock.perf_counter()
            transferred = xp.asnumpy(xp.stack((*arrays, p_value)))
            arrays = [transferred[index] for index in range(4)]
            p_value = transferred[4]
            transfer_seconds = clock.perf_counter() - started
            pvalue_timing = {"gpu_pvalue_seconds": pvalue_seconds}
        self.last_timings = dict(index_transfer_seconds=index_seconds,
                                 model_compute_seconds=compute_seconds,
                                 result_transfer_seconds=transfer_seconds,
                                 **pvalue_timing)
        result = (*arrays[:3], p_value)
        return (*result, arrays[3]) if diagnostics else result

    def fit(self, predictors, targets):
        """Return coefficient, standard error, t and p arrays; invalid fits are NaN."""
        return self._fit(predictors, targets, False)

    def fit_with_diagnostics(self, predictors, targets):
        """Return the usual test statistics plus residual sum of squares."""
        return self._fit(predictors, targets, True)
