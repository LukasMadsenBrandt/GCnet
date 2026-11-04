"""Storey q-values with the closed-form bootstrap estimator of pi0.

Equations follow the original R qvalue package (pi0est, bootstrap option).
This implements pfdr=False, lfdr.out=False; it does not implement local FDR.
Reference: https://github.com/jdstorey/qvalue/blob/master/R/pi0est.R
"""

from __future__ import annotations

import numpy as np

from gene_analysis.analysis.fdr import benjamini_hochberg


def bootstrap_pi0(p_values, lambdas=None):
    """Estimate pi0 from finite p-values, using >= lambda and R type-7 quantiles.

    Unlike the R package's warning/fallback, a nonpositive estimate raises an
    error. A degenerate input must not silently produce a published report.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("pi0 estimation requires a nonempty vector of finite p-values in [0, 1].")
    grid = np.arange(1, 20, dtype=float) / 20 if lambdas is None else np.sort(np.asarray(lambdas, dtype=float))
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or np.any((grid < 0) | (grid >= 1)) or len(np.unique(grid)) != len(grid)
            or 1 < len(grid) < 4):
        raise ValueError("Use one or at least four unique lambda values in [0, 1).")
    if p.max() < grid.max():
        raise ValueError("Maximum p-value is smaller than the lambda range; input may be truncated.")
    ordered = np.sort(p)
    tail_counts = len(p) - np.searchsorted(ordered, grid, side="left")
    estimates = tail_counts / (len(p) * (1 - grid))
    reference = np.quantile(estimates, 0.1)
    mse = (tail_counts / (len(p)**2 * (1 - grid)**2) * (1 - tail_counts / len(p))
           + (estimates - reference)**2)
    best = mse == mse.min()
    pi0 = float(min(1.0, estimates[best].min()))
    if pi0 <= 0:
        raise ValueError("Nonpositive pi0 estimate; inspect the input and lambda range.")
    return {
        "pi0": pi0, "method": "bootstrap_closed_form", "lambda": grid.tolist(),
        "pi0_lambda": estimates.tolist(), "bootstrap_mse": mse.tolist(),
        "selected_lambda": grid[best].tolist(), "p_values_used": len(p),
    }


def storey_qvalues(p_values, *, bh_adjusted=None):
    """Keep failed tests as p=1 in N and pi0 estimation, then mask their output.

    This explicitly retains the existing pipeline's planned-test convention.
    The atom at 1 is bookkeeping, not an observed p-value. Report both the
    primary pi0 and a sensitivity estimate excluding these placeholders.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.isinf(p).any():
        raise ValueError("p-values must be one-dimensional and may contain NaN, but not infinity.")
    valid = np.isfinite(p)
    if not valid.any():
        raise ValueError("At least one estimable test is required.")
    estimate = bootstrap_pi0(np.where(valid, p, 1.0))
    if valid.all():
        sensitivity = estimate["pi0"]
    else:
        try:
            sensitivity = bootstrap_pi0(p[valid])["pi0"]
        except ValueError:
            sensitivity = None
    bh = benjamini_hochberg(p) if bh_adjusted is None else np.asarray(bh_adjusted, dtype=float)
    if bh.shape != p.shape or not np.array_equal(np.isfinite(bh), valid):
        raise ValueError("BH-adjusted values must match the input shape and failed-test mask.")
    if np.any((bh[valid] < 0) | (bh[valid] > 1)):
        raise ValueError("BH-adjusted values must lie in [0, 1].")
    estimate.update(failed_tests_as_p_one=int((~valid).sum()),
                    pi0_sensitivity_excluding_failed_tests=sensitivity,
                    pfdr=False, local_fdr_computed=False)
    return bh * estimate["pi0"], estimate
