"""Utilities for Sobol recovery of complete temporal response fields.

The primary estimand is a pair of matrices with shape ``(T, d)``:
``S1[t, i]`` and ``ST[t, i]``.  This keeps the time-resolved response as the
GSA target instead of replacing it with a small set of scalar descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from gsa.pce_sobol import _orthonormal_legendre_design, _total_degree_indices


@dataclass
class MultiOutputPCESurrogate:
    """Second-stage PCE surrogate for one or many response coordinates."""

    lower: np.ndarray
    upper: np.ndarray
    active: np.ndarray
    order: int
    multi_indices: list[tuple[int, ...]]
    target_mean: np.ndarray
    coefficients: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.lower.size:
            raise ValueError(f"X must have shape (N, {self.lower.size}).")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values.")
        if np.any(self.active):
            spans = self.upper[self.active] - self.lower[self.active]
            unit = 2.0 * (X[:, self.active] - self.lower[self.active]) / spans - 1.0
            if np.any(unit < -1.0 - 1e-10) or np.any(unit > 1.0 + 1e-10):
                raise ValueError("X contains values outside the fitted bounds.")
            unit = np.clip(unit, -1.0, 1.0)
        else:
            unit = np.empty((X.shape[0], 0), dtype=float)
        design = _orthonormal_legendre_design(unit, self.multi_indices, self.order)
        return design @ self.coefficients + self.target_mean


def fit_multioutput_pce(
    X: np.ndarray,
    targets: np.ndarray,
    *,
    order: int = 2,
    bounds: Optional[np.ndarray] = None,
    ridge_alpha: float = 1e-6,
) -> MultiOutputPCESurrogate:
    """Fit the same orthonormal Ridge-PCE protocol used by formal experiments."""
    X = np.asarray(X, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.ndim != 2 or targets.shape[0] != X.shape[0]:
        raise ValueError("targets must have shape (N,) or (N, q).")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(targets)):
        raise ValueError("X and targets must contain finite values.")
    d = X.shape[1]
    if bounds is None:
        lower = np.min(X, axis=0)
        upper = np.max(X, axis=0)
    else:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (d, 2):
            raise ValueError(f"bounds must have shape ({d}, 2).")
        lower = bounds[:, 0].copy()
        upper = bounds[:, 1].copy()
    spans = upper - lower
    if np.any(spans < 0):
        raise ValueError("Upper bounds must not be smaller than lower bounds.")
    active = spans > 1e-12
    if np.any(active):
        unit = 2.0 * (X[:, active] - lower[active]) / spans[active] - 1.0
        if np.any(unit < -1.0 - 1e-10) or np.any(unit > 1.0 + 1e-10):
            raise ValueError("X contains values outside the supplied bounds.")
        unit = np.clip(unit, -1.0, 1.0)
    else:
        unit = np.empty((X.shape[0], 0), dtype=float)

    order = int(order)
    multi_indices = _total_degree_indices(int(np.sum(active)), order)
    design = _orthonormal_legendre_design(unit, multi_indices, order)
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += float(ridge_alpha)
    target_mean = np.mean(targets, axis=0, keepdims=True)
    coefficients = np.linalg.solve(gram, design.T @ (targets - target_mean))
    return MultiOutputPCESurrogate(
        lower=lower,
        upper=upper,
        active=active,
        order=order,
        multi_indices=multi_indices,
        target_mean=target_mean,
        coefficients=coefficients,
    )


def timewise_pick_freeze_reference(
    response_evaluator: Callable[[np.ndarray], np.ndarray],
    *,
    n_inputs: int,
    n_base: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute time-resolved Saltelli/Jansen indices from the response model.

    The implementation streams the hybrid designs so memory scales with
    ``N_base * T`` rather than ``d * N_base * T``.
    """
    rng = np.random.default_rng(int(seed))
    A = rng.random((int(n_base), int(n_inputs)))
    B = rng.random((int(n_base), int(n_inputs)))
    y_a = np.asarray(response_evaluator(A), dtype=float)
    y_b = np.asarray(response_evaluator(B), dtype=float)
    if y_a.ndim != 2 or y_a.shape != y_b.shape or y_a.shape[0] != int(n_base):
        raise ValueError("response_evaluator must return matching arrays with shape (N, T).")
    pooled = np.concatenate([y_a, y_b], axis=0)
    means = np.mean(pooled, axis=0)
    variances = np.var(pooled, axis=0)
    safe = variances > 1e-14
    first = np.zeros((y_a.shape[1], int(n_inputs)), dtype=float)
    total = np.zeros_like(first)
    centered_b = y_b - means[None, :]
    for input_index in range(int(n_inputs)):
        hybrid_inputs = A.copy()
        hybrid_inputs[:, input_index] = B[:, input_index]
        hybrid = np.asarray(response_evaluator(hybrid_inputs), dtype=float)
        if hybrid.shape != y_a.shape:
            raise ValueError("Hybrid response shape changed across evaluations.")
        first[safe, input_index] = (
            np.mean(centered_b[:, safe] * (hybrid[:, safe] - y_a[:, safe]), axis=0)
            / variances[safe]
        )
        total[safe, input_index] = (
            0.5 * np.mean((y_a[:, safe] - hybrid[:, safe]) ** 2, axis=0)
            / variances[safe]
        )
    return (
        np.clip(np.nan_to_num(first), 0.0, 1.0),
        np.clip(np.nan_to_num(total), 0.0, 1.0),
        variances,
    )


def functional_sobol_metrics(
    estimated_s1: np.ndarray,
    estimated_st: np.ndarray,
    reference_s1: np.ndarray,
    reference_st: np.ndarray,
    *,
    temporal_variances: Optional[np.ndarray] = None,
    active_variance_fractions: tuple[float, ...] = (1e-4, 1e-3, 1e-2),
) -> dict[str, object]:
    """Evaluate complete ``parameter x time`` Sobol recovery.

    ``RMSE_func`` keeps the declared uniform-in-time metric.  When temporal
    variances are supplied, the returned variance-weighted and active-time
    metrics audit whether near-zero-variance response intervals drive that
    result.  Active sets are fixed only from the direct-reference variance,
    never from a compared method.
    """
    estimated_s1 = np.asarray(estimated_s1, dtype=float)
    estimated_st = np.asarray(estimated_st, dtype=float)
    reference_s1 = np.asarray(reference_s1, dtype=float)
    reference_st = np.asarray(reference_st, dtype=float)
    if not (
        estimated_s1.shape
        == estimated_st.shape
        == reference_s1.shape
        == reference_st.shape
    ):
        raise ValueError("All Sobol matrices must have the same shape (T, d).")
    if estimated_s1.ndim != 2:
        raise ValueError("Sobol matrices must have shape (T, d).")
    rmse_s1 = float(np.sqrt(np.mean((estimated_s1 - reference_s1) ** 2)))
    rmse_st = float(np.sqrt(np.mean((estimated_st - reference_st) ** 2)))
    rmse_combined = float(np.sqrt(0.5 * (rmse_s1**2 + rmse_st**2)))
    rmse_s1_by_input = np.sqrt(np.mean((estimated_s1 - reference_s1) ** 2, axis=0))
    rmse_st_by_input = np.sqrt(np.mean((estimated_st - reference_st) ** 2, axis=0))

    if temporal_variances is None:
        nonnegative_variances = np.ones(estimated_s1.shape[0], dtype=float)
        weights = np.full(estimated_s1.shape[0], 1.0 / estimated_s1.shape[0])
    else:
        temporal_variances = np.asarray(temporal_variances, dtype=float).reshape(-1)
        if temporal_variances.size != estimated_s1.shape[0]:
            raise ValueError("temporal_variances must have length T.")
        if not np.all(np.isfinite(temporal_variances)):
            raise ValueError("temporal_variances must contain finite values.")
        nonnegative_variances = np.maximum(temporal_variances, 0.0)
        total_variance = float(np.sum(nonnegative_variances))
        weights = (
            nonnegative_variances / total_variance
            if total_variance > 1e-24
            else np.full(estimated_s1.shape[0], 1.0 / estimated_s1.shape[0])
        )
    squared_s1 = (estimated_s1 - reference_s1) ** 2
    squared_st = (estimated_st - reference_st) ** 2
    weighted_rmse_s1 = float(np.sqrt(np.mean(weights @ squared_s1)))
    weighted_rmse_st = float(np.sqrt(np.mean(weights @ squared_st)))
    weighted_rmse = float(
        np.sqrt(0.5 * (weighted_rmse_s1**2 + weighted_rmse_st**2))
    )
    integrated_s1_est = weights @ estimated_s1
    integrated_st_est = weights @ estimated_st
    integrated_s1_ref = weights @ reference_s1
    integrated_st_ref = weights @ reference_st
    metrics: dict[str, object] = {
        "RMSE_func": rmse_combined,
        "RMSE_func_S1": rmse_s1,
        "RMSE_func_ST": rmse_st,
        "RMSE_func_variance_weighted": weighted_rmse,
        "RMSE_func_S1_variance_weighted": weighted_rmse_s1,
        "RMSE_func_ST_variance_weighted": weighted_rmse_st,
        "RMSE_func_S1_by_input": rmse_s1_by_input,
        "RMSE_func_ST_by_input": rmse_st_by_input,
        "G1_est": integrated_s1_est,
        "GT_est": integrated_st_est,
        "G1_ref": integrated_s1_ref,
        "GT_ref": integrated_st_ref,
        "RMSE_functional_index": float(
            np.sqrt(
                0.5
                * (
                    np.mean((integrated_s1_est - integrated_s1_ref) ** 2)
                    + np.mean((integrated_st_est - integrated_st_ref) ** 2)
                )
            )
        ),
    }
    max_variance = float(np.max(nonnegative_variances))
    for fraction in active_variance_fractions:
        fraction = float(fraction)
        if not np.isfinite(fraction) or fraction < 0.0:
            raise ValueError("active_variance_fractions must be finite and nonnegative.")
        threshold = fraction * max_variance
        active = nonnegative_variances >= threshold
        if not np.any(active):
            active = np.ones_like(nonnegative_variances, dtype=bool)
        active_rmse_s1 = float(np.sqrt(np.mean(squared_s1[active])))
        active_rmse_st = float(np.sqrt(np.mean(squared_st[active])))
        label = f"{fraction:.0e}"
        metrics[f"RMSE_func_active_{label}"] = float(
            np.sqrt(0.5 * (active_rmse_s1**2 + active_rmse_st**2))
        )
        metrics[f"RMSE_func_S1_active_{label}"] = active_rmse_s1
        metrics[f"RMSE_func_ST_active_{label}"] = active_rmse_st
        metrics[f"active_time_fraction_{label}"] = float(np.mean(active))
        metrics[f"active_time_count_{label}"] = int(np.sum(active))
    return metrics


def estimate_timewise_sobol_from_surrogate(
    response_predictor: Callable[[np.ndarray], np.ndarray],
    *,
    n_inputs: int,
    n_base: int = 8192,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate response-field Sobol matrices from a fitted response surrogate.

    This uses the same Saltelli/Jansen pick-freeze estimands as the direct
    reference, but simulator calls are replaced by cheap surrogate queries.
    """
    return timewise_pick_freeze_reference(
        response_predictor,
        n_inputs=int(n_inputs),
        n_base=int(n_base),
        seed=int(seed),
    )


__all__ = [
    "MultiOutputPCESurrogate",
    "fit_multioutput_pce",
    "functional_sobol_metrics",
    "estimate_timewise_sobol_from_surrogate",
    "timewise_pick_freeze_reference",
]
