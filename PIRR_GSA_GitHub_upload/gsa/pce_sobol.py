"""Minimal orthonormal Legendre basis helpers used by functional PCE."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.polynomial.legendre import legval


def _total_degree_indices(n_variables: int, order: int) -> List[Tuple[int, ...]]:
    """Return all non-negative multi-indices with total degree at most order."""
    if n_variables < 0:
        raise ValueError("n_variables must be non-negative.")
    if order < 0:
        raise ValueError("order must be non-negative.")
    if n_variables == 0:
        return [tuple()]
    indices: List[Tuple[int, ...]] = []

    def append_prefix(prefix: Tuple[int, ...], remaining: int, degree_left: int) -> None:
        if remaining == 1:
            for degree in range(degree_left + 1):
                indices.append(prefix + (degree,))
            return
        for degree in range(degree_left + 1):
            append_prefix(prefix + (degree,), remaining - 1, degree_left - degree)

    append_prefix(tuple(), n_variables, order)
    return indices


def _orthonormal_legendre_design(
    inputs_unit: np.ndarray,
    multi_indices: List[Tuple[int, ...]],
    order: int,
) -> np.ndarray:
    """Evaluate a tensor-product orthonormal Legendre basis on [-1, 1]."""
    n_samples, n_variables = inputs_unit.shape
    univariate = np.empty((n_variables, order + 1, n_samples), dtype=float)
    for variable in range(n_variables):
        for degree in range(order + 1):
            coefficients = np.zeros(degree + 1, dtype=float)
            coefficients[-1] = 1.0
            univariate[variable, degree] = np.sqrt(2 * degree + 1) * legval(
                inputs_unit[:, variable], coefficients
            )
    design = np.ones((n_samples, len(multi_indices)), dtype=float)
    for column, alpha in enumerate(multi_indices):
        for variable, degree in enumerate(alpha):
            if degree:
                design[:, column] *= univariate[variable, degree]
    return design
