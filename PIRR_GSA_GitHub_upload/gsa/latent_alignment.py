"""Training-only semantic alignment for anonymous latent coordinates."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _as_2d_finite(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _column_correlation_matrix(latent: np.ndarray, descriptors: np.ndarray) -> np.ndarray:
    latent_centered = latent - np.mean(latent, axis=0, keepdims=True)
    descriptor_centered = descriptors - np.mean(descriptors, axis=0, keepdims=True)
    numerator = latent_centered.T @ descriptor_centered
    latent_norm = np.sqrt(np.sum(latent_centered**2, axis=0))
    descriptor_norm = np.sqrt(np.sum(descriptor_centered**2, axis=0))
    denominator = latent_norm[:, None] * descriptor_norm[None, :]
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 1e-12,
    )


@dataclass(frozen=True)
class SemanticAlignment:
    """One-to-one latent-to-descriptor mapping fitted on training data."""

    latent_indices: np.ndarray
    signs: np.ndarray
    training_correlations: np.ndarray
    n_latent: int

    def transform(self, latent) -> np.ndarray:
        """Apply the fitted coordinate permutation/signs without refitting."""
        latent_array = _as_2d_finite("latent", latent)
        if latent_array.shape[1] != self.n_latent:
            raise ValueError(
                f"latent has {latent_array.shape[1]} columns; expected {self.n_latent}."
            )
        return latent_array[:, self.latent_indices] * self.signs

    def to_dict(self) -> Dict[str, object]:
        return {
            "latent_indices": self.latent_indices.astype(int).tolist(),
            "signs": self.signs.astype(float).tolist(),
            "training_correlations": self.training_correlations.astype(float).tolist(),
            "fit_scope": "training_only",
        }


def fit_semantic_alignment(
    latent_train,
    descriptors_train,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SemanticAlignment:
    """
    Fit a one-to-one semantic mapping using training samples only.

    The Hungarian assignment maximizes the sum of absolute Pearson
    correlations. Signs are then fixed so every matched training correlation
    is non-negative. No scale fitting is used because Sobol indices are
    invariant to non-zero scalar rescaling.
    """
    latent = _as_2d_finite("latent_train", latent_train)
    descriptors = _as_2d_finite("descriptors_train", descriptors_train)
    if latent.shape[0] != descriptors.shape[0]:
        raise ValueError(
            "latent_train and descriptors_train must contain the same number of samples."
        )

    if candidate_indices is None:
        candidates = np.arange(latent.shape[1], dtype=int)
    else:
        candidates = np.asarray(candidate_indices, dtype=int).reshape(-1)
        if candidates.size == 0:
            raise ValueError("candidate_indices must not be empty.")
        if np.unique(candidates).size != candidates.size:
            raise ValueError("candidate_indices must be unique.")
        if np.any(candidates < 0) or np.any(candidates >= latent.shape[1]):
            raise ValueError("candidate_indices contains an out-of-range column index.")

    n_descriptors = descriptors.shape[1]
    if candidates.size < n_descriptors:
        raise ValueError(
            "At least as many candidate latent coordinates as descriptors are required."
        )

    correlations = _column_correlation_matrix(latent[:, candidates], descriptors)
    candidate_rows, descriptor_columns = linear_sum_assignment(-np.abs(correlations))

    latent_indices = np.empty(n_descriptors, dtype=int)
    signs = np.ones(n_descriptors, dtype=float)
    matched_correlations = np.zeros(n_descriptors, dtype=float)
    for candidate_row, descriptor_column in zip(candidate_rows, descriptor_columns):
        correlation = float(correlations[candidate_row, descriptor_column])
        latent_indices[descriptor_column] = int(candidates[candidate_row])
        signs[descriptor_column] = -1.0 if correlation < 0.0 else 1.0
        matched_correlations[descriptor_column] = abs(correlation)

    return SemanticAlignment(
        latent_indices=latent_indices,
        signs=signs,
        training_correlations=matched_correlations,
        n_latent=int(latent.shape[1]),
    )


def semantic_r2_score(aligned_latent, descriptors) -> float:
    """Mean squared diagonal correlation for already aligned coordinates."""
    aligned = _as_2d_finite("aligned_latent", aligned_latent)
    descriptor_array = _as_2d_finite("descriptors", descriptors)
    if aligned.shape != descriptor_array.shape:
        raise ValueError("aligned_latent and descriptors must have identical shapes.")
    correlations = _column_correlation_matrix(aligned, descriptor_array)
    return float(np.mean(np.diag(correlations) ** 2))
