"""Controlled nonlinear battery-discharge benchmark for Case 2.

The benchmark is intentionally separate from the observational NASA loader.
Inputs are independent variables on
``[0, 1]^9`` and are mapped to identifiable physical quantities before a
constant-current discharge with three diagnostic current pulses is simulated.
The response is terminal voltage; the fixed descriptor interface contains a
plateau level, a pulse-induced voltage drop, and a late-discharge decline.

The model is lightweight and vectorized so that an independent pick-freeze
Monte Carlo reference can be audited at large sample sizes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np


PARAM_NAMES = [
    "R0",
    "Rct",
    "tau_ct",
    "Q",
    "V_plateau",
    "OCV_slope",
    "R_growth",
    "knee_gain",
    "knee_soc",
]

DESC_NAMES = ["Plateau", "PulseDrop", "TailDrop"]

PARAM_BOUNDS = np.asarray(
    [
        [0.012, 0.045],   # R0 [ohm]
        [0.008, 0.040],   # Rct [ohm]
        [40.0, 320.0],    # polarization time constant [s]
        [2.20, 3.40],     # usable capacity [Ah]
        [3.55, 3.85],     # central OCV plateau [V]
        [0.20, 0.55],     # smooth OCV slope [V / SOC]
        [0.15, 0.85],     # relative ohmic growth toward low SOC
        [0.15, 0.45],     # low-SOC knee magnitude [V]
        [0.10, 0.25],     # knee location [SOC]
    ],
    dtype=float,
)


def map_unit_inputs(x_unit: np.ndarray) -> np.ndarray:
    """Map independent unit-hypercube inputs to physical parameters."""
    x_unit = np.asarray(x_unit, dtype=float)
    if x_unit.ndim != 2 or x_unit.shape[1] != len(PARAM_NAMES):
        raise ValueError(f"x_unit must have shape (N, {len(PARAM_NAMES)}).")
    if np.any((x_unit < 0.0) | (x_unit > 1.0)):
        raise ValueError("x_unit must lie in [0, 1].")
    lo = PARAM_BOUNDS[:, 0]
    hi = PARAM_BOUNDS[:, 1]
    return lo + x_unit * (hi - lo)


def current_profile(t: np.ndarray) -> np.ndarray:
    """Fixed discharge protocol with three short diagnostic pulses."""
    t = np.asarray(t, dtype=float)
    frac = t / max(float(t[-1]), 1.0)
    current = np.full_like(frac, 1.65, dtype=float)
    for left, right, increment in (
        (0.14, 0.21, 1.10),
        (0.46, 0.54, 0.90),
        (0.73, 0.80, 0.70),
    ):
        current[(frac >= left) & (frac < right)] += increment
    return current


@lru_cache(maxsize=8)
def battery_descriptor_matrix(T: int) -> np.ndarray:
    """Return the linear operator matrix for the three fixed descriptors."""
    n_steps = int(T)
    if n_steps < 2:
        raise ValueError("T must be at least 2.")

    def segment_weights(left: float, right: float) -> np.ndarray:
        i0 = max(0, min(n_steps - 1, int(round(left * (n_steps - 1)))))
        i1 = max(i0 + 1, min(n_steps, int(round(right * (n_steps - 1))) + 1))
        weights = np.zeros(n_steps, dtype=float)
        weights[i0:i1] = 1.0 / float(i1 - i0)
        return weights

    plateau = segment_weights(0.27, 0.42)
    pulse_drop = segment_weights(0.115, 0.135) - segment_weights(0.175, 0.195)
    tail_drop = segment_weights(0.64, 0.71) - segment_weights(0.92, 0.98)
    return np.vstack([plateau, pulse_drop, tail_drop])


def battery_descriptors(voltage: np.ndarray) -> np.ndarray:
    """Return the fixed Plateau--PulseDrop--TailDrop interface."""
    voltage = np.asarray(voltage, dtype=float)
    if voltage.ndim != 2:
        raise ValueError("voltage must have shape (N, T).")
    return voltage @ battery_descriptor_matrix(voltage.shape[1]).T


@lru_cache(maxsize=8)
def battery_descriptor_lift(T: int, smoothness: float = 1.0e4) -> np.ndarray:
    """Return a smooth right inverse ``B`` satisfying ``P B = I``.

    The fixed lifting functions minimize ``||B||^2 + smoothness*||D2 B||^2``
    under the descriptor-consistency constraint.  Unlike the discontinuous
    minimum-norm lift, this does not inject artificial steps at descriptor
    window boundaries into the residual.  ``smoothness`` is a declared
    operator-level constant, not fitted from benchmark outcomes.
    """
    operator = battery_descriptor_matrix(int(T))
    identity = np.eye(int(T), dtype=float)
    second_difference = np.diff(identity, n=2, axis=0)
    metric = identity + float(smoothness) * (
        second_difference.T @ second_difference
    )
    metric_inverse_operator_t = np.linalg.solve(metric, operator.T)
    return metric_inverse_operator_t @ np.linalg.pinv(
        operator @ metric_inverse_operator_t
    )


def battery_residual_response(voltage: np.ndarray) -> np.ndarray:
    """Remove the exact linear descriptor component from each response.

    For ``P = battery_descriptor_matrix(T)`` and its fixed right inverse ``B``,
    this returns ``R = Y - P(Y) B^T``.  Consequently ``P(R)=0`` up to floating
    point precision.  This is an operator-nullspace residual, not a
    probabilistic conditional response.
    """
    voltage = np.asarray(voltage, dtype=float)
    if voltage.ndim != 2:
        raise ValueError("voltage must have shape (N, T).")
    descriptors = battery_descriptors(voltage)
    return voltage - descriptors @ battery_descriptor_lift(voltage.shape[1]).T


def eval_controlled_battery_from_x(
    x_unit: np.ndarray,
    T: int = 160,
    noise_scale: float = 0.0,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Simulate voltage curves and descriptors for unit-hypercube inputs.

    Returns ``(Y, P, t, states)`` where ``Y`` has shape ``(N, T)`` and ``P``
    has shape ``(N, 3)``.  Noise is applied only to the returned voltage and
    hence to its descriptors; the state audit remains noise free.
    """
    if int(T) < 80:
        raise ValueError("T must be at least 80 to resolve the pulse windows.")
    physical = map_unit_inputs(x_unit)
    R0, Rct, tau_ct, capacity, v_plateau, ocv_slope, r_growth, knee_gain, knee_soc = physical.T
    n_samples = physical.shape[0]
    t = np.linspace(0.0, 3600.0, int(T), dtype=float)
    dt = float(t[1] - t[0])
    current = current_profile(t)

    soc = np.empty((n_samples, int(T)), dtype=float)
    v_pol = np.empty_like(soc)
    soc[:, 0] = 1.0
    v_pol[:, 0] = 0.0
    alpha = np.exp(-dt / tau_ct)
    for j in range(1, int(T)):
        soc[:, j] = np.maximum(
            soc[:, j - 1] - current[j - 1] * dt / (3600.0 * capacity),
            0.015,
        )
        v_pol[:, j] = alpha * v_pol[:, j - 1] + Rct * (1.0 - alpha) * current[j - 1]

    knee_width = 0.028
    knee = knee_gain[:, None] / (
        1.0 + np.exp(np.clip((soc - knee_soc[:, None]) / knee_width, -40.0, 40.0))
    )
    ocv = (
        v_plateau[:, None]
        + ocv_slope[:, None] * (soc - 0.50)
        - knee
    )
    effective_r0 = R0[:, None] * (1.0 + r_growth[:, None] * (1.0 - soc))
    voltage_clean = ocv - current[None, :] * effective_r0 - v_pol

    voltage = voltage_clean.copy()
    if float(noise_scale) > 0.0:
        rng = np.random.default_rng(seed)
        voltage += rng.normal(0.0, float(noise_scale), size=voltage.shape)

    descriptors = battery_descriptors(voltage)
    states = {
        "physical_parameters": physical,
        "soc": soc,
        "polarization_voltage": v_pol,
        "voltage_clean": voltage_clean,
        "current": current,
    }
    return voltage, descriptors, t, states


def generate_controlled_battery_data(
    N: int = 200,
    T: int = 160,
    seed: int = 0,
    noise_scale: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate an independent training design on the declared unit bounds."""
    rng = np.random.default_rng(int(seed))
    x_unit = rng.random((int(N), len(PARAM_NAMES)))
    voltage, descriptors, t, _ = eval_controlled_battery_from_x(
        x_unit,
        T=int(T),
        noise_scale=float(noise_scale),
        seed=int(seed) + 100_003,
    )
    return x_unit, voltage, descriptors, t


__all__ = [
    "PARAM_NAMES",
    "DESC_NAMES",
    "PARAM_BOUNDS",
    "battery_descriptors",
    "battery_descriptor_matrix",
    "battery_descriptor_lift",
    "battery_residual_response",
    "current_profile",
    "eval_controlled_battery_from_x",
    "generate_controlled_battery_data",
    "map_unit_inputs",
]
