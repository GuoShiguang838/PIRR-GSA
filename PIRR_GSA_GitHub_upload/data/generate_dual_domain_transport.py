"""Dual-domain advection--dispersion--reaction breakthrough benchmark.

The response is a mass-weighted combination of a mobile-domain first-passage
Green function and a retarded matrix-domain contribution with first-order
release.  It is a controlled surrogate of preferential/matrix transport, not
an empirical curve assembled after method comparison.  The predeclared
Mass--MeanArrival--Spread interface contains classical temporal moments;
secondary shoulders, multimodality, skewness, and release tails remain in the
residual.  The older Peak--Arrival--Mass interface can still be audited by the
screening script as a deliberately less smooth alternative.
"""

from __future__ import annotations

import numpy as np

from data.generate_transport import transport_descriptors


PARAM_NAMES = ["v", "D_m", "R_i", "D_i", "f_i", "alpha", "k", "M", "tau"]
TIME_MIN = 0.02
TIME_MAX = 7.0


def dual_domain_moment_descriptors(
    Y: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Return Mass, mean arrival time, and temporal spread."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    t = np.asarray(t, dtype=float).reshape(-1)
    if Y.ndim != 2 or Y.shape[1] != t.size:
        raise ValueError("Y must have shape (N, T) matching t.")
    nonnegative = np.maximum(Y, 0.0)
    mass = np.trapz(nonnegative, t, axis=1)
    safe_mass = np.maximum(mass, 1e-14)
    mean_arrival = np.trapz(nonnegative * t[None, :], t, axis=1) / safe_mass
    centered = t[None, :] - mean_arrival[:, None]
    variance = (
        np.trapz(nonnegative * centered**2, t, axis=1) / safe_mass
    )
    spread = np.sqrt(np.maximum(variance, 1e-12))
    return np.column_stack([mass, mean_arrival, spread])


def dual_domain_moment_template(
    descriptors: np.ndarray,
    t: np.ndarray,
    *,
    iterations: int = 14,
) -> np.ndarray:
    """Maximum-entropy lift matching three discrete temporal moments.

    The density belongs to ``exp(a*t + b*t^2)`` on the reported time grid.
    A vectorized Newton solve matches the target first and second raw moments;
    normalization then matches Mass exactly under trapezoidal quadrature.
    """
    descriptors = np.asarray(descriptors, dtype=float)
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(1, -1)
    if descriptors.ndim != 2 or descriptors.shape[1] != 3:
        raise ValueError("descriptors must have shape (N, 3).")
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size < 3 or np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be a strictly increasing grid with at least 3 points.")
    mass = np.maximum(descriptors[:, 0], 1e-14)
    target_mean = np.clip(descriptors[:, 1], float(t[0]), float(t[-1]))
    grid_span = float(t[-1] - t[0])
    target_spread = np.clip(
        descriptors[:, 2], grid_span * 1e-4, grid_span * 0.45
    )
    target_second = target_mean**2 + target_spread**2

    sigma2 = np.maximum(target_spread**2, 1e-8)
    a = target_mean / sigma2
    b = -0.5 / sigma2
    weights = np.empty_like(t)
    weights[1:-1] = 0.5 * (t[2:] - t[:-2])
    weights[0] = 0.5 * (t[1] - t[0])
    weights[-1] = 0.5 * (t[-1] - t[-2])
    t1 = t[None, :]
    t2 = t1**2
    for _ in range(int(iterations)):
        log_density = a[:, None] * t1 + b[:, None] * t2
        log_density -= np.max(log_density, axis=1, keepdims=True)
        weighted = np.exp(np.clip(log_density, -745.0, 0.0)) * weights[None, :]
        probabilities = weighted / np.maximum(
            np.sum(weighted, axis=1, keepdims=True), 1e-300
        )
        m1 = np.sum(probabilities * t1, axis=1)
        m2 = np.sum(probabilities * t2, axis=1)
        m3 = np.sum(probabilities * t1**3, axis=1)
        m4 = np.sum(probabilities * t1**4, axis=1)
        j11 = np.maximum(m2 - m1**2, 1e-12)
        j12 = m3 - m1 * m2
        j22 = np.maximum(m4 - m2**2, 1e-12)
        det = j11 * j22 - j12**2
        safe_det = np.where(np.abs(det) > 1e-16, det, np.sign(det) * 1e-16 + (det == 0) * 1e-16)
        error1 = m1 - target_mean
        error2 = m2 - target_second
        delta_a = (j22 * error1 - j12 * error2) / safe_det
        delta_b = (-j12 * error1 + j11 * error2) / safe_det
        step_scale = np.minimum(
            1.0,
            5.0 / np.maximum(np.maximum(np.abs(delta_a), np.abs(delta_b)), 1e-12),
        )
        a -= step_scale * delta_a
        b -= step_scale * delta_b
        b = np.minimum(b, -1e-8)

    log_density = a[:, None] * t1 + b[:, None] * t2
    log_density -= np.max(log_density, axis=1, keepdims=True)
    density = np.exp(np.clip(log_density, -745.0, 0.0))
    area = np.trapz(density, t, axis=1)
    return density * (mass / np.maximum(area, 1e-300))[:, None]


def _physical_parameters(X: np.ndarray) -> tuple[np.ndarray, ...]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(PARAM_NAMES):
        raise ValueError(f"X must have shape (N, {len(PARAM_NAMES)}).")
    if not np.all(np.isfinite(X)) or np.any(X < 0.0) or np.any(X > 1.0):
        raise ValueError("X must contain finite values in [0, 1].")
    v = 0.85 + 0.70 * X[:, 0]
    d_mobile = 0.012 + 0.055 * X[:, 1]
    retardation = 1.65 + 1.70 * X[:, 2]
    d_immobile = 0.020 + 0.110 * X[:, 3]
    immobile_fraction = 0.16 + 0.34 * X[:, 4]
    exchange_rate = 0.45 + 1.55 * X[:, 5]
    decay = 0.00 + 0.075 * X[:, 6]
    mass = 0.75 + 0.75 * X[:, 7]
    pulse_duration = 0.035 + 0.165 * X[:, 8]
    return (
        v,
        d_mobile,
        retardation,
        d_immobile,
        immobile_fraction,
        exchange_rate,
        decay,
        mass,
        pulse_duration,
    )


def _first_passage_density(
    time: np.ndarray,
    velocity: np.ndarray,
    dispersion: np.ndarray,
    *,
    distance: float = 1.0,
) -> np.ndarray:
    """Inverse-Gaussian first-passage density for a 1-D ADE path."""
    positive_time = np.maximum(np.asarray(time, dtype=float), 1e-10)
    velocity = np.asarray(velocity, dtype=float)[:, None]
    dispersion = np.asarray(dispersion, dtype=float)[:, None]
    prefactor = float(distance) / np.sqrt(
        4.0 * np.pi * dispersion * positive_time**3
    )
    exponent = -(
        (float(distance) - velocity * positive_time) ** 2
        / (4.0 * dispersion * positive_time)
    )
    density = prefactor * np.exp(np.clip(exponent, -745.0, 20.0))
    density[~np.isfinite(density)] = 0.0
    return density


def _pulse_averaged_path(
    t: np.ndarray,
    velocity: np.ndarray,
    dispersion: np.ndarray,
    pulse_duration: np.ndarray,
) -> np.ndarray:
    """Five-point midpoint quadrature for a finite uniform inlet pulse."""
    offsets = (np.arange(5, dtype=float) + 0.5) / 5.0
    response = np.zeros((velocity.size, t.size), dtype=float)
    for offset in offsets:
        shifted = t[None, :] - offset * pulse_duration[:, None]
        active = shifted > 0.0
        density = _first_passage_density(
            np.maximum(shifted, 1e-10), velocity, dispersion
        )
        response += np.where(active, density, 0.0) / offsets.size
    return response


def _normalize_area(curves: np.ndarray, t: np.ndarray) -> np.ndarray:
    areas = np.trapz(curves, t, axis=1)
    return curves / np.maximum(areas[:, None], 1e-14)


def _exchange_release(
    source: np.ndarray,
    exchange_rate: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """First-order mobile--immobile release filter with unit DC gain."""
    output = np.zeros_like(source)
    if t.size < 2:
        return source.copy()
    for index in range(1, t.size):
        dt = float(t[index] - t[index - 1])
        memory = np.exp(-exchange_rate * dt)
        output[:, index] = (
            memory * output[:, index - 1]
            + (1.0 - memory) * source[:, index]
        )
    return output


def eval_dual_domain_transport_from_x(
    X: np.ndarray,
    *,
    T: int = 192,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Evaluate clean breakthrough curves for normalized uncertain inputs."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(PARAM_NAMES):
        raise ValueError(f"X must have shape (N, {len(PARAM_NAMES)}).")
    if X.shape[0] > int(chunk_size):
        return np.concatenate(
            [
                eval_dual_domain_transport_from_x(
                    X[start : start + int(chunk_size)], T=T, chunk_size=chunk_size
                )
                for start in range(0, X.shape[0], int(chunk_size))
            ],
            axis=0,
        )
    (
        velocity,
        d_mobile,
        retardation,
        d_immobile,
        immobile_fraction,
        exchange_rate,
        decay,
        mass,
        pulse_duration,
    ) = _physical_parameters(X)
    t = np.linspace(TIME_MIN, TIME_MAX, int(T))
    mobile = _normalize_area(
        _pulse_averaged_path(t, velocity, d_mobile, pulse_duration), t
    )
    matrix_velocity = velocity / retardation
    matrix_dispersion = d_immobile / retardation
    matrix = _normalize_area(
        _pulse_averaged_path(
            t, matrix_velocity, matrix_dispersion, pulse_duration
        ),
        t,
    )
    released = _normalize_area(_exchange_release(matrix, exchange_rate, t), t)
    matrix_contribution = 0.55 * matrix + 0.45 * released
    curve = (
        (1.0 - immobile_fraction[:, None]) * mobile
        + immobile_fraction[:, None] * matrix_contribution
    )
    curve *= mass[:, None] * np.exp(-decay[:, None] * t[None, :])
    return np.maximum(curve, 0.0)


def generate_dual_domain_transport_data(
    N: int = 200,
    T: int = 192,
    seed: int = 42,
    noise_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    X = rng.random((int(N), len(PARAM_NAMES)))
    clean = eval_dual_domain_transport_from_x(X, T=int(T))
    Y = clean.copy()
    if float(noise_scale) > 0.0:
        amplitudes = np.std(clean, axis=1, keepdims=True) + 1e-12
        Y += rng.normal(0.0, float(noise_scale) * amplitudes, size=Y.shape)
        Y = np.maximum(Y, 0.0)
    t = np.linspace(TIME_MIN, TIME_MAX, int(T))
    descriptors = dual_domain_moment_descriptors(Y, t)
    return X, Y, descriptors, {
        "t": t,
        "param_names": list(PARAM_NAMES),
        "clean_response": clean,
        "model": "dual-domain ADE first-passage mixture with first-order release",
    }


if __name__ == "__main__":
    arrays = generate_dual_domain_transport_data(N=16, T=192, seed=0)
    print(arrays[0].shape, arrays[1].shape, arrays[2].shape)
