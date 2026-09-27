import numpy as np
from scipy.integrate import solve_ivp

from data.generate_damped_oscillator import (
    duffing_oscillator,
    oscillator_harmonic_descriptors,
    oscillator_harmonic_lift,
    sample_parameters,
    simulate_from_parameters_vectorized,
)


def test_vectorized_rk4_matches_solve_ivp():
    X = sample_parameters(N=3, seed=77)
    t_reference = np.linspace(0.0, 40.0, 192)
    reference = np.asarray(
        [
            solve_ivp(
                duffing_oscillator,
                [0.0, 40.0],
                [0.0, 0.0],
                args=tuple(row),
                t_eval=t_reference,
                rtol=1e-10,
                atol=1e-12,
            ).y[0]
            for row in X
        ]
    )
    actual, t_actual = simulate_from_parameters_vectorized(
        X, T_steps=192, T_end=40.0, internal_substeps=2
    )
    assert np.allclose(t_actual, t_reference)
    normalized_rmse = np.sqrt(np.mean((actual - reference) ** 2)) / np.std(reference)
    assert normalized_rmse < 2e-4


def test_harmonic_interface_and_lift_are_finite():
    X = sample_parameters(N=8, seed=11)
    Y, t = simulate_from_parameters_vectorized(
        X, T_steps=192, T_end=40.0, internal_substeps=2
    )
    descriptors = oscillator_harmonic_descriptors(Y, t)
    lifted = oscillator_harmonic_lift(descriptors, t)
    recovered = oscillator_harmonic_descriptors(lifted, t)
    assert descriptors.shape == (8, 3)
    assert lifted.shape == Y.shape
    assert np.all(np.isfinite(descriptors))
    assert np.all(np.isfinite(lifted))
    assert np.all((descriptors[:, 0] >= 0.65) & (descriptors[:, 0] <= 1.35))
    assert np.max(np.abs(recovered - descriptors)) < 2e-6
