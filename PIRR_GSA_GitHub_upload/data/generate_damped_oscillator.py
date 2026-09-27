import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

OSCILLATOR_LB = np.array([0.8, 0.05, 0.8, 0.1, 0.5, 0.8], dtype=float)
OSCILLATOR_UB = np.array([1.2, 0.3, 1.5, 1.0, 1.5, 1.2], dtype=float)

def duffing_oscillator(t, y, m, c, k, alpha, F, omega):
    x, v = y
    dxdt = v
    dvdt = (F * np.sin(omega * t) - c * v - k * x - alpha * (x**3)) / m
    return [dxdt, dvdt]


def get_parameter_bounds():
    return OSCILLATOR_LB.copy(), OSCILLATOR_UB.copy()


def sample_parameters(N=500, seed=42, dist: str = "uniform", beta_ab=(2.0, 2.0), rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    else:
        seed = None

    if str(dist).lower() == "beta":
        a, b = float(beta_ab[0]), float(beta_ab[1])
        U = rng.beta(a, b, size=(int(N), 6))
    else:
        U = rng.random((int(N), 6))
    return OSCILLATOR_LB + (OSCILLATOR_UB - OSCILLATOR_LB) * U


def simulate_from_parameters(
    X,
    T_steps=2000,
    T_end=20,
    noise_std: float = 0.0,
    progress_every: int = 100,
    progress_prefix: str = "",
):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 6:
        raise ValueError("X must have shape (N, 6) for Duffing parameters.")

    t_vec = np.linspace(0, T_end, int(T_steps))
    Y = []
    prefix = f"{progress_prefix} " if progress_prefix else ""
    print(f"{prefix}Simulating {X.shape[0]} Duffing trajectories (T={int(T_steps)})...")
    for i in range(X.shape[0]):
        m, c, k, alpha, F, omega = X[i]
        sol = solve_ivp(
            duffing_oscillator,
            [0, T_end],
            [0, 0],
            args=(m, c, k, alpha, F, omega),
            t_eval=t_vec,
            method='RK45'
        )
        y_i = sol.y[0]
        if noise_std is not None and float(noise_std) > 0.0:
            y_i = y_i + np.random.normal(0.0, float(noise_std), size=y_i.shape)
        Y.append(y_i)
        if int(progress_every) > 0 and (i + 1) % int(progress_every) == 0:
            print(f"{prefix}  Progress: {i+1}/{X.shape[0]}")
    return np.asarray(Y, dtype=float), t_vec


def simulate_from_parameters_vectorized(
    X,
    T_steps=192,
    T_end=40.0,
    internal_substeps: int = 2,
):
    """Vectorized fixed-step RK4 solver for large direct-MC audits.

    The ``solve_ivp`` route remains available as the scalar data generator. This
    implementation evaluates the same Duffing equation for many parameter
    samples at once, making time-wise pick-freeze references practical.  The
    intended formal configuration uses two RK4 substeps per reported time
    interval; its agreement with ``solve_ivp`` is covered by a regression test.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 6:
        raise ValueError("X must have shape (N, 6) for Duffing parameters.")
    if int(T_steps) < 2 or float(T_end) <= 0.0 or int(internal_substeps) < 1:
        raise ValueError("T_steps >= 2, T_end > 0, and internal_substeps >= 1 are required.")
    m, c, k, alpha, force, omega = X.T
    t_vec = np.linspace(0.0, float(T_end), int(T_steps))
    dt_output = float(t_vec[1] - t_vec[0])
    dt = dt_output / int(internal_substeps)
    position = np.zeros(X.shape[0], dtype=float)
    velocity = np.zeros_like(position)
    response = np.empty((X.shape[0], int(T_steps)), dtype=float)
    response[:, 0] = position

    def acceleration(time, pos, vel):
        return (force * np.sin(omega * time) - c * vel - k * pos - alpha * pos**3) / m

    time = 0.0
    for output_index in range(1, int(T_steps)):
        for _ in range(int(internal_substeps)):
            k1_x = velocity
            k1_v = acceleration(time, position, velocity)
            k2_x = velocity + 0.5 * dt * k1_v
            k2_v = acceleration(
                time + 0.5 * dt,
                position + 0.5 * dt * k1_x,
                velocity + 0.5 * dt * k1_v,
            )
            k3_x = velocity + 0.5 * dt * k2_v
            k3_v = acceleration(
                time + 0.5 * dt,
                position + 0.5 * dt * k2_x,
                velocity + 0.5 * dt * k2_v,
            )
            k4_x = velocity + dt * k3_v
            k4_v = acceleration(
                time + dt,
                position + dt * k3_x,
                velocity + dt * k3_v,
            )
            position = position + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            velocity = velocity + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
            time += dt
        response[:, output_index] = position
    return response, t_vec


def oscillator_harmonic_descriptors(
    Y,
    t,
    fit_fraction: float = 0.55,
    onset_time: float = 4.0,
):
    """Return dominant frequency and quadrature amplitudes of each response.

    ``[omega_d, a_sin, a_cos]`` is a continuous alternative to a wrapped
    amplitude-phase coordinate.  It describes the late-time fundamental
    oscillation, while nonlinear harmonics and start-up transients remain in
    the residual field.
    """
    Y = np.asarray(Y, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    if Y.ndim == 1:
        Y = Y[None, :]
    if Y.ndim != 2 or Y.shape[1] != t.size:
        raise ValueError("Y must have shape (N, T) and t must have length T.")
    if not (0.25 <= float(fit_fraction) <= 0.9):
        raise ValueError("fit_fraction must lie in [0.25, 0.9].")
    start = int(np.floor((1.0 - float(fit_fraction)) * t.size))
    t_fit = t[start:]
    onset = 1.0 - np.exp(-t_fit / max(float(onset_time), 1e-6))
    coarse_grid = np.linspace(0.65, 1.35, 71)
    result = np.empty((Y.shape[0], 3), dtype=float)

    def fit_at_frequency(values, omega):
        design = np.column_stack(
            [
                np.ones_like(t_fit),
                onset * np.sin(float(omega) * t_fit),
                onset * np.cos(float(omega) * t_fit),
            ]
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        residual = values - design @ coefficients
        return float(residual @ residual), coefficients

    for index, values in enumerate(Y[:, start:]):
        coarse_error = np.asarray(
            [fit_at_frequency(values, omega)[0] for omega in coarse_grid]
        )
        best = int(np.argmin(coarse_error))
        lower = coarse_grid[max(best - 1, 0)]
        upper = coarse_grid[min(best + 1, coarse_grid.size - 1)]
        optimized = minimize_scalar(
            lambda omega: fit_at_frequency(values, omega)[0],
            bounds=(float(lower), float(upper)),
            method="bounded",
            options={"xatol": 1e-10},
        )
        omega = float(optimized.x)
        _, coefficients = fit_at_frequency(values, omega)
        result[index] = [omega, coefficients[1], coefficients[2]]
    return result


def oscillator_harmonic_lift(descriptors, t, onset_time: float = 4.0):
    """Construct a smooth fundamental-response template from the interface."""
    descriptors = np.asarray(descriptors, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    if descriptors.ndim == 1:
        descriptors = descriptors[None, :]
    if descriptors.ndim != 2 or descriptors.shape[1] != 3:
        raise ValueError("descriptors must have shape (N, 3).")
    omega = np.clip(descriptors[:, 0], 0.55, 1.45)
    a_sin = descriptors[:, 1]
    a_cos = descriptors[:, 2]
    onset = 1.0 - np.exp(-t / max(float(onset_time), 1e-6))
    return onset[None, :] * (
        a_sin[:, None] * np.sin(omega[:, None] * t[None, :])
        + a_cos[:, None] * np.cos(omega[:, None] * t[None, :])
    )

def generate(N=500, T_steps=2000, T_end=20, seed=42, dist: str = "uniform", beta_ab=(2.0, 2.0), noise_std: float = 0.0):
    """
    Generate Forced Duffing Oscillator samples.
    m*x'' + c*x' + k*x + alpha*x^3 = F*sin(omega*t)
    X = [m, c, k, alpha, F, omega]
    """
    X = sample_parameters(N=N, seed=seed, dist=dist, beta_ab=beta_ab)
    print(f"Generating {N} Duffing Oscillator samples (T={T_steps})...")
    Y, t_vec = simulate_from_parameters(
        X,
        T_steps=T_steps,
        T_end=T_end,
        noise_std=noise_std,
        progress_every=100,
    )
    return X, Y, t_vec
