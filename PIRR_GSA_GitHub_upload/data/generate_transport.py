import numpy as np


def generate_transport_data(N=1000, T=160, seed=42, dist="uniform", beta_ab=(2.0, 2.0), noise_scale=0.01):
    """Generate advection-dispersion-reaction breakthrough curves.

    Parameters are returned normalized in [0, 1]. Physical parameters are:
    v: velocity, D: dispersion, k: decay, R: retardation, M: source mass, tau: injection duration.
    """
    rng = np.random.default_rng(seed)
    if str(dist).lower() == "beta":
        X = rng.beta(float(beta_ab[0]), float(beta_ab[1]), size=(N, 6))
    else:
        X = rng.random((N, 6))

    v = 0.5 + 1.5 * X[:, 0]
    D = 0.02 + 0.23 * X[:, 1]
    k = 0.00 + 0.08 * X[:, 2]
    R = 1.0 + 2.0 * X[:, 3]
    M = 0.5 + 1.5 * X[:, 4]
    tau = 0.02 + 0.13 * X[:, 5]

    t = np.linspace(0.02, 4.0, T)
    L = 1.0
    Y = np.zeros((N, T), dtype=float)

    for i in range(N):
        arrival = L * R[i] / (v[i] + 1e-12) + 0.35 * tau[i]
        width = 0.035 + 0.30 * np.sqrt(D[i]) * np.sqrt(R[i]) + 0.45 * tau[i]
        amp = M[i] / (np.sqrt(2.0 * np.pi) * width + 1e-12)
        amp *= np.exp(-k[i] * arrival) / (1.0 + 0.15 * (R[i] - 1.0))
        core = amp * np.exp(-0.5 * ((t - arrival) / width) ** 2)
        tail_scale = 0.12 + 0.55 * D[i] * R[i] + 1.20 * tau[i]
        tail = 0.28 * amp * np.exp(-(t - arrival) / (tail_scale + 1e-12))
        tail[t < arrival] = 0.0
        curve = (core + tail) * np.exp(-0.25 * k[i] * t)
        if noise_scale > 0:
            curve = curve + rng.normal(0.0, noise_scale * (np.std(curve) + 1e-8), size=T)
        Y[i] = np.maximum(curve, 0.0)

    descriptors = transport_descriptors(Y, t)
    return X, Y, descriptors, {"t": t, "param_names": ["v", "D", "k", "R", "M", "tau"]}


def transport_descriptors(Y, t=None):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    if t is None:
        t = np.linspace(0.0, 1.0, Y.shape[1])
    t = np.asarray(t, dtype=float)
    peak = np.max(Y, axis=1)
    peak_idx = np.argmax(Y, axis=1)
    t_peak = t[peak_idx]
    mass = np.trapz(Y, t, axis=1)
    return np.vstack([peak, t_peak, mass]).T


def transport_descriptor_template(descriptors, t=None, iterations=60, family="gaussian"):
    """Build a fixed template from Peak--Arrival--Mass descriptors.

    The center is the observed peak-grid time, the amplitude equals ``Peak``,
    and a deterministic bisection selects the width so that the trapezoidal
    integral equals ``Mass``.  The construction is used only to remove the
    known descriptor component before residual encoding; it is not fitted to
    method-comparison results. ``family`` can be ``gaussian`` or ``laplace``;
    comparing them audits whether the result depends on one lift shape.
    """
    descriptors = np.asarray(descriptors, dtype=float)
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(1, -1)
    if descriptors.ndim != 2 or descriptors.shape[1] != 3:
        raise ValueError("descriptors must have shape (N, 3).")
    if t is None:
        t = np.linspace(0.0, 1.0, 160)
    t = np.asarray(t, dtype=float).reshape(-1)
    family = str(family).strip().lower()
    if family not in {"gaussian", "laplace"}:
        raise ValueError("family must be 'gaussian' or 'laplace'.")
    peak = np.maximum(descriptors[:, 0], 1e-14)
    arrival = np.clip(descriptors[:, 1], float(t[0]), float(t[-1]))
    target_ratio = np.maximum(descriptors[:, 2] / peak, 0.0)
    grid_span = max(float(t[-1] - t[0]), 1e-8)
    low = np.full(descriptors.shape[0], grid_span * 1e-8)
    high = np.full(descriptors.shape[0], grid_span * 10.0)
    for _ in range(int(iterations)):
        width = 0.5 * (low + high)
        distance = np.abs(t[None, :] - arrival[:, None]) / width[:, None]
        kernel = (
            np.exp(-0.5 * distance**2)
            if family == "gaussian"
            else np.exp(-distance)
        )
        ratio = np.trapz(kernel, t, axis=1)
        increase = ratio < target_ratio
        low[increase] = width[increase]
        high[~increase] = width[~increase]
    width = 0.5 * (low + high)
    distance = np.abs(t[None, :] - arrival[:, None]) / width[:, None]
    kernel = (
        np.exp(-0.5 * distance**2)
        if family == "gaussian"
        else np.exp(-distance)
    )
    return peak[:, None] * kernel


def transport_residual_response(Y, t=None, family="gaussian"):
    """Return descriptor-conditioned residual dynamics ``Y-L(P(Y))``."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    if t is None:
        t = np.linspace(0.0, 1.0, Y.shape[1])
    descriptors = transport_descriptors(Y, t)
    return Y - transport_descriptor_template(descriptors, t, family=family)


if __name__ == "__main__":
    X, Y, P, meta = generate_transport_data(N=10, T=160, seed=0)
    print(X.shape, Y.shape, P.shape, meta["t"].shape)
