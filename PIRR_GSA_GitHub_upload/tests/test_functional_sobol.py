import numpy as np

from gsa.functional_sobol import (
    fit_multioutput_pce,
    functional_sobol_metrics,
    timewise_pick_freeze_reference,
)


def _additive_temporal_response(X):
    X = np.asarray(X, dtype=float)
    t = np.linspace(0.2, 1.0, 7)
    return X[:, 0, None] + t[None, :] * X[:, 1, None]


def test_timewise_pick_freeze_recovers_additive_temporal_indices():
    s1, st, variances = timewise_pick_freeze_reference(
        _additive_temporal_response,
        n_inputs=2,
        n_base=60_000,
        seed=81,
    )
    t = np.linspace(0.2, 1.0, 7)
    expected_x0 = 1.0 / (1.0 + t**2)
    expected_x1 = t**2 / (1.0 + t**2)
    expected = np.column_stack([expected_x0, expected_x1])
    np.testing.assert_allclose(s1, expected, atol=0.025)
    np.testing.assert_allclose(st, expected, atol=0.025)
    assert np.all(variances > 0.0)


def test_multioutput_pce_predicts_quadratic_temporal_response():
    rng = np.random.default_rng(4)
    X = rng.random((200, 2))
    targets = np.column_stack(
        [X[:, 0] + 0.5 * X[:, 1] ** 2, X[:, 0] * X[:, 1]]
    )
    surrogate = fit_multioutput_pce(
        X,
        targets,
        order=2,
        bounds=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
    )
    X_test = rng.random((50, 2))
    expected = np.column_stack(
        [X_test[:, 0] + 0.5 * X_test[:, 1] ** 2, X_test[:, 0] * X_test[:, 1]]
    )
    np.testing.assert_allclose(surrogate.predict(X_test), expected, atol=2e-5)


def test_functional_metrics_are_zero_for_identical_matrices():
    rng = np.random.default_rng(9)
    s1 = rng.random((10, 3))
    st = rng.random((10, 3))
    metrics = functional_sobol_metrics(
        s1,
        st,
        s1,
        st,
        temporal_variances=np.linspace(1.0, 2.0, 10),
    )
    assert metrics["RMSE_func"] == 0.0
    assert metrics["RMSE_func_variance_weighted"] == 0.0
    assert metrics["RMSE_func_active_1e-03"] == 0.0
    assert metrics["RMSE_functional_index"] == 0.0


def test_functional_metrics_report_variance_weighted_and_active_time_audits():
    reference_s1 = np.zeros((4, 1))
    reference_st = np.zeros((4, 1))
    estimated_s1 = np.asarray([[1.0], [0.0], [0.0], [0.0]])
    estimated_st = estimated_s1.copy()
    metrics = functional_sobol_metrics(
        estimated_s1,
        estimated_st,
        reference_s1,
        reference_st,
        temporal_variances=np.asarray([1e-8, 1.0, 1.0, 1.0]),
    )
    assert metrics["RMSE_func"] == 0.5
    assert metrics["RMSE_func_variance_weighted"] < 1e-3
    assert metrics["RMSE_func_active_1e-03"] == 0.0
    assert metrics["active_time_count_1e-03"] == 3
