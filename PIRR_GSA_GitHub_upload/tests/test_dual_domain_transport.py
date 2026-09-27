import numpy as np

from data.generate_dual_domain_transport import (
    PARAM_NAMES,
    dual_domain_moment_descriptors,
    dual_domain_moment_template,
    eval_dual_domain_transport_from_x,
    generate_dual_domain_transport_data,
)


def test_dual_domain_generator_is_deterministic_and_finite():
    X = np.random.default_rng(4).random((12, len(PARAM_NAMES)))
    first = eval_dual_domain_transport_from_x(X, T=96, chunk_size=5)
    second = eval_dual_domain_transport_from_x(X, T=96, chunk_size=20)
    assert first.shape == (12, 96)
    assert np.all(np.isfinite(first))
    assert np.all(first >= 0.0)
    np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-13)


def test_dual_domain_moment_lift_contract():
    _, responses, descriptors, meta = generate_dual_domain_transport_data(
        N=16, T=160, seed=8, noise_scale=0.0
    )
    assert responses.shape == (16, 160)
    lifted = dual_domain_moment_template(descriptors, meta["t"])
    recovered = dual_domain_moment_descriptors(lifted, meta["t"])
    np.testing.assert_allclose(recovered, descriptors, rtol=0.0, atol=2e-6)


def test_dual_domain_response_contains_material_interface_residual():
    _, responses, descriptors, meta = generate_dual_domain_transport_data(
        N=64, T=192, seed=11, noise_scale=0.0
    )
    lifted = dual_domain_moment_template(descriptors, meta["t"])
    normalized_residual = np.sqrt(np.mean((responses - lifted) ** 2)) / np.std(responses)
    assert 0.15 < normalized_residual < 1.5
