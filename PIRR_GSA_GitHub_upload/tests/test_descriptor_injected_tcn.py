import pytest
import numpy as np

torch = pytest.importorskip("torch")

from model.tcn_ae import DescriptorInjectedResidualTCNAE
from data.generate_battery_controlled import (
    battery_descriptor_matrix,
    battery_descriptor_lift,
    battery_residual_response,
)


def test_descriptor_coordinates_are_exact_and_response_shape_is_preserved():
    torch.manual_seed(3)
    model = DescriptorInjectedResidualTCNAE(
        T=160,
        descriptor_dim=3,
        residual_dim=2,
    )
    response = torch.randn(6, 1, 160)
    descriptors = torch.randn(6, 3)
    latent, reconstructed = model(response, descriptors)
    assert latent.shape == (6, 5)
    assert reconstructed.shape == (6, 160)
    torch.testing.assert_close(latent[:, :3], descriptors)


def test_descriptor_dimension_mismatch_is_rejected():
    model = DescriptorInjectedResidualTCNAE(T=160, descriptor_dim=3, residual_dim=2)
    with pytest.raises(ValueError):
        model(torch.randn(4, 1, 160), torch.randn(4, 2))


def test_linear_descriptor_projection_is_exact_in_reconstruction():
    torch.manual_seed(8)
    operator = torch.zeros(2, 32)
    operator[0, :8] = 1.0 / 8.0
    operator[1, 8:16] = 1.0 / 8.0
    model = DescriptorInjectedResidualTCNAE(
        T=32,
        descriptor_dim=2,
        residual_dim=2,
        descriptor_operator=operator,
        descriptor_mean=torch.tensor([0.2, -0.1]),
        descriptor_scale=torch.tensor([0.5, 0.25]),
    )


def test_battery_operator_residual_has_zero_descriptors():
    rng = np.random.default_rng(21)
    response = rng.normal(size=(7, 160))
    residual = battery_residual_response(response)
    np.testing.assert_allclose(
        residual @ battery_descriptor_matrix(160).T,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        battery_descriptor_matrix(160) @ battery_descriptor_lift(160),
        np.eye(3),
        atol=1e-10,
    )
    response = torch.randn(5, 1, 32)
    descriptor_z = torch.randn(5, 2)
    latent, reconstructed = model(response, descriptor_z)
    expected = descriptor_z * model.descriptor_scale + model.descriptor_mean
    torch.testing.assert_close(reconstructed @ operator.T, expected, atol=2e-5, rtol=0.0)
    residual = model.project_residual(reconstructed)
    torch.testing.assert_close(
        residual @ operator.T,
        torch.zeros(5, 2),
        atol=2e-5,
        rtol=0.0,
    )
