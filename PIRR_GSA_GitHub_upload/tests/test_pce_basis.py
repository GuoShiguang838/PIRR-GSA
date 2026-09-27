import numpy as np

from gsa.pce_sobol import _orthonormal_legendre_design, _total_degree_indices


def test_total_degree_basis_size_for_formal_case():
    indices = _total_degree_indices(9, 2)
    assert len(indices) == 55
    assert all(sum(index) <= 2 for index in indices)


def test_legendre_design_is_finite_and_has_constant_column():
    inputs = np.linspace(-1.0, 1.0, 18).reshape(6, 3)
    indices = _total_degree_indices(3, 2)
    design = _orthonormal_legendre_design(inputs, indices, 2)
    assert design.shape == (6, 10)
    assert np.all(np.isfinite(design))
    constant = indices.index((0, 0, 0))
    assert np.allclose(design[:, constant], 1.0)
