import unittest

import numpy as np

from gsa.latent_alignment import fit_semantic_alignment, semantic_r2_score


class TestSemanticAlignment(unittest.TestCase):
    def test_hungarian_matching_recovers_permutation_and_sign(self):
        rng = np.random.default_rng(20260810)
        descriptors_train = rng.normal(size=(1000, 2))
        latent_train = np.column_stack(
            [
                rng.normal(size=1000),
                -descriptors_train[:, 1] + 0.01 * rng.normal(size=1000),
                2.0 * descriptors_train[:, 0] + 0.01 * rng.normal(size=1000),
                rng.normal(size=1000),
            ]
        )

        alignment = fit_semantic_alignment(latent_train, descriptors_train)

        np.testing.assert_array_equal(alignment.latent_indices, [2, 1])
        np.testing.assert_array_equal(alignment.signs, [1.0, -1.0])
        self.assertTrue(np.all(alignment.training_correlations > 0.99))

    def test_fitted_training_mapping_applies_to_new_samples(self):
        rng = np.random.default_rng(42)
        descriptors_train = rng.normal(size=(500, 2))
        latent_train = np.column_stack(
            [descriptors_train[:, 1], rng.normal(size=500), -descriptors_train[:, 0]]
        )
        alignment = fit_semantic_alignment(latent_train, descriptors_train)

        descriptors_eval = rng.normal(size=(250, 2))
        latent_eval = np.column_stack(
            [descriptors_eval[:, 1], rng.normal(size=250), -descriptors_eval[:, 0]]
        )
        aligned_eval = alignment.transform(latent_eval)

        np.testing.assert_allclose(aligned_eval, descriptors_eval, atol=1e-12)
        self.assertAlmostEqual(semantic_r2_score(aligned_eval, descriptors_eval), 1.0)

    def test_rejects_too_few_candidate_coordinates(self):
        latent = np.ones((10, 1))
        descriptors = np.ones((10, 2))
        with self.assertRaises(ValueError):
            fit_semantic_alignment(latent, descriptors)


if __name__ == "__main__":
    unittest.main()
