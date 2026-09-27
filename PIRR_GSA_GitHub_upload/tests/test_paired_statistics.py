import unittest

from gsa.paired_statistics import paired_strongest_baseline_comparison


class TestPairedStatistics(unittest.TestCase):
    def test_positive_difference_means_proposed_is_better(self):
        rows = []
        for seed in range(8):
            rows.extend(
                [
                    {"Method": "Phy-TCN-AE-PCE", "Seed": seed, "RMSE_P": 0.10 + seed * 0.001},
                    {"Method": "Baseline-A", "Seed": seed, "RMSE_P": 0.16 + seed * 0.001},
                    {"Method": "Baseline-B", "Seed": seed, "RMSE_P": 0.20 + seed * 0.001},
                ]
            )
        result = paired_strongest_baseline_comparison(rows, metric="RMSE_P")
        self.assertEqual(result["strongest_baseline"], "Baseline-A")
        self.assertGreater(result["paired_difference_mean"], 0.0)
        self.assertGreater(result["paired_difference_95ci"][0], 0.0)
        self.assertEqual(result["n_pairs"], 8)


if __name__ == "__main__":
    unittest.main()
