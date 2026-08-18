from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_phase1a_conformal import finite_sample_quantile, weighted_thresholds


class ConformalMathTests(unittest.TestCase):
    def test_finite_sample_quantile_uses_higher_order_statistic(self) -> None:
        scores = np.arange(1, 10, dtype=float)
        self.assertEqual(finite_sample_quantile(scores, alpha=0.1), 9.0)

    def test_weighted_threshold_matches_uniform_case_without_large_test_mass(self) -> None:
        scores = np.arange(1, 11, dtype=float)
        thresholds = weighted_thresholds(scores, np.ones(10), np.ones(2), alpha=0.2)
        np.testing.assert_allclose(thresholds, [9.0, 9.0])

    def test_weighted_threshold_can_signal_support_failure(self) -> None:
        scores = np.arange(1, 11, dtype=float)
        thresholds = weighted_thresholds(scores, np.ones(10), np.asarray([100.0]), alpha=0.1)
        self.assertTrue(np.isinf(thresholds[0]))


if __name__ == "__main__":
    unittest.main()
