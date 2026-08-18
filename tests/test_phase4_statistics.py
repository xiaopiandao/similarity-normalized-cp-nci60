import unittest

import numpy as np

from scripts.analyze_phase4_statistics import (
    bh_adjust,
    paired_compound_inference,
    wilson_interval,
)


class Phase4StatisticsTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_fraction(self):
        low, high = wilson_interval(38, 40)
        self.assertLess(low, 0.95)
        self.assertGreater(high, 0.95)

    def test_bh_adjust_is_monotone_in_rank(self):
        p = np.array([0.04, 0.001, 0.02])
        q = bh_adjust(p)
        ranked = q[np.argsort(p)]
        self.assertTrue(np.all(np.diff(ranked) >= -1e-12))

    def test_paired_bootstrap_uses_compound_clusters(self):
        rng = np.random.default_rng(7)
        a = np.array([3, 2, 4, 1], dtype=float)
        b = np.array([1, 2, 2, 1], dtype=float)
        observed = np.array([4, 4, 5, 3], dtype=float)
        result = paired_compound_inference(a, b, observed, rng, 200, 400)
        self.assertAlmostEqual(result["difference"], 4 / 16)
        self.assertEqual(result["n_compounds"], 4)
        self.assertEqual(result["n_labels"], 16)


if __name__ == "__main__":
    unittest.main()
