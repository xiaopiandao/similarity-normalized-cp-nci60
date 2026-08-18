from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_phase1b_similarity import (
    fit_similarity_scale,
    predict_similarity_scale,
    similarity_normalized_intervals,
)


class SimilarityConformalTests(unittest.TestCase):
    def test_scale_is_positive_and_decreases_with_similarity(self) -> None:
        similarity = np.linspace(0.2, 0.9, 20)
        truth = np.full((20, 2), 2.0)
        prediction = np.repeat(similarity[:, None], 2, axis=1)
        model, _, _ = fit_similarity_scale(similarity, truth, prediction)
        scale = predict_similarity_scale(model, similarity)
        self.assertTrue(np.all(scale > 0))
        self.assertTrue(np.all(np.diff(scale) <= 1e-12))

    def test_normalized_intervals_expand_for_low_similarity(self) -> None:
        similarity = np.linspace(0.2, 0.9, 20)
        truth = np.full((20, 1), 2.0)
        prediction = similarity[:, None]
        model, _, _ = fit_similarity_scale(similarity, truth, prediction)
        lower, upper, _, test_scale = similarity_normalized_intervals(
            truth,
            prediction,
            similarity,
            np.asarray([0.2, 0.9]),
            model,
            np.asarray([[1.0], [1.0]]),
            0.2,
        )
        widths = (upper - lower).ravel()
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(test_scale[0], test_scale[1])


if __name__ == "__main__":
    unittest.main()
