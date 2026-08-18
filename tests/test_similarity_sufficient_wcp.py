import numpy as np

from scripts.evaluate_similarity_sufficient_wcp import (
    estimate_similarity_domain_weights,
    weighted_ks_distance,
    weighted_scaled_intervals,
)


def test_weighted_ks_is_zero_for_identical_samples() -> None:
    values = np.asarray([0.1, 0.2, 0.4, 0.7])
    assert weighted_ks_distance(values, values) == 0.0


def test_similarity_weights_are_positive_and_aligned() -> None:
    rng = np.random.default_rng(42)
    source = rng.beta(4, 3, size=120)
    target = rng.beta(3, 4, size=100)
    source_weights, target_weights, report = estimate_similarity_domain_weights(
        source, target, seed=11
    )
    assert source_weights.shape == source.shape
    assert target_weights.shape == target.shape
    assert np.isfinite(source_weights).all()
    assert np.isfinite(target_weights).all()
    assert (source_weights > 0).all()
    assert (target_weights > 0).all()
    assert 0 < report["ess_fraction"] <= 1


def test_weighted_scaled_intervals_have_expected_shape() -> None:
    truth_cal = np.arange(24, dtype=float).reshape(8, 3)
    pred_cal = truth_cal + 0.5
    pred_test = np.zeros((2, 3), dtype=float)
    lower, upper = weighted_scaled_intervals(
        truth_cal,
        pred_cal,
        pred_test,
        np.ones(8),
        np.asarray([1.0, 1.2]),
        np.ones(8),
        np.ones(2),
        alpha=0.2,
    )
    assert lower.shape == pred_test.shape
    assert upper.shape == pred_test.shape
    assert np.all(lower <= upper)

