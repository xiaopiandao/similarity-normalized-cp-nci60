import numpy as np

from scripts.evaluate_hts384_comparators import (
    DAD_K,
    METHODS,
    compound_bootstrap,
    interval_score_metrics,
)


def test_locked_dad_k_and_method_set() -> None:
    assert DAD_K == 50
    assert "UACQR_P" in METHODS
    assert "Estimated_WCP" in METHODS
    assert "Proposed_shared_monotone" in METHODS


def test_interval_score_is_zero_for_degenerate_correct_interval() -> None:
    truth = np.asarray([[1.0, 2.0]])
    result = interval_score_metrics(
        truth, truth.copy(), truth.copy(), 0.1, np.ones_like(truth, dtype=bool)
    )
    assert result["mean_interval_score"] == 0.0


def test_compound_bootstrap_preserves_equal_methods() -> None:
    truth = np.asarray([[1.0], [2.0], [3.0]])
    interval = (truth - 1.0, truth + 1.0)
    run = {method: interval for method in METHODS}
    result = compound_bootstrap(truth, [run, run], replicates=50, seed=1)
    assert np.allclose(result["coverage_difference_mean"], 0.0)
    assert np.allclose(result["ace_difference_mean"], 0.0)
