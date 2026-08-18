import numpy as np

from scripts.analyze_endpoint_boundary_sensitivity import (
    exclude_boundary_truth,
    one_sided_compatible_metrics,
)


def test_exclude_boundary_truth_only_removes_exact_4_and_8():
    truth = np.asarray([[3.9, 4.0, 5.0, 8.0, 8.1, np.nan]])
    filtered = exclude_boundary_truth(truth)
    assert np.isnan(filtered[0, 1])
    assert np.isnan(filtered[0, 3])
    assert filtered[0, 0] == 3.9
    assert filtered[0, 2] == 5.0
    assert filtered[0, 4] == 8.1


def test_one_sided_compatibility_differs_from_exact_endpoint_coverage():
    truth = np.asarray([[4.0, 8.0, 6.0]])
    lower = np.asarray([[3.0, 9.0, 5.0]])
    upper = np.asarray([[3.5, 10.0, 7.0]])
    metrics = one_sided_compatible_metrics(truth, lower, upper, alpha=0.1)
    # y<=4 is compatible with [3,3.5], y>=8 is compatible with [9,10],
    # and the exact interior value is covered.
    assert metrics["coverage"] == 1.0
    assert metrics["n_exact_4"] == 1
    assert metrics["n_exact_8"] == 1
    assert np.isnan(metrics["mean_interval_score"])
