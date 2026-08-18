import numpy as np

from scripts.run_uacqrp_exploratory import (
    uacqrp_calibration_scores,
    uacqrp_test_intervals,
    uacqrp_thresholds,
)


def test_uacqrp_scores_find_minimal_joint_expansion_rank():
    truth = np.asarray([[0.0], [5.0], [np.nan]])
    lower = np.asarray(
        [
            [[-1.0], [3.0], [0.0]],
            [[0.5], [4.0], [1.0]],
            [[1.0], [6.0], [2.0]],
        ]
    )
    upper = np.asarray(
        [
            [[0.2], [5.5], [2.0]],
            [[0.8], [4.5], [3.0]],
            [[1.2], [4.0], [4.0]],
        ]
    )
    scores = uacqrp_calibration_scores(truth, lower, upper)
    assert scores[:, 0].tolist() == [2, 2, -1]


def test_uacqrp_thresholds_apply_finite_sample_order_statistic():
    scores = np.asarray([[0], [1], [2], [3], [4], [-1]], dtype=np.int16)
    # Reference UACQR code uses zero-based ceil((n+1)*(1-alpha))-1.
    threshold = uacqrp_thresholds(scores, alpha=0.5)
    assert threshold.tolist() == [2]


def test_uacqrp_intervals_use_percentile_rank_and_virtual_infinity():
    lower = np.asarray(
        [
            [[0.0, 10.0]],
            [[1.0, 11.0]],
            [[2.0, 12.0]],
        ]
    )
    upper = np.asarray(
        [
            [[4.0, 14.0]],
            [[5.0, 15.0]],
            [[6.0, 16.0]],
        ]
    )
    lo, hi = uacqrp_test_intervals(lower, upper, np.asarray([1, 3]))
    assert lo[0, 0] == 1.0
    assert hi[0, 0] == 5.0
    assert np.isneginf(lo[0, 1])
    assert np.isposinf(hi[0, 1])
