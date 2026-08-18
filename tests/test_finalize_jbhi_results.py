import numpy as np

from scripts.finalize_jbhi_results import (
    calibration_error_bootstrap,
    screening_rows,
)


def test_screening_rows_uses_compound_level_cell_count_rule():
    truth = np.asarray([[7.0, 7.0, 7.0], [7.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    prediction = truth.copy()
    intervals = {"Global_CP": (truth - 0.5, truth + 0.5)}
    rows = screening_rows("leader_cluster", 2, truth, prediction, intervals)
    row = next(
        item
        for item in rows
        if item["method"] == "Global_CP" and item["strong_cell_threshold"] == 3
    )
    assert row["selected"] == 1
    assert row["true_positive"] == 1
    assert row["false_discovery_rate"] == 0.0


def test_calibration_error_bootstrap_sign_is_proposed_minus_comparator():
    truth = np.zeros((20, 1))
    proposed = (np.full((20, 1), -1.0), np.full((20, 1), 1.0))
    comparator = (np.full((20, 1), 0.5), np.full((20, 1), 1.0))
    result = calibration_error_bootstrap(
        truth,
        proposed,
        comparator,
        alpha=0.1,
        mask=np.ones(20, dtype=bool),
        rng=np.random.default_rng(3),
        reps=100,
    )
    assert result["ace_difference"] < 0
    assert result["bootstrap_probability_proposed_lower_ace"] == 1.0
