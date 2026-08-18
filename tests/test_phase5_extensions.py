import numpy as np
from scipy import sparse

from scripts.evaluate_phase1a_conformal import finite_sample_quantile
from scripts.evaluate_phase5_extensions import (
    conformal_order_index,
    cqr_intervals,
    dad_intervals,
    fit_unconstrained_cubic_scale,
    predict_unconstrained_cubic_scale,
    top_k_tanimoto_indices,
)


def test_conformal_order_index_matches_existing_quantile():
    rng = np.random.default_rng(7)
    for n in (8, 17, 100, 250):
        values = rng.normal(size=n)
        ordered = np.sort(values)
        for alpha in (0.05, 0.10, 0.15, 0.20):
            index = int(conformal_order_index(np.asarray([n]), alpha)[0])
            assert ordered[index] == finite_sample_quantile(values, alpha)


def test_top_k_tanimoto_indices_returns_expected_neighbours():
    calibration = sparse.csr_matrix(
        np.asarray([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]], dtype=np.float32)
    )
    test = sparse.csr_matrix(np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32))
    indices = top_k_tanimoto_indices(calibration, test, k=1)
    assert indices[:, 0].tolist() == [0, 1]


def test_dad_intervals_use_local_residual_quantile():
    truth = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    prediction = np.zeros_like(truth)
    test_prediction = np.asarray([[10.0]])
    neighbours = np.asarray([[1, 2, 3]], dtype=np.int32)
    lower, upper = dad_intervals(truth, prediction, test_prediction, neighbours, alpha=0.5)
    expected = finite_sample_quantile(np.asarray([1.0, 2.0, 3.0]), 0.5)
    assert lower[0, 0] == 10.0 - expected
    assert upper[0, 0] == 10.0 + expected


def test_cqr_intervals_apply_per_cell_conformal_correction():
    levels = np.asarray([0.05, 0.95])
    cal = np.asarray([[[0.0, 2.0]], [[0.0, 2.0]], [[0.0, 2.0]]])
    truth = np.asarray([[1.0], [3.0], [1.0]])
    test = np.asarray([[[5.0, 7.0]]])
    lower, upper = cqr_intervals(truth, cal, test, levels, alpha=0.10)
    correction = finite_sample_quantile(np.asarray([-1.0, 1.0, -1.0]), 0.10)
    assert lower[0, 0] == 5.0 - correction
    assert upper[0, 0] == 7.0 + correction


def test_unconstrained_scale_is_positive_and_median_normalized():
    similarity = np.linspace(0.1, 0.9, 80)
    truth = np.column_stack((2.0 - similarity, 4.0 - 2.0 * similarity))
    prediction = np.zeros_like(truth)
    model, _ = fit_unconstrained_cubic_scale(similarity, truth, prediction)
    fitted = predict_unconstrained_cubic_scale(model, similarity)
    assert np.all(np.isfinite(fitted))
    assert np.all(fitted > 0)
    assert np.isclose(np.median(fitted), 1.0)
