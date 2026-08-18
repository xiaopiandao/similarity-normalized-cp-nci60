import numpy as np
from scipy import sparse

from scripts.evaluate_factorized_shift_wcp import (
    choose_factorized_weight,
    fingerprint_balance,
    geometric_factorized_weights,
)


def test_geometric_factorization_matches_endpoints_after_normalization() -> None:
    sim_cal = np.asarray([0.5, 1.0, 1.5])
    sim_test = np.asarray([0.8, 1.2])
    hd_cal = np.asarray([0.25, 1.0, 2.0])
    hd_test = np.asarray([0.6, 1.8])
    cal_zero, test_zero = geometric_factorized_weights(
        sim_cal, sim_test, hd_cal, hd_test, 0.0
    )
    cal_one, test_one = geometric_factorized_weights(
        sim_cal, sim_test, hd_cal, hd_test, 1.0
    )
    assert np.allclose(cal_zero, sim_cal / sim_cal.mean())
    assert np.allclose(test_zero, sim_test / sim_cal.mean())
    assert np.allclose(cal_one, hd_cal / hd_cal.mean())
    assert np.allclose(test_one, hd_test / hd_cal.mean())


def test_geometric_factorization_midpoint_is_geometric_mean() -> None:
    sim_cal = np.asarray([1.0, 4.0])
    sim_test = np.asarray([9.0])
    hd_cal = np.asarray([4.0, 1.0])
    hd_test = np.asarray([1.0])
    cal, test = geometric_factorized_weights(
        sim_cal, sim_test, hd_cal, hd_test, 0.5
    )
    expected_cal = np.sqrt(sim_cal * hd_cal)
    expected_test = np.sqrt(sim_test * hd_test)
    assert np.allclose(cal, expected_cal / expected_cal.mean())
    assert np.allclose(test, expected_test / expected_cal.mean())


def test_fingerprint_balance_is_zero_for_identical_weighted_means() -> None:
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            dtype=float,
        )
    )
    result = fingerprint_balance(matrix, matrix, np.ones(matrix.shape[0]))
    assert result["fingerprint_rms_smd"] == 0.0
    assert result["fingerprint_max_abs_smd"] == 0.0


def test_invalid_absolute_ess_floor_is_rejected() -> None:
    matrix = sparse.csr_matrix(np.eye(4, dtype=float))
    similarity = np.linspace(0.1, 0.4, 4)
    weights = np.ones(4)
    try:
        choose_factorized_weight(
            matrix,
            matrix,
            similarity,
            similarity,
            weights,
            weights,
            weights,
            weights,
            absolute_minimum_ess_fraction=0.0,
        )
    except ValueError as exc:
        assert "absolute_minimum_ess_fraction" in str(exc)
    else:
        raise AssertionError("invalid ESS floor should raise ValueError")
