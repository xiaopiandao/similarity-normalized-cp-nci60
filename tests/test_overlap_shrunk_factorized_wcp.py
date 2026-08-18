import numpy as np
import pandas as pd

from scripts.evaluate_factorized_shift_wcp import geometric_factorized_weights
from scripts.evaluate_overlap_shrunk_factorized_wcp import (
    CURRENT_METHOD,
    SHRUNK_METHOD,
    choose_kappa,
    weight_diagnostics,
)


def test_kappa_endpoints_recover_similarity_and_selected_factorization():
    sim_cal = np.array([0.8, 1.0, 1.2])
    sim_test = np.array([0.9, 1.1])
    hd_cal = np.array([0.5, 1.0, 1.5])
    hd_test = np.array([0.7, 1.3])
    selected_lambda = 0.75

    cal_zero, test_zero = geometric_factorized_weights(
        sim_cal, sim_test, hd_cal, hd_test, 0.0 * selected_lambda
    )
    expected_cal_zero, expected_test_zero = geometric_factorized_weights(
        sim_cal, sim_test, hd_cal, hd_test, 0.0
    )
    np.testing.assert_allclose(cal_zero, expected_cal_zero)
    np.testing.assert_allclose(test_zero, expected_test_zero)

    cal_one, test_one = geometric_factorized_weights(
        sim_cal, sim_test, hd_cal, hd_test, 1.0 * selected_lambda
    )
    expected_cal_one, expected_test_one = geometric_factorized_weights(
        sim_cal, sim_test, hd_cal, hd_test, selected_lambda
    )
    np.testing.assert_allclose(cal_one, expected_cal_one)
    np.testing.assert_allclose(test_one, expected_test_one)


def test_weight_diagnostics_detects_ess_and_support_failures():
    stable = weight_diagnostics(
        np.ones(10), np.ones(3), minimum_ess_fraction=0.5, alpha=0.1
    )
    assert stable["ess_fraction"] == 1.0
    assert stable["support_failure_fraction"] == 0.0
    assert stable["feasible"]

    unstable = weight_diagnostics(
        np.array([9.0] + [0.01] * 9),
        np.array([100.0, 1.0]),
        minimum_ess_fraction=0.5,
        alpha=0.1,
    )
    assert unstable["ess_fraction"] < 0.5
    assert unstable["support_failure_fraction"] > 0.0
    assert not unstable["feasible"]


def test_choose_kappa_uses_robust_maximum_then_secondary_criteria():
    metric_rows = []
    for family in ("scaffold", "leader_cluster"):
        for scenario in (
            "reference_all",
            "exclude_exact_4_or_8",
            "one_sided_compatible",
        ):
            metric_rows.append(
                {
                    "domain": "internal_ood",
                    "family": family,
                    "seed": 1,
                    "method": CURRENT_METHOD,
                    "kappa": np.nan,
                    "scenario": scenario,
                    "coverage_error_abs": 0.04,
                    "mean_width": 2.0,
                    "finite_interval_fraction": 1.0,
                }
            )
            for kappa, ace, width in (
                (0.0, 0.03, 1.9),
                (0.5, 0.02, 2.0),
                (1.0, 0.025, 1.8),
            ):
                metric_rows.append(
                    {
                        "domain": "internal_ood",
                        "family": family,
                        "seed": 1,
                        "method": SHRUNK_METHOD,
                        "kappa": kappa,
                        "scenario": scenario,
                        "coverage_error_abs": ace,
                        "mean_width": width,
                        "finite_interval_fraction": 1.0,
                    }
                )
    diagnostics = pd.DataFrame(
        [
            {
                "domain": "internal_ood",
                "kappa": kappa,
                "ess_fraction": 0.8,
                "support_failure_fraction": 0.0,
                "feasible": True,
            }
            for kappa in (0.0, 0.5, 1.0)
        ]
    )
    selected, frame = choose_kappa(pd.DataFrame(metric_rows), diagnostics)
    assert selected == 0.5
    assert frame.loc[frame["selected_internal"], "kappa"].item() == 0.5
