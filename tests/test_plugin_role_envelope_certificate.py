import numpy as np

from scripts.evaluate_plugin_role_envelope_certificate import (
    role_envelope_certificate,
)


def test_common_scale_factor_has_unit_certificate():
    reference_cal = np.array([1.0, 2.0, 3.0])
    reference_test = np.array([0.8, 1.5])
    result = role_envelope_certificate(
        reference_cal,
        2.5 * reference_cal,
        reference_test,
        2.5 * reference_test,
    )
    assert np.isclose(result["certificate_lower"], 1.0)
    assert np.isclose(result["certificate_upper"], 1.0)
    assert np.isclose(result["certificate_factor"], 1.0)


def test_role_envelope_matches_hand_calculation():
    result = role_envelope_certificate(
        np.ones(2),
        np.array([0.9, 1.1]),
        np.ones(2),
        np.array([0.95, 1.05]),
    )
    assert np.isclose(result["certificate_lower"], 0.95 / 1.1)
    assert np.isclose(result["certificate_upper"], 1.05 / 0.9)
    assert np.isclose(result["certificate_factor"], 1.05 / 0.9)


def test_certificate_is_dominated_by_global_projective_range():
    lambda_min, lambda_max = 0.8, 1.25
    lower_scale_ratio = 1.0 / lambda_max
    upper_scale_ratio = 1.0 / lambda_min
    result = role_envelope_certificate(
        np.ones(3),
        np.array([lower_scale_ratio, 0.95, upper_scale_ratio]),
        np.ones(3),
        np.array([0.85, 1.0, 1.20]),
    )
    kappa = lambda_max / lambda_min
    assert result["certificate_factor"] <= kappa
