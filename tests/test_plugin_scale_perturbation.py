from __future__ import annotations

import numpy as np

from scripts.validate_plugin_scale_perturbation import (
    compound_difficulty,
    evaluate_perturbation,
    task_normalizers,
)


def _payload(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_validation, n_calibration, n_test, tasks = 64, 48, 31, 8
    similarity_validation = rng.uniform(0.0, 1.0, n_validation)
    similarity_calibration = rng.uniform(0.0, 1.0, n_calibration)
    similarity_test = rng.uniform(0.0, 1.0, n_test)
    task_scale = np.exp(rng.normal(0.0, 0.3, tasks))

    def residual(similarity: np.ndarray) -> np.ndarray:
        shared = 0.6 + 1.2 * (1.0 - similarity)
        return (
            shared[:, None]
            * task_scale[None, :]
            * np.exp(rng.normal(0.0, 0.25, (len(similarity), tasks)))
        )

    validation_residual = residual(similarity_validation)
    calibration_residual = residual(similarity_calibration)
    validation_residual[1, 0] = np.nan
    validation_residual[3, 2] = np.nan
    calibration_residual[2, 1] = np.nan
    return {
        "similarity_validation": similarity_validation,
        "validation_residual": validation_residual,
        "similarity_calibration": similarity_calibration,
        "calibration_residual": calibration_residual,
        "similarity_test": similarity_test,
    }


def test_compound_median_multiplicative_sandwich_with_missing_and_even_tasks() -> None:
    residual = np.asarray(
        [
            [1.0, 2.0, 4.0, 8.0],
            [2.0, np.nan, 3.0, 9.0],
            [5.0, 1.0, np.nan, 2.0],
        ]
    )
    reference = np.asarray([1.0, 2.0, 2.0, 4.0])
    lambdas = np.asarray([0.8, 1.2, 0.9, 1.1])
    baseline = compound_difficulty(residual, reference)
    perturbed = compound_difficulty(residual, reference * lambdas)
    assert np.all(perturbed >= baseline / lambdas.max() - 1e-14)
    assert np.all(perturbed <= baseline / lambdas.min() + 1e-14)


def test_projective_bounds_hold_through_isotonic_fit_and_conformal_radius() -> None:
    payload = _payload()
    reference, fallback = task_normalizers(payload["validation_residual"])
    assert fallback == 0
    lambdas = np.asarray([0.82, 1.18, 0.91, 1.06, 0.87, 1.13, 0.96, 1.02])
    result = evaluate_perturbation(
        **payload,
        reference_normalizers=reference,
        lambdas=lambdas,
    )
    assert result["compound_violation"] <= 1e-12
    assert result["raw_scale_violation"] <= 1e-12
    assert result["normalized_scale_violation"] <= 1e-12
    assert result["radius_violation"] <= 1e-12
    assert result["max_abs_log_radius_ratio"] <= result["log_kappa"] + 1e-12


def test_common_task_scale_factor_cancels_exactly() -> None:
    payload = _payload(seed=11)
    reference, fallback = task_normalizers(payload["validation_residual"])
    assert fallback == 0
    result = evaluate_perturbation(
        **payload,
        reference_normalizers=reference,
        lambdas=np.full(len(reference), 1.37),
    )
    assert abs(result["kappa"] - 1.0) <= 1e-15
    assert result["normalized_scale_max_relative_difference"] <= 1e-12
    assert result["radius_max_relative_difference"] <= 1e-12

