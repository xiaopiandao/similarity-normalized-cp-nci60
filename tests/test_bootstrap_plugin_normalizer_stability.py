import numpy as np
import pandas as pd

from scripts.bootstrap_plugin_normalizer_stability import (
    FAMILIES,
    bootstrap_normalizers,
    evaluate_gates,
    quantile_summary,
)


def test_bootstrap_normalizers_are_deterministic_for_fixed_rng():
    residual = np.arange(1, 121, dtype=float).reshape(20, 6)
    left, left_fallback = bootstrap_normalizers(
        residual, np.random.default_rng(20260816)
    )
    right, right_fallback = bootstrap_normalizers(
        residual, np.random.default_rng(20260816)
    )
    assert np.array_equal(left, right)
    assert left_fallback == right_fallback == 0


def test_quantile_summary_uses_prespecified_quantiles():
    result = quantile_summary(pd.Series(np.arange(1.0, 101.0)))
    assert np.isclose(result["median"], 50.5)
    assert np.isclose(result["q95"], 95.05)
    assert np.isclose(result["maximum"], 100.0)


def test_gates_require_all_three_confirmation_families():
    trials = pd.DataFrame(
        {
            "radius_violation": [0.0],
            "bootstrap_fallback_count": [0],
            "min_unfloored_perturbed_scale": [0.5],
        }
    )
    units = pd.DataFrame(
        {
            "reference_fallback_count": [0],
            "min_unfloored_reference_scale": [0.5],
        }
    )
    summary = pd.DataFrame(
        {
            "phase": ["confirmation"] * len(FAMILIES),
            "family": list(FAMILIES),
            "kappa_m_q95": [1.1] * len(FAMILIES),
            "radius_factor_q95": [1.05] * len(FAMILIES),
        }
    )
    result = evaluate_gates(trials, units, summary)
    assert result["all_prespecified_gates_passed"] is True

    incomplete = evaluate_gates(trials, units, summary.iloc[:-1])
    assert incomplete["B4_projective_concentration"]["passed"] is False
    assert incomplete["B5_radius_concentration"]["passed"] is False
