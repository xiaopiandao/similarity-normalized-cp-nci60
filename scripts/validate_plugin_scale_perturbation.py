"""Validate the frozen projective-stability theorem for plugin task scales.

The protocol was frozen before results in
``docs/Plugin尺度扰动定理_先验分析与验证协议_20260816.md``.
This script checks deterministic inequalities on synthetic data and on frozen
NCI-60 validation/source-calibration predictions.  Test outcomes are not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

try:
    from scripts.evaluate_phase1a_conformal import EvaluationData, finite_sample_quantile
    from scripts.evaluate_phase3_local_and_screening import sim
except ModuleNotFoundError:
    from evaluate_phase1a_conformal import (  # type: ignore[no-redef]
        EvaluationData,
        finite_sample_quantile,
    )
    from evaluate_phase3_local_and_screening import sim  # type: ignore[no-redef]


FAMILIES = ("scaffold", "leader_cluster")
SEEDS = (2, 3, 4, 5)
DELTAS = (0.01, 0.02, 0.05, 0.10, 0.20)
ALPHA = 0.10
TOLERANCE = 1e-10
FLOOR = 1e-3
DEFAULT_OUTPUT = "results/plugin_scale_perturbation_20260816"


@dataclass
class RawScale:
    model: IsotonicRegression
    valid_mask: np.ndarray
    fitted_validation: np.ndarray
    normalization: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_normalizers(residual: np.ndarray) -> tuple[np.ndarray, int]:
    """Return positive task medians and the number that would need fallback."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        values = np.nanmedian(np.asarray(residual, dtype=float), axis=0)
    invalid = ~(np.isfinite(values) & (values > 0))
    if invalid.all():
        raise ValueError("No finite positive task median")
    fallback = float(np.median(values[~invalid]))
    return np.where(invalid, fallback, values), int(invalid.sum())


def compound_difficulty(residual: np.ndarray, normalizers: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        return np.nanmedian(
            np.asarray(residual, dtype=float) / np.asarray(normalizers, dtype=float)[None, :],
            axis=1,
        )


def fit_raw_scale(
    similarity: np.ndarray, residual: np.ndarray, normalizers: np.ndarray
) -> tuple[RawScale, np.ndarray]:
    difficulty = compound_difficulty(residual, normalizers)
    valid = np.isfinite(similarity) & np.isfinite(difficulty)
    if valid.sum() < 20:
        raise ValueError("Insufficient finite compounds for isotonic scale")
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(np.asarray(similarity, dtype=float)[valid], difficulty[valid])
    fitted = model.predict(np.asarray(similarity, dtype=float)[valid])
    normalization = float(np.median(fitted))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Invalid raw-scale median normalization")
    return RawScale(model, valid, fitted, normalization), difficulty


def predict_raw(scale: RawScale, similarity: np.ndarray) -> np.ndarray:
    return np.asarray(scale.model.predict(np.asarray(similarity, dtype=float)), dtype=float)


def predict_normalized(scale: RawScale, similarity: np.ndarray) -> np.ndarray:
    return predict_raw(scale, similarity) / scale.normalization


def conformal_radii(
    calibration_residual: np.ndarray,
    calibration_scale: np.ndarray,
    test_scale: np.ndarray,
    alpha: float = ALPHA,
) -> np.ndarray:
    scores = np.asarray(calibration_residual, dtype=float) / np.asarray(
        calibration_scale, dtype=float
    )[:, None]
    quantiles = np.asarray(
        [finite_sample_quantile(scores[:, task], alpha) for task in range(scores.shape[1])],
        dtype=float,
    )
    return np.asarray(test_scale, dtype=float)[:, None] * quantiles[None, :]


def max_sandwich_violation(
    observed: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    observed = np.asarray(observed, dtype=float)
    lower = np.broadcast_to(np.asarray(lower, dtype=float), observed.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), observed.shape)
    finite = np.isfinite(observed) & np.isfinite(lower) & np.isfinite(upper)
    if not finite.any():
        return np.nan
    below = lower[finite] - observed[finite]
    above = observed[finite] - upper[finite]
    return float(max(0.0, np.max(below), np.max(above)))


def max_relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right) & (np.abs(right) > 0)
    if not finite.any():
        return np.nan
    return float(np.max(np.abs(left[finite] / right[finite] - 1.0)))


def max_abs_log_ratio(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
    if not finite.any():
        return np.nan
    return float(np.max(np.abs(np.log(left[finite] / right[finite]))))


def evaluate_perturbation(
    similarity_validation: np.ndarray,
    validation_residual: np.ndarray,
    similarity_calibration: np.ndarray,
    calibration_residual: np.ndarray,
    similarity_test: np.ndarray,
    reference_normalizers: np.ndarray,
    lambdas: np.ndarray,
) -> dict[str, float]:
    lambdas = np.asarray(lambdas, dtype=float)
    if np.any(~np.isfinite(lambdas)) or np.any(lambdas <= 0):
        raise ValueError("All perturbation factors must be finite and positive")
    perturbed_normalizers = reference_normalizers * lambdas
    reference, difficulty_reference = fit_raw_scale(
        similarity_validation, validation_residual, reference_normalizers
    )
    perturbed, difficulty_perturbed = fit_raw_scale(
        similarity_validation, validation_residual, perturbed_normalizers
    )
    lambda_min = float(lambdas.min())
    lambda_max = float(lambdas.max())
    kappa = lambda_max / lambda_min

    compound_violation = max_sandwich_violation(
        difficulty_perturbed,
        difficulty_reference / lambda_max,
        difficulty_reference / lambda_min,
    )

    roles = {
        "validation": np.asarray(similarity_validation, dtype=float),
        "calibration": np.asarray(similarity_calibration, dtype=float),
        "test": np.asarray(similarity_test, dtype=float),
    }
    raw_violation = 0.0
    normalized_violation = 0.0
    max_log_raw = 0.0
    max_log_normalized = 0.0
    min_unfloored_reference = np.inf
    min_unfloored_perturbed = np.inf
    raw_reference: dict[str, np.ndarray] = {}
    raw_perturbed: dict[str, np.ndarray] = {}
    normalized_reference: dict[str, np.ndarray] = {}
    normalized_perturbed: dict[str, np.ndarray] = {}
    for role, similarities in roles.items():
        raw_reference[role] = predict_raw(reference, similarities)
        raw_perturbed[role] = predict_raw(perturbed, similarities)
        normalized_reference[role] = raw_reference[role] / reference.normalization
        normalized_perturbed[role] = raw_perturbed[role] / perturbed.normalization
        raw_violation = max(
            raw_violation,
            max_sandwich_violation(
                raw_perturbed[role],
                raw_reference[role] / lambda_max,
                raw_reference[role] / lambda_min,
            ),
        )
        normalized_violation = max(
            normalized_violation,
            max_sandwich_violation(
                normalized_perturbed[role],
                normalized_reference[role] / kappa,
                normalized_reference[role] * kappa,
            ),
        )
        max_log_raw = max(
            max_log_raw,
            max_abs_log_ratio(raw_perturbed[role], raw_reference[role]),
        )
        max_log_normalized = max(
            max_log_normalized,
            max_abs_log_ratio(normalized_perturbed[role], normalized_reference[role]),
        )
        min_unfloored_reference = min(
            min_unfloored_reference, float(np.min(normalized_reference[role]))
        )
        min_unfloored_perturbed = min(
            min_unfloored_perturbed, float(np.min(normalized_perturbed[role]))
        )

    radius_reference = conformal_radii(
        calibration_residual,
        raw_reference["calibration"],
        raw_reference["test"],
    )
    radius_perturbed = conformal_radii(
        calibration_residual,
        raw_perturbed["calibration"],
        raw_perturbed["test"],
    )
    radius_violation = max_sandwich_violation(
        radius_perturbed, radius_reference / kappa, radius_reference * kappa
    )
    return {
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "kappa": kappa,
        "log_kappa": float(np.log(kappa)),
        "compound_violation": compound_violation,
        "raw_scale_violation": raw_violation,
        "normalized_scale_violation": normalized_violation,
        "radius_violation": radius_violation,
        "max_abs_log_raw_scale_ratio": max_log_raw,
        "max_abs_log_normalized_scale_ratio": max_log_normalized,
        "max_abs_log_radius_ratio": max_abs_log_ratio(
            radius_perturbed, radius_reference
        ),
        "normalized_scale_max_relative_difference": max_relative_difference(
            normalized_perturbed["test"], normalized_reference["test"]
        ),
        "radius_max_relative_difference": max_relative_difference(
            radius_perturbed, radius_reference
        ),
        "min_unfloored_reference_scale": min_unfloored_reference,
        "min_unfloored_perturbed_scale": min_unfloored_perturbed,
    }


def synthetic_payload(rng: np.random.Generator) -> dict[str, np.ndarray]:
    n_validation, n_calibration, n_test, tasks = 160, 120, 90, 60
    task_scale = np.exp(rng.normal(0.0, 0.35, size=tasks))

    def shared(similarity: np.ndarray) -> np.ndarray:
        return 0.55 + 1.45 * (1.0 - similarity) ** 1.25

    similarity_validation = rng.uniform(0.05, 0.98, size=n_validation)
    similarity_calibration = rng.uniform(0.05, 0.98, size=n_calibration)
    similarity_test = rng.uniform(0.0, 1.0, size=n_test)
    validation_residual = (
        shared(similarity_validation)[:, None]
        * task_scale[None, :]
        * np.exp(rng.normal(0.0, 0.45, size=(n_validation, tasks)))
    )
    calibration_residual = (
        shared(similarity_calibration)[:, None]
        * task_scale[None, :]
        * np.exp(rng.normal(0.0, 0.45, size=(n_calibration, tasks)))
    )
    validation_residual[rng.random(validation_residual.shape) < 0.06] = np.nan
    calibration_residual[rng.random(calibration_residual.shape) < 0.06] = np.nan
    return {
        "similarity_validation": similarity_validation,
        "validation_residual": validation_residual,
        "similarity_calibration": similarity_calibration,
        "calibration_residual": calibration_residual,
        "similarity_test": similarity_test,
    }


def run_synthetic(trials_per_delta: int) -> pd.DataFrame:
    rows: list[dict] = []
    for delta_index, delta in enumerate(DELTAS):
        for trial in range(trials_per_delta):
            rng = np.random.default_rng(20260816 + delta_index * 100_000 + trial)
            payload = synthetic_payload(rng)
            reference, fallback_count = task_normalizers(payload["validation_residual"])
            log_lambdas = rng.uniform(
                np.log(1.0 - delta), np.log(1.0 + delta), size=len(reference)
            )
            random_result = evaluate_perturbation(
                **payload,
                reference_normalizers=reference,
                lambdas=np.exp(log_lambdas),
            )
            rows.append(
                {
                    "delta": delta,
                    "trial": trial,
                    "mode": "random_log_uniform",
                    "fallback_count": fallback_count,
                    **random_result,
                }
            )
            common_result = evaluate_perturbation(
                **payload,
                reference_normalizers=reference,
                lambdas=np.full(len(reference), 1.0 + delta),
            )
            rows.append(
                {
                    "delta": delta,
                    "trial": trial,
                    "mode": "common",
                    "fallback_count": fallback_count,
                    **common_result,
                }
            )
    return pd.DataFrame(rows)


def correlation_aligned_lambdas(
    similarity: np.ndarray,
    residual: np.ndarray,
    normalizers: np.ndarray,
    delta: float,
) -> np.ndarray:
    normalized = residual / normalizers[None, :]
    correlations = np.zeros(normalized.shape[1], dtype=float)
    for task in range(normalized.shape[1]):
        valid = np.isfinite(similarity) & np.isfinite(normalized[:, task])
        if valid.sum() >= 3:
            result = stats.spearmanr(similarity[valid], normalized[valid, task])
            correlations[task] = float(result.statistic) if np.isfinite(result.statistic) else 0.0
    order = np.argsort(correlations, kind="mergesort")
    lambdas = np.full(len(correlations), 1.0 + delta)
    lambdas[order[: len(order) // 2]] = 1.0 - delta
    return lambdas


def run_real(root: Path, random_repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = EvaluationData(root)
    rows: list[dict] = []
    units: list[dict] = []
    for family_index, family in enumerate(FAMILIES):
        for seed in SEEDS:
            split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
            phase1_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
            phase3_dir = root / "results" / "phase3" / f"{family}_seed_{seed}"
            stored = np.load(phase1_dir / "mlp_predictions.npz")
            names = ("source_cal", "valid", "test")
            ids = {name: stored[f"nsc_{name}"].astype("int64") for name in names}
            prediction = {name: stored[f"pred_{name}"].astype(float) for name in names}
            truth = {
                "valid": data.truth(ids["valid"]),
                "source_cal": data.truth(ids["source_cal"]),
            }
            residual = {
                "valid": np.abs(truth["valid"] - prediction["valid"]),
                "source_cal": np.abs(truth["source_cal"] - prediction["source_cal"]),
            }
            fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
            similarity = {
                name: sim(phase3_dir, name, data, fit_ids, ids[name]) for name in names
            }
            reference, fallback_count = task_normalizers(residual["valid"])
            reference_scale, _ = fit_raw_scale(
                similarity["valid"], residual["valid"], reference
            )
            min_reference = float(
                min(
                    np.min(predict_normalized(reference_scale, similarity[name]))
                    for name in names
                )
            )
            units.append(
                {
                    "family": family,
                    "seed": seed,
                    "validation_compounds": int(len(ids["valid"])),
                    "fallback_count": fallback_count,
                    "min_unfloored_reference_scale": min_reference,
                }
            )

            for delta_index, delta in enumerate(DELTAS):
                patterns: list[tuple[str, int, np.ndarray]] = [
                    ("common", 0, np.full(len(reference), 1.0 + delta)),
                    (
                        "alternating_extremes",
                        0,
                        np.where(np.arange(len(reference)) % 2 == 0, 1.0 - delta, 1.0 + delta),
                    ),
                    (
                        "correlation_aligned",
                        0,
                        correlation_aligned_lambdas(
                            similarity["valid"], residual["valid"], reference, delta
                        ),
                    ),
                ]
                for repeat in range(random_repeats):
                    rng = np.random.default_rng(
                        20260816
                        + family_index * 1_000_000
                        + seed * 10_000
                        + delta_index * 100
                        + repeat
                    )
                    patterns.append(
                        (
                            "random_log_uniform",
                            repeat,
                            np.exp(
                                rng.uniform(
                                    np.log(1.0 - delta),
                                    np.log(1.0 + delta),
                                    size=len(reference),
                                )
                            ),
                        )
                    )
                for mode, repeat, lambdas in patterns:
                    result = evaluate_perturbation(
                        similarity_validation=similarity["valid"],
                        validation_residual=residual["valid"],
                        similarity_calibration=similarity["source_cal"],
                        calibration_residual=residual["source_cal"],
                        similarity_test=similarity["test"],
                        reference_normalizers=reference,
                        lambdas=lambdas,
                    )
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "delta": delta,
                            "mode": mode,
                            "repeat": repeat,
                            "fallback_count": fallback_count,
                            **result,
                        }
                    )
            print(f"perturbation family={family} seed={seed}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(units)


def evaluate_gates(
    synthetic: pd.DataFrame, real: pd.DataFrame, units: pd.DataFrame
) -> dict:
    combined = pd.concat(
        [
            synthetic.assign(source="synthetic"),
            real.assign(source="real"),
        ],
        ignore_index=True,
        sort=False,
    )
    violation_columns = {
        "T1_compound_sandwich": "compound_violation",
        "T2_raw_scale_sandwich": "raw_scale_violation",
        "T2_normalized_scale_sandwich": "normalized_scale_violation",
        "T3_radius_sandwich": "radius_violation",
    }
    gates: dict[str, dict | bool | str] = {}
    for gate, column in violation_columns.items():
        maximum = float(combined[column].max())
        gates[gate] = {
            "passed": bool(maximum <= TOLERANCE),
            "maximum_violation": maximum,
            "threshold": TOLERANCE,
        }
    common = combined[combined["mode"] == "common"]
    common_scale = float(common["normalized_scale_max_relative_difference"].max())
    common_radius = float(common["radius_max_relative_difference"].max())
    gates["T4_common_factor_invariance"] = {
        "passed": bool(common_scale <= TOLERANCE and common_radius <= TOLERANCE),
        "maximum_normalized_scale_relative_difference": common_scale,
        "maximum_radius_relative_difference": common_radius,
        "threshold": TOLERANCE,
    }
    minimum_scale = float(units["min_unfloored_reference_scale"].min())
    gates["T5_floor_inactivity"] = {
        "passed": bool(minimum_scale > FLOOR),
        "minimum_unfloored_scale": minimum_scale,
        "floor": FLOOR,
        "n_units": int(len(units)),
    }
    maximum_fallback = int(units["fallback_count"].max())
    gates["T6_fallback_inactivity"] = {
        "passed": bool(maximum_fallback == 0),
        "maximum_fallback_count": maximum_fallback,
        "n_units": int(len(units)),
    }
    gates["log_bound_diagnostic"] = {
        "maximum_excess_over_log_kappa": float(
            (combined["max_abs_log_radius_ratio"] - combined["log_kappa"]).max()
        )
    }
    hard_keys = [key for key in gates if key.startswith("T")]
    all_passed = all(bool(gates[key]["passed"]) for key in hard_keys)  # type: ignore[index]
    gates["all_frozen_gates_passed"] = all_passed
    gates["decision"] = (
        "advance_projective_stability_theorem"
        if all_passed
        else "stop_and_reconcile_theorem_with_implementation"
    )
    return gates


def make_report(
    output: Path,
    gates: dict,
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    units: pd.DataFrame,
    duration: float,
    command: str,
) -> None:
    noncommon_real = real[real["mode"] != "common"]
    lines = [
        "# Plugin Scale Perturbation: Validation Report",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-08-16",
        "- Verification Status: ANALYZED",
        "- Version Label: plugin_projective_stability_validation_v1",
        "- Manuscript Status: internal only",
        "",
        "## Run",
        "",
        f"- Command: `{command}`",
        f"- Duration: {duration:.1f} seconds",
        f"- Synthetic rows: {len(synthetic)}",
        f"- Real-data rows: {len(real)}",
        f"- Real family-seed units: {len(units)}",
        "- Test outcomes read: no",
        "",
        "## Frozen Gates",
        "",
    ]
    for key, value in gates.items():
        if key.startswith("T"):
            lines.append(f"- **{key}**: {'PASS' if value['passed'] else 'FAIL'} — `{json.dumps(value, ensure_ascii=False)}`")
    lines.extend(
        [
            "",
            f"**Decision:** `{gates['decision']}`",
            "",
            "## Diagnostics",
            "",
            f"- Maximum real-data non-common $|\\log(\\widehat r/r^*)|$: {noncommon_real['max_abs_log_radius_ratio'].max():.8f}",
            f"- Maximum corresponding theoretical $\\log\\kappa_m$: {noncommon_real['log_kappa'].max():.8f}",
            f"- Minimum unfloored real reference scale: {units['min_unfloored_reference_scale'].min():.8f}",
            f"- Maximum task fallback count: {int(units['fallback_count'].max())}",
            "",
            "## Evidence Boundary",
            "",
            "- These checks validate the deterministic code path and search for counterexamples; they do not replace the proof.",
            "- The empirical validation median is used only as a perturbation reference and is not treated as the unknown population median.",
            "- No ACE or test-label performance was computed in this experiment.",
            "",
        ]
    )
    (output / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--synthetic-trials-per-delta", type=int, default=200)
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.synthetic_trials_per_delta < 1 or args.random_repeats < 1:
        parser.error("trial and repeat counts must be positive")
    root = Path(args.root)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    synthetic = run_synthetic(args.synthetic_trials_per_delta)
    real, units = run_real(root, args.random_repeats)
    gates = evaluate_gates(synthetic, real, units)
    duration = time.perf_counter() - start

    synthetic.to_csv(output / "synthetic_trials.csv", index=False)
    real.to_csv(output / "real_perturbation_stress.csv", index=False)
    units.to_csv(output / "real_unit_diagnostics.csv", index=False)
    (output / "decision_gates.json").write_text(
        json.dumps(gates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    command = (
        f"python scripts/validate_plugin_scale_perturbation.py --root {root.as_posix()} "
        f"--synthetic-trials-per-delta {args.synthetic_trials_per_delta} "
        f"--random-repeats {args.random_repeats}"
    )
    make_report(output, gates, synthetic, real, units, duration, command)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "duration_seconds": duration,
        "python": sys.version,
        "files": {
            str(path.relative_to(root)): sha256(path)
            for path in (
                root / "scripts" / "validate_plugin_scale_perturbation.py",
                root / "docs" / "Plugin尺度扰动定理_先验分析与验证协议_20260816.md",
                root / "docs" / "Plugin任务尺度扰动定理_正式证明_v2_20260816.md",
                output / "synthetic_trials.csv",
                output / "real_perturbation_stress.csv",
                output / "real_unit_diagnostics.csv",
                output / "decision_gates.json",
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(gates, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

