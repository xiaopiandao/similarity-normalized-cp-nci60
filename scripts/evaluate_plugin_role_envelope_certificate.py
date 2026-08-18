"""Evaluate a calibration/test role-envelope certificate for plugin scales."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.bootstrap_plugin_normalizer_stability import (
        FAMILIES,
        SEEDS,
        bootstrap_normalizers,
        load_unit,
        quantile_summary,
    )
    from scripts.evaluate_phase1a_conformal import EvaluationData
    from scripts.validate_plugin_scale_perturbation import (
        TOLERANCE,
        fit_raw_scale,
        predict_raw,
        task_normalizers,
    )
except ModuleNotFoundError:
    from bootstrap_plugin_normalizer_stability import (  # type: ignore[no-redef]
        FAMILIES,
        SEEDS,
        bootstrap_normalizers,
        load_unit,
        quantile_summary,
    )
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]
    from validate_plugin_scale_perturbation import (  # type: ignore[no-redef]
        TOLERANCE,
        fit_raw_scale,
        predict_raw,
        task_normalizers,
    )


DEFAULT_BOOTSTRAP_RESULTS = (
    "results/plugin_bootstrap_stability_20260816/bootstrap_trials.csv"
)
DEFAULT_OUTPUT = "results/plugin_role_envelope_certificate_20260816"
CERTIFICATE_Q95_THRESHOLD = 1.08
LOG_FRACTION_MEDIAN_THRESHOLD = 0.50


def role_envelope_certificate(
    reference_calibration: np.ndarray,
    perturbed_calibration: np.ndarray,
    reference_test: np.ndarray,
    perturbed_test: np.ndarray,
) -> dict[str, float]:
    """Return the asymmetric and symmetric calibration/test scale certificate."""
    reference_calibration = np.asarray(reference_calibration, dtype=float)
    perturbed_calibration = np.asarray(perturbed_calibration, dtype=float)
    reference_test = np.asarray(reference_test, dtype=float)
    perturbed_test = np.asarray(perturbed_test, dtype=float)
    rho_cal = perturbed_calibration / reference_calibration
    rho_test = perturbed_test / reference_test
    finite_cal = np.isfinite(rho_cal) & (rho_cal > 0)
    finite_test = np.isfinite(rho_test) & (rho_test > 0)
    if not finite_cal.any() or not finite_test.any():
        raise ValueError("No finite positive scale ratios")
    a_cal = float(np.min(rho_cal[finite_cal]))
    b_cal = float(np.max(rho_cal[finite_cal]))
    a_test = float(np.min(rho_test[finite_test]))
    b_test = float(np.max(rho_test[finite_test]))
    lower = a_test / b_cal
    upper = b_test / a_cal
    factor = max(upper, 1.0 / lower)
    return {
        "rho_cal_min": a_cal,
        "rho_cal_max": b_cal,
        "rho_test_min": a_test,
        "rho_test_max": b_test,
        "certificate_lower": lower,
        "certificate_upper": upper,
        "certificate_factor": factor,
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "kappa_m",
        "certificate_factor",
        "radius_factor",
        "certificate_log_fraction",
        "certificate_utilization",
    )
    output: list[dict] = []
    for (phase, family), part in rows.groupby(["phase", "family"], sort=False):
        row: dict[str, object] = {
            "phase": phase,
            "family": family,
            "n_rows": int(len(part)),
            "n_seeds": int(part["seed"].nunique()),
        }
        for metric in metrics:
            for name, value in quantile_summary(part[metric]).items():
                row[f"{metric}_{name}"] = value
        output.append(row)
    return pd.DataFrame(output)


def evaluate_gates(rows: pd.DataFrame, summary: pd.DataFrame) -> dict:
    max_certificate_violation = float(
        np.max(rows["radius_factor"] - rows["certificate_factor"])
    )
    max_dominance_violation = float(
        np.max(rows["certificate_factor"] - rows["kappa_m"])
    )
    gates: dict[str, object] = {
        "C1_certificate_validity": {
            "passed": bool(max_certificate_violation <= TOLERANCE),
            "maximum_violation": max(0.0, max_certificate_violation),
            "threshold": TOLERANCE,
        },
        "C2_dominance_by_kappa": {
            "passed": bool(max_dominance_violation <= TOLERANCE),
            "maximum_violation": max(0.0, max_dominance_violation),
            "threshold": TOLERANCE,
        },
    }
    confirmation = summary[summary["phase"] == "confirmation"]
    certificate_q95 = {
        str(row.family): float(row.certificate_factor_q95)
        for row in confirmation.itertuples(index=False)
    }
    log_fraction_median = {
        str(row.family): float(row.certificate_log_fraction_median)
        for row in confirmation.itertuples(index=False)
    }
    gates["C3_certificate_concentration"] = {
        "passed": bool(
            len(certificate_q95) == len(FAMILIES)
            and all(
                value <= CERTIFICATE_Q95_THRESHOLD
                for value in certificate_q95.values()
            )
        ),
        "family_q95_certificate_factor": certificate_q95,
        "threshold": CERTIFICATE_Q95_THRESHOLD,
    }
    gates["C4_log_scale_tightening"] = {
        "passed": bool(
            len(log_fraction_median) == len(FAMILIES)
            and all(
                value <= LOG_FRACTION_MEDIAN_THRESHOLD
                for value in log_fraction_median.values()
            )
        ),
        "family_median_log_certificate_over_log_kappa": log_fraction_median,
        "threshold": LOG_FRACTION_MEDIAN_THRESHOLD,
    }
    gate_names = (
        "C1_certificate_validity",
        "C2_dominance_by_kappa",
        "C3_certificate_concentration",
        "C4_log_scale_tightening",
    )
    all_passed = all(bool(gates[name]["passed"]) for name in gate_names)  # type: ignore[index]
    gates["all_exploratory_gates_passed"] = all_passed
    gates["decision"] = (
        "advance_role_envelope_corollary_for_proof_audit"
        if all_passed
        else "retain_global_kappa_theorem_only"
    )
    return gates


def run(
    root: Path, bootstrap_results: Path, n_bootstraps: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stored = pd.read_csv(bootstrap_results)
    stored = stored[stored["bootstrap"] < n_bootstraps].copy()
    lookup = stored.set_index(["family", "seed", "bootstrap"])
    data = EvaluationData(root)
    rows: list[dict] = []
    for family_index, family in enumerate(FAMILIES):
        for seed in SEEDS:
            payload = load_unit(root, data, family, seed)
            validation_residual = np.asarray(
                payload["validation_residual"], dtype=float
            )
            reference_normalizers, _ = task_normalizers(validation_residual)
            reference_scale, _ = fit_raw_scale(
                np.asarray(payload["similarity_validation"], dtype=float),
                validation_residual,
                reference_normalizers,
            )
            reference_calibration = predict_raw(
                reference_scale,
                np.asarray(payload["similarity_calibration"], dtype=float),
            )
            reference_test = predict_raw(
                reference_scale, np.asarray(payload["similarity_test"], dtype=float)
            )
            phase = "development" if seed == 1 else "confirmation"
            for bootstrap in range(n_bootstraps):
                rng = np.random.default_rng(
                    20260816
                    + family_index * 10_000_000
                    + seed * 100_000
                    + bootstrap
                )
                boot_normalizers, _ = bootstrap_normalizers(
                    validation_residual, rng
                )
                lambdas = boot_normalizers / reference_normalizers
                perturbed_scale, _ = fit_raw_scale(
                    np.asarray(payload["similarity_validation"], dtype=float),
                    validation_residual,
                    boot_normalizers,
                )
                perturbed_calibration = predict_raw(
                    perturbed_scale,
                    np.asarray(payload["similarity_calibration"], dtype=float),
                )
                perturbed_test = predict_raw(
                    perturbed_scale,
                    np.asarray(payload["similarity_test"], dtype=float),
                )
                certificate = role_envelope_certificate(
                    reference_calibration,
                    perturbed_calibration,
                    reference_test,
                    perturbed_test,
                )
                kappa_m = float(np.max(lambdas) / np.min(lambdas))
                radius_factor = float(
                    lookup.loc[(family, seed, bootstrap), "radius_factor"]
                )
                log_kappa = float(np.log(kappa_m))
                log_certificate = float(np.log(certificate["certificate_factor"]))
                rows.append(
                    {
                        "phase": phase,
                        "family": family,
                        "seed": seed,
                        "bootstrap": bootstrap,
                        "kappa_m": kappa_m,
                        "radius_factor": radius_factor,
                        "certificate_log_fraction": (
                            log_certificate / log_kappa if log_kappa > 1e-15 else 0.0
                        ),
                        "certificate_utilization": (
                            np.log(radius_factor) / log_certificate
                            if log_certificate > 1e-15
                            else 0.0
                        ),
                        **certificate,
                    }
                )
            print(
                f"certificate family={family} seed={seed} repeats={n_bootstraps}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    return frame, summary, evaluate_gates(frame, summary)


def make_report(output: Path, gates: dict, summary: pd.DataFrame, duration: float) -> None:
    confirmation = summary[summary["phase"] == "confirmation"]
    lines = [
        "# Plugin Role-Envelope Certificate: Exploratory Validation",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: exploratory validate",
        "- Origin Date: 2026-08-16",
        "- Verification Status: ANALYZED",
        "- Version Label: plugin_role_envelope_certificate_v1",
        "- Manuscript Status: internal only",
        "",
        f"- Duration: {duration:.1f} seconds",
        "- Test outcomes read: no",
        "",
        "## Exploratory Gates",
        "",
    ]
    for key in (
        "C1_certificate_validity",
        "C2_dominance_by_kappa",
        "C3_certificate_concentration",
        "C4_log_scale_tightening",
    ):
        value = gates[key]
        lines.append(
            f"- **{key}**: {'PASS' if value['passed'] else 'FAIL'} — "
            f"`{json.dumps(value, ensure_ascii=False)}`"
        )
    lines.extend(
        [
            "",
            f"**Decision:** `{gates['decision']}`",
            "",
            "| Family | q95 kappa | q95 certificate | q95 radius | median log certificate / log kappa |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in confirmation.itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.kappa_m_q95:.6f} | "
            f"{row.certificate_factor_q95:.6f} | {row.radius_factor_q95:.6f} | "
            f"{row.certificate_log_fraction_median:.6f} |"
        )
    lines.extend(
        [
            "",
            "This is a post-bootstrap exploratory refinement. It must not be described as part of the original confirmatory B1--B5 analysis.",
            "",
        ]
    )
    (output / "EXPLORATORY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--bootstrap-results", default=DEFAULT_BOOTSTRAP_RESULTS)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.bootstraps < 1:
        parser.error("bootstraps must be positive")
    root = Path(args.root)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    bootstrap_results = root / args.bootstrap_results
    start = time.perf_counter()
    rows, summary, gates = run(root, bootstrap_results, args.bootstraps)
    duration = time.perf_counter() - start
    rows.to_csv(output / "certificate_trials.csv", index=False)
    summary.to_csv(output / "certificate_summary.csv", index=False)
    (output / "decision_gates.json").write_text(
        json.dumps(gates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_report(output, gates, summary, duration)
    print(json.dumps(gates, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

