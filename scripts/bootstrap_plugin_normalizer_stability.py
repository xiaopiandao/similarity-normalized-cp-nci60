"""Bootstrap the empirical stability of plugin task normalizers.

The prespecified protocol is documented in
``docs/Plugin任务尺度Bootstrap稳定性_预注册协议_20260816.md``.
The experiment bootstraps validation compounds to estimate task normalizers,
then propagates only that normalizer perturbation through a fixed downstream
compound-median/isotonic/split-conformal pipeline. Test outcomes are not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.evaluate_phase1a_conformal import EvaluationData
    from scripts.evaluate_phase3_local_and_screening import sim
    from scripts.validate_plugin_scale_perturbation import (
        FLOOR,
        TOLERANCE,
        evaluate_perturbation,
        fit_raw_scale,
        predict_normalized,
        task_normalizers,
    )
except ModuleNotFoundError:
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]
    from evaluate_phase3_local_and_screening import sim  # type: ignore[no-redef]
    from validate_plugin_scale_perturbation import (  # type: ignore[no-redef]
        FLOOR,
        TOLERANCE,
        evaluate_perturbation,
        fit_raw_scale,
        predict_normalized,
        task_normalizers,
    )


FAMILIES = ("random", "scaffold", "leader_cluster")
SEEDS = (1, 2, 3, 4, 5)
ALPHA = 0.10
DEFAULT_BOOTSTRAPS = 500
DEFAULT_OUTPUT = "results/plugin_bootstrap_stability_20260816"
KAPPA_Q95_THRESHOLD = 1.20
RADIUS_FACTOR_Q95_THRESHOLD = 1.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_normalizers(
    residual: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Resample compound rows and return task-wise residual medians."""
    residual = np.asarray(residual, dtype=float)
    indices = rng.integers(0, residual.shape[0], size=residual.shape[0])
    return task_normalizers(residual[indices])


def quantile_summary(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(values) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "q90": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "maximum": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = (
        "kappa_m",
        "radius_factor",
        "bound_utilization",
        "normalizer_median_abs_log_ratio",
    )
    for (phase, family), part in trials.groupby(["phase", "family"], sort=False):
        row: dict[str, object] = {
            "phase": phase,
            "family": family,
            "n_bootstrap_rows": int(len(part)),
            "n_seeds": int(part["seed"].nunique()),
        }
        for metric in metrics:
            for statistic, value in quantile_summary(part[metric]).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_gates(
    trials: pd.DataFrame, units: pd.DataFrame, summary: pd.DataFrame
) -> dict:
    gates: dict[str, object] = {}
    max_violation = float(trials["radius_violation"].max())
    gates["B1_radius_sandwich"] = {
        "passed": bool(max_violation <= TOLERANCE),
        "maximum_violation": max_violation,
        "threshold": TOLERANCE,
    }
    max_reference_fallback = int(units["reference_fallback_count"].max())
    max_bootstrap_fallback = int(trials["bootstrap_fallback_count"].max())
    gates["B2_fallback_inactivity"] = {
        "passed": bool(max_reference_fallback == 0 and max_bootstrap_fallback == 0),
        "maximum_reference_fallback_count": max_reference_fallback,
        "maximum_bootstrap_fallback_count": max_bootstrap_fallback,
    }
    minimum_scale = float(
        min(
            units["min_unfloored_reference_scale"].min(),
            trials["min_unfloored_perturbed_scale"].min(),
        )
    )
    gates["B3_floor_inactivity"] = {
        "passed": bool(minimum_scale > FLOOR),
        "minimum_unfloored_scale": minimum_scale,
        "floor": FLOOR,
    }

    confirmation = summary[summary["phase"] == "confirmation"].copy()
    kappa_values = {
        str(row.family): float(row.kappa_m_q95)
        for row in confirmation.itertuples(index=False)
    }
    radius_values = {
        str(row.family): float(row.radius_factor_q95)
        for row in confirmation.itertuples(index=False)
    }
    gates["B4_projective_concentration"] = {
        "passed": bool(
            len(kappa_values) == len(FAMILIES)
            and all(value <= KAPPA_Q95_THRESHOLD for value in kappa_values.values())
        ),
        "family_q95_kappa_m": kappa_values,
        "threshold": KAPPA_Q95_THRESHOLD,
    }
    gates["B5_radius_concentration"] = {
        "passed": bool(
            len(radius_values) == len(FAMILIES)
            and all(
                value <= RADIUS_FACTOR_Q95_THRESHOLD
                for value in radius_values.values()
            )
        ),
        "family_q95_radius_factor": radius_values,
        "threshold": RADIUS_FACTOR_Q95_THRESHOLD,
    }
    hard_keys = [
        "B1_radius_sandwich",
        "B2_fallback_inactivity",
        "B3_floor_inactivity",
        "B4_projective_concentration",
        "B5_radius_concentration",
    ]
    all_passed = all(bool(gates[key]["passed"]) for key in hard_keys)  # type: ignore[index]
    gates["all_prespecified_gates_passed"] = all_passed
    gates["decision"] = (
        "eligible_for_supplementary_empirical_stability_evidence"
        if all_passed
        else "retain_theorem_but_do_not_promote_empirical_stability_claim"
    )
    return gates


def load_unit(
    root: Path, data: EvaluationData, family: str, seed: int
) -> dict[str, object]:
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    phase1_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    phase3_dir = root / "results" / "phase3" / f"{family}_seed_{seed}"
    stored = np.load(phase1_dir / "mlp_predictions.npz")
    names = ("source_cal", "valid", "test")
    ids = {name: stored[f"nsc_{name}"].astype("int64") for name in names}
    prediction = {name: stored[f"pred_{name}"].astype(float) for name in names}
    validation_residual = np.abs(data.truth(ids["valid"]) - prediction["valid"])
    calibration_residual = np.abs(
        data.truth(ids["source_cal"]) - prediction["source_cal"]
    )
    fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
    similarities = {
        name: sim(phase3_dir, name, data, fit_ids, ids[name]) for name in names
    }
    return {
        "validation_residual": validation_residual,
        "calibration_residual": calibration_residual,
        "similarity_validation": similarities["valid"],
        "similarity_calibration": similarities["source_cal"],
        "similarity_test": similarities["test"],
    }


def run_bootstrap(root: Path, n_bootstraps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = EvaluationData(root)
    rows: list[dict] = []
    units: list[dict] = []
    for family_index, family in enumerate(FAMILIES):
        for seed in SEEDS:
            payload = load_unit(root, data, family, seed)
            residual = np.asarray(payload["validation_residual"], dtype=float)
            reference, reference_fallback = task_normalizers(residual)
            reference_scale, _ = fit_raw_scale(
                np.asarray(payload["similarity_validation"], dtype=float),
                residual,
                reference,
            )
            similarities = (
                np.asarray(payload["similarity_validation"], dtype=float),
                np.asarray(payload["similarity_calibration"], dtype=float),
                np.asarray(payload["similarity_test"], dtype=float),
            )
            minimum_reference_scale = float(
                min(np.min(predict_normalized(reference_scale, values)) for values in similarities)
            )
            phase = "development" if seed == 1 else "confirmation"
            units.append(
                {
                    "phase": phase,
                    "family": family,
                    "seed": seed,
                    "validation_compounds": int(residual.shape[0]),
                    "tasks": int(residual.shape[1]),
                    "reference_fallback_count": reference_fallback,
                    "min_unfloored_reference_scale": minimum_reference_scale,
                }
            )

            for bootstrap in range(n_bootstraps):
                rng = np.random.default_rng(
                    20260816
                    + family_index * 10_000_000
                    + seed * 100_000
                    + bootstrap
                )
                boot, bootstrap_fallback = bootstrap_normalizers(residual, rng)
                lambdas = boot / reference
                result = evaluate_perturbation(
                    similarity_validation=np.asarray(
                        payload["similarity_validation"], dtype=float
                    ),
                    validation_residual=residual,
                    similarity_calibration=np.asarray(
                        payload["similarity_calibration"], dtype=float
                    ),
                    calibration_residual=np.asarray(
                        payload["calibration_residual"], dtype=float
                    ),
                    similarity_test=np.asarray(payload["similarity_test"], dtype=float),
                    reference_normalizers=reference,
                    lambdas=lambdas,
                )
                log_kappa = float(result["log_kappa"])
                max_log_radius = float(result["max_abs_log_radius_ratio"])
                rows.append(
                    {
                        "phase": phase,
                        "family": family,
                        "seed": seed,
                        "bootstrap": bootstrap,
                        "validation_compounds": int(residual.shape[0]),
                        "bootstrap_fallback_count": bootstrap_fallback,
                        "kappa_m": float(result["kappa"]),
                        "log_kappa_m": log_kappa,
                        "radius_factor": float(np.exp(max_log_radius)),
                        "bound_utilization": (
                            max_log_radius / log_kappa if log_kappa > 1e-15 else 0.0
                        ),
                        "normalizer_median_abs_log_ratio": float(
                            np.median(np.abs(np.log(lambdas)))
                        ),
                        **result,
                    }
                )
            print(
                f"bootstrap family={family} seed={seed} repeats={n_bootstraps}",
                flush=True,
            )
    return pd.DataFrame(rows), pd.DataFrame(units)


def make_report(
    output: Path,
    gates: dict,
    trials: pd.DataFrame,
    units: pd.DataFrame,
    summary: pd.DataFrame,
    duration: float,
    command: str,
) -> None:
    confirmation = summary[summary["phase"] == "confirmation"]
    lines = [
        "# Plugin Task-Normalizer Bootstrap Stability",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-08-16",
        "- Verification Status: ANALYZED",
        "- Version Label: plugin_bootstrap_stability_v1",
        "- Manuscript Status: internal gate",
        "",
        "## Run",
        "",
        f"- Command: `{command}`",
        f"- Duration: {duration:.1f} seconds",
        f"- Bootstrap rows: {len(trials)}",
        f"- Family-seed units: {len(units)}",
        "- Test outcomes read: no",
        "",
        "## Prespecified Gates",
        "",
    ]
    for key in [
        "B1_radius_sandwich",
        "B2_fallback_inactivity",
        "B3_floor_inactivity",
        "B4_projective_concentration",
        "B5_radius_concentration",
    ]:
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
            "## Confirmation Summary",
            "",
            "| Family | q95 kappa_m | q95 radius factor | median bound utilization |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in confirmation.itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.kappa_m_q95:.6f} | "
            f"{row.radius_factor_q95:.6f} | {row.bound_utilization_median:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "- The bootstrap targets empirical validation-distribution stability, not exact error to an unknown population median.",
            "- Downstream validation design is fixed to isolate plugin-normalizer perturbation.",
            "- The experiment does not estimate OOD coverage and reads no test outcomes.",
            "- Failure of B4/B5 would not invalidate the deterministic theorem; it would block an empirical concentration claim.",
            "",
        ]
    )
    (output / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.bootstraps < 1:
        parser.error("bootstraps must be positive")
    root = Path(args.root)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    trials, units = run_bootstrap(root, args.bootstraps)
    summary = summarize_trials(trials)
    gates = evaluate_gates(trials, units, summary)
    duration = time.perf_counter() - start

    trials.to_csv(output / "bootstrap_trials.csv", index=False)
    units.to_csv(output / "unit_diagnostics.csv", index=False)
    summary.to_csv(output / "bootstrap_summary.csv", index=False)
    (output / "decision_gates.json").write_text(
        json.dumps(gates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    command = (
        "python scripts/bootstrap_plugin_normalizer_stability.py "
        f"--root {root.as_posix()} --bootstraps {args.bootstraps} "
        f"--output {args.output}"
    )
    make_report(output, gates, trials, units, summary, duration, command)
    files = (
        root / "scripts" / "bootstrap_plugin_normalizer_stability.py",
        root / "docs" / "Plugin任务尺度Bootstrap稳定性_预注册协议_20260816.md",
        output / "bootstrap_trials.csv",
        output / "unit_diagnostics.csv",
        output / "bootstrap_summary.csv",
        output / "decision_gates.json",
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "duration_seconds": duration,
        "python": sys.version,
        "files": {str(path.relative_to(root)): sha256(path) for path in files},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(gates, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

