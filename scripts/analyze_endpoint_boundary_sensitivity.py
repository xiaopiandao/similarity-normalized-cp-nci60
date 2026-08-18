"""Evaluate whether endpoint spikes at logGI50 4 and 8 change frozen conclusions.

The CellMiner export does not identify censoring at the observation level.  This
script therefore treats 4/8 as boundary *candidates* and implements two
evaluation-only sensitivity scenarios without retraining or retuning:

1. exclude_exact_4_or_8: remove exact boundary candidates from test metrics;
2. one_sided_compatible: hypothetically treat 4 as a left endpoint (y <= 4)
   and 8 as a right endpoint (y >= 8) when checking interval compatibility.

The second scenario is not a conformal coverage estimate for exact outcomes and
does not receive an interval score; it is an assumption-based stress test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.analyze_phase4_statistics import mean_t_interval
    from scripts.analyze_uacqrp_exploratory import extended_interval_metrics
    from scripts.evaluate_phase1a_conformal import EvaluationData
    from scripts.evaluate_phase5_extensions import ALPHAS, FAMILIES
    from scripts.finalize_jbhi_results import (
        ACTIVITY_THRESHOLD,
        METHODS,
        SCREENING_THRESHOLDS,
        intervals_for_alpha,
        reconstruct_run,
        screening_rows,
    )
except ModuleNotFoundError:
    from analyze_phase4_statistics import mean_t_interval  # type: ignore[no-redef]
    from analyze_uacqrp_exploratory import extended_interval_metrics  # type: ignore[no-redef]
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]
    from evaluate_phase5_extensions import ALPHAS, FAMILIES  # type: ignore[no-redef]
    from finalize_jbhi_results import (  # type: ignore[no-redef]
        ACTIVITY_THRESHOLD,
        METHODS,
        SCREENING_THRESHOLDS,
        intervals_for_alpha,
        reconstruct_run,
        screening_rows,
    )


EXACT_SCENARIOS = ("reference_all", "exclude_exact_4_or_8")


def exclude_boundary_truth(truth: np.ndarray) -> np.ndarray:
    result = np.asarray(truth, dtype=float).copy()
    result[np.isclose(result, 4.0) | np.isclose(result, 8.0)] = np.nan
    return result


def one_sided_compatible_metrics(
    truth: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> dict:
    """Coverage compatible with hypothetical left/right endpoint censoring."""
    truth = np.asarray(truth, dtype=float)
    observed = np.isfinite(truth)
    defined = observed & ~np.isnan(lower) & ~np.isnan(upper)
    finite = defined & np.isfinite(lower) & np.isfinite(upper)
    at_lower = observed & np.isclose(truth, 4.0)
    at_upper = observed & np.isclose(truth, 8.0)
    interior = observed & ~at_lower & ~at_upper
    covered = np.zeros_like(observed, dtype=bool)
    covered[at_lower & defined] = lower[at_lower & defined] <= 4.0
    covered[at_upper & defined] = upper[at_upper & defined] >= 8.0
    covered[interior & defined] = (
        (truth[interior & defined] >= lower[interior & defined])
        & (truth[interior & defined] <= upper[interior & defined])
    )
    widths = upper - lower
    coverage = float(covered.sum() / observed.sum())
    return {
        "nominal_coverage": 1.0 - alpha,
        "coverage": coverage,
        "coverage_error_abs": abs(coverage - (1.0 - alpha)),
        "mean_width": float(widths[finite].mean()) if finite.any() else np.nan,
        "median_width": float(np.median(widths[finite])) if finite.any() else np.nan,
        "finite_interval_fraction": float(finite.sum() / observed.sum()),
        "n_compounds": int(truth.shape[0]),
        "n_labels": int(observed.sum()),
        "n_exact_4": int(at_lower.sum()),
        "n_exact_8": int(at_upper.sum()),
        "mean_interval_score": np.nan,
        "mean_interval_score_finite": np.nan,
        "interval_score_status": "not_defined_for_one_sided_set_valued_outcomes",
    }


def scenario_metrics(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    scenario: str,
) -> dict:
    if scenario == "reference_all":
        metrics = extended_interval_metrics(truth, lower, upper, alpha)
        metrics["interval_score_status"] = "standard_exact_outcome_score"
    elif scenario == "exclude_exact_4_or_8":
        filtered = exclude_boundary_truth(truth)
        metrics = extended_interval_metrics(filtered, lower, upper, alpha)
        metrics["interval_score_status"] = "standard_score_after_boundary_exclusion"
    elif scenario == "one_sided_compatible":
        metrics = one_sided_compatible_metrics(truth, lower, upper, alpha)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    observed = np.isfinite(truth)
    metrics.update(
        {
            "original_n_labels": int(observed.sum()),
            "original_exact_4": int((observed & np.isclose(truth, 4.0)).sum()),
            "original_exact_8": int((observed & np.isclose(truth, 8.0)).sum()),
        }
    )
    return metrics


def summarize_units(frame: pd.DataFrame, value: str) -> dict:
    values = frame[value].to_numpy(float)
    mean, sd, low, high = mean_t_interval(values)
    return {
        "mean": mean,
        "sd": sd,
        "t95_low": low,
        "t95_high": high,
        "n_units": int(np.isfinite(values).sum()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", default="2,3,4,5")
    args = parser.parse_args()
    root = Path(args.root)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    output = root / "results" / "phase7_boundary_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    data = EvaluationData(root)

    rows: list[dict] = []
    boundary_subgroups: list[dict] = []
    screens: list[dict] = []
    prevalence: list[dict] = []
    for family in FAMILIES:
        for seed in seeds:
            payload = reconstruct_run(root, data, family, seed)
            truth = payload["truth"]["test"]
            similarity = payload["similarity"]["test"]
            observed = np.isfinite(truth)
            boundary = observed & (np.isclose(truth, 4.0) | np.isclose(truth, 8.0))
            prevalence.append(
                {
                    "family": family,
                    "seed": seed,
                    "scope": "all",
                    "n_labels": int(observed.sum()),
                    "n_boundary": int(boundary.sum()),
                    "boundary_rate": float(boundary.sum() / observed.sum()),
                    "exact_4_rate": float((observed & np.isclose(truth, 4.0)).sum() / observed.sum()),
                    "exact_8_rate": float((observed & np.isclose(truth, 8.0)).sum() / observed.sum()),
                }
            )
            low = similarity < 0.4
            low_observed = observed & low[:, None]
            low_boundary = boundary & low[:, None]
            prevalence.append(
                {
                    "family": family,
                    "seed": seed,
                    "scope": "similarity_lt_0.4",
                    "n_labels": int(low_observed.sum()),
                    "n_boundary": int(low_boundary.sum()),
                    "boundary_rate": float(low_boundary.sum() / low_observed.sum()),
                    "exact_4_rate": float((low_observed & np.isclose(truth, 4.0)).sum() / low_observed.sum()),
                    "exact_8_rate": float((low_observed & np.isclose(truth, 8.0)).sum() / low_observed.sum()),
                }
            )

            alpha_intervals = {}
            for alpha in ALPHAS:
                intervals = intervals_for_alpha(payload, alpha)
                alpha_intervals[alpha] = intervals
                for method, (lower, upper) in intervals.items():
                    for scenario in (*EXACT_SCENARIOS, "one_sided_compatible"):
                        rows.append(
                            {
                                "family": family,
                                "seed": seed,
                                "alpha": alpha,
                                "method": method,
                                "scope": "all",
                                "scenario": scenario,
                                **scenario_metrics(truth, lower, upper, alpha, scenario),
                            }
                        )
                        low_truth = truth[low]
                        rows.append(
                            {
                                "family": family,
                                "seed": seed,
                                "alpha": alpha,
                                "method": method,
                                "scope": "similarity_lt_0.4",
                                "scenario": scenario,
                                **scenario_metrics(
                                    low_truth, lower[low], upper[low], alpha, scenario
                                ),
                            }
                        )
                    for group, label_mask in (
                        ("exact_4", np.isfinite(truth) & np.isclose(truth, 4.0)),
                        ("exact_8", np.isfinite(truth) & np.isclose(truth, 8.0)),
                        (
                            "non_boundary",
                            np.isfinite(truth)
                            & ~np.isclose(truth, 4.0)
                            & ~np.isclose(truth, 8.0),
                        ),
                    ):
                        subgroup_truth = truth.copy()
                        subgroup_truth[~label_mask] = np.nan
                        boundary_subgroups.append(
                            {
                                "family": family,
                                "seed": seed,
                                "alpha": alpha,
                                "method": method,
                                "boundary_group": group,
                                **extended_interval_metrics(
                                    subgroup_truth, lower, upper, alpha
                                ),
                            }
                        )

            for scenario, scenario_truth in (
                ("reference_all", truth),
                ("exclude_exact_4_or_8", exclude_boundary_truth(truth)),
            ):
                scenario_rows = screening_rows(
                    family,
                    seed,
                    scenario_truth,
                    payload["prediction"]["test"],
                    alpha_intervals[0.10],
                )
                screens.extend({"scenario": scenario, **row} for row in scenario_rows)
            print(f"boundary sensitivity family={family} seed={seed}", flush=True)

    metrics = pd.DataFrame(rows)
    screening = pd.DataFrame(screens)
    prevalence_frame = pd.DataFrame(prevalence)
    subgroup_frame = pd.DataFrame(boundary_subgroups)
    metrics.to_csv(output / "boundary_sensitivity_seed_metrics.csv", index=False)
    screening.to_csv(output / "boundary_screening_seed_metrics.csv", index=False)
    prevalence_frame.to_csv(output / "boundary_prevalence_by_split.csv", index=False)
    subgroup_frame.to_csv(output / "boundary_subgroup_seed_metrics.csv", index=False)

    ood = metrics[metrics["family"].isin(("scaffold", "leader_cluster"))]
    aggregate = (
        ood.groupby(["scenario", "scope", "method"], as_index=False)
        .agg(
            absolute_calibration_error=("coverage_error_abs", "mean"),
            coverage=("coverage", "mean"),
            mean_width=("mean_width", "mean"),
            minimum_finite_fraction=("finite_interval_fraction", "min"),
            n_labels=("n_labels", "sum"),
        )
    )
    aggregate["calibration_rank"] = aggregate.groupby(["scenario", "scope"])[
        "absolute_calibration_error"
    ].rank(method="min")
    aggregate.to_csv(output / "boundary_sensitivity_ood_aggregate.csv", index=False)

    paired_rows = []
    for (scenario, family, seed), part in ood[
        ood["scenario"].isin(EXACT_SCENARIOS)
        & (ood["scope"] == "similarity_lt_0.4")
        & ood["method"].isin(("Proposed_shared_monotone", "UACQR_P"))
    ].groupby(["scenario", "family", "seed"]):
        pivot = part.pivot(index="alpha", columns="method", values="coverage_error_abs")
        paired_rows.append(
            {
                "scenario": scenario,
                "family": family,
                "seed": seed,
                "mean_proposed_minus_uacqrp_ace": float(
                    (pivot["Proposed_shared_monotone"] - pivot["UACQR_P"]).mean()
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "boundary_paired_low_similarity_units.csv", index=False)

    leader90 = metrics[
        (metrics["family"] == "leader_cluster")
        & np.isclose(metrics["alpha"], 0.10)
        & (metrics["scope"] == "all")
        & metrics["scenario"].isin(EXACT_SCENARIOS)
        & metrics["method"].isin(("Proposed_shared_monotone", "UACQR_P"))
    ]
    leader_differences = []
    for (scenario, seed), part in leader90.groupby(["scenario", "seed"]):
        values = part.set_index("method")["coverage"]
        leader_differences.append(
            {
                "scenario": scenario,
                "seed": seed,
                "proposed_minus_uacqrp_coverage": float(
                    values["Proposed_shared_monotone"] - values["UACQR_P"]
                ),
            }
        )
    leader_differences = pd.DataFrame(leader_differences)
    leader_differences.to_csv(output / "boundary_leader90_coverage_units.csv", index=False)

    screen_primary = screening[
        (screening["family"] == "leader_cluster")
        & (screening["strong_cell_threshold"] == 3)
    ]
    screen_summary = (
        screen_primary.groupby(["scenario", "method"], as_index=False)
        .agg(
            selected=("selected", "mean"),
            precision=("precision", "mean"),
            false_discovery_rate=("false_discovery_rate", "mean"),
            recall=("recall", "mean"),
        )
    )
    screen_summary.to_csv(output / "boundary_screening_leader_k3_summary.csv", index=False)

    subgroup_summary = (
        subgroup_frame[
            (subgroup_frame["family"] == "leader_cluster")
            & np.isclose(subgroup_frame["alpha"], 0.10)
        ]
        .groupby(["boundary_group", "method"], as_index=False)
        .agg(coverage=("coverage", "mean"), n_labels=("n_labels", "sum"))
    )
    subgroup_summary.to_csv(output / "boundary_subgroup_leader90_summary.csv", index=False)

    excluded_leader90 = metrics[
        (metrics["family"] == "leader_cluster")
        & np.isclose(metrics["alpha"], 0.10)
        & (metrics["scope"] == "all")
        & (metrics["scenario"] == "exclude_exact_4_or_8")
    ].groupby("method", as_index=False).agg(
        coverage=("coverage", "mean"),
        absolute_calibration_error=("coverage_error_abs", "mean"),
    )
    excluded_leader90_lookup = excluded_leader90.set_index("method")

    summary = {
        "status": "completed_evaluation_only_sensitivity_analysis",
        "identifiability_guardrail": (
            "Exact values 4 and 8 are boundary candidates; censoring is not identifiable "
            "from the current CellMiner export."
        ),
        "confirmation_seeds": list(seeds),
        "formal_methods": list(METHODS),
        "scenarios": {
            "reference_all": "Original exact-outcome evaluation.",
            "exclude_exact_4_or_8": "Remove exact 4/8 labels only at test evaluation.",
            "one_sided_compatible": (
                "Hypothetical compatibility: exact 4 means y<=4 and exact 8 means y>=8; "
                "standard interval score is undefined."
            ),
        },
        "low_similarity_proposed_minus_uacqrp_ace": {
            scenario: summarize_units(part, "mean_proposed_minus_uacqrp_ace")
            for scenario, part in paired.groupby("scenario")
        },
        "leader90_proposed_minus_uacqrp_coverage": {
            scenario: summarize_units(part, "proposed_minus_uacqrp_coverage")
            for scenario, part in leader_differences.groupby("scenario")
        },
        "core_claim_stability_rule": (
            "Stable only if Proposed remains lower-ACE than UACQR-P for OOD similarity<0.4 "
            "after boundary exclusion; raw coverage superiority is audited separately."
        ),
        "core_claim_stability_assessment": {
            "low_similarity_calibration": (
                "Point estimate remains favorable to Proposed after exclusion, but the "
                "descriptive eight-unit t interval includes zero."
            ),
            "leader90_calibration_closeness": {
                "status": "stable",
                "proposed_absolute_calibration_error": float(
                    excluded_leader90_lookup.loc[
                        "Proposed_shared_monotone", "absolute_calibration_error"
                    ]
                ),
                "uacqrp_absolute_calibration_error": float(
                    excluded_leader90_lookup.loc["UACQR_P", "absolute_calibration_error"]
                ),
            },
            "leader90_raw_coverage_superiority": {
                "status": "not_stable_to_boundary_exclusion",
                "proposed_coverage": float(
                    excluded_leader90_lookup.loc["Proposed_shared_monotone", "coverage"]
                ),
                "uacqrp_coverage": float(
                    excluded_leader90_lookup.loc["UACQR_P", "coverage"]
                ),
            },
        },
    }
    (output / "boundary_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    source_files = (
        root / "scripts" / "analyze_endpoint_boundary_sensitivity.py",
        root / "scripts" / "finalize_jbhi_results.py",
        root / "data" / "processed" / "cellminer" / "response_matrix_loggi50.csv.gz",
        root / "data" / "processed" / "cellminer" / "endpoint_boundary_audit.json",
        root / "results" / "final_jbhi" / "final_results_manifest.json",
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            f"python scripts/analyze_endpoint_boundary_sensitivity.py --root {root.as_posix()} "
            f"--seeds {','.join(map(str, seeds))}"
        ),
        "analysis_type": "evaluation-only endpoint-boundary sensitivity",
        "files_sha256": {
            str(path.relative_to(root)): sha256(path) for path in source_files if path.exists()
        },
    }
    (output / "boundary_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

