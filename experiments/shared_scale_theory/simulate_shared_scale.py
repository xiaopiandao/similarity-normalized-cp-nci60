"""Simulate shared versus task-specific monotone conformal scales.

This experiment mirrors the manuscript implementation: task residuals are first
normalized by task-wise validation medians, a compound-level median difficulty
is formed, and one decreasing isotonic scale is learned from those compound
summaries.  Tasks within a compound may be correlated; compounds, not response
cells, are the primary independent units.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.evaluate_phase1b_similarity import (
    fit_similarity_scale,
    predict_similarity_scale,
)
from scripts.evaluate_phase5_extensions import (
    fit_per_cell_monotone_scale,
    predict_per_cell_monotone_scale,
    scaled_intervals,
)


METHODS = ("global", "shared_monotone", "per_cell_monotone", "partial_blend_025")
METRICS = (
    "scale_log_mae",
    "low_similarity_scale_log_mae",
    "coverage",
    "macro_cell_ace",
    "q90_cell_ace",
    "low_similarity_macro_cell_ace",
    "max_similarity_bin_gap",
    "mean_width",
)


@dataclass(frozen=True)
class Scenario:
    n_validation: int
    n_tasks: int
    correlation: float
    heterogeneity: float
    missing_rate: float


@dataclass(frozen=True)
class SimulationConfig:
    n_calibration: int = 600
    n_test: int = 2000
    alpha: float = 0.10
    base_scale_slope: float = 1.20
    partial_task_weight: float = 0.25
    heterogeneity_mode: str = "additive"


@dataclass(frozen=True)
class TaskParameters:
    levels: np.ndarray
    shape_offsets: np.ndarray
    observation_probabilities: np.ndarray


def parse_number_list(value: str, cast) -> list:
    """Parse a comma-delimited CLI grid while rejecting empty grids."""
    parsed = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Grid cannot be empty")
    return parsed


def _rescale_probabilities(raw: np.ndarray, target_mean: float) -> np.ndarray:
    """Scale heterogeneous observation probabilities to an approximate mean."""
    probabilities = np.asarray(raw, dtype=float)
    for _ in range(8):
        current = float(probabilities.mean())
        if current <= 0:
            break
        probabilities = np.clip(probabilities * target_mean / current, 0.08, 1.0)
    return probabilities


def make_task_parameters(
    rng: np.random.Generator,
    scenario: Scenario,
    config: SimulationConfig | None = None,
) -> TaskParameters:
    if config is None:
        config = SimulationConfig()
    levels = np.exp(rng.normal(0.0, 0.35, scenario.n_tasks))
    latent_offsets = rng.normal(0.0, scenario.heterogeneity, scenario.n_tasks)
    latent_offsets -= float(latent_offsets.mean())
    if config.heterogeneity_mode == "additive":
        offsets = latent_offsets
    elif config.heterogeneity_mode == "log_slope":
        # Every task remains monotone decreasing, while slope ratios can vary
        # substantially at high heterogeneity.
        task_slopes = config.base_scale_slope * np.exp(latent_offsets)
        offsets = task_slopes - config.base_scale_slope
    else:
        raise ValueError(f"Unknown heterogeneity mode: {config.heterogeneity_mode}")
    if scenario.missing_rate <= 0:
        observation = np.ones(scenario.n_tasks, dtype=float)
    else:
        target = 1.0 - scenario.missing_rate
        raw = target * np.exp(rng.normal(0.0, 0.60, scenario.n_tasks))
        observation = _rescale_probabilities(raw, target)
    return TaskParameters(levels, offsets, observation)


def draw_similarity(
    rng: np.random.Generator, n_rows: int, role: str
) -> np.ndarray:
    """Draw source-like or target-like nearest-fit similarity coordinates."""
    if role == "source_calibration":
        return rng.beta(5.0, 2.0, n_rows)
    if role in {"validation", "test"}:
        return rng.beta(2.5, 3.5, n_rows)
    raise ValueError(f"Unknown role: {role}")


def relative_shape(
    similarity: np.ndarray,
    parameters: TaskParameters,
    base_scale_slope: float,
) -> np.ndarray:
    """Return the task-specific conditional residual scale before noise."""
    coordinate = 0.55 - np.asarray(similarity, dtype=float)[:, None]
    slopes = base_scale_slope + parameters.shape_offsets[None, :]
    return parameters.levels[None, :] * np.exp(coordinate * slopes)


def draw_responses(
    rng: np.random.Generator,
    similarity: np.ndarray,
    parameters: TaskParameters,
    scenario: Scenario,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate correlated, heteroscedastic responses and an observation mask."""
    n_rows = len(similarity)
    common = rng.normal(size=(n_rows, 1))
    task_noise = rng.normal(size=(n_rows, scenario.n_tasks))
    rho = float(np.clip(scenario.correlation, 0.0, 0.999999))
    standardized = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * task_noise
    truth = relative_shape(similarity, parameters, config.base_scale_slope) * standardized

    if scenario.missing_rate <= 0:
        observed = np.ones_like(truth, dtype=bool)
    else:
        # Missingness may depend on observed chemistry and task, but not on Y.
        chemistry_factor = 0.85 + 0.15 * similarity[:, None]
        probability = np.clip(
            parameters.observation_probabilities[None, :] * chemistry_factor,
            0.02,
            1.0,
        )
        observed = rng.random(size=truth.shape) < probability
        truth = truth.copy()
        truth[~observed] = np.nan
    return truth, observed


def true_normalized_task_scale(
    evaluation_similarity: np.ndarray,
    validation_similarity: np.ndarray,
    validation_mask: np.ndarray,
    parameters: TaskParameters,
    config: SimulationConfig,
) -> np.ndarray:
    """Normalize each oracle task shape on its observed validation chemistry."""
    raw_evaluation = relative_shape(evaluation_similarity, parameters, config.base_scale_slope)
    raw_validation = relative_shape(validation_similarity, parameters, config.base_scale_slope)
    normalizations = np.empty(raw_validation.shape[1], dtype=float)
    for task in range(raw_validation.shape[1]):
        observed = validation_mask[:, task]
        if not observed.any():
            normalizations[task] = float(np.median(raw_validation[:, task]))
        else:
            normalizations[task] = float(np.median(raw_validation[observed, task]))
    return raw_evaluation / normalizations[None, :]


def scale_error_metrics(
    estimated: np.ndarray,
    oracle: np.ndarray,
    evaluation_similarity: np.ndarray,
) -> dict[str, float]:
    if estimated.ndim == 1:
        estimated = np.repeat(estimated[:, None], oracle.shape[1], axis=1)
    log_error = np.abs(np.log(np.maximum(estimated, 1e-8)) - np.log(oracle))
    low = evaluation_similarity < 0.4
    return {
        "scale_log_mae": float(np.mean(log_error)),
        "low_similarity_scale_log_mae": float(np.mean(log_error[low])),
    }


def interval_metrics(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    similarity: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    observed = np.isfinite(truth)
    covered = observed & (truth >= lower) & (truth <= upper)
    target = 1.0 - alpha
    task_coverage = np.full(truth.shape[1], np.nan, dtype=float)
    low_task_coverage = np.full(truth.shape[1], np.nan, dtype=float)
    low_rows = similarity < 0.4
    for task in range(truth.shape[1]):
        task_mask = observed[:, task]
        if task_mask.any():
            task_coverage[task] = float(covered[task_mask, task].mean())
        low_mask = task_mask & low_rows
        if low_mask.any():
            low_task_coverage[task] = float(covered[low_mask, task].mean())

    bin_gaps: list[float] = []
    for left, right in zip(np.linspace(0.0, 1.0, 6)[:-1], np.linspace(0.0, 1.0, 6)[1:]):
        if right == 1.0:
            rows = (similarity >= left) & (similarity <= right)
        else:
            rows = (similarity >= left) & (similarity < right)
        bin_mask = observed & rows[:, None]
        if bin_mask.any():
            bin_gaps.append(abs(float(covered[bin_mask].mean()) - target))

    width = np.where(observed, upper - lower, np.nan)
    return {
        "coverage": float(covered[observed].mean()),
        "macro_cell_ace": float(np.nanmean(np.abs(task_coverage - target))),
        "q90_cell_ace": float(np.nanquantile(np.abs(task_coverage - target), 0.90)),
        "low_similarity_macro_cell_ace": float(
            np.nanmean(np.abs(low_task_coverage - target))
        ),
        "max_similarity_bin_gap": float(max(bin_gaps)) if bin_gaps else np.nan,
        "mean_width": float(np.nanmean(width)),
    }


def _estimated_scales(
    method: str,
    shared_model,
    per_cell_model,
    similarity: np.ndarray,
    partial_task_weight: float,
) -> np.ndarray:
    if method == "global":
        return np.ones(len(similarity), dtype=float)
    shared = predict_similarity_scale(shared_model, similarity)
    if method == "shared_monotone":
        return shared
    if per_cell_model is None:
        raise ValueError("Per-cell model was not estimable")
    per_cell = predict_per_cell_monotone_scale(per_cell_model, similarity)
    if method == "per_cell_monotone":
        return per_cell
    if method == "partial_blend_025":
        return np.power(shared[:, None], 1.0 - partial_task_weight) * np.power(
            per_cell, partial_task_weight
        )
    raise ValueError(f"Unknown method: {method}")


def run_one_replicate(
    scenario: Scenario,
    config: SimulationConfig,
    seed: int,
    replicate: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    parameters = make_task_parameters(rng, scenario, config)
    similarities = {
        "validation": draw_similarity(rng, scenario.n_validation, "validation"),
        "source_calibration": draw_similarity(
            rng, config.n_calibration, "source_calibration"
        ),
        "test": draw_similarity(rng, config.n_test, "test"),
    }
    validation_truth, validation_mask = draw_responses(
        rng, similarities["validation"], parameters, scenario, config
    )
    calibration_truth, _ = draw_responses(
        rng, similarities["source_calibration"], parameters, scenario, config
    )
    test_truth, _ = draw_responses(rng, similarities["test"], parameters, scenario, config)
    validation_prediction = np.zeros_like(validation_truth)
    calibration_prediction = np.zeros_like(calibration_truth)
    test_prediction = np.zeros_like(test_truth)

    shared_model = None
    per_cell_model = None
    shared_error = ""
    per_cell_error = ""
    with warnings.catch_warnings():
        # All-missing compound rows are permitted and excluded by the fitters.
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        try:
            shared_model, _, _ = fit_similarity_scale(
                similarities["validation"], validation_truth, validation_prediction
            )
        except Exception as exc:  # Recorded rather than retried by design.
            shared_error = f"{type(exc).__name__}: {exc}"
        try:
            per_cell_model, _ = fit_per_cell_monotone_scale(
                similarities["validation"], validation_truth, validation_prediction
            )
        except Exception as exc:  # Recorded rather than retried by design.
            per_cell_error = f"{type(exc).__name__}: {exc}"

    evaluation_similarity = np.linspace(0.02, 0.98, 129)
    oracle_scale = true_normalized_task_scale(
        evaluation_similarity,
        similarities["validation"],
        validation_mask,
        parameters,
        config,
    )

    records: list[dict] = []
    scenario_fields = asdict(scenario)
    for method in METHODS:
        requires_shared = method != "global"
        requires_per_cell = method in {"per_cell_monotone", "partial_blend_025"}
        failure = shared_error if requires_shared and shared_model is None else ""
        if requires_per_cell and per_cell_model is None:
            failure = per_cell_error or "Per-cell model was not estimable"
        base = {
            **scenario_fields,
            "replicate": int(replicate),
            "seed": int(seed),
            "method": method,
            "status": "failed" if failure else "estimated",
            "error": failure,
            "mean_observation_probability": float(
                parameters.observation_probabilities.mean()
            ),
            "min_observation_probability": float(
                parameters.observation_probabilities.min()
            ),
            "observed_validation_fraction": float(validation_mask.mean()),
        }
        if failure:
            records.append({**base, **{metric: np.nan for metric in METRICS}})
            continue

        evaluation_scale = _estimated_scales(
            method,
            shared_model,
            per_cell_model,
            evaluation_similarity,
            config.partial_task_weight,
        )
        calibration_scale = _estimated_scales(
            method,
            shared_model,
            per_cell_model,
            similarities["source_calibration"],
            config.partial_task_weight,
        )
        test_scale = _estimated_scales(
            method,
            shared_model,
            per_cell_model,
            similarities["test"],
            config.partial_task_weight,
        )
        lower, upper = scaled_intervals(
            calibration_truth,
            calibration_prediction,
            test_prediction,
            calibration_scale,
            test_scale,
            config.alpha,
        )
        records.append(
            {
                **base,
                **scale_error_metrics(
                    evaluation_scale, oracle_scale, evaluation_similarity
                ),
                **interval_metrics(
                    test_truth,
                    lower,
                    upper,
                    similarities["test"],
                    config.alpha,
                ),
            }
        )
    return records


def summarize_scenarios(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "n_validation",
        "n_tasks",
        "correlation",
        "heterogeneity",
        "missing_rate",
        "method",
    ]
    rows: list[dict] = []
    for values, part in raw.groupby(keys, dropna=False):
        row = dict(zip(keys, values))
        estimated = part[part["status"] == "estimated"]
        row["attempted_repeats"] = int(len(part))
        row["estimated_repeats"] = int(len(estimated))
        row["estimation_rate"] = float(len(estimated) / len(part))
        for metric in METRICS:
            values_array = estimated[metric].dropna().to_numpy(dtype=float)
            count = len(values_array)
            mean = float(np.mean(values_array)) if count else np.nan
            sd = float(np.std(values_array, ddof=1)) if count > 1 else np.nan
            se = float(sd / np.sqrt(count)) if count > 1 else np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_se"] = se
            row[f"{metric}_ci_low"] = mean - 1.96 * se if count > 1 else np.nan
            row[f"{metric}_ci_high"] = mean + 1.96 * se if count > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def paired_contrasts(raw: pd.DataFrame) -> pd.DataFrame:
    index = [
        "n_validation",
        "n_tasks",
        "correlation",
        "heterogeneity",
        "missing_rate",
        "replicate",
        "seed",
    ]
    comparable = raw[
        raw["method"].isin(["shared_monotone", "per_cell_monotone"])
        & (raw["status"] == "estimated")
    ]
    rows: list[pd.DataFrame] = []
    for metric in METRICS:
        wide = comparable.pivot(index=index, columns="method", values=metric).dropna()
        if wide.empty:
            continue
        difference = (
            wide["shared_monotone"] - wide["per_cell_monotone"]
        ).rename(f"shared_minus_per_cell_{metric}")
        rows.append(difference.to_frame())
    if not rows:
        return pd.DataFrame(columns=index)
    return pd.concat(rows, axis=1).reset_index()


def evaluate_decision_gates(
    raw: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame
) -> dict:
    scenario_keys = [
        "n_validation",
        "n_tasks",
        "correlation",
        "heterogeneity",
        "missing_rate",
    ]
    low_heterogeneity = paired[
        (paired["heterogeneity"] <= 0.2) & (paired["n_validation"] <= 250)
    ]
    scenario_difference = (
        low_heterogeneity.groupby(scenario_keys)[
            "shared_minus_per_cell_scale_log_mae"
        ]
        .mean()
        .dropna()
    )
    winning_fraction = float((scenario_difference < 0).mean()) if len(scenario_difference) else np.nan
    gate1 = bool(np.isfinite(winning_fraction) and winning_fraction >= 0.75)

    ace_difference = low_heterogeneity[
        "shared_minus_per_cell_macro_cell_ace"
    ].dropna()
    mean_ace_difference = float(ace_difference.mean()) if len(ace_difference) else np.nan
    gate2 = bool(np.isfinite(mean_ace_difference) and mean_ace_difference <= 0.005)

    advantage = (
        paired.assign(
            shared_advantage=lambda frame: -frame[
                "shared_minus_per_cell_scale_log_mae"
            ]
        )
        .groupby(scenario_keys)["shared_advantage"]
        .mean()
        .unstack("correlation")
    )
    if 0.0 in advantage.columns and 0.5 in advantage.columns:
        correlation_ok = advantage[0.5] <= advantage[0.0] + 0.01
        correlation_consistency = float(correlation_ok.mean())
    else:
        correlation_consistency = np.nan
    gate3 = bool(
        np.isfinite(correlation_consistency) and correlation_consistency >= (2.0 / 3.0)
    )

    highest_heterogeneity = float(paired["heterogeneity"].max())
    largest_validation = int(paired["n_validation"].max())
    high_heterogeneity = paired[
        (paired["heterogeneity"] == highest_heterogeneity)
        & (paired["n_validation"] == largest_validation)
    ]
    high_scenario_difference = (
        high_heterogeneity.groupby(scenario_keys)[
            "shared_minus_per_cell_scale_log_mae"
        ]
        .mean()
        .dropna()
    )
    per_cell_winning_fraction = (
        float((high_scenario_difference > 0).mean())
        if len(high_scenario_difference)
        else np.nan
    )
    gate4 = bool(
        np.isfinite(per_cell_winning_fraction) and per_cell_winning_fraction >= 0.50
    )

    stability = summary[
        (summary["n_tasks"] == 60)
        & (summary["n_validation"] == 50)
        & (summary["missing_rate"] == 0.3)
        & summary["method"].isin(["shared_monotone", "per_cell_monotone"])
    ]
    rates = stability.groupby("method")["estimation_rate"].mean().to_dict()
    shared_rate = float(rates.get("shared_monotone", np.nan))
    per_cell_rate = float(rates.get("per_cell_monotone", np.nan))
    gate5 = bool(
        np.isfinite(shared_rate)
        and np.isfinite(per_cell_rate)
        and shared_rate > per_cell_rate
    )

    gates = {
        "gate_1_low_heterogeneity_scale_efficiency": {
            "passed": gate1,
            "threshold": "shared wins >= 75% of comparable scenario means",
            "value": winning_fraction,
            "n_scenarios": int(len(scenario_difference)),
        },
        "gate_2_macro_ace_noninferiority": {
            "passed": gate2,
            "threshold": "mean shared-minus-per-cell macro ACE <= 0.005",
            "value": mean_ace_difference,
            "n_paired_replicates": int(len(ace_difference)),
        },
        "gate_3_correlation_mechanism": {
            "passed": gate3,
            "threshold": "rho=0.5 advantage <= rho=0 advantage + 0.01 in >= 2/3 matches",
            "value": correlation_consistency,
            "n_matches": int(len(advantage)),
        },
        "gate_4_heterogeneity_boundary": {
            "passed": gate4,
            "threshold": (
                "per-cell scale wins >= 50% of scenario means at the highest "
                "heterogeneity and largest validation size"
            ),
            "value": per_cell_winning_fraction,
            "n_scenarios": int(len(high_scenario_difference)),
            "heterogeneity_level": highest_heterogeneity,
            "n_validation": largest_validation,
        },
        "gate_5_missingness_estimability": {
            "passed": gate5,
            "threshold": "shared estimation rate > per-cell rate at J=60, n_val=50, missing=0.3",
            "shared_rate": shared_rate,
            "per_cell_rate": per_cell_rate,
        },
    }
    gates["required_go_gate"] = {
        "passed": bool(gate1 and gate2 and gate5),
        "rule": "gates 1, 2, and 5 must all pass",
    }
    return gates


def make_validation_report(
    output: Path,
    command: str,
    duration_seconds: float,
    raw: pd.DataFrame,
    gates: dict,
) -> None:
    estimated = raw[raw["status"] == "estimated"]
    lines = [
        "# Shared-Scale Theory Simulation: Validation Report",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-08-16",
        "- Verification Status: UNVERIFIED",
        "- Version Label: shared_scale_theory_sim_v1",
        "",
        "## Experiment Result",
        "",
        "- **ID**: shared_scale_theory_20260816",
        "- **Type**: simulation",
        "- **Status**: completed",
        f"- **Command**: `{command}`",
        "- **Working Directory**: `E:\\Manba-revise`",
        f"- **Duration**: {duration_seconds:.1f} seconds",
        "- **Exit Code**: 0",
        f"- **Attempted method fits**: {len(raw)}",
        f"- **Estimated method fits**: {len(estimated)}",
        "",
        "## Prespecified Decision Gates",
        "",
        "| Gate | Result | Observed | Threshold |",
        "|---|---|---:|---|",
    ]
    for key, item in gates.items():
        if key == "required_go_gate":
            continue
        observed = item.get("value")
        if observed is None:
            observed = (
                f"shared={item.get('shared_rate', float('nan')):.3f}; "
                f"per-cell={item.get('per_cell_rate', float('nan')):.3f}"
            )
        elif isinstance(observed, float):
            observed = f"{observed:.6f}"
        lines.append(
            f"| {key} | {'PASS' if item['passed'] else 'FAIL'} | {observed} | {item['threshold']} |"
        )
    go = gates["required_go_gate"]["passed"]
    lines.extend(
        [
            "",
            "## Stage Decision",
            "",
            (
                "- **GO**: required gates 1, 2, and 5 passed. Proceed to formal proof development and a confirmatory real-data uncertainty analysis."
                if go
                else "- **NO-GO FOR THEORETICAL CLAIM AS CURRENTLY FORMULATED**: at least one required gate failed. Retain the empirical method contribution and revise the theory before manuscript insertion."
            ),
            "- Gates 3 and 4 are mechanism checks. Their failure does not automatically reject the empirical method, but it invalidates the proposed effective-task or heterogeneity interpretation.",
            "- This synthetic experiment is not external validation and does not establish exact coverage under scaffold, leader-cluster, or assay shift.",
            "",
            "## Output Files",
            "",
            "- `raw_results.csv`",
            "- `scenario_summary.csv`",
            "- `paired_contrasts.csv`",
            "- `decision_gates.json`",
            "- `VALIDATION_REPORT.md`",
            "",
        ]
    )
    (output / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def scenario_grid(
    n_validation: Iterable[int],
    n_tasks: Iterable[int],
    correlations: Iterable[float],
    heterogeneity: Iterable[float],
    missing_rates: Iterable[float],
) -> list[Scenario]:
    return [
        Scenario(*values)
        for values in product(
            n_validation, n_tasks, correlations, heterogeneity, missing_rates
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/shared_scale_theory_20260816")
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--n-validation", default="50,250,1000")
    parser.add_argument("--n-tasks", default="5,15,60")
    parser.add_argument("--correlations", default="0.0,0.5")
    parser.add_argument("--heterogeneity", default="0.0,0.2,0.5")
    parser.add_argument("--missing-rates", default="0.0,0.3")
    parser.add_argument("--n-calibration", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument(
        "--heterogeneity-mode",
        choices=("additive", "log_slope"),
        default="additive",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    validation_grid = parse_number_list(args.n_validation, int)
    task_grid = parse_number_list(args.n_tasks, int)
    correlation_grid = parse_number_list(args.correlations, float)
    heterogeneity_grid = parse_number_list(args.heterogeneity, float)
    missing_grid = parse_number_list(args.missing_rates, float)
    scenarios = scenario_grid(
        validation_grid,
        task_grid,
        correlation_grid,
        heterogeneity_grid,
        missing_grid,
    )
    config = SimulationConfig(
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        heterogeneity_mode=args.heterogeneity_mode,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    seed_sequence = np.random.SeedSequence(args.seed)
    child_sequences = seed_sequence.spawn(len(scenarios) * args.repeats)

    records: list[dict] = []
    child_index = 0
    for scenario in scenarios:
        for replicate in range(args.repeats):
            child_seed = int(child_sequences[child_index].generate_state(1)[0])
            child_index += 1
            records.extend(
                run_one_replicate(scenario, config, child_seed, replicate)
            )

    raw = pd.DataFrame(records)
    summary = summarize_scenarios(raw)
    paired = paired_contrasts(raw)
    gates = evaluate_decision_gates(raw, summary, paired)
    raw.to_csv(args.output / "raw_results.csv", index=False)
    summary.to_csv(args.output / "scenario_summary.csv", index=False)
    paired.to_csv(args.output / "paired_contrasts.csv", index=False)
    (args.output / "decision_gates.json").write_text(
        json.dumps(gates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    duration = time.perf_counter() - start
    command = "python " + " ".join(sys.argv)
    make_validation_report(args.output, command, duration, raw, gates)
    print(json.dumps({"duration_seconds": duration, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
