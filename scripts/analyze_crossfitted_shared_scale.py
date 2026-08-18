"""Evaluate proof-friendly cross-fitted task normalization on frozen predictions.

The protocol is frozen in
``docs/交叉拟合任务尺度_先验分析与实验协议_20260816.md``.  Validation
folds are assigned without outcomes, keep Morgan fingerprint groups intact,
and are stratified over nearest-fit similarity.  This is an exploratory
sensitivity analysis because the confirmation test labels were used previously.
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
    from scripts.analyze_scale_sample_efficiency import (
        metrics_for_intervals,
        stratified_subsample,
    )
    from scripts.evaluate_phase1a_conformal import EvaluationData
    from scripts.evaluate_phase3_local_and_screening import sim
    from scripts.evaluate_phase5_extensions import (
        fit_per_cell_monotone_scale,
        fit_similarity_scale,
        predict_per_cell_monotone_scale,
        predict_similarity_scale,
        scaled_intervals,
    )
except ModuleNotFoundError:
    from analyze_scale_sample_efficiency import (  # type: ignore[no-redef]
        metrics_for_intervals,
        stratified_subsample,
    )
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]
    from evaluate_phase3_local_and_screening import sim  # type: ignore[no-redef]
    from evaluate_phase5_extensions import (  # type: ignore[no-redef]
        fit_per_cell_monotone_scale,
        fit_similarity_scale,
        predict_per_cell_monotone_scale,
        predict_similarity_scale,
        scaled_intervals,
    )


FAMILIES = ("scaffold", "leader_cluster")
METHODS = (
    "plugin_shared",
    "single_split_shared",
    "crossfit2_shared",
    "per_cell_monotone",
)
PRIMARY_ALPHA = 0.10
DEFAULT_OUTPUT = "results/crossfit_shared_scale_20260816"


@dataclass
class ExternalNormalizerScale:
    model: IsotonicRegression
    normalization: float


@dataclass
class CrossFittedScale:
    components: tuple[ExternalNormalizerScale, ExternalNormalizerScale]
    ensemble_normalization: float


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    result = stats.spearmanr(left, right, nan_policy="omit")
    return float(result.statistic)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_task_normalizers(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, dict]:
    residual = np.abs(truth - prediction)
    label_counts = np.isfinite(residual).sum(axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        normalizers = np.nanmedian(residual, axis=0)
    positive = normalizers[np.isfinite(normalizers) & (normalizers > 0)]
    if not len(positive):
        raise ValueError("No positive task residual median in normalizer fold")
    fallback = float(np.median(positive))
    invalid = ~(np.isfinite(normalizers) & (normalizers > 0))
    normalizers = np.where(invalid, fallback, normalizers)
    return normalizers, {
        "normalizer_min_labels": int(label_counts.min()),
        "normalizer_cells_below_20": int((label_counts < 20).sum()),
        "normalizer_fallback_cells": int(invalid.sum()),
        "normalizer_median": float(np.median(normalizers)),
    }


def fit_scale_with_external_normalizer(
    similarity: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    task_normalizers: np.ndarray,
) -> tuple[ExternalNormalizerScale, dict]:
    residual = np.abs(truth - prediction)
    if residual.shape[1] != len(task_normalizers):
        raise ValueError("Task normalizer count does not match response columns")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        difficulty = np.nanmedian(residual / task_normalizers[None, :], axis=1)
    valid = np.isfinite(similarity) & np.isfinite(difficulty)
    if valid.sum() < 20:
        raise ValueError("Insufficient scale-fold compounds with finite difficulty")
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(similarity[valid], difficulty[valid])
    raw = np.maximum(model.predict(similarity[valid]), 1e-3)
    normalization = float(np.median(raw))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("External-normalizer scale normalization is invalid")
    fitted = raw / normalization
    return ExternalNormalizerScale(model, normalization), {
        "scale_valid_compounds": int(valid.sum()),
        "scale_min": float(fitted.min()),
        "scale_median": float(np.median(fitted)),
        "scale_max": float(fitted.max()),
        "difficulty_spearman": spearman_correlation(
            similarity[valid], difficulty[valid]
        ),
    }


def predict_external_scale(
    scale: ExternalNormalizerScale, similarity: np.ndarray
) -> np.ndarray:
    return np.maximum(scale.model.predict(similarity) / scale.normalization, 1e-3)


def group_stratified_twofold(
    similarity: np.ndarray,
    group_ids: np.ndarray,
    rng: np.random.Generator,
    strata: int = 5,
) -> np.ndarray:
    """Assign intact groups to two approximately balanced similarity-stratified folds."""
    frame = pd.DataFrame(
        {
            "row": np.arange(len(similarity), dtype=int),
            "similarity": np.asarray(similarity, dtype=float),
            "group": np.asarray(group_ids).astype(str),
        }
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="np.find_common_type is deprecated", category=DeprecationWarning
        )
        grouped = (
            frame.groupby("group", as_index=False)
            .agg(group_similarity=("similarity", "median"), group_size=("row", "size"))
            .sort_values("group")
            .reset_index(drop=True)
        )
    edges = np.unique(
        np.quantile(grouped["group_similarity"], np.linspace(0.0, 1.0, strata + 1))
    )
    grouped["stratum"] = np.digitize(
        grouped["group_similarity"], edges[1:-1], right=True
    )
    assignment: dict[str, int] = {}
    total_counts = np.zeros(2, dtype=int)
    for stratum in sorted(grouped["stratum"].unique()):
        part = grouped[grouped["stratum"] == stratum].copy()
        order = rng.permutation(len(part))
        stratum_counts = np.zeros(2, dtype=int)
        for position in order:
            position = int(position)
            if stratum_counts[0] == stratum_counts[1]:
                fold = int(total_counts[1] < total_counts[0])
                if total_counts[0] == total_counts[1]:
                    fold = int(rng.integers(0, 2))
            else:
                fold = int(np.argmin(stratum_counts))
            assignment[str(part["group"].iat[position])] = fold
            count = int(part["group_size"].iat[position])
            stratum_counts[fold] += count
            total_counts[fold] += count
    folds = frame["group"].map(assignment).to_numpy(dtype=int)
    if set(np.unique(folds)) != {0, 1}:
        raise ValueError("Two-fold assignment did not produce both folds")
    return folds


def fit_crossfitted_scale(
    similarity: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    folds: np.ndarray,
) -> tuple[CrossFittedScale, ExternalNormalizerScale, dict]:
    components: list[ExternalNormalizerScale] = []
    diagnostics: dict[str, float | int] = {}
    for scale_fold in (0, 1):
        scale_mask = folds == scale_fold
        normalizer_mask = ~scale_mask
        normalizers, normalizer_diag = estimate_task_normalizers(
            truth[normalizer_mask], prediction[normalizer_mask]
        )
        component, scale_diag = fit_scale_with_external_normalizer(
            similarity[scale_mask],
            truth[scale_mask],
            prediction[scale_mask],
            normalizers,
        )
        components.append(component)
        diagnostics.update(
            {
                f"fold_{scale_fold}_compounds": int(scale_mask.sum()),
                f"fold_{scale_fold}_normalizer_compounds": int(normalizer_mask.sum()),
                **{
                    f"fold_{scale_fold}_{key}": value
                    for key, value in {**normalizer_diag, **scale_diag}.items()
                },
            }
        )
    component_tuple = (components[0], components[1])
    raw_ensemble = 0.5 * (
        predict_external_scale(component_tuple[0], similarity)
        + predict_external_scale(component_tuple[1], similarity)
    )
    ensemble_normalization = float(np.median(raw_ensemble))
    if not np.isfinite(ensemble_normalization) or ensemble_normalization <= 0:
        raise ValueError("Cross-fitted ensemble normalization is invalid")
    diagnostics["ensemble_normalization"] = ensemble_normalization
    diagnostics["ensemble_scale_min"] = float(
        (raw_ensemble / ensemble_normalization).min()
    )
    diagnostics["ensemble_scale_max"] = float(
        (raw_ensemble / ensemble_normalization).max()
    )
    return (
        CrossFittedScale(component_tuple, ensemble_normalization),
        component_tuple[0],
        diagnostics,
    )


def predict_crossfitted_scale(
    scale: CrossFittedScale, similarity: np.ndarray
) -> np.ndarray:
    raw = 0.5 * (
        predict_external_scale(scale.components[0], similarity)
        + predict_external_scale(scale.components[1], similarity)
    )
    return np.maximum(raw / scale.ensemble_normalization, 1e-3)


def scale_agreement(
    plugin: np.ndarray,
    crossfit: np.ndarray,
    component_0: np.ndarray,
    component_1: np.ndarray,
) -> dict[str, float]:
    log_ratio = np.abs(
        np.log(np.maximum(crossfit, 1e-8)) - np.log(np.maximum(plugin, 1e-8))
    )
    component_log_ratio = np.abs(
        np.log(np.maximum(component_0, 1e-8))
        - np.log(np.maximum(component_1, 1e-8))
    )
    return {
        "plugin_crossfit_spearman": spearman_correlation(plugin, crossfit),
        "plugin_crossfit_median_abs_log_ratio": float(np.median(log_ratio)),
        "plugin_crossfit_q95_abs_log_ratio": float(np.quantile(log_ratio, 0.95)),
        "component_spearman": spearman_correlation(component_0, component_1),
        "component_median_abs_log_ratio": float(np.median(component_log_ratio)),
        "component_q95_abs_log_ratio": float(np.quantile(component_log_ratio, 0.95)),
    }


def mean_t_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, np.nan, np.nan, np.nan
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return mean, sd, mean - half, mean + half


def _unit_metrics(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "coverage",
        "coverage_error_abs",
        "mean_width",
        "macro_cell_ace",
        "q90_cell_ace",
        "cell_coverage_gap_sd",
    )
    keys = ["family", "seed", "validation_n", "method", "scope"]
    rows: list[dict] = []
    for values, part in repeat_metrics.groupby(keys, dropna=False):
        estimated = part[part["status"] == "estimated"]
        row = dict(zip(keys, values))
        row["attempted_repeats"] = int(len(part))
        row["estimated_repeats"] = int(len(estimated))
        row["estimation_rate"] = float(len(estimated) / len(part))
        for metric in metrics:
            row[metric] = float(estimated[metric].mean()) if len(estimated) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_units(units: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "coverage_error_abs",
        "macro_cell_ace",
        "q90_cell_ace",
        "cell_coverage_gap_sd",
        "mean_width",
    )
    index = ["family", "seed", "validation_n", "scope"]
    source = units[units["method"].isin(["plugin_shared", "crossfit2_shared"])]
    frames: list[pd.DataFrame] = []
    for metric in metrics:
        wide = source.pivot(index=index, columns="method", values=metric)
        wide[f"crossfit_minus_plugin_{metric}"] = (
            wide["crossfit2_shared"] - wide["plugin_shared"]
        )
        if metric == "mean_width":
            wide["crossfit_div_plugin_mean_width"] = (
                wide["crossfit2_shared"] / wide["plugin_shared"]
            )
        frames.append(
            wide[
                [
                    "plugin_shared",
                    "crossfit2_shared",
                    f"crossfit_minus_plugin_{metric}",
                ]
                + (["crossfit_div_plugin_mean_width"] if metric == "mean_width" else [])
            ].rename(
                columns={
                    "plugin_shared": f"plugin_{metric}",
                    "crossfit2_shared": f"crossfit_{metric}",
                }
            )
        )
    return pd.concat(frames, axis=1).reset_index()


def evaluate_gates(
    repeat_metrics: pd.DataFrame,
    paired: pd.DataFrame,
    agreement: pd.DataFrame,
) -> dict:
    full_n = int(repeat_metrics["validation_n"].max())
    full = paired[(paired["validation_n"] == full_n) & (paired["scope"] == "all")]

    macro_mean, _, macro_low, macro_high = mean_t_interval(
        full["crossfit_minus_plugin_macro_cell_ace"].to_numpy(float)
    )
    g1 = bool(macro_mean <= 0.002 and macro_high <= 0.005)

    q90_mean, _, q90_low, q90_high = mean_t_interval(
        full["crossfit_minus_plugin_q90_cell_ace"].to_numpy(float)
    )
    g2 = bool(q90_mean <= 0.003 and q90_high <= 0.007)

    width_ratio = full["crossfit_div_plugin_mean_width"].dropna().to_numpy(float)
    width_median = float(np.median(width_ratio)) if len(width_ratio) else np.nan
    width_max = float(np.max(width_ratio)) if len(width_ratio) else np.nan
    g3 = bool(width_median <= 1.05 and width_max <= 1.10)

    full_agreement = (
        agreement[agreement["validation_n"] == full_n]
        .groupby(["family", "seed"], as_index=False)
        .agg(
            plugin_crossfit_spearman=("plugin_crossfit_spearman", "mean"),
            plugin_crossfit_median_abs_log_ratio=(
                "plugin_crossfit_median_abs_log_ratio",
                "mean",
            ),
        )
    )
    scale_spearman_median = float(
        full_agreement["plugin_crossfit_spearman"].median()
    )
    scale_log_ratio_median = float(
        full_agreement["plugin_crossfit_median_abs_log_ratio"].median()
    )
    g4 = bool(scale_spearman_median >= 0.98 and scale_log_ratio_median <= 0.05)

    mid_repeat = repeat_metrics[
        (repeat_metrics["validation_n"] >= 250)
        & (repeat_metrics["method"] == "crossfit2_shared")
        & (repeat_metrics["scope"] == "all")
    ]
    mid_failure_count = int((mid_repeat["status"] != "estimated").sum())
    mid_paired = paired[
        (paired["validation_n"] >= 250) & (paired["scope"] == "all")
    ]
    mid_macro_difference = float(
        mid_paired["crossfit_minus_plugin_macro_cell_ace"].mean()
    )
    g5 = bool(mid_failure_count == 0 and mid_macro_difference <= 0.003)

    small = repeat_metrics[
        repeat_metrics["validation_n"].isin([50, 100])
        & (repeat_metrics["method"] == "crossfit2_shared")
        & (repeat_metrics["scope"] == "all")
    ]
    small_summary = {
        str(int(n)): {
            "estimation_rate": float((part["status"] == "estimated").mean()),
            "attempted": int(len(part)),
        }
        for n, part in small.groupby("validation_n")
    }

    all_hard = bool(g1 and g2 and g3 and g4 and g5)
    full_only = bool(g1 and g2 and g3 and g4)
    if all_hard:
        decision = "crossfit_candidate_for_proof_not_manuscript_replacement"
    elif full_only:
        decision = "crossfit_full_validation_sensitivity_only"
    else:
        decision = "retain_plugin_and_prove_plugin_perturbation"
    return {
        "G1_full_macro_ace": {
            "passed": g1,
            "mean_difference": macro_mean,
            "t95_low": macro_low,
            "t95_high": macro_high,
            "threshold": "mean <= 0.002 and t95 upper <= 0.005",
            "n_units": int(len(full)),
        },
        "G2_full_q90_cell_ace": {
            "passed": g2,
            "mean_difference": q90_mean,
            "t95_low": q90_low,
            "t95_high": q90_high,
            "threshold": "mean <= 0.003 and t95 upper <= 0.007",
            "n_units": int(len(full)),
        },
        "G3_full_width": {
            "passed": g3,
            "median_ratio": width_median,
            "max_ratio": width_max,
            "threshold": "median <= 1.05 and max <= 1.10",
        },
        "G4_full_scale_agreement": {
            "passed": g4,
            "median_spearman": scale_spearman_median,
            "median_abs_log_ratio": scale_log_ratio_median,
            "threshold": "median Spearman >= 0.98 and median abs log ratio <= 0.05",
            "n_units": int(len(full_agreement)),
        },
        "G5_mid_sample_usability": {
            "passed": g5,
            "failure_count": mid_failure_count,
            "mean_macro_ace_difference": mid_macro_difference,
            "threshold": "zero failures and mean macro ACE difference <= 0.003 for n >= 250",
        },
        "G6_small_sample_diagnostic": {
            "hard_gate": False,
            "by_validation_n": small_summary,
        },
        "hard_gates_all_passed": all_hard,
        "full_validation_gates_passed": full_only,
        "decision": decision,
    }


def make_report(
    output: Path,
    duration: float,
    command: str,
    gates: dict,
    repeat_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Cross-Fitted Shared Scale: Validation Report",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-08-16",
        "- Verification Status: UNVERIFIED",
        "- Version Label: crossfit_normalizer_result_v1",
        "",
        "## Experiment Result",
        "",
        "- **ID**: crossfit_shared_scale_20260816",
        "- **Type**: analysis",
        "- **Status**: completed",
        f"- **Command**: `{command}`",
        "- **Working Directory**: project root supplied with `--root`",
        f"- **Duration**: {duration:.1f} seconds",
        "- **Exit Code**: 0",
        f"- **Attempted method-scope records**: {len(repeat_metrics)}",
        "",
        "## Prespecified Gates",
        "",
        "| Gate | Result | Key value | Threshold |",
        "|---|---|---|---|",
    ]
    for key in (
        "G1_full_macro_ace",
        "G2_full_q90_cell_ace",
        "G3_full_width",
        "G4_full_scale_agreement",
        "G5_mid_sample_usability",
    ):
        item = gates[key]
        if key in {"G1_full_macro_ace", "G2_full_q90_cell_ace"}:
            value = f"mean={item['mean_difference']:.6f}; upper={item['t95_high']:.6f}"
        elif key == "G3_full_width":
            value = f"median={item['median_ratio']:.6f}; max={item['max_ratio']:.6f}"
        elif key == "G4_full_scale_agreement":
            value = (
                f"rho={item['median_spearman']:.6f}; "
                f"log-ratio={item['median_abs_log_ratio']:.6f}"
            )
        else:
            value = (
                f"failures={item['failure_count']}; "
                f"ACE diff={item['mean_macro_ace_difference']:.6f}"
            )
        lines.append(
            f"| {key} | {'PASS' if item['passed'] else 'FAIL'} | {value} | {item['threshold']} |"
        )
    lines.extend(
        [
            "",
            "## Stage Decision",
            "",
            f"- **Decision code**: `{gates['decision']}`",
            "- This analysis is exploratory because the test labels were previously inspected.",
            "- No cross-fitting result is eligible for manuscript insertion without a new confirmation source.",
            "- Failed fits were retained as failures and were not automatically retried.",
            "",
        ]
    )
    (output / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", default="2,3,4,5")
    parser.add_argument("--sizes", default="50,100,250,500,1000,2524")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(args.root)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    sizes = tuple(int(value) for value in args.sizes.split(",") if value)
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    split_groups = pd.read_csv(
        root / "data" / "processed" / "cellminer" / "chemical_split_groups.csv",
        usecols=["nsc", "morgan_fp_group_id"],
    ).set_index("nsc")["morgan_fp_group_id"]
    data = EvaluationData(root)
    metric_rows: list[dict] = []
    agreement_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    start = time.perf_counter()

    for family_index, family in enumerate(FAMILIES):
        for seed in seeds:
            split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
            phase1_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
            phase3_dir = root / "results" / "phase3" / f"{family}_seed_{seed}"
            stored = np.load(phase1_dir / "mlp_predictions.npz")
            names = ("source_cal", "valid", "test")
            ids = {name: stored[f"nsc_{name}"].astype("int64") for name in names}
            prediction = {name: stored[f"pred_{name}"].astype(float) for name in names}
            truth = {name: data.truth(ids[name]) for name in names}
            fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
            similarity = {
                name: sim(phase3_dir, name, data, fit_ids, ids[name]) for name in names
            }
            valid_groups = split_groups.reindex(ids["valid"]).fillna(
                pd.Series(ids["valid"], index=ids["valid"]).map(lambda value: f"NSC::{value}")
            ).to_numpy()
            low_test = similarity["test"] < 0.4

            for n_requested in sizes:
                n = min(n_requested, len(ids["valid"]))
                for repeat in range(args.repeats):
                    rng = np.random.default_rng(
                        20260816
                        + family_index * 1_000_000
                        + seed * 10_000
                        + n * 10
                        + repeat
                    )
                    subset = stratified_subsample(similarity["valid"], n, rng)
                    subset_similarity = similarity["valid"][subset]
                    subset_truth = truth["valid"][subset]
                    subset_prediction = prediction["valid"][subset]
                    subset_groups = valid_groups[subset]
                    folds = group_stratified_twofold(
                        subset_similarity, subset_groups, rng
                    )

                    method_error = {method: "" for method in METHODS}
                    cal_scales: dict[str, np.ndarray] = {}
                    test_scales: dict[str, np.ndarray] = {}
                    crossfit_diag: dict = {}

                    try:
                        plugin_model, _, plugin_diag = fit_similarity_scale(
                            subset_similarity, subset_truth, subset_prediction
                        )
                        cal_scales["plugin_shared"] = predict_similarity_scale(
                            plugin_model, similarity["source_cal"]
                        )
                        test_scales["plugin_shared"] = predict_similarity_scale(
                            plugin_model, similarity["test"]
                        )
                    except Exception as exc:
                        method_error["plugin_shared"] = f"{type(exc).__name__}: {exc}"
                        plugin_diag = {}

                    try:
                        crossfit_model, single_model, crossfit_diag = fit_crossfitted_scale(
                            subset_similarity,
                            subset_truth,
                            subset_prediction,
                            folds,
                        )
                        for role in ("source_cal", "test"):
                            role_similarity = similarity[role]
                            cal_or_test = (
                                cal_scales if role == "source_cal" else test_scales
                            )
                            cal_or_test["single_split_shared"] = predict_external_scale(
                                single_model, role_similarity
                            )
                            cal_or_test["crossfit2_shared"] = predict_crossfitted_scale(
                                crossfit_model, role_similarity
                            )
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        method_error["single_split_shared"] = message
                        method_error["crossfit2_shared"] = message
                        crossfit_model = None
                        single_model = None

                    try:
                        per_cell_model, per_cell_diag = fit_per_cell_monotone_scale(
                            subset_similarity, subset_truth, subset_prediction
                        )
                        cal_scales["per_cell_monotone"] = predict_per_cell_monotone_scale(
                            per_cell_model, similarity["source_cal"]
                        )
                        test_scales["per_cell_monotone"] = predict_per_cell_monotone_scale(
                            per_cell_model, similarity["test"]
                        )
                    except Exception as exc:
                        method_error["per_cell_monotone"] = f"{type(exc).__name__}: {exc}"
                        per_cell_diag = {}

                    diagnostic_rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "validation_n": n,
                            "repeat": repeat,
                            "fold_0_compounds": int((folds == 0).sum()),
                            "fold_1_compounds": int((folds == 1).sum()),
                            "fold_0_groups": int(len(np.unique(subset_groups[folds == 0]))),
                            "fold_1_groups": int(len(np.unique(subset_groups[folds == 1]))),
                            "groups_split_across_folds": int(
                                pd.DataFrame({"group": subset_groups, "fold": folds})
                                .groupby("group")["fold"]
                                .nunique()
                                .gt(1)
                                .sum()
                            ),
                            **crossfit_diag,
                            **{
                                f"{method}_status": "failed" if error else "estimated"
                                for method, error in method_error.items()
                            },
                            **{
                                f"{method}_error": error
                                for method, error in method_error.items()
                            },
                        }
                    )

                    if (
                        not method_error["plugin_shared"]
                        and not method_error["crossfit2_shared"]
                        and crossfit_model is not None
                    ):
                        component_test = [
                            predict_external_scale(component, similarity["test"])
                            for component in crossfit_model.components
                        ]
                        agreement_rows.append(
                            {
                                "family": family,
                                "seed": seed,
                                "validation_n": n,
                                "repeat": repeat,
                                **scale_agreement(
                                    test_scales["plugin_shared"],
                                    test_scales["crossfit2_shared"],
                                    component_test[0],
                                    component_test[1],
                                ),
                            }
                        )

                    for method in METHODS:
                        error = method_error[method]
                        if error:
                            for scope in ("all", "similarity_lt_0.4"):
                                metric_rows.append(
                                    {
                                        "family": family,
                                        "seed": seed,
                                        "validation_n": n,
                                        "repeat": repeat,
                                        "method": method,
                                        "scope": scope,
                                        "status": "failed",
                                        "error": error,
                                    }
                                )
                            continue
                        lower, upper = scaled_intervals(
                            truth["source_cal"],
                            prediction["source_cal"],
                            prediction["test"],
                            cal_scales[method],
                            test_scales[method],
                            PRIMARY_ALPHA,
                        )
                        for scope, mask in (
                            ("all", np.ones(len(ids["test"]), dtype=bool)),
                            ("similarity_lt_0.4", low_test),
                        ):
                            metric_rows.append(
                                {
                                    "family": family,
                                    "seed": seed,
                                    "validation_n": n,
                                    "repeat": repeat,
                                    "method": method,
                                    "scope": scope,
                                    "status": "estimated",
                                    "error": "",
                                    **metrics_for_intervals(
                                        truth["test"][mask],
                                        lower[mask],
                                        upper[mask],
                                        PRIMARY_ALPHA,
                                    ),
                                }
                            )
                print(
                    f"crossfit family={family} seed={seed} n={n} repeats={args.repeats}",
                    flush=True,
                )

    repeat_metrics = pd.DataFrame(metric_rows)
    agreement = pd.DataFrame(agreement_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    units = _unit_metrics(repeat_metrics)
    paired = _paired_units(units)
    gates = evaluate_gates(repeat_metrics, paired, agreement)

    repeat_metrics.to_csv(output / "repeat_metrics.csv", index=False)
    agreement.to_csv(output / "scale_agreement.csv", index=False)
    diagnostics.to_csv(output / "fold_diagnostics.csv", index=False)
    units.to_csv(output / "family_seed_units.csv", index=False)
    paired.to_csv(output / "paired_contrasts.csv", index=False)
    (output / "decision_gates.json").write_text(
        json.dumps(gates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    duration = time.perf_counter() - start
    command = "python " + " ".join(sys.argv)
    make_report(output, duration, command, gates, repeat_metrics)
    manifest_sources = (
        root / "scripts" / "analyze_crossfitted_shared_scale.py",
        root / "scripts" / "evaluate_phase1b_similarity.py",
        root / "scripts" / "evaluate_phase5_extensions.py",
        root / "data" / "processed" / "cellminer" / "chemical_split_groups.csv",
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "duration_seconds": duration,
        "files_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in manifest_sources
            if path.exists()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"duration_seconds": duration, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()

