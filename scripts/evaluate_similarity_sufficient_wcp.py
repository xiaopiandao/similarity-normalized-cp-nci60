"""Explore similarity-sufficient weighted conformal prediction.

This diagnostic keeps the frozen point predictions and split roles unchanged.
It compares unweighted, high-dimensional domain-weighted, and one-dimensional
similarity-weighted conformal intervals, both with and without the existing
shared similarity normalization. Test responses are used only for evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

try:
    from scripts.evaluate_phase1a_conformal import (
        EvaluationData,
        finite_sample_quantile,
        weighted_thresholds,
    )
    from scripts.evaluate_phase1b_similarity import (
        fit_similarity_scale,
        predict_similarity_scale,
    )
except ModuleNotFoundError:
    from evaluate_phase1a_conformal import (  # type: ignore[no-redef]
        EvaluationData,
        finite_sample_quantile,
        weighted_thresholds,
    )
    from evaluate_phase1b_similarity import (  # type: ignore[no-redef]
        fit_similarity_scale,
        predict_similarity_scale,
    )


FAMILIES = ("random", "scaffold", "leader_cluster")
OOD_FAMILIES = ("scaffold", "leader_cluster")
ALPHAS = (0.05, 0.10, 0.15, 0.20)
METHODS = (
    "Global_CP",
    "HD_WCP",
    "Sim1D_WCP",
    "SNCP_unweighted",
    "SNCP_HD_WCP",
    "SNCP_Sim1D_WCP",
)
SIMILARITY_EDGES = np.asarray((0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001))
SIMILARITY_LABELS = (
    "[0,.3)",
    "[.3,.4)",
    "[.4,.5)",
    "[.5,.6)",
    "[.6,.7)",
    "[.7,.8)",
    "[.8,1]",
)


def load_similarity(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    observed_ids = frame["nsc"].to_numpy("int64")
    if not np.array_equal(observed_ids, expected_ids):
        raise ValueError(f"Similarity IDs do not match frozen predictions: {path}")
    values = frame["nearest_fit_tanimoto"].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite similarity values: {path}")
    return values


def estimate_similarity_domain_weights(
    source_similarity: np.ndarray,
    target_similarity: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate dQ_S/dP_S with cross-fitted one-dimensional classifiers."""
    source = np.asarray(source_similarity, dtype=float).reshape(-1, 1)
    target = np.asarray(target_similarity, dtype=float).reshape(-1, 1)
    features = np.vstack((source, target))
    labels = np.concatenate(
        (np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int))
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates: list[dict] = []
    probabilities_by_candidate: list[np.ndarray] = []
    for knots, c_value in itertools.product((3, 5, 7), (0.03, 0.3, 3.0)):
        classifier = Pipeline(
            steps=(
                (
                    "spline",
                    SplineTransformer(
                        n_knots=knots,
                        degree=3,
                        include_bias=False,
                        extrapolation="linear",
                    ),
                ),
                ("scale", StandardScaler()),
                (
                    "domain",
                    LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            )
        )
        probability = cross_val_predict(
            classifier,
            features,
            labels,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        probability = np.clip(probability, 1e-6, 1 - 1e-6)
        candidates.append(
            {
                "n_knots": knots,
                "C": c_value,
                "oof_auc": float(roc_auc_score(labels, probability)),
                "oof_log_loss": float(log_loss(labels, probability)),
                "oof_brier": float(brier_score_loss(labels, probability)),
            }
        )
        probabilities_by_candidate.append(probability)
    selected = int(np.argmin([row["oof_log_loss"] for row in candidates]))
    probability = probabilities_by_candidate[selected]
    prior_correction = len(source) / len(target)
    weights = probability / (1.0 - probability) * prior_correction
    source_weights = weights[: len(source)]
    target_weights = weights[len(source) :]
    ess = float(source_weights.sum() ** 2 / np.sum(source_weights**2))
    report = {
        "estimator": "5-fold cross-fitted cubic-spline logistic domain classifier on nearest-fit Tanimoto only",
        "selected_n_knots": candidates[selected]["n_knots"],
        "selected_C": candidates[selected]["C"],
        "oof_domain_auc": candidates[selected]["oof_auc"],
        "oof_log_loss": candidates[selected]["oof_log_loss"],
        "oof_brier": candidates[selected]["oof_brier"],
        "source_weight_min": float(source_weights.min()),
        "source_weight_median": float(np.median(source_weights)),
        "source_weight_mean": float(source_weights.mean()),
        "source_weight_q95": float(np.quantile(source_weights, 0.95)),
        "source_weight_q99": float(np.quantile(source_weights, 0.99)),
        "source_weight_max": float(source_weights.max()),
        "target_weight_median": float(np.median(target_weights)),
        "target_weight_q99": float(np.quantile(target_weights, 0.99)),
        "target_weight_max": float(target_weights.max()),
        "ess": ess,
        "ess_fraction": ess / len(source_weights),
        "regularization_grid": candidates,
    }
    return source_weights, target_weights, report


def standard_intervals(
    truth_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    cal_scale: np.ndarray,
    test_scale: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.abs(truth_cal - pred_cal) / cal_scale[:, None]
    quantiles = np.asarray(
        [finite_sample_quantile(scores[:, cell], alpha) for cell in range(scores.shape[1])]
    )
    radius = test_scale[:, None] * quantiles[None, :]
    return pred_test - radius, pred_test + radius


def weighted_scaled_intervals(
    truth_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    cal_scale: np.ndarray,
    test_scale: np.ndarray,
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.abs(truth_cal - pred_cal) / cal_scale[:, None]
    normalized_radius = np.column_stack(
        [
            weighted_thresholds(
                scores[:, cell], cal_weights, test_weights, alpha
            )
            for cell in range(scores.shape[1])
        ]
    )
    radius = test_scale[:, None] * normalized_radius
    return pred_test - radius, pred_test + radius


def interval_metrics(
    truth: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> dict:
    observed = np.isfinite(truth)
    finite = observed & np.isfinite(lower) & np.isfinite(upper)
    covered = finite & (truth >= lower) & (truth <= upper)
    width = upper - lower
    cell_coverages = []
    cell_aces = []
    for cell in range(truth.shape[1]):
        cell_observed = observed[:, cell]
        if not cell_observed.any():
            continue
        coverage = float(covered[:, cell].sum() / cell_observed.sum())
        cell_coverages.append(coverage)
        cell_aces.append(abs(coverage - (1.0 - alpha)))
    coverage = float(covered.sum() / observed.sum())
    return {
        "nominal_coverage": 1.0 - alpha,
        "coverage": coverage,
        "coverage_error_abs": abs(coverage - (1.0 - alpha)),
        "macro_cell_ace": float(np.mean(cell_aces)),
        "minimum_cell_coverage": float(np.min(cell_coverages)),
        "mean_width": float(width[finite].mean()) if finite.any() else np.nan,
        "median_width": float(np.median(width[finite])) if finite.any() else np.nan,
        "finite_interval_fraction": float(finite.sum() / observed.sum()),
        "n_compounds": int(truth.shape[0]),
        "n_labels": int(observed.sum()),
    }


def similarity_metrics(
    similarity: np.ndarray,
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> list[dict]:
    groups = np.digitize(similarity, SIMILARITY_EDGES[1:-1])
    rows = []
    for group, label in enumerate(SIMILARITY_LABELS):
        selected = groups == group
        if not selected.any():
            continue
        rows.append(
            {
                "similarity_bin": label,
                "bin_order": group,
                "mean_similarity": float(similarity[selected].mean()),
                **interval_metrics(truth[selected], lower[selected], upper[selected], alpha),
            }
        )
    return rows


def weighted_ks_distance(
    source: np.ndarray,
    target: np.ndarray,
    source_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
) -> float:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_weights = (
        np.ones(len(source), dtype=float)
        if source_weights is None
        else np.asarray(source_weights, dtype=float)
    )
    target_weights = (
        np.ones(len(target), dtype=float)
        if target_weights is None
        else np.asarray(target_weights, dtype=float)
    )
    source_valid = np.isfinite(source) & np.isfinite(source_weights) & (source_weights > 0)
    target_valid = np.isfinite(target) & np.isfinite(target_weights) & (target_weights > 0)
    source = source[source_valid]
    target = target[target_valid]
    source_weights = source_weights[source_valid]
    target_weights = target_weights[target_valid]
    if not len(source) or not len(target):
        return np.nan
    source_order = np.argsort(source, kind="mergesort")
    target_order = np.argsort(target, kind="mergesort")
    source = source[source_order]
    target = target[target_order]
    source_cdf = np.cumsum(source_weights[source_order]) / source_weights.sum()
    target_cdf = np.cumsum(target_weights[target_order]) / target_weights.sum()
    grid = np.unique(np.concatenate((source, target)))
    source_index = np.searchsorted(source, grid, side="right") - 1
    target_index = np.searchsorted(target, grid, side="right") - 1
    source_values = np.where(source_index >= 0, source_cdf[np.maximum(source_index, 0)], 0.0)
    target_values = np.where(target_index >= 0, target_cdf[np.maximum(target_index, 0)], 0.0)
    return float(np.max(np.abs(source_values - target_values)))


def weight_diagnostics(
    family: str,
    seed: int,
    estimator: str,
    cal_similarity: np.ndarray,
    test_similarity: np.ndarray,
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    report: dict,
) -> dict:
    ess = float(cal_weights.sum() ** 2 / np.sum(cal_weights**2))
    support_limit = (0.10 / 0.90) * cal_weights.sum()
    return {
        "family": family,
        "seed": seed,
        "estimator": estimator,
        "domain_auc": float(report.get("oof_domain_auc", np.nan)),
        "domain_log_loss": float(report.get("oof_log_loss", np.nan)),
        "ess": ess,
        "ess_fraction": ess / len(cal_weights),
        "cal_weight_min": float(cal_weights.min()),
        "cal_weight_median": float(np.median(cal_weights)),
        "cal_weight_q99": float(np.quantile(cal_weights, 0.99)),
        "cal_weight_max": float(cal_weights.max()),
        "test_weight_q99": float(np.quantile(test_weights, 0.99)),
        "test_weight_max": float(test_weights.max()),
        "support_failure_fraction_alpha_0_10": float((test_weights > support_limit).mean()),
        "weight_similarity_spearman": float(
            pd.Series(cal_weights).corr(pd.Series(cal_similarity), method="spearman")
        ),
        "similarity_ks_unweighted": weighted_ks_distance(cal_similarity, test_similarity),
        "similarity_ks_weighted": weighted_ks_distance(
            cal_similarity, test_similarity, cal_weights, np.ones(len(test_weights))
        ),
    }


def standardized_scores(
    truth: np.ndarray,
    prediction: np.ndarray,
    compound_scale: np.ndarray,
    reference_cell_medians: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.abs(truth - prediction) / compound_scale[:, None]
    if reference_cell_medians is None:
        reference_cell_medians = np.nanmedian(scores, axis=0)
    positive = reference_cell_medians[
        np.isfinite(reference_cell_medians) & (reference_cell_medians > 1e-8)
    ]
    fallback = float(np.nanmedian(positive))
    reference_cell_medians = np.where(
        np.isfinite(reference_cell_medians) & (reference_cell_medians > 1e-8),
        reference_cell_medians,
        fallback,
    )
    scores = scores / reference_cell_medians[None, :]
    return scores, reference_cell_medians


def score_shift_row(
    family: str,
    seed: int,
    scale_name: str,
    weight_name: str,
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_weights: np.ndarray,
) -> dict:
    per_cell_ks = []
    per_cell_wasserstein = []
    for cell in range(cal_scores.shape[1]):
        cal = cal_scores[:, cell]
        test = test_scores[:, cell]
        cal_valid = np.isfinite(cal)
        test_valid = np.isfinite(test)
        if not cal_valid.any() or not test_valid.any():
            continue
        per_cell_ks.append(
            weighted_ks_distance(
                cal[cal_valid],
                test[test_valid],
                cal_weights[cal_valid],
                np.ones(test_valid.sum()),
            )
        )
        per_cell_wasserstein.append(
            wasserstein_distance(
                cal[cal_valid],
                test[test_valid],
                u_weights=cal_weights[cal_valid],
                v_weights=np.ones(test_valid.sum()),
            )
        )
    return {
        "family": family,
        "seed": seed,
        "scale": scale_name,
        "weight": weight_name,
        "mean_cell_ks": float(np.mean(per_cell_ks)),
        "median_cell_ks": float(np.median(per_cell_ks)),
        "max_cell_ks": float(np.max(per_cell_ks)),
        "mean_cell_wasserstein": float(np.mean(per_cell_wasserstein)),
        "median_cell_wasserstein": float(np.median(per_cell_wasserstein)),
        "cells": len(per_cell_ks),
    }


def conditional_score_gap(
    family: str,
    seed: int,
    cal_similarity: np.ndarray,
    test_similarity: np.ndarray,
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    strata: int = 5,
    minimum_per_domain: int = 40,
) -> dict:
    pooled = np.concatenate((cal_similarity, test_similarity))
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, strata + 1)))
    if len(edges) < 3:
        return {
            "family": family,
            "seed": seed,
            "conditional_ks_mean": np.nan,
            "conditional_ks_median": np.nan,
            "strata_used": 0,
        }
    edges[0] = -np.inf
    edges[-1] = np.inf
    values = []
    weights = []
    for index in range(len(edges) - 1):
        cal_selected = (cal_similarity >= edges[index]) & (cal_similarity < edges[index + 1])
        test_selected = (test_similarity >= edges[index]) & (test_similarity < edges[index + 1])
        if cal_selected.sum() < minimum_per_domain or test_selected.sum() < minimum_per_domain:
            continue
        for cell in range(cal_scores.shape[1]):
            cal = cal_scores[cal_selected, cell]
            test = test_scores[test_selected, cell]
            cal = cal[np.isfinite(cal)]
            test = test[np.isfinite(test)]
            if len(cal) < minimum_per_domain or len(test) < minimum_per_domain:
                continue
            values.append(weighted_ks_distance(cal, test))
            weights.append(2.0 * len(cal) * len(test) / (len(cal) + len(test)))
    return {
        "family": family,
        "seed": seed,
        "conditional_ks_mean": float(np.average(values, weights=weights)) if values else np.nan,
        "conditional_ks_median": float(np.median(values)) if values else np.nan,
        "conditional_ks_max": float(np.max(values)) if values else np.nan,
        "strata_used": int(len(values) / cal_scores.shape[1]) if values else 0,
        "cell_stratum_comparisons": len(values),
        "quantile_strata_requested": strata,
        "minimum_per_domain": minimum_per_domain,
    }


def load_high_dimensional_weights(
    root: Path, family: str, seed: int, ids_cal: np.ndarray, ids_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    result_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    frame = pd.read_csv(result_dir / "domain_weights.csv")
    cal_frame = frame.loc[frame["domain"] == "source_cal"]
    test_frame = frame.loc[frame["domain"] == "test"]
    if not np.array_equal(cal_frame["nsc"].to_numpy("int64"), ids_cal):
        raise ValueError(f"High-dimensional source weights are misaligned for {family} seed {seed}")
    if not np.array_equal(test_frame["nsc"].to_numpy("int64"), ids_test):
        raise ValueError(f"High-dimensional test weights are misaligned for {family} seed {seed}")
    report = json.loads((result_dir / "domain_shift_metrics.json").read_text(encoding="utf-8"))
    return (
        cal_frame["weight"].to_numpy(float),
        test_frame["weight"].to_numpy(float),
        report,
    )


def evaluate_one(
    root: Path, data: EvaluationData, family: str, seed: int
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    prediction_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    similarity_dir = root / "results" / "phase1b" / f"{family}_seed_{seed}"
    point = np.load(prediction_dir / "mlp_predictions.npz")
    ids_cal = point["nsc_source_cal"].astype("int64")
    ids_valid = point["nsc_valid"].astype("int64")
    ids_test = point["nsc_test"].astype("int64")
    pred_cal = point["pred_source_cal"].astype(float)
    pred_valid = point["pred_valid"].astype(float)
    pred_test = point["pred_test"].astype(float)
    truth_cal = data.truth(ids_cal)
    truth_valid = data.truth(ids_valid)
    truth_test = data.truth(ids_test)
    sim_cal = load_similarity(similarity_dir / "nearest_fit_tanimoto_source_cal.csv", ids_cal)
    sim_valid = load_similarity(similarity_dir / "nearest_fit_tanimoto_valid.csv", ids_valid)
    sim_test = load_similarity(similarity_dir / "nearest_fit_tanimoto_test.csv", ids_test)

    scale_model, _, scale_report = fit_similarity_scale(sim_valid, truth_valid, pred_valid)
    cal_scale = predict_similarity_scale(scale_model, sim_cal)
    test_scale = predict_similarity_scale(scale_model, sim_test)
    unit_cal_scale = np.ones(len(ids_cal), dtype=float)
    unit_test_scale = np.ones(len(ids_test), dtype=float)

    hd_cal_weights, hd_test_weights, hd_report = load_high_dimensional_weights(
        root, family, seed, ids_cal, ids_test
    )
    sim_cal_weights, sim_test_weights, sim_report = estimate_similarity_domain_weights(
        sim_cal, sim_test, seed=80_000 + seed
    )

    weight_rows = [
        weight_diagnostics(
            family,
            seed,
            "HD_Morgan2048",
            sim_cal,
            sim_test,
            hd_cal_weights,
            hd_test_weights,
            hd_report,
        ),
        weight_diagnostics(
            family,
            seed,
            "Sim1D_Tanimoto",
            sim_cal,
            sim_test,
            sim_cal_weights,
            sim_test_weights,
            sim_report,
        ),
    ]

    raw_cal_scores, raw_cell_medians = standardized_scores(
        truth_cal, pred_cal, unit_cal_scale
    )
    raw_test_scores, _ = standardized_scores(
        truth_test, pred_test, unit_test_scale, raw_cell_medians
    )
    normalized_cal_scores, normalized_cell_medians = standardized_scores(
        truth_cal, pred_cal, cal_scale
    )
    normalized_test_scores, _ = standardized_scores(
        truth_test, pred_test, test_scale, normalized_cell_medians
    )
    score_rows = []
    for scale_name, source_scores, target_scores in (
        ("raw", raw_cal_scores, raw_test_scores),
        ("similarity_normalized", normalized_cal_scores, normalized_test_scores),
    ):
        for weight_name, weights in (
            ("unweighted", np.ones(len(ids_cal))),
            ("HD_Morgan2048", hd_cal_weights),
            ("Sim1D_Tanimoto", sim_cal_weights),
        ):
            score_rows.append(
                score_shift_row(
                    family,
                    seed,
                    scale_name,
                    weight_name,
                    source_scores,
                    target_scores,
                    weights,
                )
            )
    conditional_report = conditional_score_gap(
        family,
        seed,
        sim_cal,
        sim_test,
        normalized_cal_scores,
        normalized_test_scores,
    )

    metric_rows = []
    bin_rows = []
    for alpha in ALPHAS:
        method_intervals = {
            "Global_CP": standard_intervals(
                truth_cal, pred_cal, pred_test, unit_cal_scale, unit_test_scale, alpha
            ),
            "HD_WCP": weighted_scaled_intervals(
                truth_cal,
                pred_cal,
                pred_test,
                unit_cal_scale,
                unit_test_scale,
                hd_cal_weights,
                hd_test_weights,
                alpha,
            ),
            "Sim1D_WCP": weighted_scaled_intervals(
                truth_cal,
                pred_cal,
                pred_test,
                unit_cal_scale,
                unit_test_scale,
                sim_cal_weights,
                sim_test_weights,
                alpha,
            ),
            "SNCP_unweighted": standard_intervals(
                truth_cal, pred_cal, pred_test, cal_scale, test_scale, alpha
            ),
            "SNCP_HD_WCP": weighted_scaled_intervals(
                truth_cal,
                pred_cal,
                pred_test,
                cal_scale,
                test_scale,
                hd_cal_weights,
                hd_test_weights,
                alpha,
            ),
            "SNCP_Sim1D_WCP": weighted_scaled_intervals(
                truth_cal,
                pred_cal,
                pred_test,
                cal_scale,
                test_scale,
                sim_cal_weights,
                sim_test_weights,
                alpha,
            ),
        }
        for method in METHODS:
            lower, upper = method_intervals[method]
            metric_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "alpha": alpha,
                    "method": method,
                    **interval_metrics(truth_test, lower, upper, alpha),
                }
            )
            bin_rows.extend(
                {
                    "family": family,
                    "seed": seed,
                    "alpha": alpha,
                    "method": method,
                    **row,
                }
                for row in similarity_metrics(sim_test, truth_test, lower, upper, alpha)
            )

    details = {
        "family": family,
        "seed": seed,
        "scale": scale_report,
        "similarity_weight_estimator": sim_report,
        "conditional_similarity_sufficiency_diagnostic": conditional_report,
    }
    return metric_rows, bin_rows, weight_rows, score_rows, details


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    clean = values.to_numpy(float)
    clean = clean[np.isfinite(clean)]
    mean = float(np.mean(clean)) if len(clean) else np.nan
    sd = float(np.std(clean, ddof=1)) if len(clean) > 1 else np.nan
    if len(clean) > 1:
        half = float(student_t.ppf(0.975, len(clean) - 1) * sd / np.sqrt(len(clean)))
        return mean, sd, mean - half, mean + half
    return mean, sd, np.nan, np.nan


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = (
        "coverage",
        "coverage_error_abs",
        "macro_cell_ace",
        "minimum_cell_coverage",
        "mean_width",
        "finite_interval_fraction",
    )
    for (family, method, alpha), part in metrics.groupby(["family", "method", "alpha"]):
        row = {
            "family": family,
            "method": method,
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
            "seeds": int(part["seed"].nunique()),
        }
        for column in numeric:
            mean, sd, low, high = mean_ci(part[column])
            row[f"{column}_mean"] = mean
            row[f"{column}_sd"] = sd
            row[f"{column}_ci95_low"] = low
            row[f"{column}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    observed = abs(values.mean())
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    permuted = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def paired_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        metrics["family"].isin(OOD_FAMILIES) & np.isclose(metrics["alpha"], 0.10)
    ]
    candidate = primary[primary["method"] == "SNCP_Sim1D_WCP"].set_index(
        ["family", "seed"]
    )
    rows = []
    for comparator in (
        "Global_CP",
        "HD_WCP",
        "Sim1D_WCP",
        "SNCP_unweighted",
        "SNCP_HD_WCP",
    ):
        reference = primary[primary["method"] == comparator].set_index(["family", "seed"])
        for metric in (
            "coverage_error_abs",
            "macro_cell_ace",
            "mean_width",
            "finite_interval_fraction",
        ):
            differences = candidate[metric] - reference[metric]
            mean, sd, low, high = mean_ci(differences)
            rows.append(
                {
                    "candidate": "SNCP_Sim1D_WCP",
                    "comparator": comparator,
                    "metric": metric,
                    "difference_definition": "candidate_minus_comparator",
                    "units": len(differences),
                    "mean_difference": mean,
                    "sd_difference": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(
                        differences.to_numpy(float)
                    ),
                    "candidate_better_units": int(
                        (differences < 0).sum()
                        if metric != "finite_interval_fraction"
                        else (differences > 0).sum()
                    ),
                    "ties": int(np.isclose(differences, 0.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_decision(
    metrics: pd.DataFrame,
    weights: pd.DataFrame,
    score_shift: pd.DataFrame,
    conditional: pd.DataFrame,
) -> dict:
    primary = metrics[
        metrics["family"].isin(OOD_FAMILIES) & np.isclose(metrics["alpha"], 0.10)
    ]
    candidate = primary[primary["method"] == "SNCP_Sim1D_WCP"].set_index(
        ["family", "seed"]
    )
    current = primary[primary["method"] == "SNCP_unweighted"].set_index(
        ["family", "seed"]
    )
    ood_weights = weights[weights["family"].isin(OOD_FAMILIES)]
    hd_ess = float(
        ood_weights[ood_weights["estimator"] == "HD_Morgan2048"]["ess_fraction"].mean()
    )
    sim_ess = float(
        ood_weights[ood_weights["estimator"] == "Sim1D_Tanimoto"]["ess_fraction"].mean()
    )
    sim_diag = ood_weights[ood_weights["estimator"] == "Sim1D_Tanimoto"]
    score_ood = score_shift[
        score_shift["family"].isin(OOD_FAMILIES)
        & (score_shift["scale"] == "similarity_normalized")
    ]
    unweighted_score_ks = float(
        score_ood[score_ood["weight"] == "unweighted"]["mean_cell_ks"].mean()
    )
    sim_weighted_score_ks = float(
        score_ood[score_ood["weight"] == "Sim1D_Tanimoto"]["mean_cell_ks"].mean()
    )
    candidate_ace = float(candidate["coverage_error_abs"].mean())
    current_ace = float(current["coverage_error_abs"].mean())
    width_ratio = float(candidate["mean_width"].mean() / current["mean_width"].mean())
    minimum_finite = float(candidate["finite_interval_fraction"].min())
    similarity_ks_before = float(sim_diag["similarity_ks_unweighted"].mean())
    similarity_ks_after = float(sim_diag["similarity_ks_weighted"].mean())
    conditions = {
        "minimum_finite_interval_fraction_ge_0_999": minimum_finite >= 0.999,
        "ace_noninferiority_margin_0_005": candidate_ace - current_ace <= 0.005,
        "mean_width_ratio_le_1_15": width_ratio <= 1.15,
        "sim1d_ess_not_lower_than_hd": sim_ess >= hd_ess,
        "similarity_balance_improves": similarity_ks_after < similarity_ks_before,
    }
    paper_conditions = {
        "candidate_mean_ace_lower_than_current": candidate_ace < current_ace,
        "mean_width_ratio_le_1_10": width_ratio <= 1.10,
        "normalized_score_ks_improves": sim_weighted_score_ks < unweighted_score_ks,
    }
    conditional_ood = conditional[conditional["family"].isin(OOD_FAMILIES)]
    conditional_mean = float(conditional_ood["conditional_ks_mean"].mean())
    return {
        "status": "exploratory_internal_only",
        "primary_scope": "scaffold and leader-cluster, five seeds each, nominal 90% coverage",
        "observed": {
            "candidate_mean_ace": candidate_ace,
            "current_sncp_mean_ace": current_ace,
            "ace_candidate_minus_current": candidate_ace - current_ace,
            "candidate_to_current_mean_width_ratio": width_ratio,
            "candidate_minimum_finite_interval_fraction": minimum_finite,
            "sim1d_mean_ess_fraction": sim_ess,
            "high_dimensional_mean_ess_fraction": hd_ess,
            "sim1d_similarity_ks_before": similarity_ks_before,
            "sim1d_similarity_ks_after": similarity_ks_after,
            "normalized_score_mean_cell_ks_unweighted": unweighted_score_ks,
            "normalized_score_mean_cell_ks_sim1d_weighted": sim_weighted_score_ks,
            "conditional_normalized_score_ks_mean": conditional_mean,
        },
        "engineering_feasibility_conditions": conditions,
        "engineering_feasibility_pass": bool(all(conditions.values())),
        "paper_value_conditions": paper_conditions,
        "paper_value_pass": bool(all(paper_conditions.values())),
        "assumption_note": (
            "The conditional normalized-score KS is a retrospective diagnostic, not a proof "
            "of R independent of domain given similarity."
        ),
    }


def dataframe_to_markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def format_value(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_internal_report(
    output: Path,
    decision: dict,
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    score_shift: pd.DataFrame,
) -> None:
    primary = summary[
        summary["family"].isin(OOD_FAMILIES)
        & np.isclose(summary["alpha"], 0.10)
    ][
        [
            "family",
            "method",
            "coverage_mean",
            "coverage_error_abs_mean",
            "macro_cell_ace_mean",
            "mean_width_mean",
            "finite_interval_fraction_mean",
        ]
    ]
    weight_summary = (
        weights[weights["family"].isin(OOD_FAMILIES)]
        .groupby("estimator")
        .agg(
            ess_fraction=("ess_fraction", "mean"),
            cal_weight_max=("cal_weight_max", "max"),
            test_weight_max=("test_weight_max", "max"),
            support_failure=("support_failure_fraction_alpha_0_10", "max"),
            similarity_ks_before=("similarity_ks_unweighted", "mean"),
            similarity_ks_after=("similarity_ks_weighted", "mean"),
        )
        .reset_index()
    )
    score_summary = (
        score_shift[
            score_shift["family"].isin(OOD_FAMILIES)
            & (score_shift["scale"] == "similarity_normalized")
        ]
        .groupby("weight")
        .agg(mean_cell_ks=("mean_cell_ks", "mean"), mean_cell_wasserstein=("mean_cell_wasserstein", "mean"))
        .reset_index()
    )
    observed = decision["observed"]
    lines = [
        "# 相似度充分加权保形预测：内部探索性结果",
        "",
        "日期：2026-08-16  ",
        "验证状态：ANALYZED（冻结预测上的回顾性探索，不是外部确认）",
        "",
        "## 结论摘要",
        "",
        f"- 工程可行性门槛：**{'通过' if decision['engineering_feasibility_pass'] else '未通过'}**。",
        f"- 论文价值门槛：**{'通过' if decision['paper_value_pass'] else '未通过'}**。",
        f"- OOD 90% 工作点：候选方法 ACE={observed['candidate_mean_ace']:.4f}，当前 SNCP ACE={observed['current_sncp_mean_ace']:.4f}，差值={observed['ace_candidate_minus_current']:+.4f}。",
        f"- 候选方法/当前方法平均宽度比={observed['candidate_to_current_mean_width_ratio']:.4f}；最低有限区间比例={observed['candidate_minimum_finite_interval_fraction']:.6f}。",
        f"- 一维相似度权重 ESS 比例={observed['sim1d_mean_ess_fraction']:.4f}，高维权重 ESS 比例={observed['high_dimensional_mean_ess_fraction']:.4f}。",
        f"- 相似度分布 KS 经一维加权由 {observed['sim1d_similarity_ks_before']:.4f} 变为 {observed['sim1d_similarity_ks_after']:.4f}。",
        f"- 标准化残差平均逐细胞系 KS 经一维加权由 {observed['normalized_score_mean_cell_ks_unweighted']:.4f} 变为 {observed['normalized_score_mean_cell_ks_sim1d_weighted']:.4f}。",
        f"- 相似度分层后的条件标准化残差 KS 均值={observed['conditional_normalized_score_ks_mean']:.4f}；该值只用于诊断，不能证明相似度充分性。",
        "",
        "## 90% OOD 主结果",
        "",
        dataframe_to_markdown(primary),
        "",
        "## 权重稳定性",
        "",
        dataframe_to_markdown(weight_summary),
        "",
        "## 标准化残差的源域—目标域差异",
        "",
        dataframe_to_markdown(score_summary),
        "",
        "## 判定条件",
        "",
    ]
    for name, passed in decision["engineering_feasibility_conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}：`{name}`")
    lines.extend(("", "### 论文价值条件", ""))
    for name, passed in decision["paper_value_conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}：`{name}`")
    lines.extend(
        (
            "",
            "## 解释边界",
            "",
            "- 本实验允许无标签 test 分子结构参与密度比估计，但 test 响应标签只用于最终评估。",
            "- 这是一项探索性测试；只有在独立外部数据或预先冻结的新切分上复现后，才可升级为确认性证据。",
            "- 无论结果方向如何，完整数据均保存在该内部目录；不因论文叙述需要删除不利结果。",
            "",
        )
    )
    (output / "INTERNAL_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument(
        "--output",
        default="results/similarity_sufficient_wcp_20260816",
    )
    args = parser.parse_args()
    root = Path(args.root)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    families = tuple(value for value in args.families.split(",") if value)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    data = EvaluationData(root)

    metric_rows: list[dict] = []
    bin_rows: list[dict] = []
    weight_rows: list[dict] = []
    score_rows: list[dict] = []
    details = []
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for seed in seeds:
            metrics, bins, weights, scores, detail = evaluate_one(
                root, data, family, seed
            )
            metric_rows.extend(metrics)
            bin_rows.extend(bins)
            weight_rows.extend(weights)
            score_rows.extend(scores)
            details.append(detail)
            print(f"completed family={family} seed={seed}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    bins = pd.DataFrame(bin_rows)
    weights = pd.DataFrame(weight_rows)
    score_shift = pd.DataFrame(score_rows)
    conditional = pd.DataFrame(
        [row["conditional_similarity_sufficiency_diagnostic"] for row in details]
    )
    summary = summarize_metrics(metrics)
    contrasts = paired_contrasts(metrics)
    decision = build_decision(metrics, weights, score_shift, conditional)

    metrics.to_csv(output / "method_seed_metrics.csv", index=False)
    bins.to_csv(output / "similarity_bin_metrics.csv", index=False)
    weights.to_csv(output / "weight_diagnostics.csv", index=False)
    score_shift.to_csv(output / "score_shift_diagnostics.csv", index=False)
    conditional.to_csv(output / "conditional_sufficiency_diagnostics.csv", index=False)
    summary.to_csv(output / "method_summary_ci.csv", index=False)
    contrasts.to_csv(output / "paired_candidate_contrasts.csv", index=False)
    (output / "run_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "families": families,
        "seeds": seeds,
        "alphas": ALPHAS,
        "methods": METHODS,
        "point_predictor": "frozen Phase 1A MLP predictions",
        "scale": "shared decreasing isotonic scale fitted on validation responses only",
        "target_features_used_for_weighting": "unlabeled test structures/similarities only",
        "target_responses_used": "evaluation only",
        "output_files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_internal_report(output, decision, summary, weights, score_shift)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

