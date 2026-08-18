"""Evaluate similarity-normalized conformal prediction (Phase 1B).

M5 learns a monotone compound-level residual scale from the validation set:
nearer compounds are expected to have no larger uncertainty scale.  The scale
model uses no source-calibration or test response labels.  Per-cell conformal
quantiles are then fit only on source-calibration residuals normalized by this
scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

try:  # Support both ``python -m`` and direct script execution.
    from scripts.evaluate_phase1a_conformal import (
        EvaluationData,
        finite_sample_quantile,
        interval_metrics,
        nearest_fit_similarity,
        similarity_bin_metrics,
    )
except ModuleNotFoundError:
    from evaluate_phase1a_conformal import (  # type: ignore[no-redef]
        EvaluationData,
        finite_sample_quantile,
        interval_metrics,
        nearest_fit_similarity,
        similarity_bin_metrics,
    )


FAMILIES = ("random", "scaffold", "leader_cluster")
MODELS = ("ridge", "mlp")


def fit_similarity_scale(
    similarity: np.ndarray, truth: np.ndarray, prediction: np.ndarray
) -> tuple[IsotonicRegression, np.ndarray, dict]:
    """Fit a positive, decreasing scale on one compound-level residual per row."""
    residual = np.abs(truth - prediction)
    per_cell_scale = np.nanmedian(residual, axis=0)
    fallback = float(np.nanmedian(per_cell_scale[np.isfinite(per_cell_scale) & (per_cell_scale > 0)]))
    per_cell_scale = np.where(
        np.isfinite(per_cell_scale) & (per_cell_scale > 0), per_cell_scale, fallback
    )
    normalized = residual / per_cell_scale[None, :]
    compound_difficulty = np.nanmedian(normalized, axis=1)
    valid = np.isfinite(similarity) & np.isfinite(compound_difficulty)
    if valid.sum() < 20:
        raise ValueError("Insufficient validation compounds for similarity-scale fitting")
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(similarity[valid], compound_difficulty[valid])
    fitted = model.predict(similarity)
    normalization = float(np.median(fitted[valid]))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Similarity scale normalization is invalid")
    fitted = np.maximum(fitted / normalization, 1e-3)
    diagnostics = {
        "validation_compounds": int(valid.sum()),
        "validation_difficulty_spearman": float(
            pd.Series(similarity[valid]).corr(
                pd.Series(compound_difficulty[valid]), method="spearman"
            )
        ),
        "scale_min": float(fitted[valid].min()),
        "scale_median": float(np.median(fitted[valid])),
        "scale_max": float(fitted[valid].max()),
        "cell_median_residual_min": float(per_cell_scale.min()),
        "cell_median_residual_median": float(np.median(per_cell_scale)),
        "cell_median_residual_max": float(per_cell_scale.max()),
    }
    # Store the normalization on the model so prediction is on the same scale.
    model._phase1b_normalization = normalization  # type: ignore[attr-defined]
    return model, per_cell_scale, diagnostics


def predict_similarity_scale(model: IsotonicRegression, similarity: np.ndarray) -> np.ndarray:
    normalization = float(getattr(model, "_phase1b_normalization"))
    return np.maximum(model.predict(similarity) / normalization, 1e-3)


def standard_intervals(
    truth_cal: np.ndarray, pred_cal: np.ndarray, pred_test: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.abs(truth_cal - pred_cal)
    quantiles = np.asarray(
        [finite_sample_quantile(residual[:, column], alpha) for column in range(residual.shape[1])]
    )
    return pred_test - quantiles, pred_test + quantiles


def similarity_normalized_intervals(
    truth_cal: np.ndarray,
    pred_cal: np.ndarray,
    cal_similarity: np.ndarray,
    test_similarity: np.ndarray,
    scale_model: IsotonicRegression,
    pred_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cal_scale = predict_similarity_scale(scale_model, cal_similarity)
    test_scale = predict_similarity_scale(scale_model, test_similarity)
    residual_score = np.abs(truth_cal - pred_cal) / cal_scale[:, None]
    quantiles = np.asarray(
        [
            finite_sample_quantile(residual_score[:, column], alpha)
            for column in range(residual_score.shape[1])
        ]
    )
    radius = test_scale[:, None] * quantiles[None, :]
    return pred_test - radius, pred_test + radius, cal_scale, test_scale


def load_or_compute_similarity(
    output_dir: Path,
    label: str,
    data: EvaluationData,
    fit_ids: np.ndarray,
    ids: np.ndarray,
) -> np.ndarray:
    path = output_dir / f"nearest_fit_tanimoto_{label}.csv"
    if path.exists():
        return pd.read_csv(path)["nearest_fit_tanimoto"].to_numpy(dtype=float)
    similarity = nearest_fit_similarity(
        data.x[data.rows(fit_ids)], data.x[data.rows(ids)]
    )
    pd.DataFrame({"nsc": ids, "nearest_fit_tanimoto": similarity}).to_csv(path, index=False)
    return similarity


def evaluate_one(
    root: Path, data: EvaluationData, family: str, seed: int, model: str, alpha: float
) -> tuple[list[dict], list[dict], dict]:
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    source_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    output_dir = root / "results" / "phase1b" / f"{family}_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    stored = np.load(source_dir / f"{model}_predictions.npz")
    ids_cal = stored["nsc_source_cal"].astype("int64")
    ids_valid = stored["nsc_valid"].astype("int64")
    ids_test = stored["nsc_test"].astype("int64")
    pred_cal = stored["pred_source_cal"].astype(float)
    pred_valid = stored["pred_valid"].astype(float)
    pred_test = stored["pred_test"].astype(float)
    truth_cal = data.truth(ids_cal)
    truth_valid = data.truth(ids_valid)
    truth_test = data.truth(ids_test)
    fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")

    sim_cal = load_or_compute_similarity(output_dir, "source_cal", data, fit_ids, ids_cal)
    sim_valid = load_or_compute_similarity(output_dir, "valid", data, fit_ids, ids_valid)
    sim_test = load_or_compute_similarity(output_dir, "test", data, fit_ids, ids_test)
    scale_model, _, diagnostics = fit_similarity_scale(sim_valid, truth_valid, pred_valid)
    lower_m1, upper_m1 = standard_intervals(truth_cal, pred_cal, pred_test, alpha)
    lower_m5, upper_m5, cal_scale, test_scale = similarity_normalized_intervals(
        truth_cal,
        pred_cal,
        sim_cal,
        sim_test,
        scale_model,
        pred_test,
        alpha,
    )

    metric_rows = []
    bins = []
    for method, lower, upper in (
        ("M1_marginal_unweighted", lower_m1, upper_m1),
        ("M5_similarity_normalized", lower_m5, upper_m5),
    ):
        metric_rows.append(
            {
                "family": family,
                "seed": seed,
                "model": model,
                "method": method,
                "alpha": alpha,
                "nominal_coverage": 1 - alpha,
                **interval_metrics(truth_test, lower, upper, "marginal"),
            }
        )
        bins.extend(
            {
                "family": family,
                "seed": seed,
                "model": model,
                **row,
            }
            for row in similarity_bin_metrics(sim_test, truth_test, lower, upper, method)
        )

    detail = {
        "family": family,
        "seed": seed,
        "model": model,
        "alpha": alpha,
        "scale_training": (
            "validation labels only; one normalized median residual per compound; "
            "monotone decreasing isotonic regression against nearest-fit Tanimoto"
        ),
        "scale_diagnostics": diagnostics,
        "source_cal_scale": {
            "min": float(cal_scale.min()),
            "median": float(np.median(cal_scale)),
            "max": float(cal_scale.max()),
        },
        "test_scale": {
            "min": float(test_scale.min()),
            "median": float(np.median(test_scale)),
            "max": float(test_scale.max()),
        },
        "test_similarity": {
            "mean": float(sim_test.mean()),
            "median": float(np.median(sim_test)),
            "fraction_below_0_4": float((sim_test < 0.4).mean()),
        },
    }
    (output_dir / f"{model}_similarity_detail.json").write_text(
        json.dumps(detail, indent=2), encoding="utf-8"
    )
    return metric_rows, bins, detail


def development_gate(metrics: pd.DataFrame, bins: pd.DataFrame) -> dict:
    mlp = metrics.loc[metrics["model"] == "mlp"].copy()
    coverage = mlp.pivot(index="family", columns="method", values="coverage")
    width = mlp.pivot(index="family", columns="method", values="mean_width_finite")
    ood = coverage.loc[["scaffold", "leader_cluster"]]
    baseline_error = np.abs(ood["M1_marginal_unweighted"] - 0.9).mean()
    proposed_error = np.abs(ood["M5_similarity_normalized"] - 0.9).mean()
    error_reduction = 1 - proposed_error / baseline_error
    low = bins.loc[
        (bins["model"] == "mlp")
        & (bins["family"].isin(["scaffold", "leader_cluster"]))
        & (bins["similarity_bin"] == "[0,.4)")
    ].pivot(index="family", columns="method", values="marginal_coverage")
    low_similarity_gain = (
        low["M5_similarity_normalized"] - low["M1_marginal_unweighted"]
    ).mean()
    width_increase = (
        width.loc[["scaffold", "leader_cluster"], "M5_similarity_normalized"]
        / width.loc[["scaffold", "leader_cluster"], "M1_marginal_unweighted"]
        - 1
    ).mean()
    random_coverage = float(coverage.loc["random", "M5_similarity_normalized"])
    passed = bool(
        error_reduction >= 0.25
        and low_similarity_gain >= 0.015
        and width_increase <= 0.20
        and random_coverage >= 0.885
    )
    return {
        "gate": "seed1_development",
        "criteria": {
            "ood_mean_absolute_coverage_error_reduction_min": 0.25,
            "low_similarity_coverage_gain_min": 0.015,
            "ood_mean_width_increase_max": 0.20,
            "random_coverage_min": 0.885,
        },
        "observed": {
            "ood_mean_absolute_coverage_error_reduction": float(error_reduction),
            "low_similarity_coverage_gain": float(low_similarity_gain),
            "ood_mean_width_increase": float(width_increase),
            "random_coverage": random_coverage,
        },
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--families", default="random,scaffold,leader_cluster")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--models", default="ridge,mlp")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    root = Path(args.root)
    data = EvaluationData(root)
    metrics: list[dict] = []
    bins: list[dict] = []
    for family in [value for value in args.families.split(",") if value]:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for seed in [int(value) for value in args.seeds.split(",") if value]:
            for model in [value for value in args.models.split(",") if value]:
                rows, bin_rows, _ = evaluate_one(root, data, family, seed, model, args.alpha)
                metrics.extend(rows)
                bins.extend(bin_rows)
                print(f"completed family={family} seed={seed} model={model}", flush=True)

    output_dir = root / "results" / "phase1b"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics)
    bins_frame = pd.DataFrame(bins)
    metrics_frame.to_csv(output_dir / "similarity_conformal_summary.csv", index=False)
    bins_frame.to_csv(output_dir / "similarity_coverage_by_tanimoto.csv", index=False)
    if set(args.seeds.split(",")) == {"1"} and "mlp" in args.models.split(","):
        gate = development_gate(metrics_frame, bins_frame)
        (output_dir / "seed1_development_gate.json").write_text(
            json.dumps(gate, indent=2), encoding="utf-8"
        )
        print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()

