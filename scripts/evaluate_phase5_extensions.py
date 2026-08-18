"""Evaluate Phase 5 multi-alpha baselines and scale-function ablations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

try:
    from scripts.evaluate_phase1a_conformal import (
        EvaluationData,
        estimate_domain_weights,
        finite_sample_quantile,
        weighted_thresholds,
    )
    from scripts.evaluate_phase1b_similarity import (
        fit_similarity_scale,
        predict_similarity_scale,
    )
    from scripts.evaluate_phase3_local_and_screening import sim
except ModuleNotFoundError:
    from evaluate_phase1a_conformal import (  # type: ignore[no-redef]
        EvaluationData,
        estimate_domain_weights,
        finite_sample_quantile,
        weighted_thresholds,
    )
    from evaluate_phase1b_similarity import (  # type: ignore[no-redef]
        fit_similarity_scale,
        predict_similarity_scale,
    )
    from evaluate_phase3_local_and_screening import sim  # type: ignore[no-redef]


FAMILIES = ("random", "scaffold", "leader_cluster")
ALPHAS = (0.05, 0.10, 0.15, 0.20)
DAD_K = 250
SIMILARITY_EDGES = np.asarray((0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001))
SIMILARITY_LABELS = ("[0,.3)", "[.3,.4)", "[.4,.5)", "[.5,.6)", "[.6,.7)", "[.7,.8)", "[.8,1]")


def interval_metrics(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> dict:
    observed = np.isfinite(truth)
    finite = observed & np.isfinite(lower) & np.isfinite(upper)
    covered = finite & (truth >= lower) & (truth <= upper)
    width = upper - lower
    coverage = float(covered.sum() / observed.sum())
    return {
        "nominal_coverage": 1.0 - alpha,
        "coverage": coverage,
        "coverage_error_abs": abs(coverage - (1.0 - alpha)),
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
        row = interval_metrics(truth[selected], lower[selected], upper[selected], alpha)
        rows.append(
            {
                "similarity_bin": label,
                "bin_order": group,
                "mean_similarity": float(similarity[selected].mean()),
                **row,
            }
        )
    return rows


def cell_metrics(
    truth: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> list[dict]:
    """Per-cell calibration metrics for macro and tail-stability analyses."""
    rows = []
    for cell in range(truth.shape[1]):
        observed = np.isfinite(truth[:, cell])
        finite = observed & np.isfinite(lower[:, cell]) & np.isfinite(upper[:, cell])
        covered = finite & (truth[:, cell] >= lower[:, cell]) & (truth[:, cell] <= upper[:, cell])
        coverage = float(covered.sum() / observed.sum()) if observed.any() else np.nan
        width = upper[:, cell] - lower[:, cell]
        rows.append(
            {
                "cell_index": cell,
                "coverage": coverage,
                "coverage_error_abs": abs(coverage - (1.0 - alpha)),
                "mean_width": float(width[finite].mean()) if finite.any() else np.nan,
                "finite_interval_fraction": float(finite.sum() / observed.sum()) if observed.any() else np.nan,
                "n_compounds": int(observed.sum()),
            }
        )
    return rows


def standard_intervals(
    truth_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.abs(truth_cal - pred_cal)
    quantiles = np.asarray(
        [finite_sample_quantile(residual[:, cell], alpha) for cell in range(residual.shape[1])]
    )
    return pred_test - quantiles, pred_test + quantiles


def weighted_intervals(
    truth_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.abs(truth_cal - pred_cal)
    radius = np.column_stack(
        [
            weighted_thresholds(residual[:, cell], cal_weights, test_weights, alpha)
            for cell in range(residual.shape[1])
        ]
    )
    return pred_test - radius, pred_test + radius


def cqr_intervals(
    truth_cal: np.ndarray,
    quantile_cal: np.ndarray,
    quantile_test: np.ndarray,
    quantile_levels: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower_level, upper_level = alpha / 2.0, 1.0 - alpha / 2.0
    lower_index = int(np.argmin(np.abs(quantile_levels - lower_level)))
    upper_index = int(np.argmin(np.abs(quantile_levels - upper_level)))
    if not np.isclose(quantile_levels[lower_index], lower_level) or not np.isclose(
        quantile_levels[upper_index], upper_level
    ):
        raise ValueError(f"CQR predictions do not contain quantiles for alpha={alpha}")
    lower_cal = quantile_cal[:, :, lower_index]
    upper_cal = quantile_cal[:, :, upper_index]
    scores = np.maximum(lower_cal - truth_cal, truth_cal - upper_cal)
    correction = np.asarray(
        [finite_sample_quantile(scores[:, cell], alpha) for cell in range(scores.shape[1])]
    )
    return (
        quantile_test[:, :, lower_index] - correction,
        quantile_test[:, :, upper_index] + correction,
    )


def top_k_tanimoto_indices(cal_x, test_x, k: int, block_size: int = 32) -> np.ndarray:
    if k <= 0 or k > cal_x.shape[0]:
        raise ValueError("k must be between 1 and the number of calibration compounds")
    cal_counts = np.asarray(cal_x.sum(axis=1)).ravel()
    test_counts = np.asarray(test_x.sum(axis=1)).ravel()
    result = np.empty((test_x.shape[0], k), dtype=np.int32)
    for start in range(0, test_x.shape[0], block_size):
        stop = min(start + block_size, test_x.shape[0])
        intersection = (test_x[start:stop] @ cal_x.T).toarray()
        union = test_counts[start:stop, None] + cal_counts[None, :] - intersection
        similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union > 0,
        )
        result[start:stop] = np.argpartition(similarity, -k, axis=1)[:, -k:]
    return result


def conformal_order_index(n: np.ndarray, alpha: float) -> np.ndarray:
    """Index matching finite_sample_quantile(..., method='higher')."""
    n = np.asarray(n, dtype=int)
    level_numerator = np.ceil((n + 1) * (1.0 - alpha))
    level = np.minimum(1.0, level_numerator / np.maximum(n, 1))
    index = np.ceil(level * np.maximum(n - 1, 0)).astype(int)
    return np.minimum(index, np.maximum(n - 1, 0))


def dad_intervals(
    truth_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    neighbour_indices: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.abs(truth_cal - pred_cal)
    radius = np.full_like(pred_test, np.nan, dtype=float)
    cells = np.arange(residual.shape[1])
    for row, neighbours in enumerate(neighbour_indices):
        local = residual[neighbours]
        ordered = np.sort(local, axis=0)
        counts = np.isfinite(local).sum(axis=0)
        valid = counts > 0
        indices = conformal_order_index(counts, alpha)
        radius[row, valid] = ordered[indices[valid], cells[valid]]
    return pred_test - radius, pred_test + radius


def _validation_difficulty(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = np.abs(truth - prediction)
    per_cell = np.nanmedian(residual, axis=0)
    positive = per_cell[np.isfinite(per_cell) & (per_cell > 0)]
    fallback = float(np.nanmedian(positive))
    per_cell = np.where(np.isfinite(per_cell) & (per_cell > 0), per_cell, fallback)
    normalized = residual / per_cell[None, :]
    return residual, per_cell, np.nanmedian(normalized, axis=1)


@dataclass
class CubicScale:
    coefficients: np.ndarray
    similarity_mean: float
    similarity_scale: float
    similarity_min: float
    similarity_max: float
    log_min: float
    log_max: float
    normalization: float


def fit_unconstrained_cubic_scale(
    similarity: np.ndarray, truth: np.ndarray, prediction: np.ndarray, ridge: float = 1e-2
) -> tuple[CubicScale, dict]:
    _, _, difficulty = _validation_difficulty(truth, prediction)
    valid = np.isfinite(similarity) & np.isfinite(difficulty) & (difficulty > 0)
    if valid.sum() < 20:
        raise ValueError("Insufficient validation compounds for cubic scale fitting")
    x = similarity[valid].astype(float)
    y = np.log(np.maximum(difficulty[valid], 1e-3))
    mean = float(x.mean())
    scale = float(x.std()) or 1.0
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z, z**2, z**3))
    penalty = np.diag((0.0, ridge, ridge, ridge))
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    raw_log = design @ coefficients
    log_min, log_max = np.quantile(y, (0.01, 0.99))
    raw = np.exp(np.clip(raw_log, log_min, log_max))
    normalization = float(np.median(raw))
    model = CubicScale(
        coefficients=coefficients,
        similarity_mean=mean,
        similarity_scale=scale,
        similarity_min=float(x.min()),
        similarity_max=float(x.max()),
        log_min=float(log_min),
        log_max=float(log_max),
        normalization=normalization,
    )
    fitted = predict_unconstrained_cubic_scale(model, similarity[valid])
    diagnostics = {
        "validation_compounds": int(valid.sum()),
        "validation_difficulty_spearman": float(
            pd.Series(similarity[valid]).corr(pd.Series(difficulty[valid]), method="spearman")
        ),
        "scale_min": float(fitted.min()),
        "scale_median": float(np.median(fitted)),
        "scale_max": float(fitted.max()),
        "cubic_coefficients": coefficients.tolist(),
    }
    return model, diagnostics


def predict_unconstrained_cubic_scale(model: CubicScale, similarity: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(similarity, dtype=float), model.similarity_min, model.similarity_max)
    z = (x - model.similarity_mean) / model.similarity_scale
    design = np.column_stack((np.ones(len(z)), z, z**2, z**3))
    log_scale = np.clip(design @ model.coefficients, model.log_min, model.log_max)
    return np.maximum(np.exp(log_scale) / model.normalization, 1e-3)


@dataclass
class PerCellMonotoneScale:
    models: list[IsotonicRegression]
    normalizations: np.ndarray


def fit_per_cell_monotone_scale(
    similarity: np.ndarray, truth: np.ndarray, prediction: np.ndarray
) -> tuple[PerCellMonotoneScale, dict]:
    residual, per_cell, _ = _validation_difficulty(truth, prediction)
    models: list[IsotonicRegression] = []
    normalizations = np.empty(residual.shape[1], dtype=float)
    scale_min, scale_max = [], []
    for cell in range(residual.shape[1]):
        difficulty = residual[:, cell] / per_cell[cell]
        valid = np.isfinite(similarity) & np.isfinite(difficulty)
        if valid.sum() < 20:
            raise ValueError(f"Insufficient validation compounds for cell {cell}")
        model = IsotonicRegression(increasing=False, out_of_bounds="clip")
        model.fit(similarity[valid], difficulty[valid])
        raw = np.maximum(model.predict(similarity[valid]), 1e-3)
        normalization = float(np.median(raw))
        models.append(model)
        normalizations[cell] = normalization
        normalized = raw / normalization
        scale_min.append(float(normalized.min()))
        scale_max.append(float(normalized.max()))
    diagnostics = {
        "cell_models": len(models),
        "scale_min_across_cells": float(np.min(scale_min)),
        "scale_max_across_cells": float(np.max(scale_max)),
        "normalization_median": float(np.median(normalizations)),
    }
    return PerCellMonotoneScale(models, normalizations), diagnostics


def predict_per_cell_monotone_scale(
    model: PerCellMonotoneScale, similarity: np.ndarray
) -> np.ndarray:
    columns = [
        np.maximum(estimator.predict(similarity) / normalization, 1e-3)
        for estimator, normalization in zip(model.models, model.normalizations)
    ]
    return np.column_stack(columns)


def scaled_intervals(
    truth_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    cal_scale: np.ndarray,
    test_scale: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if cal_scale.ndim == 1:
        cal_scale = cal_scale[:, None]
    if test_scale.ndim == 1:
        test_scale = test_scale[:, None]
    scores = np.abs(truth_cal - pred_cal) / cal_scale
    quantiles = np.asarray(
        [finite_sample_quantile(scores[:, cell], alpha) for cell in range(scores.shape[1])]
    )
    radius = test_scale * quantiles[None, :]
    return pred_test - radius, pred_test + radius


def load_weights(
    root: Path,
    data: EvaluationData,
    family: str,
    seed: int,
    ids_cal: np.ndarray,
    ids_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    result_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    weights_path = result_dir / "domain_weights.csv"
    report_path = result_dir / "domain_shift_metrics.json"
    if weights_path.exists() and report_path.exists():
        frame = pd.read_csv(weights_path)
        cal = frame.loc[frame["domain"] == "source_cal", "weight"].to_numpy(float)
        test = frame.loc[frame["domain"] == "test", "weight"].to_numpy(float)
        if len(cal) == len(ids_cal) and len(test) == len(ids_test):
            return cal, test, json.loads(report_path.read_text(encoding="utf-8"))
    cal, test, report = estimate_domain_weights(
        data.x[data.rows(ids_cal)], data.x[data.rows(ids_test)], seed=30_000 + seed
    )
    return cal, test, report


def evaluate_one(root: Path, data: EvaluationData, family: str, seed: int) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    phase1_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    phase3_dir = root / "results" / "phase3" / f"{family}_seed_{seed}"
    output_dir = root / "results" / "phase5" / f"{family}_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    point = np.load(phase1_dir / "mlp_predictions.npz")
    cqr = np.load(output_dir / "cqr_predictions.npz")
    names = ("source_cal", "valid", "test")
    ids = {name: point[f"nsc_{name}"].astype("int64") for name in names}
    prediction = {name: point[f"pred_{name}"].astype(float) for name in names}
    truth = {name: data.truth(ids[name]) for name in names}
    for name in names:
        if not np.array_equal(ids[name], cqr[f"nsc_{name}"].astype("int64")):
            raise ValueError(f"CQR and point-prediction IDs disagree for {family} seed {seed} {name}")

    fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
    similarity = {name: sim(phase3_dir, name, data, fit_ids, ids[name]) for name in names}
    cal_weights, test_weights, shift_report = load_weights(
        root, data, family, seed, ids["source_cal"], ids["test"]
    )

    neighbour_path = output_dir / f"dad_k{DAD_K}_neighbours.npz"
    if neighbour_path.exists():
        neighbour_indices = np.load(neighbour_path)["indices"].astype(np.int32)
    else:
        neighbour_indices = top_k_tanimoto_indices(
            data.x[data.rows(ids["source_cal"])], data.x[data.rows(ids["test"])], DAD_K
        )
        np.savez_compressed(neighbour_path, indices=neighbour_indices)

    shared_model, _, shared_diag = fit_similarity_scale(
        similarity["valid"], truth["valid"], prediction["valid"]
    )
    shared_cal = predict_similarity_scale(shared_model, similarity["source_cal"])
    shared_test = predict_similarity_scale(shared_model, similarity["test"])

    multi_rows: list[dict] = []
    multi_bins: list[dict] = []
    multi_cells: list[dict] = []
    for alpha in ALPHAS:
        methods = {
            "Reference_global_CP": standard_intervals(
                truth["source_cal"], prediction["source_cal"], prediction["test"], alpha
            ),
            "B1_CQR": cqr_intervals(
                truth["source_cal"],
                cqr["pred_source_cal"].astype(float),
                cqr["pred_test"].astype(float),
                cqr["quantiles"].astype(float),
                alpha,
            ),
            "B2_dAD_style_k250": dad_intervals(
                truth["source_cal"],
                prediction["source_cal"],
                prediction["test"],
                neighbour_indices,
                alpha,
            ),
            "B3_estimated_WCP": weighted_intervals(
                truth["source_cal"],
                prediction["source_cal"],
                prediction["test"],
                cal_weights,
                test_weights,
                alpha,
            ),
            "Proposed_shared_monotone": scaled_intervals(
                truth["source_cal"],
                prediction["source_cal"],
                prediction["test"],
                shared_cal,
                shared_test,
                alpha,
            ),
        }
        for method, (lower, upper) in methods.items():
            multi_rows.append(
                {"family": family, "seed": seed, "alpha": alpha, "method": method, **interval_metrics(truth["test"], lower, upper, alpha)}
            )
            multi_bins.extend(
                {"family": family, "seed": seed, "alpha": alpha, "method": method, **row}
                for row in similarity_metrics(similarity["test"], truth["test"], lower, upper, alpha)
            )
            multi_cells.extend(
                {"family": family, "seed": seed, "alpha": alpha, "method": method, **row}
                for row in cell_metrics(truth["test"], lower, upper, alpha)
            )

    cubic_model, cubic_diag = fit_unconstrained_cubic_scale(
        similarity["valid"], truth["valid"], prediction["valid"]
    )
    per_cell_model, per_cell_diag = fit_per_cell_monotone_scale(
        similarity["valid"], truth["valid"], prediction["valid"]
    )
    ablation_scales = {
        "shared_monotone_isotonic": (shared_cal, shared_test),
        "shared_unconstrained_cubic": (
            predict_unconstrained_cubic_scale(cubic_model, similarity["source_cal"]),
            predict_unconstrained_cubic_scale(cubic_model, similarity["test"]),
        ),
        "per_cell_monotone_isotonic": (
            predict_per_cell_monotone_scale(per_cell_model, similarity["source_cal"]),
            predict_per_cell_monotone_scale(per_cell_model, similarity["test"]),
        ),
    }
    ablation_rows: list[dict] = []
    ablation_bins: list[dict] = []
    ablation_cells: list[dict] = []
    for alpha in ALPHAS:
        for method, (cal_scale, test_scale) in ablation_scales.items():
            lower, upper = scaled_intervals(
                truth["source_cal"], prediction["source_cal"], prediction["test"], cal_scale, test_scale, alpha
            )
            ablation_rows.append(
                {"family": family, "seed": seed, "alpha": alpha, "method": method, **interval_metrics(truth["test"], lower, upper, alpha)}
            )
            ablation_bins.extend(
                {"family": family, "seed": seed, "alpha": alpha, "method": method, **row}
                for row in similarity_metrics(similarity["test"], truth["test"], lower, upper, alpha)
            )
            ablation_cells.extend(
                {"family": family, "seed": seed, "alpha": alpha, "method": method, **row}
                for row in cell_metrics(truth["test"], lower, upper, alpha)
            )

    diagnostics = {
        "family": family,
        "seed": seed,
        "dad": {"k_compounds": DAD_K, "target_neighbour_rule": "same cell line (q=1)"},
        "wcp": shift_report,
        "scale": {
            "shared_monotone_isotonic": shared_diag,
            "shared_unconstrained_cubic": cubic_diag,
            "per_cell_monotone_isotonic": per_cell_diag,
        },
    }
    (output_dir / "extension_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return multi_rows, multi_bins, multi_cells, ablation_rows, ablation_bins, ablation_cells, diagnostics


def summarize_seed1(multi: pd.DataFrame, ablation: pd.DataFrame) -> dict:
    ood = multi[multi["family"].isin(("scaffold", "leader_cluster"))]
    at_90 = ood[np.isclose(ood["alpha"], 0.10)]
    baseline = at_90[at_90["method"] == "Reference_global_CP"]["coverage_error_abs"].mean()
    method_summary = []
    for method, part in ood.groupby("method"):
        method_summary.append(
            {
                "method": method,
                "mean_absolute_calibration_error_across_ood_and_alphas": float(part["coverage_error_abs"].mean()),
                "mean_width_across_ood_and_alphas": float(part["mean_width"].replace(np.inf, np.nan).mean()),
                "coverage_error_reduction_vs_global_at_90": float(
                    1.0 - at_90[at_90["method"] == method]["coverage_error_abs"].mean() / baseline
                ) if baseline > 0 else np.nan,
            }
        )
    ablation_summary = []
    for method, part in ablation.groupby("method"):
        ood_part = part[part["family"].isin(("scaffold", "leader_cluster"))]
        ablation_summary.append(
            {
                "method": method,
                "ood_mean_coverage": float(ood_part["coverage"].mean()),
                "ood_mean_coverage_error_abs": float(ood_part["coverage_error_abs"].mean()),
                "ood_mean_width": float(ood_part["mean_width"].mean()),
            }
        )
    return {"status": "development_seed_only_not_confirmatory", "multi_alpha": method_summary, "scale_ablation": ablation_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--families", default="random,scaffold,leader_cluster")
    parser.add_argument("--seeds", default="1")
    args = parser.parse_args()
    root = Path(args.root)
    data = EvaluationData(root)
    families = tuple(value for value in args.families.split(",") if value)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    multi_rows, multi_bins, multi_cells, ablation_rows, ablation_bins, ablation_cells = [], [], [], [], [], []
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for seed in seeds:
            m, mb, mc, a, ab, ac, _ = evaluate_one(root, data, family, seed)
            multi_rows.extend(m)
            multi_bins.extend(mb)
            multi_cells.extend(mc)
            ablation_rows.extend(a)
            ablation_bins.extend(ab)
            ablation_cells.extend(ac)
            print(f"completed family={family} seed={seed}", flush=True)

    output_dir = root / "results" / "phase5"
    output_dir.mkdir(parents=True, exist_ok=True)
    multi = pd.DataFrame(multi_rows)
    ablation = pd.DataFrame(ablation_rows)
    multi.to_csv(output_dir / "multi_alpha_seed_metrics.csv", index=False)
    pd.DataFrame(multi_bins).to_csv(output_dir / "multi_alpha_similarity_metrics.csv", index=False)
    pd.DataFrame(multi_cells).to_csv(output_dir / "multi_alpha_cell_metrics.csv", index=False)
    ablation.to_csv(output_dir / "scale_ablation_seed_metrics.csv", index=False)
    pd.DataFrame(ablation_bins).to_csv(output_dir / "scale_ablation_similarity_metrics.csv", index=False)
    pd.DataFrame(ablation_cells).to_csv(output_dir / "scale_ablation_cell_metrics.csv", index=False)
    if seeds == (1,):
        summary = summarize_seed1(multi, ablation)
        (output_dir / "seed1_development_findings.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

