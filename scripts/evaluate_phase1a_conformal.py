"""Evaluate marginal and compound-profile conformal prediction under shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


FAMILIES = ("random", "scaffold", "leader_cluster")
MODELS = ("ridge", "mlp")


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return np.nan
    level = min(1.0, np.ceil((len(clean) + 1) * (1 - alpha)) / len(clean))
    return float(np.quantile(clean, level, method="higher"))


def weighted_thresholds(
    scores: np.ndarray,
    calibration_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    calibration_weights = np.asarray(calibration_weights, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(calibration_weights) & (calibration_weights > 0)
    scores = scores[valid]
    calibration_weights = calibration_weights[valid]
    if not len(scores):
        return np.full(len(test_weights), np.nan)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    cumulative = np.cumsum(calibration_weights[order])
    targets = (1 - alpha) * (cumulative[-1] + np.asarray(test_weights, dtype=float))
    indices = np.searchsorted(cumulative, targets, side="left")
    result = np.full(len(targets), np.inf)
    finite = indices < len(sorted_scores)
    result[finite] = sorted_scores[indices[finite]]
    return result


def load_ids(path: Path) -> np.ndarray:
    return pd.read_csv(path)["nsc"].to_numpy("int64")


class EvaluationData:
    def __init__(self, root: Path) -> None:
        feature_dir = root / "data" / "features" / "cellminer"
        self.x = sparse.load_npz(feature_dir / "morgan_r2_2048.npz").astype(np.float32)
        feature_index = pd.read_csv(feature_dir / "morgan_r2_2048_index.csv")
        self.row_by_nsc = {
            int(nsc): row for row, nsc in enumerate(feature_index["nsc"].astype(int))
        }
        self.responses = pd.read_csv(
            root / "data" / "processed" / "cellminer" / "response_matrix_loggi50.csv.gz"
        ).set_index("nsc")
        self.cell_lines = self.responses.columns.tolist()

    def rows(self, ids: np.ndarray) -> np.ndarray:
        return np.fromiter((self.row_by_nsc[int(value)] for value in ids), dtype=np.int64)

    def truth(self, ids: np.ndarray) -> np.ndarray:
        return self.responses.loc[ids].to_numpy(dtype=np.float32)


def estimate_domain_weights(
    x_source: sparse.csr_matrix,
    x_target: sparse.csr_matrix,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x = sparse.vstack([x_source, x_target], format="csr")
    labels = np.concatenate(
        [np.zeros(x_source.shape[0], dtype=int), np.ones(x_target.shape[0], dtype=int)]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    candidate_c = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
    candidates = []
    candidate_probabilities = []
    for c_value in candidate_c:
        classifier = LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        )
        candidate_probability = cross_val_predict(
            classifier, x, labels, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        candidate_probability = np.clip(candidate_probability, 1e-6, 1 - 1e-6)
        candidates.append(
            {
                "C": c_value,
                "oof_auc": float(roc_auc_score(labels, candidate_probability)),
                "oof_log_loss": float(log_loss(labels, candidate_probability)),
                "oof_brier": float(brier_score_loss(labels, candidate_probability)),
            }
        )
        candidate_probabilities.append(candidate_probability)
    best_index = int(np.argmin([item["oof_log_loss"] for item in candidates]))
    probabilities = candidate_probabilities[best_index]
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    prior_correction = x_source.shape[0] / x_target.shape[0]
    weights = probabilities / (1 - probabilities) * prior_correction
    source_weights = weights[: x_source.shape[0]]
    target_weights = weights[x_source.shape[0] :]
    ess = float(source_weights.sum() ** 2 / np.sum(source_weights**2))
    report = {
        "oof_domain_auc": float(roc_auc_score(labels, probabilities)),
        "source_weight_min": float(source_weights.min()),
        "source_weight_median": float(np.median(source_weights)),
        "source_weight_mean": float(source_weights.mean()),
        "source_weight_q95": float(np.quantile(source_weights, 0.95)),
        "source_weight_q99": float(np.quantile(source_weights, 0.99)),
        "source_weight_max": float(source_weights.max()),
        "ess": ess,
        "ess_fraction": ess / len(source_weights),
        "target_weight_median": float(np.median(target_weights)),
        "target_weight_q99": float(np.quantile(target_weights, 0.99)),
        "target_weight_max": float(target_weights.max()),
        "selected_C": candidates[best_index]["C"],
        "regularization_grid": candidates,
        "estimation": (
            "5-fold out-of-fold logistic domain classifier; L2 regularization "
            "selected by minimum domain-label OOF log loss without response labels"
        ),
    }
    return source_weights, target_weights, report


def nearest_fit_similarity(
    x_fit: sparse.csr_matrix, x_test: sparse.csr_matrix, block_size: int = 64
) -> np.ndarray:
    fit_counts = np.asarray(x_fit.sum(axis=1)).ravel().astype(np.float32)
    result = np.zeros(x_test.shape[0], dtype=np.float32)
    for start in range(0, x_test.shape[0], block_size):
        stop = min(start + block_size, x_test.shape[0])
        block = x_test[start:stop]
        intersections = (block @ x_fit.T).toarray().astype(np.float32)
        test_counts = np.asarray(block.sum(axis=1)).ravel().astype(np.float32)
        unions = test_counts[:, None] + fit_counts[None, :] - intersections
        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections),
            where=unions > 0,
        )
        result[start:stop] = similarities.max(axis=1)
    return result


def interval_metrics(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    level: str,
) -> dict:
    observed = np.isfinite(truth)
    covered = (truth >= lower) & (truth <= upper) & observed
    width = upper - lower
    finite_width = width[np.isfinite(width) & observed]
    if level == "marginal":
        coverage = float(covered.sum() / observed.sum())
        n_observations = int(observed.sum())
        n_compounds = int(truth.shape[0])
    elif level == "simultaneous_profile":
        complete = observed.all(axis=1)
        profile_covered = covered[complete].all(axis=1)
        coverage = float(profile_covered.mean()) if complete.any() else np.nan
        n_observations = int(complete.sum() * truth.shape[1])
        n_compounds = int(complete.sum())
    else:
        raise ValueError(level)
    return {
        "level": level,
        "coverage": coverage,
        "coverage_error_abs": abs(coverage - 0.9),
        "mean_width_finite": float(finite_width.mean()) if len(finite_width) else np.nan,
        "median_width_finite": float(np.median(finite_width)) if len(finite_width) else np.nan,
        "finite_interval_fraction": float((np.isfinite(width) & observed).sum() / observed.sum()),
        "n_compounds": n_compounds,
        "n_observations": n_observations,
    }


def similarity_bin_metrics(
    similarity: np.ndarray,
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    method: str,
) -> list[dict]:
    bins = np.asarray([0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001])
    labels = ["[0,.4)", "[.4,.5)", "[.5,.6)", "[.6,.7)", "[.7,.8)", "[.8,1]"]
    groups = np.digitize(similarity, bins[1:-1], right=False)
    rows = []
    observed = np.isfinite(truth)
    covered = (truth >= lower) & (truth <= upper) & observed
    complete = observed.all(axis=1)
    for index, label in enumerate(labels):
        selected = groups == index
        marginal_n = int(observed[selected].sum())
        complete_selected = selected & complete
        rows.append(
            {
                "method": method,
                "similarity_bin": label,
                "compounds": int(selected.sum()),
                "marginal_observations": marginal_n,
                "marginal_coverage": (
                    float(covered[selected].sum() / marginal_n) if marginal_n else np.nan
                ),
                "complete_profiles": int(complete_selected.sum()),
                "simultaneous_coverage": (
                    float(covered[complete_selected].all(axis=1).mean())
                    if complete_selected.any()
                    else np.nan
                ),
            }
        )
    return rows


def evaluate_one(
    root: Path,
    data: EvaluationData,
    family: str,
    seed: int,
    model: str,
    alpha: float,
) -> tuple[list[dict], list[dict], dict]:
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    result_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    prediction_path = result_dir / f"{model}_predictions.npz"
    stored = np.load(prediction_path)
    ids_cal = stored["nsc_source_cal"].astype("int64")
    ids_valid = stored["nsc_valid"].astype("int64")
    ids_test = stored["nsc_test"].astype("int64")
    pred_cal = stored["pred_source_cal"].astype(float)
    pred_valid = stored["pred_valid"].astype(float)
    pred_test = stored["pred_test"].astype(float)
    truth_cal = data.truth(ids_cal)
    truth_valid = data.truth(ids_valid)
    truth_test = data.truth(ids_test)

    fit_ids = load_ids(split_dir / "fit_nsc.csv")
    x_fit = data.x[data.rows(fit_ids)]
    x_cal = data.x[data.rows(ids_cal)]
    x_test = data.x[data.rows(ids_test)]
    weights_path = result_dir / "domain_weights.csv"
    shift_path = result_dir / "domain_shift_metrics.json"
    similarity_path = result_dir / "nearest_fit_tanimoto.csv"
    cached_shift = (
        json.loads(shift_path.read_text(encoding="utf-8")) if shift_path.exists() else {}
    )
    cache_current = "selected_C" in cached_shift
    if weights_path.exists() and shift_path.exists() and similarity_path.exists() and cache_current:
        weights_frame = pd.read_csv(weights_path)
        cal_weights = weights_frame.loc[weights_frame["domain"] == "source_cal", "weight"].to_numpy()
        test_weights = weights_frame.loc[weights_frame["domain"] == "test", "weight"].to_numpy()
        shift_report = json.loads(shift_path.read_text(encoding="utf-8"))
        similarity = pd.read_csv(similarity_path)["nearest_fit_tanimoto"].to_numpy()
    else:
        cal_weights, test_weights, shift_report = estimate_domain_weights(
            x_cal, x_test, seed=30_000 + seed
        )
        pd.DataFrame(
            {
                "nsc": np.concatenate([ids_cal, ids_test]),
                "domain": ["source_cal"] * len(ids_cal) + ["test"] * len(ids_test),
                "weight": np.concatenate([cal_weights, test_weights]),
            }
        ).to_csv(weights_path, index=False)
        shift_path.write_text(json.dumps(shift_report, indent=2), encoding="utf-8")
        similarity = nearest_fit_similarity(x_fit, x_test)
        pd.DataFrame(
            {"nsc": ids_test, "nearest_fit_tanimoto": similarity}
        ).to_csv(similarity_path, index=False)

    cal_residual = np.abs(truth_cal - pred_cal)
    test_methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    standard_q = np.asarray(
        [finite_sample_quantile(cal_residual[:, cell], alpha) for cell in range(truth_cal.shape[1])]
    )
    test_methods["M1_marginal_unweighted"] = (pred_test - standard_q, pred_test + standard_q)

    weighted_q = np.column_stack(
        [
            weighted_thresholds(
                cal_residual[:, cell], cal_weights, test_weights, alpha
            )
            for cell in range(truth_cal.shape[1])
        ]
    )
    test_methods["M2_marginal_weighted"] = (pred_test - weighted_q, pred_test + weighted_q)

    valid_residual = np.abs(truth_valid - pred_valid)
    scales = np.nanmedian(valid_residual, axis=0)
    fallback_scale = float(np.nanmedian(scales[np.isfinite(scales) & (scales > 1e-6)]))
    scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, fallback_scale)
    complete_cal = np.isfinite(truth_cal).all(axis=1)
    profile_scores = np.max(cal_residual[complete_cal] / scales[None, :], axis=1)
    profile_q = finite_sample_quantile(profile_scores, alpha)
    test_methods["M3_profile_unweighted"] = (
        pred_test - profile_q * scales,
        pred_test + profile_q * scales,
    )
    weighted_profile_q = weighted_thresholds(
        profile_scores, cal_weights[complete_cal], test_weights, alpha
    )
    test_methods["M4_profile_weighted"] = (
        pred_test - weighted_profile_q[:, None] * scales[None, :],
        pred_test + weighted_profile_q[:, None] * scales[None, :],
    )

    metric_rows: list[dict] = []
    bin_rows: list[dict] = []
    for method, (lower, upper) in test_methods.items():
        levels = ["marginal"]
        if "profile" in method:
            levels.append("simultaneous_profile")
        for level in levels:
            metric_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "model": model,
                    "method": method,
                    "alpha": alpha,
                    "nominal_coverage": 1 - alpha,
                    **interval_metrics(truth_test, lower, upper, level),
                }
            )
        for row in similarity_bin_metrics(similarity, truth_test, lower, upper, method):
            bin_rows.append({"family": family, "seed": seed, "model": model, **row})

    detail = {
        "family": family,
        "seed": seed,
        "model": model,
        "alpha": alpha,
        "complete_source_calibration_profiles": int(complete_cal.sum()),
        "complete_test_profiles": int(np.isfinite(truth_test).all(axis=1).sum()),
        "profile_scale_source": "per-cell median absolute residual on validation set",
        "domain_shift": shift_report,
        "standard_profile_quantile": profile_q,
        "weighted_profile_infinite_fraction": float(np.isinf(weighted_profile_q).mean()),
        "test_similarity": {
            "mean": float(similarity.mean()),
            "median": float(np.median(similarity)),
            "q95": float(np.quantile(similarity, 0.95)),
            "max": float(similarity.max()),
        },
    }
    (result_dir / f"{model}_conformal_detail.json").write_text(
        json.dumps(detail, indent=2), encoding="utf-8"
    )
    return metric_rows, bin_rows, detail


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
    metric_rows: list[dict] = []
    bin_rows: list[dict] = []
    shift_rows: list[dict] = []
    for family in [item for item in args.families.split(",") if item]:
        for seed in [int(item) for item in args.seeds.split(",") if item]:
            for model in [item for item in args.models.split(",") if item]:
                rows, bins, detail = evaluate_one(
                    root, data, family, seed, model, args.alpha
                )
                metric_rows.extend(rows)
                bin_rows.extend(bins)
                shift_rows.append(
                    {"family": family, "seed": seed, **detail["domain_shift"]}
                )
                print(
                    f"completed family={family} seed={seed} model={model}", flush=True
                )

    output = root / "results" / "phase1a"
    pd.DataFrame(metric_rows).to_csv(output / "conformal_summary.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(output / "coverage_by_tanimoto.csv", index=False)
    pd.DataFrame(shift_rows).drop_duplicates(["family", "seed"]).to_csv(
        output / "domain_shift_summary.csv", index=False
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()

