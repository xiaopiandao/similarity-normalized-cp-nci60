"""Evaluate internal UACQR-P intervals without modifying manuscript artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.evaluate_phase1a_conformal import EvaluationData
    from scripts.evaluate_phase3_local_and_screening import sim
    from scripts.evaluate_phase5_extensions import (
        ALPHAS,
        FAMILIES,
        cell_metrics,
        cqr_intervals,
        evaluate_one,
        fit_similarity_scale,
        interval_metrics,
        predict_similarity_scale,
        scaled_intervals,
        similarity_metrics,
        standard_intervals,
    )
except ModuleNotFoundError:
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]
    from evaluate_phase3_local_and_screening import sim  # type: ignore[no-redef]
    from evaluate_phase5_extensions import (  # type: ignore[no-redef]
        ALPHAS,
        FAMILIES,
        cell_metrics,
        cqr_intervals,
        evaluate_one,
        fit_similarity_scale,
        interval_metrics,
        predict_similarity_scale,
        scaled_intervals,
        similarity_metrics,
        standard_intervals,
    )


METHOD = "B1_UACQR_P"
OOD_FAMILIES = ("scaffold", "leader_cluster")


def extended_interval_metrics(
    truth: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> dict:
    """Add set-coverage and proper-score diagnostics to Phase 5 metrics."""
    base = interval_metrics(truth, lower, upper, alpha)
    observed = np.isfinite(truth)
    defined = observed & ~np.isnan(lower) & ~np.isnan(upper)
    covered = defined & (truth >= lower) & (truth <= upper)
    finite = defined & np.isfinite(lower) & np.isfinite(upper)
    widths = upper - lower
    scores = np.full_like(truth, np.nan, dtype=float)
    scores[defined] = widths[defined]
    below = defined & (truth < lower)
    above = defined & (truth > upper)
    scores[below] += 2.0 / alpha * (lower[below] - truth[below])
    scores[above] += 2.0 / alpha * (truth[above] - upper[above])
    base.update(
        {
            "set_coverage_including_infinite": float(covered.sum() / observed.sum()),
            "coverage_among_finite_intervals": float(covered[finite].mean()) if finite.any() else np.nan,
            "mean_interval_score": float(np.mean(scores[defined])) if defined.any() else np.nan,
            "mean_interval_score_finite": float(np.mean(scores[finite])) if finite.any() else np.nan,
            "crossed_interval_fraction": float((defined & (lower > upper)).sum() / observed.sum()),
        }
    )
    return base


def evaluate_uacqrp_one(
    root: Path, data: EvaluationData, family: str, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    run_dir = root / "results" / "phase6_uacqrp_exploratory" / f"{family}_seed_{seed}"
    intervals = np.load(run_dir / "uacqrp_intervals.npz")
    ids = intervals["nsc_test"].astype("int64")
    truth = data.truth(ids)
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
    phase3_dir = root / "results" / "phase3" / f"{family}_seed_{seed}"
    similarity = sim(phase3_dir, "test", data, fit_ids, ids)

    rows, bins, cells = [], [], []
    stored_alphas = intervals["alphas"].astype(float)
    for alpha in ALPHAS:
        index = int(np.flatnonzero(np.isclose(stored_alphas, alpha))[0])
        lower = intervals["lower"][index].astype(float)
        upper = intervals["upper"][index].astype(float)
        rows.append(
            {
                "family": family,
                "seed": seed,
                "alpha": alpha,
                "method": METHOD,
                **extended_interval_metrics(truth, lower, upper, alpha),
            }
        )
        bins.extend(
            {"family": family, "seed": seed, "alpha": alpha, "method": METHOD, **row}
            for row in similarity_metrics(similarity, truth, lower, upper, alpha)
        )
        cells.extend(
            {"family": family, "seed": seed, "alpha": alpha, "method": METHOD, **row}
            for row in cell_metrics(truth, lower, upper, alpha)
        )
    return rows, bins, cells


def evaluate_proper_scores_one(
    root: Path, data: EvaluationData, family: str, seed: int
) -> list[dict]:
    """Reconstruct core intervals and score them with the Winkler interval score."""
    phase1_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    phase3_dir = root / "results" / "phase3" / f"{family}_seed_{seed}"
    phase5_dir = root / "results" / "phase5" / f"{family}_seed_{seed}"
    run_dir = root / "results" / "phase6_uacqrp_exploratory" / f"{family}_seed_{seed}"
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    point = np.load(phase1_dir / "mlp_predictions.npz")
    cqr = np.load(phase5_dir / "cqr_predictions.npz")
    uacqrp = np.load(run_dir / "uacqrp_intervals.npz")
    names = ("source_cal", "valid", "test")
    ids = {name: point[f"nsc_{name}"].astype("int64") for name in names}
    prediction = {name: point[f"pred_{name}"].astype(float) for name in names}
    truth = {name: data.truth(ids[name]) for name in names}
    fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
    similarity = {
        name: sim(phase3_dir, name, data, fit_ids, ids[name]) for name in names
    }
    shared_model, _, _ = fit_similarity_scale(
        similarity["valid"], truth["valid"], prediction["valid"]
    )
    shared_cal = predict_similarity_scale(shared_model, similarity["source_cal"])
    shared_test = predict_similarity_scale(shared_model, similarity["test"])
    stored_alphas = uacqrp["alphas"].astype(float)
    rows = []
    for alpha in ALPHAS:
        u_index = int(np.flatnonzero(np.isclose(stored_alphas, alpha))[0])
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
            METHOD: (
                uacqrp["lower"][u_index].astype(float),
                uacqrp["upper"][u_index].astype(float),
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
            rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "alpha": alpha,
                    "method": method,
                    **extended_interval_metrics(truth["test"], lower, upper, alpha),
                }
            )
    return rows


def low_similarity_rows(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame["bin_order"].isin((0, 1))]
    rows = []
    for keys, part in selected.groupby(["family", "seed", "alpha", "method"]):
        weights = part["n_labels"].to_numpy(float)
        coverage = float(np.average(part["coverage"], weights=weights))
        rows.append(
            {
                "family": keys[0],
                "seed": keys[1],
                "alpha": keys[2],
                "method": keys[3],
                "coverage": coverage,
                "coverage_error_abs": abs(coverage - (1.0 - keys[2])),
                "mean_width": float(np.average(part["mean_width"], weights=weights)),
                "n_labels": int(weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    metrics: pd.DataFrame,
    similarity: pd.DataFrame,
    proper_scores: pd.DataFrame,
    seeds: tuple[int, ...],
) -> dict:
    ood = metrics[metrics["family"].isin(OOD_FAMILIES)]
    method_rows = []
    for method, part in ood.groupby("method"):
        method_rows.append(
            {
                "method": method,
                "ood_multi_alpha_mean_absolute_calibration_error": float(part["coverage_error_abs"].mean()),
                "ood_multi_alpha_mean_width": float(part["mean_width"].replace(np.inf, np.nan).mean()),
                "minimum_finite_interval_fraction": float(part["finite_interval_fraction"].min()),
            }
        )
    at_90 = metrics[np.isclose(metrics["alpha"], 0.10)]
    family_90 = (
        at_90.groupby(["family", "method"], as_index=False)
        .agg(coverage=("coverage", "mean"), mean_width=("mean_width", "mean"), finite_fraction=("finite_interval_fraction", "min"))
        .to_dict(orient="records")
    )
    low = low_similarity_rows(similarity)
    low_summary = (
        low[low["family"].isin(OOD_FAMILIES)]
        .groupby("method", as_index=False)
        .agg(coverage_error_abs=("coverage_error_abs", "mean"), mean_width=("mean_width", "mean"))
        .to_dict(orient="records")
    )
    proper_ood = proper_scores[proper_scores["family"].isin(OOD_FAMILIES)]
    proper_summary = (
        proper_ood.groupby("method", as_index=False)
        .agg(
            mean_interval_score=("mean_interval_score", "mean"),
            mean_interval_score_finite=("mean_interval_score_finite", "mean"),
            mean_width=("mean_width", "mean"),
            minimum_finite_interval_fraction=("finite_interval_fraction", "min"),
        )
        .to_dict(orient="records")
    )
    return {
        "status": "internal_exploratory_not_for_manuscript",
        "seeds": list(seeds),
        "alpha_levels": list(ALPHAS),
        "method_summary": method_rows,
        "coverage_90_by_family": family_90,
        "low_similarity_below_0_4_ood_summary": low_summary,
        "ood_interval_score_summary": proper_summary,
        "interpretation_guardrail": "Results describe this frozen implementation only; no manuscript inclusion decision is made here.",
    }


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
    rows, bins, cells, proper_rows = [], [], [], []
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for seed in seeds:
            existing, existing_bins, existing_cells, _, _, _, _ = evaluate_one(
                root, data, family, seed
            )
            u_rows, u_bins, u_cells = evaluate_uacqrp_one(root, data, family, seed)
            rows.extend(existing)
            rows.extend(u_rows)
            bins.extend(existing_bins)
            bins.extend(u_bins)
            cells.extend(existing_cells)
            cells.extend(u_cells)
            proper_rows.extend(evaluate_proper_scores_one(root, data, family, seed))
            print(f"evaluated family={family} seed={seed}", flush=True)

    output = root / "results" / "phase6_uacqrp_exploratory"
    if seeds == (1,):
        prefix = "development_"
    elif seeds == (2, 3, 4, 5):
        prefix = "confirmation_"
    else:
        prefix = "all_"
    metrics = pd.DataFrame(rows)
    similarity = pd.DataFrame(bins)
    proper_scores = pd.DataFrame(proper_rows)
    metrics.to_csv(output / f"{prefix}comparison_seed_metrics.csv", index=False)
    similarity.to_csv(output / f"{prefix}comparison_similarity_metrics.csv", index=False)
    pd.DataFrame(cells).to_csv(output / f"{prefix}comparison_cell_metrics.csv", index=False)
    proper_scores.to_csv(output / f"{prefix}proper_score_seed_metrics.csv", index=False)
    summary = summarize(metrics, similarity, proper_scores, seeds)
    (output / f"{prefix}summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

