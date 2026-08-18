"""Phase 4 statistical validation and chemical-neighbourhood diagnostics.

This script does not train or tune a predictive model.  It reconstructs the
frozen M1/M3/M5 intervals for confirmation seeds 2--5 and produces:

* seed-level descriptive 95% CIs;
* compound-clustered paired bootstrap CIs and sign-randomisation p-values;
* Wilson intervals for high-confidence screening threshold scans;
* coverage/width curves over pre-specified chemical-similarity bins;
* exact Tanimoto diagnostics for the fixed-k local calibration baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats

try:
    from scripts.evaluate_phase1a_conformal import EvaluationData
    from scripts.evaluate_phase1b_similarity import (
        fit_similarity_scale,
        similarity_normalized_intervals,
        standard_intervals,
    )
    from scripts.evaluate_phase3_local_and_screening import K, mondrian, sim
except ModuleNotFoundError:
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]
    from evaluate_phase1b_similarity import (  # type: ignore[no-redef]
        fit_similarity_scale,
        similarity_normalized_intervals,
        standard_intervals,
    )
    from evaluate_phase3_local_and_screening import K, mondrian, sim  # type: ignore[no-redef]


FAMILIES = ("random", "scaffold", "leader_cluster")
METHODS = ("M1_global", "M3_mondrian", "M5_similarity_normalized")
METHOD_LABELS = {
    "M1_global": "M1 global CP",
    "M3_mondrian": "M3 Mondrian CP",
    "M5_similarity_normalized": "M5 similarity-normalized CP",
    "Point_prediction_threshold": "Point prediction",
}
SIMILARITY_EDGES = np.asarray([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001])
SIMILARITY_LABELS = (
    "[0,.3)",
    "[.3,.4)",
    "[.4,.5)",
    "[.5,.6)",
    "[.6,.7)",
    "[.7,.8)",
    "[.8,1]",
)


def mean_t_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, np.nan, np.nan, np.nan
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf((1 + confidence) / 2, len(values) - 1) * sd / np.sqrt(len(values)))
    return mean, sd, mean - half, mean + half


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z = float(stats.norm.ppf((1 + confidence) / 2))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(centre - half), float(centre + half)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    p = values[valid]
    if not len(p):
        return result
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[valid] = restored
    return result


def paired_compound_inference(
    covered_a: np.ndarray,
    covered_b: np.ndarray,
    observed: np.ndarray,
    rng: np.random.Generator,
    bootstrap_reps: int,
    randomization_reps: int,
) -> dict:
    """Paired inference with compounds as clusters and labels retained within rows."""
    covered_a = np.asarray(covered_a, dtype=float)
    covered_b = np.asarray(covered_b, dtype=float)
    observed = np.asarray(observed, dtype=float)
    valid = observed > 0
    covered_a, covered_b, observed = covered_a[valid], covered_b[valid], observed[valid]
    n = len(observed)
    observed_difference = float((covered_a.sum() - covered_b.sum()) / observed.sum())
    bootstrap = np.empty(bootstrap_reps, dtype=float)
    chunk = 200
    for start in range(0, bootstrap_reps, chunk):
        stop = min(start + chunk, bootstrap_reps)
        indices = rng.integers(0, n, size=(stop - start, n))
        num = covered_a[indices].sum(axis=1) - covered_b[indices].sum(axis=1)
        den = observed[indices].sum(axis=1)
        bootstrap[start:stop] = num / den
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    difference_by_compound = covered_a - covered_b
    denominator = observed.sum()
    exceedances = 0
    completed = 0
    for start in range(0, randomization_reps, chunk):
        stop = min(start + chunk, randomization_reps)
        signs = rng.choice((-1.0, 1.0), size=(stop - start, n))
        null_difference = (signs * difference_by_compound[None, :]).sum(axis=1) / denominator
        exceedances += int((np.abs(null_difference) >= abs(observed_difference)).sum())
        completed += stop - start
    p_value = (exceedances + 1) / (completed + 1)
    return {
        "difference": observed_difference,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "paired_randomization_p": float(p_value),
        "n_compounds": int(n),
        "n_labels": int(observed.sum()),
        "bootstrap_reps": int(bootstrap_reps),
        "randomization_reps": int(randomization_reps),
    }


def reconstruct_intervals(root: Path, data: EvaluationData, family: str, seed: int) -> dict:
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    source_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    cache_dir = root / "results" / "phase3" / f"{family}_seed_{seed}"
    stored = np.load(source_dir / "mlp_predictions.npz")
    ids = {name: stored[f"nsc_{name}"].astype("int64") for name in ("source_cal", "valid", "test")}
    prediction = {name: stored[f"pred_{name}"].astype(float) for name in ids}
    truth = {name: data.truth(ids[name]) for name in ids}
    fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
    similarity = {
        name: sim(cache_dir, name, data, fit_ids, ids[name])
        for name in ("source_cal", "valid", "test")
    }
    lower_m1, upper_m1 = standard_intervals(
        truth["source_cal"], prediction["source_cal"], prediction["test"], 0.1
    )
    lower_m3, upper_m3, _ = mondrian(
        truth["source_cal"],
        prediction["source_cal"],
        similarity["source_cal"],
        prediction["test"],
        similarity["test"],
    )
    scale_model, _, scale_diagnostics = fit_similarity_scale(
        similarity["valid"], truth["valid"], prediction["valid"]
    )
    lower_m5, upper_m5, _, test_scale = similarity_normalized_intervals(
        truth["source_cal"],
        prediction["source_cal"],
        similarity["source_cal"],
        similarity["test"],
        scale_model,
        prediction["test"],
        0.1,
    )
    return {
        "ids": ids,
        "prediction": prediction,
        "truth": truth,
        "fit_ids": fit_ids,
        "similarity": similarity,
        "intervals": {
            "M1_global": (lower_m1, upper_m1),
            "M3_mondrian": (lower_m3, upper_m3),
            "M5_similarity_normalized": (lower_m5, upper_m5),
        },
        "test_scale": test_scale,
        "scale_diagnostics": scale_diagnostics,
    }


def coverage_counts(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.isfinite(truth)
    covered = observed & (truth >= lower) & (truth <= upper)
    return covered.sum(axis=1).astype(float), observed.sum(axis=1).astype(float)


def interval_rows(family: str, seed: int, payload: dict) -> tuple[list[dict], list[dict], dict]:
    truth = payload["truth"]["test"]
    similarity = payload["similarity"]["test"]
    observed = np.isfinite(truth)
    overall_rows: list[dict] = []
    bin_rows: list[dict] = []
    compound_counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    groups = np.digitize(similarity, SIMILARITY_EDGES[1:-1])
    for method, (lower, upper) in payload["intervals"].items():
        covered = observed & (truth >= lower) & (truth <= upper)
        width = upper - lower
        per_compound_covered, per_compound_observed = coverage_counts(truth, lower, upper)
        compound_counts[method] = (per_compound_covered, per_compound_observed)
        overall_rows.append(
            {
                "family": family,
                "seed": seed,
                "method": method,
                "coverage": float(covered.sum() / observed.sum()),
                "mean_width": float(np.nanmean(width[observed])),
                "n_compounds": int(len(truth)),
                "n_labels": int(observed.sum()),
            }
        )
        for group, label in enumerate(SIMILARITY_LABELS):
            compound_mask = groups == group
            label_mask = observed & compound_mask[:, None]
            if not compound_mask.any() or not label_mask.any():
                continue
            bin_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "method": method,
                    "similarity_bin": label,
                    "bin_order": group,
                    "coverage": float(covered[label_mask].mean()),
                    "mean_width": float(width[label_mask].mean()),
                    "n_compounds": int(compound_mask.sum()),
                    "n_labels": int(label_mask.sum()),
                    "mean_similarity": float(similarity[compound_mask].mean()),
                }
            )
    return overall_rows, bin_rows, compound_counts


def screening_rows(family: str, seed: int, payload: dict) -> list[dict]:
    truth = payload["truth"]["test"]
    prediction = payload["prediction"]["test"]
    observed = np.isfinite(truth)
    truly_active = (truth >= 6.0) & observed
    lower_by_method = {
        method: intervals[0] for method, intervals in payload["intervals"].items()
    }
    lower_by_method["Point_prediction_threshold"] = prediction
    rows: list[dict] = []
    for threshold in (1, 3, 5, 10):
        true_molecule = truly_active.sum(axis=1) >= threshold
        total_true = int(true_molecule.sum())
        for method, lower in lower_by_method.items():
            selected_cell = (lower > 6.0) & observed
            selected = selected_cell.sum(axis=1) >= threshold
            selected_total = int(selected.sum())
            true_positive = int((selected & true_molecule).sum())
            false_positive = selected_total - true_positive
            precision = true_positive / selected_total if selected_total else np.nan
            recall = true_positive / total_true if total_true else np.nan
            precision_low, precision_high = wilson_interval(true_positive, selected_total)
            recall_low, recall_high = wilson_interval(true_positive, total_true)
            rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "method": method,
                    "strong_cell_threshold": threshold,
                    "selected": selected_total,
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "total_true": total_true,
                    "precision": precision,
                    "precision_wilson_low": precision_low,
                    "precision_wilson_high": precision_high,
                    "false_discovery_rate": 1 - precision if selected_total else np.nan,
                    "fdr_wilson_low": 1 - precision_high if selected_total else np.nan,
                    "fdr_wilson_high": 1 - precision_low if selected_total else np.nan,
                    "recall": recall,
                    "recall_wilson_low": recall_low,
                    "recall_wilson_high": recall_high,
                }
            )
    return rows


def m4_neighbour_diagnostics(
    data: EvaluationData, family: str, seed: int, payload: dict
) -> pd.DataFrame:
    ids_cal = payload["ids"]["source_cal"]
    ids_test = payload["ids"]["test"]
    cal_x = data.x[data.rows(ids_cal)]
    test_x = data.x[data.rows(ids_test)]
    cal_counts = np.asarray(cal_x.sum(axis=1)).ravel()
    test_counts = np.asarray(test_x.sum(axis=1)).ravel()
    cal_residual = np.abs(payload["truth"]["source_cal"] - payload["prediction"]["source_cal"])
    per_cell_median = np.nanmedian(cal_residual, axis=0)
    fallback = np.nanmedian(per_cell_median[np.isfinite(per_cell_median) & (per_cell_median > 0)])
    per_cell_median = np.where(
        np.isfinite(per_cell_median) & (per_cell_median > 0), per_cell_median, fallback
    )
    cal_difficulty = np.nanmedian(cal_residual / per_cell_median[None, :], axis=1)
    test_residual = np.abs(payload["truth"]["test"] - payload["prediction"]["test"])
    test_difficulty = np.nanmedian(test_residual / per_cell_median[None, :], axis=1)
    rows: list[dict] = []
    rank_indices = {"sim_rank1": 0, "sim_rank10": 9, "sim_rank50": 49, "sim_rank128": 127, "sim_rank512": 511}
    for start in range(0, len(ids_test), 32):
        stop = min(start + 32, len(ids_test))
        intersections = (test_x[start:stop] @ cal_x.T).toarray()
        unions = test_counts[start:stop, None] + cal_counts[None, :] - intersections
        similarity = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        indices = np.argpartition(similarity, -K, axis=1)[:, -K:]
        top_similarity = np.take_along_axis(similarity, indices, axis=1)
        top_order = np.argsort(top_similarity, axis=1)[:, ::-1]
        top_similarity = np.take_along_axis(top_similarity, top_order, axis=1)
        indices = np.take_along_axis(indices, top_order, axis=1)
        for local_row in range(stop - start):
            test_row = start + local_row
            neighbour_difficulty = cal_difficulty[indices[local_row]]
            row = {
                "family": family,
                "seed": seed,
                "nsc": int(ids_test[test_row]),
                "nearest_fit_similarity": float(payload["similarity"]["test"][test_row]),
                "mean_top512_similarity": float(top_similarity[local_row].mean()),
                "median_top512_similarity": float(np.median(top_similarity[local_row])),
                "local_difficulty_q90": float(np.quantile(neighbour_difficulty, 0.9, method="higher")),
                "local_difficulty_iqr": float(
                    np.quantile(neighbour_difficulty, 0.75)
                    - np.quantile(neighbour_difficulty, 0.25)
                ),
                "test_difficulty": float(test_difficulty[test_row]),
            }
            row.update(
                {
                    name: float(top_similarity[local_row, index])
                    for name, index in rank_indices.items()
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_across_seeds(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, part in frame.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(groups, keys))
        base["n_seeds"] = int(part["seed"].nunique())
        for metric in metrics:
            mean, sd, low, high = mean_t_interval(part[metric].to_numpy(float))
            base[f"{metric}_mean"] = mean
            base[f"{metric}_sd"] = sd
            base[f"{metric}_t95_low"] = low
            base[f"{metric}_t95_high"] = high
        rows.append(base)
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def plot_similarity_curve(summary: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    colors = {"M1_global": "#6B7280", "M5_similarity_normalized": "#2878B5"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.85), sharex=True)
    x = np.arange(len(SIMILARITY_LABELS))
    for column, family in enumerate(FAMILIES):
        part = summary[summary["family"] == family]
        for method in ("M1_global", "M5_similarity_normalized"):
            method_part = part[part["method"] == method].sort_values("bin_order")
            y = method_part["coverage_mean"].to_numpy()
            low = method_part["coverage_t95_low"].to_numpy()
            high = method_part["coverage_t95_high"].to_numpy()
            axes[0, column].plot(x, y, marker="o", lw=1.5, ms=3.2, color=colors[method], label=METHOD_LABELS[method])
            axes[0, column].fill_between(x, np.clip(low, 0, 1), np.clip(high, 0, 1), color=colors[method], alpha=0.14, linewidth=0)
            width = method_part["mean_width_mean"].to_numpy()
            width_low = method_part["mean_width_t95_low"].to_numpy()
            width_high = method_part["mean_width_t95_high"].to_numpy()
            axes[1, column].plot(x, width, marker="o", lw=1.5, ms=3.2, color=colors[method])
            axes[1, column].fill_between(x, width_low, width_high, color=colors[method], alpha=0.14, linewidth=0)
        axes[0, column].axhline(0.9, color="#B91C1C", ls="--", lw=1.0)
        axes[0, column].set_title(family.replace("_", " "))
        axes[0, column].set_ylim(0.72, 1.0)
        axes[1, column].set_xticks(x, SIMILARITY_LABELS, rotation=40, ha="right")
        axes[1, column].set_xlabel("Nearest-fit Tanimoto")
    axes[0, 0].set_ylabel("Marginal coverage")
    axes[1, 0].set_ylabel("Mean interval width")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].text(-0.20, 1.08, "a", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=8)
    axes[1, 0].text(-0.20, 1.08, "b", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def plot_screening_scan(summary: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    colors = {
        "Point_prediction_threshold": "#9CA3AF",
        "M1_global": "#6B7280",
        "M3_mondrian": "#D99036",
        "M5_similarity_normalized": "#2878B5",
    }
    markers = {"Point_prediction_threshold": "x", "M1_global": "o", "M3_mondrian": "s", "M5_similarity_normalized": "^"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), sharex=True)
    thresholds = np.asarray([1, 3, 5, 10])
    for column, family in enumerate(FAMILIES):
        part = summary[summary["family"] == family]
        for method in ("Point_prediction_threshold", "M1_global", "M3_mondrian", "M5_similarity_normalized"):
            method_part = part[part["method"] == method].sort_values("strong_cell_threshold")
            axes[0, column].plot(
                thresholds,
                method_part["false_discovery_rate_mean"],
                color=colors[method],
                marker=markers[method],
                lw=1.35,
                ms=3.4,
                label=METHOD_LABELS[method],
            )
            axes[1, column].plot(
                thresholds,
                method_part["selected_mean"],
                color=colors[method],
                marker=markers[method],
                lw=1.35,
                ms=3.4,
            )
        axes[0, column].set_title(family.replace("_", " "))
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].set_yscale("log")
        axes[1, column].set_xticks(thresholds)
        axes[1, column].set_xlabel("Minimum active cell lines")
    axes[0, 0].set_ylabel("False discovery rate")
    axes[1, 0].set_ylabel("Selected molecules (mean)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=6.1,
        handlelength=1.7,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    axes[0, 0].text(-0.20, 1.08, "a", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=8)
    axes[1, 0].text(-0.20, 1.08, "b", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=0.7, w_pad=0.9)
    save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", default="2,3,4,5")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--randomization-reps", type=int, default=10000)
    parser.add_argument("--skip-m4-diagnostics", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    output = root / "results" / "phase4"
    output.mkdir(parents=True, exist_ok=True)
    data = EvaluationData(root)
    seeds = tuple(int(value) for value in args.seeds.split(","))
    rng = np.random.default_rng(20260721)

    overall_rows: list[dict] = []
    bin_rows: list[dict] = []
    screening: list[dict] = []
    paired: list[dict] = []
    m4_frames: list[pd.DataFrame] = []
    scale_rows: list[dict] = []
    for family in FAMILIES:
        for seed in seeds:
            payload = reconstruct_intervals(root, data, family, seed)
            overall, bins, counts = interval_rows(family, seed, payload)
            overall_rows.extend(overall)
            bin_rows.extend(bins)
            screening.extend(screening_rows(family, seed, payload))
            scale_rows.append({"family": family, "seed": seed, **payload["scale_diagnostics"]})
            similarity_test = payload["similarity"]["test"]
            scopes = {
                "all": np.ones(len(similarity_test), dtype=bool),
                "similarity_lt_0.4": similarity_test < 0.4,
                "similarity_lt_0.3": similarity_test < 0.3,
            }
            scope_status = {
                "all": "confirmatory_frozen_seed_analysis",
                "similarity_lt_0.4": "confirmatory_carry_forward_from_phase1b",
                "similarity_lt_0.3": "exploratory_follow_up_to_pre_outcome_similarity_bin",
            }
            for scope, scope_mask in scopes.items():
                for comparator in ("M1_global", "M3_mondrian"):
                    m5_covered, observed = counts["M5_similarity_normalized"]
                    comparator_covered, _ = counts[comparator]
                    inference = paired_compound_inference(
                        m5_covered[scope_mask],
                        comparator_covered[scope_mask],
                        observed[scope_mask],
                        rng,
                        args.bootstrap_reps,
                        args.randomization_reps,
                    )
                    paired.append(
                        {
                            "family": family,
                            "seed": seed,
                            "analysis_scope": scope,
                            "analysis_status": scope_status[scope],
                            "contrast": f"M5_minus_{comparator}",
                            **inference,
                        }
                    )
            if not args.skip_m4_diagnostics:
                m4_frames.append(m4_neighbour_diagnostics(data, family, seed, payload))
            print(f"completed family={family} seed={seed}", flush=True)

    overall_frame = pd.DataFrame(overall_rows)
    bins_frame = pd.DataFrame(bin_rows)
    screening_frame = pd.DataFrame(screening)
    paired_frame = pd.DataFrame(paired)
    paired_frame["bh_q_value"] = bh_adjust(paired_frame["paired_randomization_p"].to_numpy())
    overall_frame.to_csv(output / "calibration_seed_metrics.csv", index=False)
    bins_frame.to_csv(output / "coverage_similarity_seed_metrics.csv", index=False)
    screening_frame.to_csv(output / "screening_threshold_scan.csv", index=False)
    paired_frame.to_csv(output / "paired_compound_inference.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(output / "m5_scale_diagnostics.csv", index=False)

    overall_summary = summarize_across_seeds(
        overall_frame, ["family", "method"], ["coverage", "mean_width"]
    )
    bin_summary = summarize_across_seeds(
        bins_frame,
        ["family", "method", "similarity_bin", "bin_order"],
        ["coverage", "mean_width", "n_compounds", "n_labels", "mean_similarity"],
    )
    screening_summary = summarize_across_seeds(
        screening_frame,
        ["family", "method", "strong_cell_threshold"],
        ["selected", "true_positive", "false_positive", "precision", "false_discovery_rate", "recall"],
    )
    paired_summary = summarize_across_seeds(
        paired_frame,
        ["family", "analysis_scope", "contrast"],
        ["difference"],
    )
    overall_summary.to_csv(output / "calibration_confirmation_summary_ci.csv", index=False)
    bin_summary.to_csv(output / "coverage_similarity_summary_ci.csv", index=False)
    screening_summary.to_csv(output / "screening_threshold_summary_ci.csv", index=False)
    paired_summary.to_csv(output / "paired_effect_seed_summary_ci.csv", index=False)

    if m4_frames:
        m4 = pd.concat(m4_frames, ignore_index=True)
        m4.to_csv(output / "m4_neighbour_diagnostics_by_compound.csv.gz", index=False, compression="gzip")
        summary_rows = []
        diagnostic_columns = [
            "sim_rank1",
            "sim_rank10",
            "sim_rank50",
            "sim_rank128",
            "sim_rank512",
            "mean_top512_similarity",
            "median_top512_similarity",
            "local_difficulty_q90",
            "local_difficulty_iqr",
            "test_difficulty",
        ]
        for (family, seed), part in m4.groupby(["family", "seed"]):
            row = {"family": family, "seed": seed, "n_compounds": len(part)}
            for column in diagnostic_columns:
                row[f"{column}_mean"] = float(part[column].mean())
                row[f"{column}_median"] = float(part[column].median())
                row[f"{column}_q10"] = float(part[column].quantile(0.1))
                row[f"{column}_q90"] = float(part[column].quantile(0.9))
            row["spearman_mean_similarity_vs_test_difficulty"] = float(
                part["mean_top512_similarity"].corr(part["test_difficulty"], method="spearman")
            )
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(output / "m4_neighbour_diagnostics_summary.csv", index=False)

    figure_dir = output / "figures"
    plot_similarity_curve(bin_summary, figure_dir / "coverage_width_vs_similarity")
    plot_screening_scan(screening_summary, figure_dir / "screening_threshold_scan")
    metadata = {
        "confirmation_seeds": list(seeds),
        "bootstrap_reps": args.bootstrap_reps,
        "randomization_reps": args.randomization_reps,
        "similarity_edges": SIMILARITY_EDGES.tolist(),
        "similarity_bin_rationale": "[0,.2) had fewer than 10 compounds per seed; merged to [0,.3) before outcome analysis",
        "bootstrap_unit": "compound; all observed cell-line labels retained within compound",
        "analysis_scope_status": {
            "all": "confirmatory frozen-seed analysis",
            "similarity_lt_0.4": "confirmatory carry-forward from Phase 1B",
            "similarity_lt_0.3": "exploratory follow-up to the pre-outcome [0,.3) similarity bin; not a new confirmatory claim",
        },
        "multiple_testing": "Benjamini-Hochberg across all seed-level paired randomisation tests; multiplicity correction does not convert the exploratory <.3 subgroup into a confirmatory result",
        "screening_interval": "Wilson 95% interval computed within each family/seed/threshold/method",
        "seed_ci": "descriptive Student-t 95% CI across four frozen confirmation split seeds; overlapping compounds mean it is not an independent-cohort CI",
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

