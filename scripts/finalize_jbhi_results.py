"""Create the frozen, manuscript-facing comparison artifacts for the JBHI rewrite.

The script reconstructs every formal interval method from the same confirmation
splits, computes a common metric set, evaluates lower-bound screening, performs
paired compound-clustered comparisons between the proposed method and UACQR-P,
and writes a provenance manifest.  It does not train or tune any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from scripts.analyze_phase4_statistics import (
        mean_t_interval,
        paired_compound_inference,
        wilson_interval,
    )
    from scripts.analyze_uacqrp_exploratory import extended_interval_metrics
    from scripts.evaluate_phase1a_conformal import EvaluationData
    from scripts.evaluate_phase3_local_and_screening import sim
    from scripts.evaluate_phase5_extensions import (
        ALPHAS,
        DAD_K,
        FAMILIES,
        dad_intervals,
        fit_similarity_scale,
        load_weights,
        predict_similarity_scale,
        scaled_intervals,
        standard_intervals,
        top_k_tanimoto_indices,
        weighted_intervals,
    )
except ModuleNotFoundError:
    from analyze_phase4_statistics import (  # type: ignore[no-redef]
        mean_t_interval,
        paired_compound_inference,
        wilson_interval,
    )
    from analyze_uacqrp_exploratory import extended_interval_metrics  # type: ignore[no-redef]
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]
    from evaluate_phase3_local_and_screening import sim  # type: ignore[no-redef]
    from evaluate_phase5_extensions import (  # type: ignore[no-redef]
        ALPHAS,
        DAD_K,
        FAMILIES,
        dad_intervals,
        fit_similarity_scale,
        load_weights,
        predict_similarity_scale,
        scaled_intervals,
        standard_intervals,
        top_k_tanimoto_indices,
        weighted_intervals,
    )


METHODS = (
    "Global_CP",
    "UACQR_P",
    "dAD_style_k250",
    "Estimated_WCP",
    "Proposed_shared_monotone",
)
METHOD_LABELS = {
    "Point_prediction": "Point",
    "Global_CP": "Global CP",
    "UACQR_P": "UACQR-P",
    "dAD_style_k250": "dAD",
    "Estimated_WCP": "WCP",
    "Proposed_shared_monotone": "Proposed",
}
COLORS = {
    "Point_prediction": "#8C8C8C",
    "Global_CP": "#4D4D4D",
    "UACQR_P": "#D55E00",
    "dAD_style_k250": "#009E73",
    "Estimated_WCP": "#CC79A7",
    "Proposed_shared_monotone": "#0072B2",
}
MARKERS = {
    "Point_prediction": "x",
    "Global_CP": "o",
    "UACQR_P": "s",
    "dAD_style_k250": "D",
    "Estimated_WCP": "v",
    "Proposed_shared_monotone": "^",
}
SCREENING_THRESHOLDS = (1, 3, 5, 10)
ACTIVITY_THRESHOLD = 6.0


def reconstruct_run(root: Path, data: EvaluationData, family: str, seed: int) -> dict:
    """Load frozen predictions and prepare every formal interval constructor."""
    phase1 = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    phase3 = root / "results" / "phase3" / f"{family}_seed_{seed}"
    phase5 = root / "results" / "phase5" / f"{family}_seed_{seed}"
    phase6 = root / "results" / "phase6_uacqrp_exploratory" / f"{family}_seed_{seed}"
    split = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"

    point = np.load(phase1 / "mlp_predictions.npz")
    uacqr = np.load(phase6 / "uacqrp_intervals.npz")
    names = ("source_cal", "valid", "test")
    ids = {name: point[f"nsc_{name}"].astype("int64") for name in names}
    prediction = {name: point[f"pred_{name}"].astype(float) for name in names}
    truth = {name: data.truth(ids[name]) for name in names}
    if not np.array_equal(ids["test"], uacqr["nsc_test"].astype("int64")):
        raise ValueError(f"UACQR-P and point IDs disagree for {family} seed {seed}")

    fit_ids = pd.read_csv(split / "fit_nsc.csv")["nsc"].to_numpy("int64")
    similarity = {name: sim(phase3, name, data, fit_ids, ids[name]) for name in names}
    cal_weights, test_weights, _ = load_weights(
        root, data, family, seed, ids["source_cal"], ids["test"]
    )
    neighbour_path = phase5 / f"dad_k{DAD_K}_neighbours.npz"
    if neighbour_path.exists():
        neighbours = np.load(neighbour_path)["indices"].astype(np.int32)
    else:
        neighbours = top_k_tanimoto_indices(
            data.x[data.rows(ids["source_cal"])], data.x[data.rows(ids["test"])], DAD_K
        )
        np.savez_compressed(neighbour_path, indices=neighbours)

    scale_model, _, _ = fit_similarity_scale(
        similarity["valid"], truth["valid"], prediction["valid"]
    )
    cal_scale = predict_similarity_scale(scale_model, similarity["source_cal"])
    test_scale = predict_similarity_scale(scale_model, similarity["test"])
    return {
        "ids": ids,
        "prediction": prediction,
        "truth": truth,
        "similarity": similarity,
        "weights": (cal_weights, test_weights),
        "neighbours": neighbours,
        "scales": (cal_scale, test_scale),
        "uacqr": uacqr,
    }


def intervals_for_alpha(payload: dict, alpha: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    truth = payload["truth"]
    prediction = payload["prediction"]
    cal_weights, test_weights = payload["weights"]
    cal_scale, test_scale = payload["scales"]
    uacqr = payload["uacqr"]
    stored_alphas = uacqr["alphas"].astype(float)
    matches = np.flatnonzero(np.isclose(stored_alphas, alpha))
    if len(matches) != 1:
        raise ValueError(f"No unique UACQR-P interval found for alpha={alpha}")
    index = int(matches[0])
    return {
        "Global_CP": standard_intervals(
            truth["source_cal"], prediction["source_cal"], prediction["test"], alpha
        ),
        "UACQR_P": (
            uacqr["lower"][index].astype(float),
            uacqr["upper"][index].astype(float),
        ),
        "dAD_style_k250": dad_intervals(
            truth["source_cal"],
            prediction["source_cal"],
            prediction["test"],
            payload["neighbours"],
            alpha,
        ),
        "Estimated_WCP": weighted_intervals(
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
            cal_scale,
            test_scale,
            alpha,
        ),
    }


def screening_rows(
    family: str, seed: int, truth: np.ndarray, prediction: np.ndarray, intervals: dict
) -> list[dict]:
    observed = np.isfinite(truth)
    true_active = (truth >= ACTIVITY_THRESHOLD) & observed
    lower_by_method = {method: bounds[0] for method, bounds in intervals.items()}
    lower_by_method["Point_prediction"] = prediction
    rows: list[dict] = []
    for threshold in SCREENING_THRESHOLDS:
        truth_positive = true_active.sum(axis=1) >= threshold
        total_positive = int(truth_positive.sum())
        for method, lower in lower_by_method.items():
            selected = ((lower > ACTIVITY_THRESHOLD) & observed).sum(axis=1) >= threshold
            n_selected = int(selected.sum())
            true_positive = int((selected & truth_positive).sum())
            false_positive = n_selected - true_positive
            precision = true_positive / n_selected if n_selected else np.nan
            recall = true_positive / total_positive if total_positive else np.nan
            p_low, p_high = wilson_interval(true_positive, n_selected)
            r_low, r_high = wilson_interval(true_positive, total_positive)
            rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "method": method,
                    "strong_cell_threshold": threshold,
                    "selected": n_selected,
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "total_true": total_positive,
                    "precision": precision,
                    "false_discovery_rate": 1.0 - precision if n_selected else np.nan,
                    "recall": recall,
                    "precision_wilson_low": p_low,
                    "precision_wilson_high": p_high,
                    "fdr_wilson_low": 1.0 - p_high if n_selected else np.nan,
                    "fdr_wilson_high": 1.0 - p_low if n_selected else np.nan,
                    "recall_wilson_low": r_low,
                    "recall_wilson_high": r_high,
                }
            )
    return rows


def compound_coverage_counts(
    truth: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.isfinite(truth)
    covered = observed & (truth >= lower) & (truth <= upper)
    return covered.sum(axis=1).astype(float), observed.sum(axis=1).astype(float)


def calibration_error_bootstrap(
    truth: np.ndarray,
    proposed: tuple[np.ndarray, np.ndarray],
    comparator: tuple[np.ndarray, np.ndarray],
    alpha: float,
    mask: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> dict:
    """Bootstrap Proposed-minus-comparator absolute calibration error by compound."""
    p_count, observed = compound_coverage_counts(truth, *proposed)
    c_count, _ = compound_coverage_counts(truth, *comparator)
    valid = mask & (observed > 0)
    p_count, c_count, observed = p_count[valid], c_count[valid], observed[valid]
    target = 1.0 - alpha

    def difference(indices: np.ndarray | None = None) -> float:
        if indices is None:
            pc, cc, den = p_count.sum(), c_count.sum(), observed.sum()
        else:
            pc, cc, den = p_count[indices].sum(), c_count[indices].sum(), observed[indices].sum()
        return float(abs(pc / den - target) - abs(cc / den - target))

    estimate = difference()
    samples = np.empty(reps, dtype=float)
    n = len(observed)
    for start in range(0, reps, 200):
        stop = min(start + 200, reps)
        indices = rng.integers(0, n, size=(stop - start, n))
        pc = p_count[indices].sum(axis=1)
        cc = c_count[indices].sum(axis=1)
        den = observed[indices].sum(axis=1)
        samples[start:stop] = np.abs(pc / den - target) - np.abs(cc / den - target)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "ace_difference": estimate,
        "bootstrap_ci_low": float(low),
        "bootstrap_ci_high": float(high),
        "bootstrap_probability_proposed_lower_ace": float(np.mean(samples < 0)),
        "n_compounds": int(n),
        "n_labels": int(observed.sum()),
        "bootstrap_reps": int(reps),
    }


def summarize_across_seeds(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["n_seeds"] = int(part["seed"].nunique())
        for metric in metrics:
            values = part[metric].to_numpy(float)
            has_positive_infinity = bool(np.isposinf(values).any())
            if has_positive_infinity:
                mean, sd, low, high = np.inf, np.nan, np.nan, np.nan
            else:
                mean, sd, low, high = mean_t_interval(values)
            row.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_sd": sd,
                    f"{metric}_t95_low": low,
                    f"{metric}_t95_high": high,
                    f"{metric}_n_finite": int(np.isfinite(values).sum()),
                    f"{metric}_has_positive_infinity": has_positive_infinity,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
        (".tiff", {"dpi": 600}),
    ):
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", **kwargs)


def plot_screening(summary: pd.DataFrame, output: Path) -> None:
    """Leader-cluster screening trade-off for all formal comparators."""
    configure_plotting()
    part = summary[summary["family"] == "leader_cluster"]
    methods = ("Point_prediction",) + METHODS
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), sharex=True)
    panels = (
        ("false_discovery_rate", "False discovery rate", False),
        ("selected", "Selected compounds", True),
        ("recall", "Recall", False),
    )
    for axis, (metric, ylabel, log_scale) in zip(axes, panels):
        for method in methods:
            values = part[part["method"] == method].sort_values("strong_cell_threshold")
            axis.plot(
                values["strong_cell_threshold"],
                values[f"{metric}_mean"],
                color=COLORS[method],
                marker=MARKERS[method],
                lw=1.25,
                ms=3.4,
                label=METHOD_LABELS[method],
            )
        axis.set_xlabel("Minimum active cell lines")
        axis.set_ylabel(ylabel)
        axis.set_xticks(SCREENING_THRESHOLDS)
        if log_scale:
            axis.set_yscale("symlog", linthresh=1.0, linscale=0.7)
            axis.set_ylim(-0.1, 650)
            axis.set_yticks([0, 1, 10, 100, 500])
            axis.set_yticklabels(["0", "1", "10", "100", "500"])
        else:
            axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.7)
    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=8.5)
    axes[0].text(
        0.98,
        0.98,
        "UACQR-P: no compounds selected\n(FDR undefined)",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=COLORS["UACQR_P"],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize=6.6)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output)
    plt.close(fig)


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
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--randomization-reps", type=int, default=10000)
    args = parser.parse_args()
    root = Path(args.root)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    output = root / "results" / "final_jbhi"
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260721)
    data = EvaluationData(root)

    metric_rows: list[dict] = []
    screening: list[dict] = []
    paired_coverage: list[dict] = []
    paired_ace: list[dict] = []
    input_files: set[Path] = set()
    for family in FAMILIES:
        for seed in seeds:
            payload = reconstruct_run(root, data, family, seed)
            truth_test = payload["truth"]["test"]
            similarity_test = payload["similarity"]["test"]
            alpha_intervals: dict[float, dict] = {}
            for alpha in ALPHAS:
                intervals = intervals_for_alpha(payload, alpha)
                alpha_intervals[alpha] = intervals
                for method, (lower, upper) in intervals.items():
                    metric_rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "alpha": alpha,
                            "method": method,
                            "scope": "all",
                            **extended_interval_metrics(truth_test, lower, upper, alpha),
                        }
                    )
                    low_mask = similarity_test < 0.4
                    metric_rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "alpha": alpha,
                            "method": method,
                            "scope": "similarity_lt_0.4",
                            **extended_interval_metrics(
                                truth_test[low_mask], lower[low_mask], upper[low_mask], alpha
                            ),
                        }
                    )
                proposed = intervals["Proposed_shared_monotone"]
                uacqr = intervals["UACQR_P"]
                # The primary coverage contrast is the frozen 90% leader-cluster
                # analysis.  The prespecified low-similarity comparison is made
                # on absolute calibration error at every nominal level.
                if family == "leader_cluster" and np.isclose(alpha, 0.10):
                    mask = np.ones(len(truth_test), dtype=bool)
                    p_covered, observed = compound_coverage_counts(truth_test, *proposed)
                    u_covered, _ = compound_coverage_counts(truth_test, *uacqr)
                    inference = paired_compound_inference(
                        p_covered[mask],
                        u_covered[mask],
                        observed[mask],
                        rng,
                        args.bootstrap_reps,
                        args.randomization_reps,
                    )
                    paired_coverage.append(
                        {
                            "family": family,
                            "seed": seed,
                            "alpha": alpha,
                            "scope": "all",
                            "contrast": "Proposed_minus_UACQR_P_coverage",
                            **inference,
                        }
                    )
                if family in ("scaffold", "leader_cluster"):
                    mask = similarity_test < 0.4
                    paired_ace.append(
                        {
                            "family": family,
                            "seed": seed,
                            "alpha": alpha,
                            "scope": "similarity_lt_0.4",
                            "contrast": "Proposed_minus_UACQR_P_absolute_calibration_error",
                            **calibration_error_bootstrap(
                                truth_test,
                                proposed,
                                uacqr,
                                alpha,
                                mask,
                                rng,
                                args.bootstrap_reps,
                            ),
                        }
                    )
            screening.extend(
                screening_rows(
                    family,
                    seed,
                    truth_test,
                    payload["prediction"]["test"],
                    alpha_intervals[0.10],
                )
            )
            run_files = (
                root / "results" / "phase1a" / f"{family}_seed_{seed}" / "mlp_predictions.npz",
                root / "results" / "phase1a" / f"{family}_seed_{seed}" / "domain_weights.csv",
                root / "results" / "phase1a" / f"{family}_seed_{seed}" / "domain_shift_metrics.json",
                root / "results" / "phase6_uacqrp_exploratory" / f"{family}_seed_{seed}" / "uacqrp_intervals.npz",
                root / "results" / "phase6_uacqrp_exploratory" / f"{family}_seed_{seed}" / "uacqrp_run_metrics.json",
                root / "results" / "phase5" / f"{family}_seed_{seed}" / f"dad_k{DAD_K}_neighbours.npz",
                root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}" / "fit_nsc.csv",
            )
            input_files.update(run_files)
            print(f"finalized family={family} seed={seed}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    screens = pd.DataFrame(screening)
    coverage_inference = pd.DataFrame(paired_coverage)
    ace_inference = pd.DataFrame(paired_ace)
    metrics.to_csv(output / "formal_method_seed_metrics.csv", index=False)
    screens.to_csv(output / "formal_screening_seed_metrics.csv", index=False)
    coverage_inference.to_csv(output / "proposed_vs_uacqrp_paired_coverage.csv", index=False)
    ace_inference.to_csv(output / "proposed_vs_uacqrp_paired_ace.csv", index=False)

    metric_summary = summarize_across_seeds(
        metrics,
        ["family", "alpha", "method", "scope"],
        [
            "coverage",
            "coverage_error_abs",
            "mean_width",
            "finite_interval_fraction",
            "mean_interval_score",
            "mean_interval_score_finite",
        ],
    )
    screen_summary = summarize_across_seeds(
        screens,
        ["family", "method", "strong_cell_threshold"],
        ["selected", "true_positive", "false_positive", "precision", "false_discovery_rate", "recall"],
    )
    metric_summary.to_csv(output / "formal_method_summary_ci.csv", index=False)
    screen_summary.to_csv(output / "formal_screening_summary_ci.csv", index=False)

    ood = metrics[
        metrics["family"].isin(("scaffold", "leader_cluster"))
        & (metrics["scope"] == "all")
    ]
    low_ood = metrics[
        metrics["family"].isin(("scaffold", "leader_cluster"))
        & (metrics["scope"] == "similarity_lt_0.4")
    ]
    def aggregate_records(frame: pd.DataFrame) -> list[dict]:
        records = []
        for method, part in frame.groupby("method"):
            scores = part["mean_interval_score"].to_numpy(float)
            unbounded = bool(np.isposinf(scores).any())
            records.append(
                {
                    "method": method,
                    "absolute_calibration_error": float(part["coverage_error_abs"].mean()),
                    "mean_width": float(part["mean_width"].mean()),
                    "interval_score": None if unbounded else float(scores.mean()),
                    "interval_score_status": (
                        "unbounded_due_to_infinite_intervals" if unbounded else "all_intervals_finite"
                    ),
                    "finite_only_interval_score": float(part["mean_interval_score_finite"].mean()),
                    "minimum_finite_interval_fraction": float(part["finite_interval_fraction"].min()),
                }
            )
        return records

    aggregate = {
        "ood_all": aggregate_records(ood),
        "ood_similarity_lt_0_4": aggregate_records(low_ood),
    }
    (output / "formal_aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    plot_screening(screen_summary, figure_dir / "screening_threshold_scan_formal")

    source_files = [
        root / "scripts" / "finalize_jbhi_results.py",
        root / "scripts" / "evaluate_phase5_extensions.py",
        root / "scripts" / "analyze_uacqrp_exploratory.py",
        root / "scripts" / "evaluate_phase1a_conformal.py",
        root / "scripts" / "evaluate_phase1b_similarity.py",
        root / "scripts" / "evaluate_phase3_local_and_screening.py",
        root / "scripts" / "run_uacqrp_exploratory.py",
        root / "data" / "features" / "cellminer" / "morgan_r2_2048.npz",
        root / "data" / "features" / "cellminer" / "morgan_r2_2048_index.csv",
        root / "data" / "processed" / "cellminer" / "response_matrix_loggi50.csv.gz",
    ]
    manifest_files = sorted(path for path in input_files.union(source_files) if path.exists())
    manifest = {
        "status": "frozen_manuscript_facing_confirmation_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            f"python scripts/finalize_jbhi_results.py --root {root.as_posix()} "
            f"--seeds {','.join(map(str, seeds))} --bootstrap-reps {args.bootstrap_reps} "
            f"--randomization-reps {args.randomization_reps}"
        ),
        "families": list(FAMILIES),
        "confirmation_seeds": list(seeds),
        "alpha_levels": list(ALPHAS),
        "formal_methods": list(METHODS),
        "screening_thresholds": list(SCREENING_THRESHOLDS),
        "activity_rule": "observed logGI50 >= 6; selected only when lower 90% bound > 6",
        "bootstrap_unit": "compound, retaining all observed cell-line labels within compound",
        "seed_interval": "descriptive Student-t 95% interval across confirmation split seeds",
        "figure_contract": {
            "core_conclusion": "Lower-bound screening exchanges shortlist size and recall for lower empirical FDR under leader-cluster shift.",
            "evidence": "All five formal interval methods and the point predictor, confirmation seeds 2-5, k in {1,3,5,10}.",
            "archetype": "three-panel screening trade-off curve",
            "exports": ["SVG", "PDF", "PNG 300 dpi", "TIFF 600 dpi"],
        },
        "files_sha256": {str(path.relative_to(root)): sha256(path) for path in manifest_files},
    }
    (output / "final_results_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()

