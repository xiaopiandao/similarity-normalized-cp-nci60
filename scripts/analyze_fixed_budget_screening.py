"""Equal-budget compound screening analysis for the JBHI revision.

This analysis addresses the confounding between false-discovery rate and
shortlist size in threshold-based screening.  Every method ranks the same test
compounds, and performance is compared at fixed experimental budgets.  For the
primary endpoint (activity in at least three cell lines), a compound's ranking
score is its third-largest point prediction or prediction-interval lower bound.

The script reuses frozen confirmation predictions and interval constructors;
it does not train, tune, or select a model on the test set.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from scripts.evaluate_phase1a_conformal import EvaluationData
    from scripts.finalize_jbhi_results import intervals_for_alpha, reconstruct_run
except ModuleNotFoundError:
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]
    from finalize_jbhi_results import (  # type: ignore[no-redef]
        intervals_for_alpha,
        reconstruct_run,
    )


FAMILY = "leader_cluster"
SEEDS = (2, 3, 4, 5)
ALPHA = 0.10
ACTIVITY_THRESHOLD = 6.0
ACTIVE_CELL_THRESHOLD = 3
BUDGETS = (40, 100, 200, 400)
METHOD_ORDER = (
    "Point_prediction",
    "Global_CP",
    "UACQR_P",
    "dAD_style_k250",
    "Estimated_WCP",
    "Proposed_shared_monotone",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kth_largest(values: np.ndarray, observed: np.ndarray, k: int) -> np.ndarray:
    """Return the kth-largest observed cell-line value for each compound."""
    masked = np.where(observed, values, -np.inf)
    index = masked.shape[1] - k
    return np.partition(masked, index, axis=1)[:, index]


def mean_t_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "n_seeds": 0,
            "mean": np.nan,
            "sd": np.nan,
            "t95_low": np.nan,
            "t95_high": np.nan,
        }
    mean = float(values.mean())
    if len(values) == 1:
        return {
            "n_seeds": 1,
            "mean": mean,
            "sd": np.nan,
            "t95_low": np.nan,
            "t95_high": np.nan,
        }
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return {
        "n_seeds": int(len(values)),
        "mean": mean,
        "sd": sd,
        "t95_low": mean - half,
        "t95_high": mean + half,
    }


def exact_sign_randomization_p(differences: np.ndarray) -> float:
    """Two-sided exact sign-randomization p-value across confirmation seeds."""
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return np.nan
    observed = abs(float(differences.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        null.append(abs(float(np.mean(np.asarray(signs) * differences))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def stable_top_k(scores: np.ndarray, compound_ids: np.ndarray, budget: int) -> np.ndarray:
    """Rank descending by score, breaking exact ties by ascending NSC ID."""
    order = np.lexsort((compound_ids, -scores))
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[: min(budget, len(order))]] = True
    return selected


def build_design_record(root: Path, output: Path, seeds: tuple[int, ...]) -> dict:
    record = {
        "analysis_id": "jbhi_equal_budget_screening_v1",
        "analysis_status": "post_hoc_diagnostic_frozen_before_execution",
        "family": FAMILY,
        "confirmation_seeds": list(seeds),
        "nominal_coverage": 1.0 - ALPHA,
        "activity_threshold_loggi50": ACTIVITY_THRESHOLD,
        "compound_endpoint": (
            f"observed activity >= {ACTIVITY_THRESHOLD} in at least "
            f"{ACTIVE_CELL_THRESHOLD} cell lines"
        ),
        "ranking_score": (
            f"{ACTIVE_CELL_THRESHOLD}rd-largest observed-cell point prediction "
            "or interval lower bound"
        ),
        "fixed_budgets": list(BUDGETS),
        "primary_comparison": "Proposed_shared_monotone versus Point_prediction at K=40",
        "success_criteria": {
            "primary": (
                "mean precision/TP gain above zero at K=40 and positive TP difference "
                "in at least three of four confirmation seeds"
            ),
            "robustness": (
                "positive mean TP difference versus point ranking at at least three of "
                "the four fixed budgets"
            ),
            "baseline_context": (
                "report all formal interval comparators; no test-set method selection"
            ),
        },
        "interpretation_boundary": (
            "A fixed-budget advantage supports ranking utility. Failure does not invalidate "
            "interval calibration but precludes a ranking-superiority claim."
        ),
        "source_scripts": [
            str(root / "scripts" / "finalize_jbhi_results.py"),
            str(root / "scripts" / "analyze_fixed_budget_screening.py"),
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis_design.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", default=",".join(str(value) for value in SEEDS))
    parser.add_argument(
        "--output",
        default="results/screening_fixed_budget_20260815",
        help="Output directory relative to root unless absolute.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    design = build_design_record(root, output, seeds)

    data = EvaluationData(root)
    budget_rows: list[dict] = []
    ranking_rows: list[dict] = []
    score_rows: list[dict] = []
    anomaly_rows: list[dict] = []
    inputs: set[Path] = set()

    for seed in seeds:
        payload = reconstruct_run(root, data, FAMILY, seed)
        truth = payload["truth"]["test"]
        prediction = payload["prediction"]["test"]
        compound_ids = payload["ids"]["test"]
        observed = np.isfinite(truth)
        truth_positive = ((truth >= ACTIVITY_THRESHOLD) & observed).sum(axis=1) >= ACTIVE_CELL_THRESHOLD
        prevalence = float(truth_positive.mean())
        intervals = intervals_for_alpha(payload, ALPHA)
        method_values = {"Point_prediction": prediction}
        method_values.update({method: bounds[0] for method, bounds in intervals.items()})

        for method in METHOD_ORDER:
            scores = kth_largest(method_values[method], observed, ACTIVE_CELL_THRESHOLD)
            if np.isnan(scores).any() or np.isposinf(scores).any():
                raise ValueError(f"NaN/+inf ranking score for {method}, seed={seed}")
            # A -inf score means that fewer than three finite lower bounds were
            # available for that comparator/compound.  It has an unambiguous
            # ranking interpretation (last place), but sklearn ranking metrics
            # require finite inputs.  Replace it only for metric computation by
            # a deterministic floor below every finite score and retain an audit
            # flag in the compound-level output.
            metric_scores = scores.copy()
            negative_infinite = np.isneginf(metric_scores)
            if negative_infinite.any():
                finite_minimum = float(metric_scores[np.isfinite(metric_scores)].min())
                finite_floor = finite_minimum - max(1.0, abs(finite_minimum) * 1e-6)
                metric_scores[negative_infinite] = finite_floor
                anomaly_rows.append(
                    {
                        "family": FAMILY,
                        "seed": seed,
                        "method": method,
                        "anomaly": "negative_infinite_compound_ranking_score",
                        "n_compounds": int(negative_infinite.sum()),
                        "handling": "rank_last_with_deterministic_finite_floor_for_ap_auc",
                    }
                )
            ap = float(average_precision_score(truth_positive.astype(int), metric_scores))
            auc = float(roc_auc_score(truth_positive.astype(int), metric_scores))
            ranking_rows.append(
                {
                    "family": FAMILY,
                    "seed": seed,
                    "method": method,
                    "n_compounds": len(scores),
                    "total_true": int(truth_positive.sum()),
                    "prevalence": prevalence,
                    "average_precision": ap,
                    "average_precision_over_prevalence": ap / prevalence,
                    "roc_auc": auc,
                }
            )
            for compound_id, score, metric_score, label in zip(
                compound_ids, scores, metric_scores, truth_positive
            ):
                score_rows.append(
                    {
                        "family": FAMILY,
                        "seed": seed,
                        "method": method,
                        "nsc": int(compound_id),
                        "score": float(score),
                        "metric_score": float(metric_score),
                        "score_was_negative_infinite": bool(np.isneginf(score)),
                        "truth_positive": int(label),
                    }
                )
            for budget in BUDGETS:
                selected = stable_top_k(scores, compound_ids, budget)
                n_selected = int(selected.sum())
                true_positive = int((selected & truth_positive).sum())
                false_positive = n_selected - true_positive
                precision = true_positive / n_selected
                recall = true_positive / int(truth_positive.sum())
                budget_rows.append(
                    {
                        "family": FAMILY,
                        "seed": seed,
                        "method": method,
                        "budget": budget,
                        "selected": n_selected,
                        "true_positive": true_positive,
                        "false_positive": false_positive,
                        "total_true": int(truth_positive.sum()),
                        "precision": precision,
                        "false_discovery_rate": 1.0 - precision,
                        "recall": recall,
                        "prevalence": prevalence,
                        "enrichment_over_prevalence": precision / prevalence,
                    }
                )

        inputs.update(
            {
                root / "results" / "phase1a" / f"{FAMILY}_seed_{seed}" / "mlp_predictions.npz",
                root
                / "results"
                / "phase6_uacqrp_exploratory"
                / f"{FAMILY}_seed_{seed}"
                / "uacqrp_intervals.npz",
                root / "results" / "phase5" / f"{FAMILY}_seed_{seed}" / "dad_k250_neighbours.npz",
                root / "data" / "splits" / "phase1a" / f"{FAMILY}_seed_{seed}" / "fit_nsc.csv",
            }
        )
        print(f"completed fixed-budget screening: family={FAMILY} seed={seed}", flush=True)

    budget_frame = pd.DataFrame(budget_rows)
    ranking_frame = pd.DataFrame(ranking_rows)
    score_frame = pd.DataFrame(score_rows)
    budget_frame.to_csv(output / "fixed_budget_seed_metrics.csv", index=False)
    ranking_frame.to_csv(output / "ranking_seed_metrics.csv", index=False)
    score_frame.to_csv(output / "compound_ranking_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(anomaly_rows).to_csv(output / "analysis_anomalies.csv", index=False)

    budget_summary_rows: list[dict] = []
    for (method, budget), group in budget_frame.groupby(["method", "budget"], sort=False):
        for metric in (
            "true_positive",
            "false_positive",
            "precision",
            "false_discovery_rate",
            "recall",
            "enrichment_over_prevalence",
        ):
            budget_summary_rows.append(
                {
                    "method": method,
                    "budget": int(budget),
                    "metric": metric,
                    **mean_t_summary(group[metric].to_numpy(float)),
                }
            )
    pd.DataFrame(budget_summary_rows).to_csv(output / "fixed_budget_summary_ci.csv", index=False)

    ranking_summary_rows: list[dict] = []
    for method, group in ranking_frame.groupby("method", sort=False):
        for metric in ("average_precision", "average_precision_over_prevalence", "roc_auc"):
            ranking_summary_rows.append(
                {"method": method, "metric": metric, **mean_t_summary(group[metric].to_numpy(float))}
            )
    pd.DataFrame(ranking_summary_rows).to_csv(output / "ranking_summary_ci.csv", index=False)

    paired_rows: list[dict] = []
    for comparator in METHOD_ORDER[:-1]:
        for budget in BUDGETS:
            subset = budget_frame[budget_frame["budget"] == budget]
            proposed = subset[subset["method"] == "Proposed_shared_monotone"].set_index("seed")
            reference = subset[subset["method"] == comparator].set_index("seed")
            for metric in ("true_positive", "precision", "false_discovery_rate", "recall"):
                differences = (
                    proposed.loc[list(seeds), metric].to_numpy(float)
                    - reference.loc[list(seeds), metric].to_numpy(float)
                )
                paired_rows.append(
                    {
                        "contrast": f"Proposed_minus_{comparator}",
                        "budget": budget,
                        "metric": metric,
                        **mean_t_summary(differences),
                        "positive_seeds": int((differences > 0).sum()),
                        "zero_seeds": int((differences == 0).sum()),
                        "negative_seeds": int((differences < 0).sum()),
                        "exact_sign_randomization_p": exact_sign_randomization_p(differences),
                    }
                )
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(output / "paired_proposed_contrasts.csv", index=False)

    primary = paired_frame[
        (paired_frame["contrast"] == "Proposed_minus_Point_prediction")
        & (paired_frame["budget"] == 40)
        & (paired_frame["metric"] == "true_positive")
    ].iloc[0]
    point_by_budget = paired_frame[
        (paired_frame["contrast"] == "Proposed_minus_Point_prediction")
        & (paired_frame["metric"] == "true_positive")
    ]
    primary_pass = bool(primary["mean"] > 0 and primary["positive_seeds"] >= 3)
    robust_budgets = int((point_by_budget["mean"] > 0).sum())
    robustness_pass = robust_budgets >= 3

    decision = {
        "primary_success": primary_pass,
        "robustness_success": robustness_pass,
        "positive_budgets_vs_point": robust_budgets,
        "primary_k40_mean_tp_difference": float(primary["mean"]),
        "primary_k40_positive_seeds": int(primary["positive_seeds"]),
        "editorial_route": (
            "eligible_for_manuscript_fixed_budget_result"
            if primary_pass and robustness_pass
            else "do_not_claim_ranking_superiority_reframe_as_precision_first_threshold_rule"
        ),
    }

    manifest = {
        "design": design,
        "decision": decision,
        "anomalies": anomaly_rows,
        "input_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in sorted(inputs)
            if path.exists()
        },
        "output_files": [
            "analysis_design.json",
            "fixed_budget_seed_metrics.csv",
            "fixed_budget_summary_ci.csv",
            "ranking_seed_metrics.csv",
            "ranking_summary_ci.csv",
            "paired_proposed_contrasts.csv",
            "compound_ranking_scores.csv.gz",
            "analysis_anomalies.csv",
        ],
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_lines = [
        "# Equal-budget screening analysis",
        "",
        "## Frozen decision",
        "",
        f"- Primary success: `{primary_pass}`",
        f"- Robustness success: `{robustness_pass}`",
        f"- Positive budgets versus point ranking: `{robust_budgets}/4`",
        f"- K=40 mean TP difference (Proposed - Point): `{float(primary['mean']):.3f}`",
        f"- K=40 positive seeds: `{int(primary['positive_seeds'])}/4`",
        f"- Editorial route: `{decision['editorial_route']}`",
        "",
        "## Interpretation boundary",
        "",
        "This post-hoc diagnostic evaluates ranking utility at matched shortlist sizes. "
        "It does not alter the prespecified interval-calibration claims and must not be "
        "described as a prospectively validated screening policy.",
    ]
    (output / "INTERNAL_FINDINGS.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

