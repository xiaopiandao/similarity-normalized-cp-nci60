"""Evaluate similarity-anchored factorized density-ratio conformal weights."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.evaluate_phase1b_similarity import (
        fit_similarity_scale,
        predict_similarity_scale,
    )
    from scripts.evaluate_similarity_sufficient_wcp import (
        ALPHAS,
        EvaluationData,
        dataframe_to_markdown,
        estimate_similarity_domain_weights,
        exact_sign_flip_pvalue,
        interval_metrics,
        load_high_dimensional_weights,
        load_similarity,
        mean_ci,
        score_shift_row,
        similarity_metrics,
        standard_intervals,
        standardized_scores,
        summarize_metrics,
        weighted_ks_distance,
        weighted_scaled_intervals,
    )
except ModuleNotFoundError:
    from evaluate_phase1b_similarity import (  # type: ignore[no-redef]
        fit_similarity_scale,
        predict_similarity_scale,
    )
    from evaluate_similarity_sufficient_wcp import (  # type: ignore[no-redef]
        ALPHAS,
        EvaluationData,
        dataframe_to_markdown,
        estimate_similarity_domain_weights,
        exact_sign_flip_pvalue,
        interval_metrics,
        load_high_dimensional_weights,
        load_similarity,
        mean_ci,
        score_shift_row,
        similarity_metrics,
        standard_intervals,
        standardized_scores,
        summarize_metrics,
        weighted_ks_distance,
        weighted_scaled_intervals,
    )


FAMILIES = ("random", "scaffold", "leader_cluster")
OOD_FAMILIES = ("scaffold", "leader_cluster")
LAMBDAS = np.asarray((0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0))
PRIMARY_ALPHA = 0.10


def geometric_factorized_weights(
    similarity_cal_weights: np.ndarray,
    similarity_test_weights: np.ndarray,
    high_dimensional_cal_weights: np.ndarray,
    high_dimensional_test_weights: np.ndarray,
    lambda_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometrically shrink the full density ratio toward the similarity ratio."""
    arrays = (
        similarity_cal_weights,
        similarity_test_weights,
        high_dimensional_cal_weights,
        high_dimensional_test_weights,
    )
    if any(np.any(~np.isfinite(values)) or np.any(values <= 0) for values in arrays):
        raise ValueError("All component weights must be finite and strictly positive")
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda_value must be between zero and one")
    cal_log = (
        (1.0 - lambda_value) * np.log(similarity_cal_weights)
        + lambda_value * np.log(high_dimensional_cal_weights)
    )
    test_log = (
        (1.0 - lambda_value) * np.log(similarity_test_weights)
        + lambda_value * np.log(high_dimensional_test_weights)
    )
    cal = np.exp(cal_log)
    test = np.exp(test_log)
    normalization = float(cal.mean())
    return cal / normalization, test / normalization


def fingerprint_balance(
    source_x,
    target_x,
    source_weights: np.ndarray,
    prevalence_floor: float = 0.01,
) -> dict:
    """Weighted Morgan-bit standardized mean differences."""
    source_mean_unweighted = np.asarray(source_x.mean(axis=0)).ravel().astype(float)
    target_mean = np.asarray(target_x.mean(axis=0)).ravel().astype(float)
    pooled = 0.5 * (source_mean_unweighted + target_mean)
    selected = (pooled >= prevalence_floor) & (pooled <= 1.0 - prevalence_floor)
    if not selected.any():
        raise ValueError("No fingerprint bits pass the prevalence filter")
    weighted_source_mean = np.asarray(
        source_x.T.dot(np.asarray(source_weights, dtype=float)) / source_weights.sum()
    ).ravel()
    denominator = np.sqrt(pooled * (1.0 - pooled) + 1e-3)
    smd = np.abs(weighted_source_mean[selected] - target_mean[selected]) / denominator[selected]
    return {
        "fingerprint_bits": int(selected.sum()),
        "fingerprint_rms_smd": float(np.sqrt(np.mean(smd**2))),
        "fingerprint_mean_abs_smd": float(np.mean(smd)),
        "fingerprint_q95_abs_smd": float(np.quantile(smd, 0.95)),
        "fingerprint_max_abs_smd": float(np.max(smd)),
    }


def choose_factorized_weight(
    source_x,
    target_x,
    cal_similarity: np.ndarray,
    test_similarity: np.ndarray,
    similarity_cal_weights: np.ndarray,
    similarity_test_weights: np.ndarray,
    high_dimensional_cal_weights: np.ndarray,
    high_dimensional_test_weights: np.ndarray,
    absolute_minimum_ess_fraction: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    unit_cal = np.ones(len(cal_similarity), dtype=float)
    baseline_fp = fingerprint_balance(source_x, target_x, unit_cal)
    baseline_ks = weighted_ks_distance(cal_similarity, test_similarity)
    sim_ess = float(
        similarity_cal_weights.sum() ** 2
        / np.sum(similarity_cal_weights**2)
        / len(similarity_cal_weights)
    )
    if not 0.0 < absolute_minimum_ess_fraction <= 1.0:
        raise ValueError("absolute_minimum_ess_fraction must be in (0, 1]")
    minimum_ess = max(absolute_minimum_ess_fraction, 0.80 * sim_ess)
    rows = []
    weights_by_lambda: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for lambda_value in LAMBDAS:
        cal_weights, test_weights = geometric_factorized_weights(
            similarity_cal_weights,
            similarity_test_weights,
            high_dimensional_cal_weights,
            high_dimensional_test_weights,
            float(lambda_value),
        )
        weights_by_lambda[float(lambda_value)] = (cal_weights, test_weights)
        ess = float(cal_weights.sum() ** 2 / np.sum(cal_weights**2))
        ess_fraction = ess / len(cal_weights)
        similarity_ks = weighted_ks_distance(
            cal_similarity, test_similarity, cal_weights, np.ones(len(test_similarity))
        )
        fp = fingerprint_balance(source_x, target_x, cal_weights)
        relative_similarity = similarity_ks / baseline_ks if baseline_ks > 0 else 0.0
        relative_fingerprint = (
            fp["fingerprint_rms_smd"] / baseline_fp["fingerprint_rms_smd"]
            if baseline_fp["fingerprint_rms_smd"] > 0
            else 0.0
        )
        support_limit = (PRIMARY_ALPHA / (1.0 - PRIMARY_ALPHA)) * cal_weights.sum()
        support_failure = float((test_weights > support_limit).mean())
        feasible = bool(ess_fraction >= minimum_ess and support_failure == 0.0)
        rows.append(
            {
                "lambda": float(lambda_value),
                "ess": ess,
                "ess_fraction": ess_fraction,
                "minimum_ess_fraction": minimum_ess,
                "similarity_ks": similarity_ks,
                "relative_similarity_imbalance": relative_similarity,
                **fp,
                "relative_fingerprint_imbalance": relative_fingerprint,
                "balance_objective": max(relative_similarity, relative_fingerprint),
                "support_failure_fraction_alpha_0_10": support_failure,
                "feasible": feasible,
            }
        )
    frame = pd.DataFrame(rows)
    feasible = frame[frame["feasible"]]
    if feasible.empty:
        raise ValueError("No factorized weight satisfies the pre-specified feasibility rules")
    selected_row = feasible.sort_values(
        ["balance_objective", "ess_fraction", "lambda"],
        ascending=[True, False, True],
        kind="mergesort",
    ).iloc[0]
    selected_lambda = float(selected_row["lambda"])
    selected_cal, selected_test = weights_by_lambda[selected_lambda]
    frame["selected"] = np.isclose(frame["lambda"], selected_lambda)
    report = {
        "selected_lambda": selected_lambda,
        "selection_uses_response_labels": False,
        "selection_objective": "minimize the worst relative imbalance across similarity KS and fingerprint RMS-SMD",
        "minimum_ess_fraction": minimum_ess,
        "absolute_minimum_ess_fraction": absolute_minimum_ess_fraction,
        "baseline_similarity_ks": baseline_ks,
        "baseline_fingerprint_rms_smd": baseline_fp["fingerprint_rms_smd"],
        "selected_balance_objective": float(selected_row["balance_objective"]),
        "selected_ess_fraction": float(selected_row["ess_fraction"]),
        "selected_similarity_ks": float(selected_row["similarity_ks"]),
        "selected_fingerprint_rms_smd": float(selected_row["fingerprint_rms_smd"]),
    }
    return selected_cal, selected_test, frame, report


def method_name(lambda_value: float) -> str:
    return f"factorized_lambda_{int(round(lambda_value * 1000)):04d}"


def evaluate_one(
    root: Path, data: EvaluationData, family: str, seed: int
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
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
    hd_cal, hd_test, hd_report = load_high_dimensional_weights(
        root, family, seed, ids_cal, ids_test
    )
    sim_cal_weights, sim_test_weights, sim_report = estimate_similarity_domain_weights(
        sim_cal, sim_test, seed=80_000 + seed
    )
    source_x = data.x[data.rows(ids_cal)]
    target_x = data.x[data.rows(ids_test)]
    selected_cal, selected_test, balance_frame, selection_report = choose_factorized_weight(
        source_x,
        target_x,
        sim_cal,
        sim_test,
        sim_cal_weights,
        sim_test_weights,
        hd_cal,
        hd_test,
    )
    balance_rows = [
        {"family": family, "seed": seed, **row}
        for row in balance_frame.to_dict(orient="records")
    ]

    metric_rows = []
    bin_rows = []
    for alpha in ALPHAS:
        methods = {
            "SNCP_unweighted": standard_intervals(
                truth_cal, pred_cal, pred_test, cal_scale, test_scale, alpha
            ),
            "SNCP_Sim1D": weighted_scaled_intervals(
                truth_cal,
                pred_cal,
                pred_test,
                cal_scale,
                test_scale,
                sim_cal_weights,
                sim_test_weights,
                alpha,
            ),
            "SNCP_HD": weighted_scaled_intervals(
                truth_cal,
                pred_cal,
                pred_test,
                cal_scale,
                test_scale,
                hd_cal,
                hd_test,
                alpha,
            ),
            "SNCP_factorized_selected": weighted_scaled_intervals(
                truth_cal,
                pred_cal,
                pred_test,
                cal_scale,
                test_scale,
                selected_cal,
                selected_test,
                alpha,
            ),
        }
        for method, (lower, upper) in methods.items():
            metric_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "alpha": alpha,
                    "method": method,
                    "selected_lambda": selection_report["selected_lambda"],
                    **interval_metrics(truth_test, lower, upper, alpha),
                }
            )
            bin_rows.extend(
                {
                    "family": family,
                    "seed": seed,
                    "alpha": alpha,
                    "method": method,
                    "selected_lambda": selection_report["selected_lambda"],
                    **row,
                }
                for row in similarity_metrics(sim_test, truth_test, lower, upper, alpha)
            )

    grid_rows = []
    for lambda_value in LAMBDAS:
        cal_weights, test_weights = geometric_factorized_weights(
            sim_cal_weights,
            sim_test_weights,
            hd_cal,
            hd_test,
            float(lambda_value),
        )
        lower, upper = weighted_scaled_intervals(
            truth_cal,
            pred_cal,
            pred_test,
            cal_scale,
            test_scale,
            cal_weights,
            test_weights,
            PRIMARY_ALPHA,
        )
        grid_rows.append(
            {
                "family": family,
                "seed": seed,
                "alpha": PRIMARY_ALPHA,
                "lambda": float(lambda_value),
                "method": method_name(float(lambda_value)),
                "selected": bool(np.isclose(lambda_value, selection_report["selected_lambda"])),
                **interval_metrics(truth_test, lower, upper, PRIMARY_ALPHA),
            }
        )

    normalized_cal_scores, cell_medians = standardized_scores(
        truth_cal, pred_cal, cal_scale
    )
    normalized_test_scores, _ = standardized_scores(
        truth_test, pred_test, test_scale, cell_medians
    )
    score_rows = []
    for weight_name, weights in (
        ("unweighted", np.ones(len(ids_cal))),
        ("Sim1D", sim_cal_weights),
        ("HD", hd_cal),
        ("factorized_selected", selected_cal),
    ):
        score_rows.append(
            score_shift_row(
                family,
                seed,
                "similarity_normalized",
                weight_name,
                normalized_cal_scores,
                normalized_test_scores,
                weights,
            )
        )
    detail = {
        "family": family,
        "seed": seed,
        "selection": selection_report,
        "scale": scale_report,
        "similarity_weight": sim_report,
        "high_dimensional_weight": hd_report,
    }
    return metric_rows, bin_rows, grid_rows, balance_rows, score_rows, detail


def paired_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        metrics["family"].isin(OOD_FAMILIES) & np.isclose(metrics["alpha"], PRIMARY_ALPHA)
    ]
    candidate = primary[primary["method"] == "SNCP_factorized_selected"].set_index(
        ["family", "seed"]
    )
    rows = []
    for comparator in ("SNCP_unweighted", "SNCP_Sim1D", "SNCP_HD"):
        reference = primary[primary["method"] == comparator].set_index(["family", "seed"])
        for metric in (
            "coverage_error_abs",
            "macro_cell_ace",
            "mean_width",
            "finite_interval_fraction",
        ):
            difference = candidate[metric] - reference[metric]
            mean, sd, low, high = mean_ci(difference)
            rows.append(
                {
                    "candidate": "SNCP_factorized_selected",
                    "comparator": comparator,
                    "metric": metric,
                    "difference_definition": "candidate_minus_comparator",
                    "units": len(difference),
                    "mean_difference": mean,
                    "sd_difference": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(
                        difference.to_numpy(float)
                    ),
                    "candidate_better_units": int(
                        (difference < 0).sum()
                        if metric != "finite_interval_fraction"
                        else (difference > 0).sum()
                    ),
                    "ties": int(np.isclose(difference, 0.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_decision(
    metrics: pd.DataFrame,
    balance: pd.DataFrame,
    score_shift: pd.DataFrame,
) -> dict:
    primary = metrics[
        metrics["family"].isin(OOD_FAMILIES) & np.isclose(metrics["alpha"], PRIMARY_ALPHA)
    ]
    candidate = primary[primary["method"] == "SNCP_factorized_selected"].set_index(
        ["family", "seed"]
    )
    current = primary[primary["method"] == "SNCP_unweighted"].set_index(
        ["family", "seed"]
    )
    selected_balance = balance[
        balance["family"].isin(OOD_FAMILIES) & balance["selected"]
    ]
    hd_balance = balance[
        balance["family"].isin(OOD_FAMILIES) & np.isclose(balance["lambda"], 1.0)
    ]
    score_ood = score_shift[score_shift["family"].isin(OOD_FAMILIES)]
    candidate_score_ks = float(
        score_ood[score_ood["weight"] == "factorized_selected"]["mean_cell_ks"].mean()
    )
    current_score_ks = float(
        score_ood[score_ood["weight"] == "unweighted"]["mean_cell_ks"].mean()
    )
    candidate_ace = float(candidate["coverage_error_abs"].mean())
    current_ace = float(current["coverage_error_abs"].mean())
    family_ace = {}
    family_noninferiority = {}
    for family in OOD_FAMILIES:
        cand = candidate.xs(family)["coverage_error_abs"].mean()
        base = current.xs(family)["coverage_error_abs"].mean()
        family_ace[family] = {
            "candidate": float(cand),
            "current": float(base),
            "difference": float(cand - base),
        }
        family_noninferiority[family] = bool(cand - base <= 0.002)
    ace_difference = candidate["coverage_error_abs"] - current["coverage_error_abs"]
    width_ratio = float(candidate["mean_width"].mean() / current["mean_width"].mean())
    minimum_finite = float(candidate["finite_interval_fraction"].min())
    selected_ess = float(selected_balance["ess_fraction"].mean())
    hd_ess = float(hd_balance["ess_fraction"].mean())
    selected_lambdas = selected_balance["lambda"].value_counts().sort_index()
    engineering = {
        "minimum_finite_interval_fraction_ge_0_999": minimum_finite >= 0.999,
        "selected_ess_higher_than_hd": selected_ess > hd_ess,
        "similarity_balance_better_than_unweighted": bool(
            (selected_balance["relative_similarity_imbalance"] < 1.0).all()
        ),
        "fingerprint_balance_better_than_unweighted": bool(
            (selected_balance["relative_fingerprint_imbalance"] < 1.0).all()
        ),
        "mean_width_ratio_le_1_10": width_ratio <= 1.10,
    }
    paper = {
        "candidate_mean_ace_lower_than_current": candidate_ace < current_ace,
        "scaffold_ace_noninferior_margin_0_002": family_noninferiority["scaffold"],
        "leader_cluster_ace_noninferior_margin_0_002": family_noninferiority["leader_cluster"],
        "normalized_score_ks_lower_than_current": candidate_score_ks < current_score_ks,
        "ace_better_in_at_least_6_of_10_units": int((ace_difference < 0).sum()) >= 6,
    }
    return {
        "status": "exploratory_internal_only",
        "observed": {
            "candidate_mean_ace": candidate_ace,
            "current_mean_ace": current_ace,
            "ace_candidate_minus_current": candidate_ace - current_ace,
            "candidate_to_current_mean_width_ratio": width_ratio,
            "candidate_minimum_finite_interval_fraction": minimum_finite,
            "candidate_mean_ess_fraction": selected_ess,
            "hd_mean_ess_fraction": hd_ess,
            "candidate_normalized_score_mean_cell_ks": candidate_score_ks,
            "current_normalized_score_mean_cell_ks": current_score_ks,
            "ace_better_units": int((ace_difference < 0).sum()),
            "family_ace": family_ace,
            "selected_lambda_counts": {
                str(float(key)): int(value) for key, value in selected_lambdas.items()
            },
        },
        "engineering_conditions": engineering,
        "engineering_pass": bool(all(engineering.values())),
        "paper_value_conditions": paper,
        "paper_value_pass": bool(all(paper.values())),
    }


def write_report(
    output: Path,
    decision: dict,
    summary: pd.DataFrame,
    balance: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    primary = summary[
        summary["family"].isin(OOD_FAMILIES) & np.isclose(summary["alpha"], PRIMARY_ALPHA)
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
    selected_balance = balance[
        balance["family"].isin(OOD_FAMILIES) & balance["selected"]
    ][
        [
            "family",
            "seed",
            "lambda",
            "ess_fraction",
            "relative_similarity_imbalance",
            "relative_fingerprint_imbalance",
            "balance_objective",
        ]
    ]
    paired = contrasts[contrasts["comparator"] == "SNCP_unweighted"]
    observed = decision["observed"]
    lines = [
        "# 因子化相似度锚定 WCP：内部迭代结果",
        "",
        "日期：2026-08-16  ",
        "状态：探索性内部开发结果",
        "",
        "## 结论摘要",
        "",
        f"- 工程门槛：**{'通过' if decision['engineering_pass'] else '未通过'}**。",
        f"- 论文价值门槛：**{'通过' if decision['paper_value_pass'] else '未通过'}**。",
        f"- 候选方法 OOD ACE={observed['candidate_mean_ace']:.5f}，当前方法 ACE={observed['current_mean_ace']:.5f}，差值={observed['ace_candidate_minus_current']:+.5f}。",
        f"- 候选/当前平均宽度比={observed['candidate_to_current_mean_width_ratio']:.5f}；最低有限区间比例={observed['candidate_minimum_finite_interval_fraction']:.6f}。",
        f"- 候选 ESS 比例={observed['candidate_mean_ess_fraction']:.4f}，纯高维 ESS 比例={observed['hd_mean_ess_fraction']:.4f}。",
        f"- 候选标准化残差 KS={observed['candidate_normalized_score_mean_cell_ks']:.5f}，未加权={observed['current_normalized_score_mean_cell_ks']:.5f}。",
        f"- ACE 改善单元={observed['ace_better_units']}/10；选择的 lambda 分布={observed['selected_lambda_counts']}。",
        "",
        "## 90% OOD 主结果",
        "",
        dataframe_to_markdown(primary),
        "",
        "## 无标签选择结果",
        "",
        dataframe_to_markdown(selected_balance),
        "",
        "## 候选对当前方法的配对比较",
        "",
        dataframe_to_markdown(paired),
        "",
        "## 工程判定",
        "",
    ]
    for name, passed in decision["engineering_conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}：`{name}`")
    lines.extend(("", "## 论文价值判定", ""))
    for name, passed in decision["paper_value_conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}：`{name}`")
    lines.extend(
        (
            "",
            "## 解释边界",
            "",
            "- lambda 的选择只使用分子结构、相似度和权重稳定性，不读取响应标签。",
            "- 由于相同五个种子的响应已经在此前开发阶段查看，本结果仍不可视为确认性证据。",
            "- 所有未达到预期的结果继续保留在内部目录，不自动进入论文。",
            "",
        )
    )
    (output / "INTERNAL_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--output", default="results/factorized_shift_wcp_20260816")
    args = parser.parse_args()
    root = Path(args.root)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    families = tuple(value for value in args.families.split(",") if value)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    data = EvaluationData(root)

    metric_rows: list[dict] = []
    bin_rows: list[dict] = []
    grid_rows: list[dict] = []
    balance_rows: list[dict] = []
    score_rows: list[dict] = []
    details = []
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for seed in seeds:
            metrics, bins, grid, balance, scores, detail = evaluate_one(
                root, data, family, seed
            )
            metric_rows.extend(metrics)
            bin_rows.extend(bins)
            grid_rows.extend(grid)
            balance_rows.extend(balance)
            score_rows.extend(scores)
            details.append(detail)
            print(
                f"completed family={family} seed={seed} selected_lambda={detail['selection']['selected_lambda']}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    bins = pd.DataFrame(bin_rows)
    grid = pd.DataFrame(grid_rows)
    balance = pd.DataFrame(balance_rows)
    score_shift = pd.DataFrame(score_rows)
    summary = summarize_metrics(metrics)
    contrasts = paired_contrasts(metrics)
    decision = build_decision(metrics, balance, score_shift)

    metrics.to_csv(output / "method_seed_metrics.csv", index=False)
    bins.to_csv(output / "similarity_bin_metrics.csv", index=False)
    grid.to_csv(output / "lambda_grid_seed_metrics.csv", index=False)
    balance.to_csv(output / "lambda_balance_diagnostics.csv", index=False)
    score_shift.to_csv(output / "score_shift_diagnostics.csv", index=False)
    summary.to_csv(output / "method_summary_ci.csv", index=False)
    contrasts.to_csv(output / "paired_candidate_contrasts.csv", index=False)
    (output / "run_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_report(output, decision, summary, balance, contrasts)
    manifest = {
        "status": "complete",
        "families": families,
        "seeds": seeds,
        "alphas": ALPHAS,
        "lambda_grid": LAMBDAS.tolist(),
        "selection_uses_response_labels": False,
        "point_predictor": "frozen Phase 1A MLP",
        "scale": "frozen-role validation-fitted shared monotone similarity scale",
        "target_responses_used": "evaluation only",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

