"""Externally evaluate fixed factorized conformal weighting on HTS384."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.evaluate_factorized_shift_wcp import (
        ALPHAS,
        choose_factorized_weight,
        dataframe_to_markdown,
        exact_sign_flip_pvalue,
    )
    from scripts.evaluate_hts384_external_lockbox import (
        HTS384Data,
        load_split_arrays,
        masked_interval_metrics,
        predict_external_mlp,
    )
    from scripts.evaluate_phase1a_conformal import (
        EvaluationData,
        estimate_domain_weights,
        nearest_fit_similarity,
    )
    from scripts.evaluate_phase1b_similarity import (
        fit_similarity_scale,
        predict_similarity_scale,
    )
    from scripts.evaluate_similarity_sufficient_wcp import (
        estimate_similarity_domain_weights,
        mean_ci,
        score_shift_row,
        standard_intervals,
        standardized_scores,
        weighted_scaled_intervals,
    )
except ModuleNotFoundError:
    from evaluate_factorized_shift_wcp import (  # type: ignore[no-redef]
        ALPHAS,
        choose_factorized_weight,
        dataframe_to_markdown,
        exact_sign_flip_pvalue,
    )
    from evaluate_hts384_external_lockbox import (  # type: ignore[no-redef]
        HTS384Data,
        load_split_arrays,
        masked_interval_metrics,
        predict_external_mlp,
    )
    from evaluate_phase1a_conformal import (  # type: ignore[no-redef]
        EvaluationData,
        estimate_domain_weights,
        nearest_fit_similarity,
    )
    from evaluate_phase1b_similarity import (  # type: ignore[no-redef]
        fit_similarity_scale,
        predict_similarity_scale,
    )
    from evaluate_similarity_sufficient_wcp import (  # type: ignore[no-redef]
        estimate_similarity_domain_weights,
        mean_ci,
        score_shift_row,
        standard_intervals,
        standardized_scores,
        weighted_scaled_intervals,
    )


FAMILIES = ("scaffold", "leader_cluster")
METHODS = ("SNCP_unweighted", "SNCP_Sim1D", "SNCP_HD", "SNCP_factorized_selected")
PRIMARY_ALPHA = 0.10


def evaluate_one(
    root: Path,
    data: EvaluationData,
    external: HTS384Data,
    family: str,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    arrays = load_split_arrays(root, data, family, seed, "mlp")
    fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
    pred_external = predict_external_mlp(root, external, family, seed)
    truth_external = external.truth(data.cell_lines)
    x_fit = data.x[data.rows(fit_ids)]
    x_cal = data.x[data.rows(arrays["ids_source_cal"])]
    sim_cal = nearest_fit_similarity(x_fit, x_cal)
    sim_valid = nearest_fit_similarity(x_fit, data.x[data.rows(arrays["ids_valid"])])
    sim_external = nearest_fit_similarity(x_fit, external.x)

    scale_model, _, scale_report = fit_similarity_scale(
        sim_valid, arrays["truth_valid"], arrays["pred_valid"]
    )
    cal_scale = predict_similarity_scale(scale_model, sim_cal)
    external_scale = predict_similarity_scale(scale_model, sim_external)
    sim_cal_weights, sim_external_weights, sim_report = estimate_similarity_domain_weights(
        sim_cal, sim_external, seed=180_000 + seed
    )
    hd_cal_weights, hd_external_weights, hd_report = estimate_domain_weights(
        x_cal, external.x, seed=190_000 + seed
    )
    selected_cal, selected_external, balance_frame, selection_report = choose_factorized_weight(
        x_cal,
        external.x,
        sim_cal,
        sim_external,
        sim_cal_weights,
        sim_external_weights,
        hd_cal_weights,
        hd_external_weights,
        absolute_minimum_ess_fraction=0.25,
    )
    balance_rows = [
        {"family": family, "seed": seed, **row}
        for row in balance_frame.to_dict(orient="records")
    ]

    label_masks = {
        "all_labels": np.isfinite(truth_external),
        "non_endpoint_4_8": np.isfinite(truth_external)
        & ~np.isclose(truth_external, 4.0)
        & ~np.isclose(truth_external, 8.0),
    }
    row_masks = {
        "all_compounds": np.ones(len(external.ids), dtype=bool),
        "similarity_lt_0.4": sim_external < 0.4,
    }
    metric_rows = []
    for alpha in ALPHAS:
        intervals = {
            "SNCP_unweighted": standard_intervals(
                arrays["truth_source_cal"],
                arrays["pred_source_cal"],
                pred_external,
                cal_scale,
                external_scale,
                alpha,
            ),
            "SNCP_Sim1D": weighted_scaled_intervals(
                arrays["truth_source_cal"],
                arrays["pred_source_cal"],
                pred_external,
                cal_scale,
                external_scale,
                sim_cal_weights,
                sim_external_weights,
                alpha,
            ),
            "SNCP_HD": weighted_scaled_intervals(
                arrays["truth_source_cal"],
                arrays["pred_source_cal"],
                pred_external,
                cal_scale,
                external_scale,
                hd_cal_weights,
                hd_external_weights,
                alpha,
            ),
            "SNCP_factorized_selected": weighted_scaled_intervals(
                arrays["truth_source_cal"],
                arrays["pred_source_cal"],
                pred_external,
                cal_scale,
                external_scale,
                selected_cal,
                selected_external,
                alpha,
            ),
        }
        for method in METHODS:
            lower, upper = intervals[method]
            for row_scope, row_mask in row_masks.items():
                for label_scope, label_mask in label_masks.items():
                    combined = label_mask & row_mask[:, None]
                    metric_rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "model": "mlp",
                            "alpha": alpha,
                            "method": method,
                            "selected_lambda": selection_report["selected_lambda"],
                            "row_scope": row_scope,
                            "label_scope": label_scope,
                            **masked_interval_metrics(
                                truth_external, lower, upper, alpha, combined
                            ),
                        }
                    )

    normalized_cal_scores, cell_medians = standardized_scores(
        arrays["truth_source_cal"], arrays["pred_source_cal"], cal_scale
    )
    normalized_external_scores, _ = standardized_scores(
        truth_external, pred_external, external_scale, cell_medians
    )
    score_rows = []
    for weight_name, weights in (
        ("unweighted", np.ones(len(sim_cal))),
        ("Sim1D", sim_cal_weights),
        ("HD", hd_cal_weights),
        ("factorized_selected", selected_cal),
    ):
        score_rows.append(
            score_shift_row(
                family,
                seed,
                "similarity_normalized",
                weight_name,
                normalized_cal_scores,
                normalized_external_scores,
                weights,
            )
        )
    detail = {
        "family": family,
        "seed": seed,
        "external_compounds": int(len(external.ids)),
        "external_labels": int(np.isfinite(truth_external).sum()),
        "selection": selection_report,
        "scale": scale_report,
        "similarity_weight": sim_report,
        "high_dimensional_weight": hd_report,
        "external_similarity": {
            "mean": float(sim_external.mean()),
            "median": float(np.median(sim_external)),
            "fraction_below_0_4": float((sim_external < 0.4).mean()),
        },
    }
    return metric_rows, balance_rows, score_rows, detail


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in metrics.groupby(
        ["alpha", "method", "row_scope", "label_scope"], sort=True
    ):
        alpha, method, row_scope, label_scope = keys
        row = {
            "alpha": alpha,
            "method": method,
            "row_scope": row_scope,
            "label_scope": label_scope,
            "units": len(part),
        }
        for column in (
            "coverage",
            "coverage_error_abs",
            "mean_width",
            "finite_interval_fraction",
        ):
            mean, sd, low, high = mean_ci(part[column])
            row[f"{column}_mean"] = mean
            row[f"{column}_sd"] = sd
            row[f"{column}_ci95_low"] = low
            row[f"{column}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[np.isclose(metrics["alpha"], PRIMARY_ALPHA)]
    rows = []
    for row_scope, label_scope in itertools.product(
        ("all_compounds", "similarity_lt_0.4"),
        ("all_labels", "non_endpoint_4_8"),
    ):
        selected = primary[
            (primary["row_scope"] == row_scope)
            & (primary["label_scope"] == label_scope)
        ]
        candidate = selected[selected["method"] == "SNCP_factorized_selected"].set_index(
            ["family", "seed"]
        )
        for comparator in ("SNCP_unweighted", "SNCP_Sim1D", "SNCP_HD"):
            reference = selected[selected["method"] == comparator].set_index(
                ["family", "seed"]
            )
            for metric in (
                "coverage_error_abs",
                "mean_width",
                "finite_interval_fraction",
            ):
                difference = candidate[metric] - reference[metric]
                mean, sd, low, high = mean_ci(difference)
                rows.append(
                    {
                        "row_scope": row_scope,
                        "label_scope": label_scope,
                        "candidate": "SNCP_factorized_selected",
                        "comparator": comparator,
                        "metric": metric,
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


def build_decision(metrics: pd.DataFrame, balance: pd.DataFrame) -> dict:
    primary = metrics[
        np.isclose(metrics["alpha"], PRIMARY_ALPHA)
        & (metrics["row_scope"] == "all_compounds")
    ]
    observed = {}
    conditions = {}
    for label_scope in ("all_labels", "non_endpoint_4_8"):
        selected = primary[primary["label_scope"] == label_scope]
        candidate = selected[selected["method"] == "SNCP_factorized_selected"].set_index(
            ["family", "seed"]
        )
        current = selected[selected["method"] == "SNCP_unweighted"].set_index(
            ["family", "seed"]
        )
        ace_difference = candidate["coverage_error_abs"] - current["coverage_error_abs"]
        observed[label_scope] = {
            "candidate_ace": float(candidate["coverage_error_abs"].mean()),
            "current_ace": float(current["coverage_error_abs"].mean()),
            "ace_difference": float(ace_difference.mean()),
            "candidate_coverage": float(candidate["coverage"].mean()),
            "current_coverage": float(current["coverage"].mean()),
            "width_ratio": float(candidate["mean_width"].mean() / current["mean_width"].mean()),
            "minimum_finite_interval_fraction": float(
                candidate["finite_interval_fraction"].min()
            ),
            "ace_better_units": int((ace_difference < 0).sum()),
        }
        conditions[f"{label_scope}_ace_lower"] = bool(ace_difference.mean() < 0)
        conditions[f"{label_scope}_width_ratio_le_1_10"] = bool(
            observed[label_scope]["width_ratio"] <= 1.10
        )
    conditions["minimum_finite_interval_fraction_ge_0_999"] = bool(
        min(
            observed["all_labels"]["minimum_finite_interval_fraction"],
            observed["non_endpoint_4_8"]["minimum_finite_interval_fraction"],
        )
        >= 0.999
    )
    conditions["all_labels_ace_better_in_at_least_5_of_8_units"] = bool(
        observed["all_labels"]["ace_better_units"] >= 5
    )
    selected_balance = balance[balance["selected"]]
    hd_balance = balance[np.isclose(balance["lambda"], 1.0)]
    selected_ess = float(selected_balance["ess_fraction"].mean())
    hd_ess = float(hd_balance["ess_fraction"].mean())
    conditions["factorized_ess_higher_than_hd"] = selected_ess > hd_ess
    return {
        "status": "assay_external_validation",
        "observed": {
            **observed,
            "selected_mean_ess_fraction": selected_ess,
            "hd_mean_ess_fraction": hd_ess,
            "selected_lambda_counts": {
                str(float(key)): int(value)
                for key, value in selected_balance["lambda"].value_counts().sort_index().items()
            },
        },
        "conditions": conditions,
        "external_validation_pass": bool(all(conditions.values())),
        "guardrail": (
            "HTS384 is assay-external, not temporal. Its base-method labels were previously inspected; "
            "the factorized method and unlabeled selection rule were fixed before this evaluation."
        ),
    }


def write_report(
    output: Path,
    decision: dict,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    balance: pd.DataFrame,
) -> None:
    primary = summary[
        np.isclose(summary["alpha"], PRIMARY_ALPHA)
        & (summary["row_scope"] == "all_compounds")
    ][
        [
            "method",
            "label_scope",
            "coverage_mean",
            "coverage_error_abs_mean",
            "mean_width_mean",
            "finite_interval_fraction_mean",
        ]
    ]
    paired = contrasts[
        (contrasts["row_scope"] == "all_compounds")
        & (contrasts["comparator"] == "SNCP_unweighted")
    ]
    selected_balance = balance[balance["selected"]][
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
    all_labels = decision["observed"]["all_labels"]
    non_endpoint = decision["observed"]["non_endpoint_4_8"]
    lines = [
        "# 因子化相似度锚定 WCP：HTS384 外部验证结果",
        "",
        "日期：2026-08-16",
        "",
        f"- 外部验证门槛：**{'通过' if decision['external_validation_pass'] else '未通过'}**。",
        f"- 全部标签：候选ACE={all_labels['candidate_ace']:.5f}，当前ACE={all_labels['current_ace']:.5f}，差值={all_labels['ace_difference']:+.5f}。",
        f"- 排除4/8端点：候选ACE={non_endpoint['candidate_ace']:.5f}，当前ACE={non_endpoint['current_ace']:.5f}，差值={non_endpoint['ace_difference']:+.5f}。",
        f"- 全部标签候选/当前宽度比={all_labels['width_ratio']:.5f}；ACE改善单元={all_labels['ace_better_units']}/8。",
        f"- 因子化ESS={decision['observed']['selected_mean_ess_fraction']:.4f}，纯高维ESS={decision['observed']['hd_mean_ess_fraction']:.4f}。",
        f"- lambda分布={decision['observed']['selected_lambda_counts']}。",
        "",
        "## 90% 全外部化合物",
        "",
        dataframe_to_markdown(primary),
        "",
        "## 候选对当前方法配对比较",
        "",
        dataframe_to_markdown(paired),
        "",
        "## 无标签lambda选择",
        "",
        dataframe_to_markdown(selected_balance),
        "",
        "## 判定条件",
        "",
    ]
    for name, passed in decision["conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}：`{name}`")
    lines.extend(
        (
            "",
            "## 解释边界",
            "",
            f"- {decision['guardrail']}",
            "- 权重与lambda不使用HTS384响应；响应只用于本文件的最终评估。",
            "- 全部结果保留在内部目录，是否进入论文需结合统计审计决定。",
            "",
        )
    )
    (output / "INTERNAL_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--families", default="scaffold,leader_cluster")
    parser.add_argument("--seeds", default="2,3,4,5")
    parser.add_argument("--output", default="results/factorized_hts384_external_v2_20260816")
    args = parser.parse_args()
    root = Path(args.root)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    data = EvaluationData(root)
    external = HTS384Data(root)

    metric_rows: list[dict] = []
    balance_rows: list[dict] = []
    score_rows: list[dict] = []
    details = []
    for family in [value for value in args.families.split(",") if value]:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for seed in [int(value) for value in args.seeds.split(",") if value]:
            metrics, balance, scores, detail = evaluate_one(
                root, data, external, family, seed
            )
            metric_rows.extend(metrics)
            balance_rows.extend(balance)
            score_rows.extend(scores)
            details.append(detail)
            print(
                f"completed family={family} seed={seed} selected_lambda={detail['selection']['selected_lambda']}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    balance = pd.DataFrame(balance_rows)
    score_shift = pd.DataFrame(score_rows)
    summary = summarize(metrics)
    contrasts = paired_contrasts(metrics)
    decision = build_decision(metrics, balance)
    metrics.to_csv(output / "external_interval_metrics.csv", index=False)
    balance.to_csv(output / "lambda_balance_diagnostics.csv", index=False)
    score_shift.to_csv(output / "score_shift_diagnostics.csv", index=False)
    summary.to_csv(output / "external_summary_ci.csv", index=False)
    contrasts.to_csv(output / "paired_external_contrasts.csv", index=False)
    (output / "run_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_report(output, decision, summary, contrasts, balance)
    manifest = {
        "status": "complete",
        "families": args.families,
        "seeds": args.seeds,
        "model": "mlp",
        "external_compounds": int(len(external.ids)),
        "selection_uses_external_response_labels": False,
        "external_absolute_minimum_ess_fraction": 0.25,
        "guardrail": decision["guardrail"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

