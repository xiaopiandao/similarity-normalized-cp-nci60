"""Develop overlap-shrunk factorized WCP and stress-test it on HTS384.

The response-free balance rule first selects a domain-specific lambda_hat.
This script evaluates lambda_eff = kappa * lambda_hat, selects a single kappa
using internal OOD outcomes only, and then applies it unchanged to HTS384.

Exact values 4 and 8 are boundary candidates, not confirmed censored labels.
The one-sided scenario is therefore an assumption-based compatibility stress
test and is never interpreted as ordinary conformal coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.analyze_endpoint_boundary_sensitivity import scenario_metrics
    from scripts.evaluate_factorized_shift_wcp import (
        choose_factorized_weight,
        dataframe_to_markdown,
        geometric_factorized_weights,
        load_high_dimensional_weights,
        load_similarity,
    )
    from scripts.evaluate_hts384_external_lockbox import (
        HTS384Data,
        load_split_arrays,
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
        standard_intervals,
        weighted_scaled_intervals,
    )
except ModuleNotFoundError:
    from analyze_endpoint_boundary_sensitivity import scenario_metrics  # type: ignore[no-redef]
    from evaluate_factorized_shift_wcp import (  # type: ignore[no-redef]
        choose_factorized_weight,
        dataframe_to_markdown,
        geometric_factorized_weights,
        load_high_dimensional_weights,
        load_similarity,
    )
    from evaluate_hts384_external_lockbox import (  # type: ignore[no-redef]
        HTS384Data,
        load_split_arrays,
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
        standard_intervals,
        weighted_scaled_intervals,
    )


ALPHA = 0.10
KAPPAS = np.asarray((0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0))
FAMILIES = ("scaffold", "leader_cluster")
SCENARIOS = (
    "reference_all",
    "exclude_exact_4_or_8",
    "one_sided_compatible",
)
CURRENT_METHOD = "SNCP_unweighted"
SHRUNK_METHOD = "SNCP_factorized_shrunk"


def weight_diagnostics(
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    minimum_ess_fraction: float,
    alpha: float = ALPHA,
) -> dict:
    """Return finite-sample overlap diagnostics for one weight vector pair."""
    cal_weights = np.asarray(cal_weights, dtype=float)
    test_weights = np.asarray(test_weights, dtype=float)
    ess = float(cal_weights.sum() ** 2 / np.sum(cal_weights**2))
    ess_fraction = ess / len(cal_weights)
    support_limit = (alpha / (1.0 - alpha)) * cal_weights.sum()
    support_failure = float((test_weights > support_limit).mean())
    return {
        "ess": ess,
        "ess_fraction": ess_fraction,
        "minimum_ess_fraction": float(minimum_ess_fraction),
        "support_failure_fraction": support_failure,
        "feasible": bool(
            ess_fraction >= minimum_ess_fraction and support_failure == 0.0
        ),
    }


def add_interval_rows(
    rows: list[dict],
    *,
    domain: str,
    family: str,
    seed: int,
    method: str,
    kappa: float | None,
    selected_lambda: float | None,
    effective_lambda: float | None,
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """Append the three frozen endpoint-sensitivity evaluations."""
    for scenario in SCENARIOS:
        rows.append(
            {
                "domain": domain,
                "family": family,
                "seed": seed,
                "alpha": ALPHA,
                "method": method,
                "kappa": kappa,
                "selected_lambda": selected_lambda,
                "effective_lambda": effective_lambda,
                "scenario": scenario,
                **scenario_metrics(truth, lower, upper, ALPHA, scenario),
            }
        )


def evaluate_internal_unit(
    root: Path,
    data: EvaluationData,
    family: str,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
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
    sim_cal = load_similarity(
        similarity_dir / "nearest_fit_tanimoto_source_cal.csv", ids_cal
    )
    sim_valid = load_similarity(
        similarity_dir / "nearest_fit_tanimoto_valid.csv", ids_valid
    )
    sim_test = load_similarity(
        similarity_dir / "nearest_fit_tanimoto_test.csv", ids_test
    )
    scale_model, _, scale_report = fit_similarity_scale(
        sim_valid, truth_valid, pred_valid
    )
    cal_scale = predict_similarity_scale(scale_model, sim_cal)
    test_scale = predict_similarity_scale(scale_model, sim_test)
    sim_cal_w, sim_test_w, sim_report = estimate_similarity_domain_weights(
        sim_cal, sim_test, seed=80_000 + seed
    )
    hd_cal_w, hd_test_w, hd_report = load_high_dimensional_weights(
        root, family, seed, ids_cal, ids_test
    )
    source_x = data.x[data.rows(ids_cal)]
    target_x = data.x[data.rows(ids_test)]
    _, _, _, selection = choose_factorized_weight(
        source_x,
        target_x,
        sim_cal,
        sim_test,
        sim_cal_w,
        sim_test_w,
        hd_cal_w,
        hd_test_w,
        absolute_minimum_ess_fraction=0.30,
    )
    selected_lambda = float(selection["selected_lambda"])

    metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    current_lower, current_upper = standard_intervals(
        truth_cal, pred_cal, pred_test, cal_scale, test_scale, ALPHA
    )
    add_interval_rows(
        metric_rows,
        domain="internal_ood",
        family=family,
        seed=seed,
        method=CURRENT_METHOD,
        kappa=None,
        selected_lambda=None,
        effective_lambda=None,
        truth=truth_test,
        lower=current_lower,
        upper=current_upper,
    )
    for kappa in KAPPAS:
        effective_lambda = float(kappa * selected_lambda)
        cal_w, test_w = geometric_factorized_weights(
            sim_cal_w,
            sim_test_w,
            hd_cal_w,
            hd_test_w,
            effective_lambda,
        )
        lower, upper = weighted_scaled_intervals(
            truth_cal,
            pred_cal,
            pred_test,
            cal_scale,
            test_scale,
            cal_w,
            test_w,
            ALPHA,
        )
        add_interval_rows(
            metric_rows,
            domain="internal_ood",
            family=family,
            seed=seed,
            method=SHRUNK_METHOD,
            kappa=float(kappa),
            selected_lambda=selected_lambda,
            effective_lambda=effective_lambda,
            truth=truth_test,
            lower=lower,
            upper=upper,
        )
        diagnostic_rows.append(
            {
                "domain": "internal_ood",
                "family": family,
                "seed": seed,
                "kappa": float(kappa),
                "selected_lambda": selected_lambda,
                "effective_lambda": effective_lambda,
                **weight_diagnostics(
                    cal_w, test_w, float(selection["minimum_ess_fraction"])
                ),
            }
        )
    detail = {
        "domain": "internal_ood",
        "family": family,
        "seed": seed,
        "selection": selection,
        "scale": scale_report,
        "similarity_weight": sim_report,
        "high_dimensional_weight": hd_report,
    }
    return metric_rows, diagnostic_rows, detail


def evaluate_external_unit(
    root: Path,
    data: EvaluationData,
    external: HTS384Data,
    family: str,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    arrays = load_split_arrays(root, data, family, seed, "mlp")
    fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
    pred_external = predict_external_mlp(root, external, family, seed)
    truth_external = external.truth(data.cell_lines)
    x_fit = data.x[data.rows(fit_ids)]
    x_cal = data.x[data.rows(arrays["ids_source_cal"])]
    sim_cal = nearest_fit_similarity(x_fit, x_cal)
    sim_valid = nearest_fit_similarity(
        x_fit, data.x[data.rows(arrays["ids_valid"])]
    )
    sim_external = nearest_fit_similarity(x_fit, external.x)
    scale_model, _, scale_report = fit_similarity_scale(
        sim_valid, arrays["truth_valid"], arrays["pred_valid"]
    )
    cal_scale = predict_similarity_scale(scale_model, sim_cal)
    external_scale = predict_similarity_scale(scale_model, sim_external)
    sim_cal_w, sim_external_w, sim_report = estimate_similarity_domain_weights(
        sim_cal, sim_external, seed=180_000 + seed
    )
    hd_cal_w, hd_external_w, hd_report = estimate_domain_weights(
        x_cal, external.x, seed=190_000 + seed
    )
    _, _, _, selection = choose_factorized_weight(
        x_cal,
        external.x,
        sim_cal,
        sim_external,
        sim_cal_w,
        sim_external_w,
        hd_cal_w,
        hd_external_w,
        absolute_minimum_ess_fraction=0.25,
    )
    selected_lambda = float(selection["selected_lambda"])

    metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    current_lower, current_upper = standard_intervals(
        arrays["truth_source_cal"],
        arrays["pred_source_cal"],
        pred_external,
        cal_scale,
        external_scale,
        ALPHA,
    )
    add_interval_rows(
        metric_rows,
        domain="hts384_external_exploratory",
        family=family,
        seed=seed,
        method=CURRENT_METHOD,
        kappa=None,
        selected_lambda=None,
        effective_lambda=None,
        truth=truth_external,
        lower=current_lower,
        upper=current_upper,
    )
    for kappa in KAPPAS:
        effective_lambda = float(kappa * selected_lambda)
        cal_w, external_w = geometric_factorized_weights(
            sim_cal_w,
            sim_external_w,
            hd_cal_w,
            hd_external_w,
            effective_lambda,
        )
        lower, upper = weighted_scaled_intervals(
            arrays["truth_source_cal"],
            arrays["pred_source_cal"],
            pred_external,
            cal_scale,
            external_scale,
            cal_w,
            external_w,
            ALPHA,
        )
        add_interval_rows(
            metric_rows,
            domain="hts384_external_exploratory",
            family=family,
            seed=seed,
            method=SHRUNK_METHOD,
            kappa=float(kappa),
            selected_lambda=selected_lambda,
            effective_lambda=effective_lambda,
            truth=truth_external,
            lower=lower,
            upper=upper,
        )
        diagnostic_rows.append(
            {
                "domain": "hts384_external_exploratory",
                "family": family,
                "seed": seed,
                "kappa": float(kappa),
                "selected_lambda": selected_lambda,
                "effective_lambda": effective_lambda,
                **weight_diagnostics(
                    cal_w, external_w, float(selection["minimum_ess_fraction"])
                ),
            }
        )
    detail = {
        "domain": "hts384_external_exploratory",
        "family": family,
        "seed": seed,
        "selection": selection,
        "scale": scale_report,
        "similarity_weight": sim_report,
        "high_dimensional_weight": hd_report,
        "boundary_guardrail": "exact 4/8 are boundary candidates, not confirmed censoring",
    }
    return metric_rows, diagnostic_rows, detail


def summarize_kappas(metrics: pd.DataFrame, domain: str) -> pd.DataFrame:
    selected = metrics[
        (metrics["domain"] == domain) & (metrics["method"] == SHRUNK_METHOD)
    ]
    rows = []
    for (kappa, family, scenario), part in selected.groupby(
        ["kappa", "family", "scenario"], sort=True
    ):
        rows.append(
            {
                "domain": domain,
                "kappa": float(kappa),
                "family": family,
                "scenario": scenario,
                "units": len(part),
                "coverage_mean": float(part["coverage"].mean()),
                "ace_mean": float(part["coverage_error_abs"].mean()),
                "width_mean": float(part["mean_width"].mean()),
                "finite_min": float(part["finite_interval_fraction"].min()),
            }
        )
    return pd.DataFrame(rows)


def current_robust_objective(metrics: pd.DataFrame) -> float:
    current = metrics[
        (metrics["domain"] == "internal_ood")
        & (metrics["method"] == CURRENT_METHOD)
    ]
    cells = current.groupby(["family", "scenario"])["coverage_error_abs"].mean()
    return float(cells.max())


def choose_kappa(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Select kappa from internal OOD outcomes only using the frozen rules."""
    internal = metrics[
        (metrics["domain"] == "internal_ood")
        & (metrics["method"] == SHRUNK_METHOD)
    ]
    current = metrics[
        (metrics["domain"] == "internal_ood")
        & (metrics["method"] == CURRENT_METHOD)
    ]
    internal_diag = diagnostics[diagnostics["domain"] == "internal_ood"]
    rows = []
    for kappa, part in internal.groupby("kappa", sort=True):
        cells = part.groupby(["family", "scenario"]).agg(
            ace_mean=("coverage_error_abs", "mean"),
            width_mean=("mean_width", "mean"),
        )
        current_cells = current.groupby(["family", "scenario"]).agg(
            width_mean=("mean_width", "mean")
        )
        width_ratios = cells["width_mean"] / current_cells["width_mean"]
        diag = internal_diag[np.isclose(internal_diag["kappa"], float(kappa))]
        rows.append(
            {
                "kappa": float(kappa),
                "robust_max_family_scenario_ace": float(cells["ace_mean"].max()),
                "mean_family_scenario_ace": float(cells["ace_mean"].mean()),
                "mean_width": float(part["mean_width"].mean()),
                "max_width_ratio_to_current": float(width_ratios.max()),
                "minimum_finite_interval_fraction": float(
                    part["finite_interval_fraction"].min()
                ),
                "minimum_ess_fraction": float(diag["ess_fraction"].min()),
                "maximum_support_failure_fraction": float(
                    diag["support_failure_fraction"].max()
                ),
                "all_weight_units_feasible": bool(diag["feasible"].all()),
                "feasible": bool(
                    part["finite_interval_fraction"].min() >= 0.999
                    and width_ratios.max() <= 1.10
                    and diag["feasible"].all()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    feasible = frame[frame["feasible"]].copy()
    if feasible.empty:
        raise ValueError("No kappa satisfies the pre-specified internal feasibility rules")
    selected = feasible.sort_values(
        [
            "robust_max_family_scenario_ace",
            "mean_family_scenario_ace",
            "mean_width",
            "kappa",
        ],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).iloc[0]
    selected_kappa = float(selected["kappa"])
    frame["selected_internal"] = np.isclose(frame["kappa"], selected_kappa)
    return selected_kappa, frame


def scenario_comparison(
    metrics: pd.DataFrame,
    domain: str,
    selected_kappa: float,
) -> pd.DataFrame:
    selected = metrics[
        (metrics["domain"] == domain)
        & (metrics["method"] == SHRUNK_METHOD)
        & np.isclose(metrics["kappa"], selected_kappa)
    ]
    current = metrics[
        (metrics["domain"] == domain) & (metrics["method"] == CURRENT_METHOD)
    ]
    rows = []
    for scenario in SCENARIOS:
        candidate_part = selected[selected["scenario"] == scenario].set_index(
            ["family", "seed"]
        )
        current_part = current[current["scenario"] == scenario].set_index(
            ["family", "seed"]
        )
        ace_diff = (
            candidate_part["coverage_error_abs"]
            - current_part["coverage_error_abs"]
        )
        rows.append(
            {
                "domain": domain,
                "scenario": scenario,
                "selected_kappa": selected_kappa,
                "units": len(candidate_part),
                "candidate_coverage": float(candidate_part["coverage"].mean()),
                "current_coverage": float(current_part["coverage"].mean()),
                "candidate_ace": float(candidate_part["coverage_error_abs"].mean()),
                "current_ace": float(current_part["coverage_error_abs"].mean()),
                "ace_difference": float(ace_diff.mean()),
                "candidate_width": float(candidate_part["mean_width"].mean()),
                "current_width": float(current_part["mean_width"].mean()),
                "width_ratio": float(
                    candidate_part["mean_width"].mean()
                    / current_part["mean_width"].mean()
                ),
                "minimum_finite_interval_fraction": float(
                    candidate_part["finite_interval_fraction"].min()
                ),
                "ace_better_units": int((ace_diff < 0).sum()),
                "ties": int(np.isclose(ace_diff, 0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_decision(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected_kappa: float,
    selection_frame: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    comparisons = pd.concat(
        [
            scenario_comparison(metrics, "internal_ood", selected_kappa),
            scenario_comparison(
                metrics, "hts384_external_exploratory", selected_kappa
            ),
        ],
        ignore_index=True,
    )
    selected_row = selection_frame[
        np.isclose(selection_frame["kappa"], selected_kappa)
    ].iloc[0]
    adaptive_row = selection_frame[np.isclose(selection_frame["kappa"], 1.0)].iloc[0]
    current_objective = current_robust_objective(metrics)
    internal_conditions = {
        "robust_objective_lower_than_unshrunk_kappa_1": bool(
            selected_row["robust_max_family_scenario_ace"]
            < adaptive_row["robust_max_family_scenario_ace"]
        ),
        "robust_objective_lower_than_current_sncp": bool(
            selected_row["robust_max_family_scenario_ace"] < current_objective
        ),
        "internal_feasibility_pass": bool(selected_row["feasible"]),
    }
    external = comparisons[
        comparisons["domain"] == "hts384_external_exploratory"
    ].set_index("scenario")
    external_diag = diagnostics[
        (diagnostics["domain"] == "hts384_external_exploratory")
        & np.isclose(diagnostics["kappa"], selected_kappa)
    ]
    external_conditions = {
        "reference_all_ace_not_higher_than_current": bool(
            external.loc["reference_all", "ace_difference"] <= 0.0
        ),
        "boundary_excluded_ace_noninferior_margin_0_002": bool(
            external.loc["exclude_exact_4_or_8", "ace_difference"] <= 0.002
        ),
        "hypothetical_one_sided_ace_not_higher_than_current": bool(
            external.loc["one_sided_compatible", "ace_difference"] <= 0.0
        ),
        "all_scenarios_width_ratio_le_1_10": bool(
            (external["width_ratio"] <= 1.10).all()
        ),
        "minimum_finite_interval_fraction_ge_0_999": bool(
            external["minimum_finite_interval_fraction"].min() >= 0.999
        ),
        "reference_all_ace_better_in_at_least_5_of_8_units": bool(
            external.loc["reference_all", "ace_better_units"] >= 5
        ),
        "external_weight_feasibility_pass": bool(external_diag["feasible"].all()),
    }
    all_conditions = {**internal_conditions, **external_conditions}
    decision = {
        "status": "exploratory_method_development",
        "selected_kappa_internal_only": selected_kappa,
        "selection_uses_hts384_outcomes": False,
        "hts384_status": (
            "exploratory pressure test because its outcomes were inspected in prior work"
        ),
        "internal_robust_objective": {
            "selected": float(selected_row["robust_max_family_scenario_ace"]),
            "unshrunk_kappa_1": float(
                adaptive_row["robust_max_family_scenario_ace"]
            ),
            "current_sncp": current_objective,
        },
        "internal_conditions": internal_conditions,
        "external_conditions": external_conditions,
        "paper_candidate_pass": bool(all(all_conditions.values())),
        "guardrails": [
            "Exact 4/8 values are boundary candidates, not confirmed censored observations.",
            "The one-sided compatibility scenario is assumption-based and is not ordinary conformal coverage.",
            "A non-passing candidate remains internal and is not added to the manuscript.",
        ],
    }
    return decision, comparisons


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(
    output: Path,
    decision: dict,
    selection_frame: pd.DataFrame,
    comparisons: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    selected_kappa = float(decision["selected_kappa_internal_only"])
    selected_diag = diagnostics[np.isclose(diagnostics["kappa"], selected_kappa)]
    compact_selection = selection_frame[
        [
            "kappa",
            "robust_max_family_scenario_ace",
            "mean_family_scenario_ace",
            "max_width_ratio_to_current",
            "minimum_ess_fraction",
            "feasible",
            "selected_internal",
        ]
    ]
    compact_comparisons = comparisons[
        [
            "domain",
            "scenario",
            "candidate_coverage",
            "current_coverage",
            "candidate_ace",
            "current_ace",
            "ace_difference",
            "width_ratio",
            "ace_better_units",
        ]
    ]
    lines = [
        "# 因子化 WCP 重叠收缩优化：内部发现",
        "",
        "日期：2026-08-16",
        "",
        f"- 内部选择的 κ：**{selected_kappa:.3f}**。",
        f"- 论文候选总门槛：**{'通过' if decision['paper_candidate_pass'] else '未通过'}**。",
        "- HTS384 已在前序工作中查看标签，本轮只属于探索性压力测试。",
        "- 精确 4/8 仅为边界候选；假设性单侧相容结果不是普通 conformal coverage。",
        "",
        "## 内部 κ 选择",
        "",
        dataframe_to_markdown(compact_selection),
        "",
        "## 选定 κ 与当前 SNCP 的场景比较",
        "",
        dataframe_to_markdown(compact_comparisons),
        "",
        "## 判定条件",
        "",
        "### 内部开发",
        "",
    ]
    for name, passed in decision["internal_conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(("", "### HTS384 探索性压力测试", ""))
    for name, passed in decision["external_conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        (
            "",
            "## 权重诊断（选定 κ）",
            "",
            dataframe_to_markdown(
                selected_diag[
                    [
                        "domain",
                        "family",
                        "seed",
                        "selected_lambda",
                        "effective_lambda",
                        "ess_fraction",
                        "support_failure_fraction",
                        "feasible",
                    ]
                ]
            ),
            "",
            "## 论文使用规则",
            "",
            "只有总门槛全部通过时，才建议把该扩展升级为论文候选。未通过的结果仅保留在内部开发记录；不得选择性改写为已确认的删失优势。",
            "",
        )
    )
    (output / "INTERNAL_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--internal-seeds", default="1,2,3,4,5")
    parser.add_argument("--external-seeds", default="2,3,4,5")
    parser.add_argument(
        "--output", default="results/overlap_shrunk_factorized_wcp_20260816"
    )
    args = parser.parse_args()
    root = Path(args.root)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    internal_seeds = tuple(
        int(value) for value in args.internal_seeds.split(",") if value
    )
    external_seeds = tuple(
        int(value) for value in args.external_seeds.split(",") if value
    )
    data = EvaluationData(root)
    external = HTS384Data(root)

    metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    details: list[dict] = []
    for family in FAMILIES:
        for seed in internal_seeds:
            metrics, diagnostics, detail = evaluate_internal_unit(
                root, data, family, seed
            )
            metric_rows.extend(metrics)
            diagnostic_rows.extend(diagnostics)
            details.append(detail)
            print(
                f"internal complete family={family} seed={seed} "
                f"lambda={detail['selection']['selected_lambda']}",
                flush=True,
            )

    internal_metrics = pd.DataFrame(metric_rows)
    internal_diagnostics = pd.DataFrame(diagnostic_rows)
    selected_kappa, selection_frame = choose_kappa(
        internal_metrics, internal_diagnostics
    )
    print(f"internal selected kappa={selected_kappa}", flush=True)

    for family in FAMILIES:
        for seed in external_seeds:
            metrics, diagnostics, detail = evaluate_external_unit(
                root, data, external, family, seed
            )
            metric_rows.extend(metrics)
            diagnostic_rows.extend(diagnostics)
            details.append(detail)
            print(
                f"external complete family={family} seed={seed} "
                f"lambda={detail['selection']['selected_lambda']}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    internal_summary = summarize_kappas(metrics, "internal_ood")
    external_summary = summarize_kappas(
        metrics, "hts384_external_exploratory"
    )
    decision, comparisons = build_decision(
        metrics, diagnostics, selected_kappa, selection_frame
    )

    metrics.to_csv(output / "unit_scenario_metrics.csv", index=False)
    diagnostics.to_csv(output / "weight_diagnostics.csv", index=False)
    selection_frame.to_csv(output / "internal_kappa_selection.csv", index=False)
    internal_summary.to_csv(output / "internal_kappa_summary.csv", index=False)
    external_summary.to_csv(output / "external_kappa_summary.csv", index=False)
    comparisons.to_csv(output / "selected_kappa_comparisons.csv", index=False)
    (output / "run_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_report(output, decision, selection_frame, comparisons, diagnostics)
    script_path = Path(__file__).resolve()
    design_path = root / "docs" / "因子化WCP_重叠收缩优化设计_20260816.md"
    manifest = {
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "design": str(design_path),
        "design_sha256": sha256(design_path),
        "internal_seeds": list(internal_seeds),
        "external_seeds": list(external_seeds),
        "alpha": ALPHA,
        "kappas": KAPPAS.tolist(),
        "selected_kappa_internal_only": selected_kappa,
        "output": str(output),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

