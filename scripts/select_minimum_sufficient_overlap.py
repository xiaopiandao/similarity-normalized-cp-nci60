"""Select the minimum sufficient overlap shrinkage from frozen grid results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.evaluate_factorized_shift_wcp import dataframe_to_markdown
    from scripts.evaluate_overlap_shrunk_factorized_wcp import (
        CURRENT_METHOD,
        SCENARIOS,
        SHRUNK_METHOD,
        current_robust_objective,
        scenario_comparison,
    )
except ModuleNotFoundError:
    from evaluate_factorized_shift_wcp import dataframe_to_markdown  # type: ignore[no-redef]
    from evaluate_overlap_shrunk_factorized_wcp import (  # type: ignore[no-redef]
        CURRENT_METHOD,
        SCENARIOS,
        SHRUNK_METHOD,
        current_robust_objective,
        scenario_comparison,
    )


NONINFERIORITY_MARGIN = 0.002


def select_minimum_sufficient_kappa(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Choose the smallest internally adequate kappa using frozen criteria."""
    internal_candidate = metrics[
        (metrics["domain"] == "internal_ood")
        & (metrics["method"] == SHRUNK_METHOD)
    ]
    internal_current = metrics[
        (metrics["domain"] == "internal_ood")
        & (metrics["method"] == CURRENT_METHOD)
    ]
    internal_diag = diagnostics[diagnostics["domain"] == "internal_ood"]
    current_scenario_ace = internal_current.groupby("scenario")[
        "coverage_error_abs"
    ].mean()
    current_cells = internal_current.groupby(["family", "scenario"])[
        "coverage_error_abs"
    ].mean()
    current_width = internal_current.groupby(["family", "scenario"])[
        "mean_width"
    ].mean()
    current_robust = float(current_cells.max())
    rows = []
    for kappa, part in internal_candidate.groupby("kappa", sort=True):
        scenario_ace = part.groupby("scenario")["coverage_error_abs"].mean()
        scenario_diffs = scenario_ace - current_scenario_ace
        cells = part.groupby(["family", "scenario"])["coverage_error_abs"].mean()
        widths = part.groupby(["family", "scenario"])["mean_width"].mean()
        diag = internal_diag[np.isclose(internal_diag["kappa"], float(kappa))]
        noninferior = bool(
            (scenario_diffs <= NONINFERIORITY_MARGIN + 1e-12).all()
        )
        robust_improvement = bool(float(cells.max()) < current_robust)
        width_pass = bool(float((widths / current_width).max()) <= 1.10)
        finite_pass = bool(part["finite_interval_fraction"].min() >= 0.999)
        weights_pass = bool(diag["feasible"].all())
        rows.append(
            {
                "kappa": float(kappa),
                "reference_ace_difference": float(
                    scenario_diffs.loc["reference_all"]
                ),
                "boundary_excluded_ace_difference": float(
                    scenario_diffs.loc["exclude_exact_4_or_8"]
                ),
                "hypothetical_one_sided_ace_difference": float(
                    scenario_diffs.loc["one_sided_compatible"]
                ),
                "robust_max_family_scenario_ace": float(cells.max()),
                "current_robust_max_family_scenario_ace": current_robust,
                "max_width_ratio_to_current": float((widths / current_width).max()),
                "minimum_finite_interval_fraction": float(
                    part["finite_interval_fraction"].min()
                ),
                "minimum_ess_fraction": float(diag["ess_fraction"].min()),
                "maximum_support_failure_fraction": float(
                    diag["support_failure_fraction"].max()
                ),
                "all_scenarios_ace_noninferior_margin_0_002": noninferior,
                "robust_objective_lower_than_current": robust_improvement,
                "width_pass": width_pass,
                "finite_interval_pass": finite_pass,
                "weight_feasibility_pass": weights_pass,
                "eligible": bool(
                    noninferior
                    and robust_improvement
                    and width_pass
                    and finite_pass
                    and weights_pass
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("kappa", kind="mergesort")
    eligible = frame[frame["eligible"]]
    if eligible.empty:
        raise ValueError("No kappa meets the minimum-sufficient correction rules")
    selected_kappa = float(eligible.iloc[0]["kappa"])
    frame["selected_minimum_sufficient"] = np.isclose(
        frame["kappa"], selected_kappa
    )
    return selected_kappa, frame


def build_external_audit(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected_kappa: float,
) -> tuple[pd.DataFrame, dict]:
    comparisons = pd.concat(
        [
            scenario_comparison(metrics, "internal_ood", selected_kappa),
            scenario_comparison(
                metrics, "hts384_external_exploratory", selected_kappa
            ),
        ],
        ignore_index=True,
    )
    external = comparisons[
        comparisons["domain"] == "hts384_external_exploratory"
    ].set_index("scenario")
    external_diag = diagnostics[
        (diagnostics["domain"] == "hts384_external_exploratory")
        & np.isclose(diagnostics["kappa"], selected_kappa)
    ]
    conditions = {
        "reference_all_ace_not_higher_than_current": bool(
            external.loc["reference_all", "ace_difference"] <= 0.0
        ),
        "boundary_excluded_ace_noninferior_margin_0_002": bool(
            external.loc["exclude_exact_4_or_8", "ace_difference"]
            <= NONINFERIORITY_MARGIN
        ),
        "hypothetical_one_sided_ace_not_higher_than_current": bool(
            external.loc["one_sided_compatible", "ace_difference"] <= 0.0
        ),
        "all_external_weight_units_feasible": bool(
            external_diag["feasible"].all()
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
        "retains_nonzero_high_dimensional_component": bool(selected_kappa > 0.0),
    }
    return comparisons, {
        "status": "post_hoc_exploratory_reselection",
        "selected_kappa_internal_only": selected_kappa,
        "selection_rule": (
            "smallest kappa with internal three-scenario ACE noninferiority, "
            "lower robust objective, and all feasibility constraints"
        ),
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "external_technical_conditions": conditions,
        "external_technical_gate_pass": bool(all(conditions.values())),
        "manuscript_promotion_status": (
            "requires a new untouched confirmation set or nested re-validation"
        ),
        "why_not_confirmatory": (
            "The full HTS384 kappa grid was visible before this second-stage rule was formalized."
        ),
        "boundary_guardrail": (
            "Exact 4/8 are boundary candidates; one-sided compatibility is not ordinary coverage."
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(
    output: Path,
    decision: dict,
    selection: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    selected = float(decision["selected_kappa_internal_only"])
    selection_table = selection[
        [
            "kappa",
            "reference_ace_difference",
            "boundary_excluded_ace_difference",
            "hypothetical_one_sided_ace_difference",
            "robust_max_family_scenario_ace",
            "minimum_ess_fraction",
            "eligible",
            "selected_minimum_sufficient",
        ]
    ]
    comparison_table = comparisons[
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
        "# 最小充分高维校正：探索性内部报告",
        "",
        "日期：2026-08-16",
        "",
        f"- 内部规则选择 κ=**{selected:.3f}**。",
        f"- 外部技术门槛：**{'通过' if decision['external_technical_gate_pass'] else '未通过'}**。",
        "- 该选择规则形成前，HTS384 κ 网格已经可见，因此不能称为独立确认。",
        "",
        "## 内部最小充分选择",
        "",
        dataframe_to_markdown(selection_table),
        "",
        "## 选定 κ 的内部与外部比较",
        "",
        dataframe_to_markdown(comparison_table),
        "",
        "## HTS384 技术条件",
        "",
    ]
    for name, passed in decision["external_technical_conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        (
            "",
            "## 结论边界",
            "",
            "该规则可以作为下一轮方法候选，但当前 HTS384 只能提供探索性支持。若要写成主要创新并声称外部验证，需要新的未触碰确认集或嵌套重验证。精确 4/8 不得表述为已确认删失。",
            "",
        )
    )
    (output / "INTERNAL_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--source", default="results/overlap_shrunk_factorized_wcp_20260816"
    )
    parser.add_argument(
        "--output", default="results/minimum_sufficient_overlap_20260816"
    )
    args = parser.parse_args()
    root = Path(args.root)
    source = root / args.source
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = source / "unit_scenario_metrics.csv"
    diagnostics_path = source / "weight_diagnostics.csv"
    metrics = pd.read_csv(metrics_path)
    diagnostics = pd.read_csv(diagnostics_path)
    selected_kappa, selection = select_minimum_sufficient_kappa(
        metrics, diagnostics
    )
    comparisons, decision = build_external_audit(
        metrics, diagnostics, selected_kappa
    )
    selection.to_csv(output / "minimum_sufficient_selection.csv", index=False)
    comparisons.to_csv(output / "selected_kappa_comparisons.csv", index=False)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_report(output, decision, selection, comparisons)
    script_path = Path(__file__).resolve()
    design_path = root / "docs" / "因子化WCP_最小充分校正选择规则_20260816.md"
    manifest = {
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "design": str(design_path),
        "design_sha256": sha256(design_path),
        "source_metrics": str(metrics_path),
        "source_metrics_sha256": sha256(metrics_path),
        "source_diagnostics": str(diagnostics_path),
        "source_diagnostics_sha256": sha256(diagnostics_path),
        "selected_kappa": selected_kappa,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

