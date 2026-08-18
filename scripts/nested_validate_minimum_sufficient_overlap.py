"""Grouped-by-seed nested audit of the minimum-sufficient kappa rule."""

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
    )
    from scripts.select_minimum_sufficient_overlap import (
        NONINFERIORITY_MARGIN,
        select_minimum_sufficient_kappa,
    )
except ModuleNotFoundError:
    from evaluate_factorized_shift_wcp import dataframe_to_markdown  # type: ignore[no-redef]
    from evaluate_overlap_shrunk_factorized_wcp import (  # type: ignore[no-redef]
        CURRENT_METHOD,
        SCENARIOS,
        SHRUNK_METHOD,
    )
    from select_minimum_sufficient_overlap import (  # type: ignore[no-redef]
        NONINFERIORITY_MARGIN,
        select_minimum_sufficient_kappa,
    )


def heldout_unit_rows(
    metrics: pd.DataFrame,
    heldout_seed: int,
    selected_kappa: float,
) -> list[dict]:
    internal = metrics[metrics["domain"] == "internal_ood"]
    rows = []
    for family in ("scaffold", "leader_cluster"):
        for scenario in SCENARIOS:
            candidate = internal[
                (internal["family"] == family)
                & (internal["seed"] == heldout_seed)
                & (internal["scenario"] == scenario)
                & (internal["method"] == SHRUNK_METHOD)
                & np.isclose(internal["kappa"], selected_kappa)
            ]
            current = internal[
                (internal["family"] == family)
                & (internal["seed"] == heldout_seed)
                & (internal["scenario"] == scenario)
                & (internal["method"] == CURRENT_METHOD)
            ]
            if len(candidate) != 1 or len(current) != 1:
                raise ValueError(
                    f"Expected one row for heldout seed={heldout_seed}, "
                    f"family={family}, scenario={scenario}"
                )
            candidate_row = candidate.iloc[0]
            current_row = current.iloc[0]
            rows.append(
                {
                    "heldout_seed": heldout_seed,
                    "family": family,
                    "scenario": scenario,
                    "selected_kappa": selected_kappa,
                    "candidate_coverage": float(candidate_row["coverage"]),
                    "current_coverage": float(current_row["coverage"]),
                    "candidate_ace": float(candidate_row["coverage_error_abs"]),
                    "current_ace": float(current_row["coverage_error_abs"]),
                    "ace_difference": float(
                        candidate_row["coverage_error_abs"]
                        - current_row["coverage_error_abs"]
                    ),
                    "candidate_width": float(candidate_row["mean_width"]),
                    "current_width": float(current_row["mean_width"]),
                    "width_ratio": float(
                        candidate_row["mean_width"] / current_row["mean_width"]
                    ),
                    "finite_interval_fraction": float(
                        candidate_row["finite_interval_fraction"]
                    ),
                }
            )
    return rows


def run_nested_audit(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    internal_metrics = metrics[metrics["domain"] == "internal_ood"].copy()
    internal_diagnostics = diagnostics[
        diagnostics["domain"] == "internal_ood"
    ].copy()
    seeds = sorted(int(value) for value in internal_metrics["seed"].unique())
    fold_rows = []
    selection_rows = []
    heldout_rows = []
    for heldout_seed in seeds:
        development_metrics = internal_metrics[
            internal_metrics["seed"] != heldout_seed
        ]
        development_diagnostics = internal_diagnostics[
            internal_diagnostics["seed"] != heldout_seed
        ]
        try:
            selected_kappa, selection = select_minimum_sufficient_kappa(
                development_metrics, development_diagnostics
            )
        except ValueError as error:
            fold_rows.append(
                {
                    "heldout_seed": heldout_seed,
                    "selection_success": False,
                    "selected_kappa": np.nan,
                    "reason": str(error),
                }
            )
            continue
        fold_rows.append(
            {
                "heldout_seed": heldout_seed,
                "selection_success": True,
                "selected_kappa": selected_kappa,
                "reason": "",
            }
        )
        selection = selection.copy()
        selection.insert(0, "heldout_seed", heldout_seed)
        selection_rows.extend(selection.to_dict(orient="records"))
        heldout_rows.extend(
            heldout_unit_rows(internal_metrics, heldout_seed, selected_kappa)
        )

    folds = pd.DataFrame(fold_rows)
    selections = pd.DataFrame(selection_rows)
    heldout = pd.DataFrame(heldout_rows)
    summary_rows = []
    if not heldout.empty:
        for scenario, part in heldout.groupby("scenario", sort=True):
            summary_rows.append(
                {
                    "scenario": scenario,
                    "units": len(part),
                    "candidate_coverage": float(part["candidate_coverage"].mean()),
                    "current_coverage": float(part["current_coverage"].mean()),
                    "candidate_ace": float(part["candidate_ace"].mean()),
                    "current_ace": float(part["current_ace"].mean()),
                    "ace_difference": float(part["ace_difference"].mean()),
                    "mean_width_ratio": float(part["width_ratio"].mean()),
                    "maximum_width_ratio": float(part["width_ratio"].max()),
                    "minimum_finite_interval_fraction": float(
                        part["finite_interval_fraction"].min()
                    ),
                    "ace_better_units": int((part["ace_difference"] < 0).sum()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    successful = folds[folds["selection_success"]]
    if successful.empty:
        mode_count = 0
        all_nonzero = False
    else:
        mode_count = int(successful["selected_kappa"].value_counts().max())
        all_nonzero = bool((successful["selected_kappa"] > 0).all())
    heldout_diag_rows = []
    for row in successful.itertuples(index=False):
        selected = internal_diagnostics[
            (internal_diagnostics["seed"] == int(row.heldout_seed))
            & np.isclose(
                internal_diagnostics["kappa"], float(row.selected_kappa)
            )
        ].copy()
        selected.insert(0, "heldout_seed", int(row.heldout_seed))
        heldout_diag_rows.extend(selected.to_dict(orient="records"))
    heldout_diagnostics = pd.DataFrame(heldout_diag_rows)
    scenario_noninferior = bool(
        not summary.empty
        and (summary["ace_difference"] <= NONINFERIORITY_MARGIN).all()
    )
    conditions = {
        "selection_success_5_of_5": bool(len(successful) == len(seeds) == 5),
        "same_kappa_selected_at_least_4_of_5": bool(mode_count >= 4),
        "all_folds_retain_nonzero_high_dimensional_component": all_nonzero,
        "all_scenarios_heldout_ace_noninferior_margin_0_002": scenario_noninferior,
        "maximum_heldout_width_ratio_le_1_10": bool(
            not summary.empty and summary["maximum_width_ratio"].max() <= 1.10
        ),
        "minimum_finite_interval_fraction_ge_0_999": bool(
            not summary.empty
            and summary["minimum_finite_interval_fraction"].min() >= 0.999
        ),
        "all_heldout_weight_units_feasible": bool(
            not heldout_diagnostics.empty
            and heldout_diagnostics["feasible"].all()
        ),
    }
    decision = {
        "status": "nested_internal_stability_audit",
        "folds": len(seeds),
        "selected_kappa_counts": {
            str(float(key)): int(value)
            for key, value in successful["selected_kappa"].value_counts().sort_index().items()
        },
        "conditions": conditions,
        "nested_stability_pass": bool(all(conditions.values())),
        "interpretation": (
            "Internal grouped holdout only; it does not restore untouched external confirmation."
        ),
    }
    return folds, selections, heldout, summary, heldout_diagnostics, decision


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(
    output: Path,
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    decision: dict,
) -> None:
    lines = [
        "# 最小充分高维校正：按 seed 嵌套审计",
        "",
        "日期：2026-08-16",
        "",
        f"- 嵌套稳定性门槛：**{'通过' if decision['nested_stability_pass'] else '未通过'}**。",
        f"- κ 选择分布：{decision['selected_kappa_counts']}。",
        "- 这是内部按 seed 分组留出审计，不是新的外部确认。",
        "",
        "## 各折选择",
        "",
        dataframe_to_markdown(folds),
        "",
        "## 汇总留出表现",
        "",
        dataframe_to_markdown(summary),
        "",
        "## 判定条件",
        "",
    ]
    for name, passed in decision["conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        (
            "",
            "精确 4/8 仅为边界候选；假设性单侧相容结果不是普通 conformal coverage。",
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
        "--output", default="results/nested_minimum_sufficient_overlap_20260816"
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
    folds, selections, heldout, summary, heldout_diagnostics, decision = (
        run_nested_audit(metrics, diagnostics)
    )
    folds.to_csv(output / "fold_selections.csv", index=False)
    selections.to_csv(output / "fold_kappa_diagnostics.csv", index=False)
    heldout.to_csv(output / "heldout_unit_metrics.csv", index=False)
    summary.to_csv(output / "heldout_summary.csv", index=False)
    heldout_diagnostics.to_csv(
        output / "heldout_weight_diagnostics.csv", index=False
    )
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_report(output, folds, summary, decision)
    script_path = Path(__file__).resolve()
    design_path = root / "docs" / "因子化WCP_按seed嵌套验证设计_20260816.md"
    manifest = {
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "design": str(design_path),
        "design_sha256": sha256(design_path),
        "source_metrics_sha256": sha256(metrics_path),
        "source_diagnostics_sha256": sha256(diagnostics_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

