"""Create a seed 2–5 confirmation summary for the frozen Phase 1B method."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(".")
    results = root / "results" / "phase1b"
    metrics = pd.read_csv(results / "similarity_conformal_summary.csv")
    bins = pd.read_csv(results / "similarity_coverage_by_tanimoto.csv")
    metrics = metrics.loc[(metrics["model"] == "mlp") & (metrics["seed"].between(2, 5))]
    bins = bins.loc[(bins["model"] == "mlp") & (bins["seed"].between(2, 5))]

    coverage = metrics.pivot(
        index=["family", "seed"], columns="method", values="coverage"
    )
    widths = metrics.pivot(
        index=["family", "seed"], columns="method", values="mean_width_finite"
    )
    summary_rows = []
    for family in ("random", "scaffold", "leader_cluster"):
        values = coverage.loc[family]
        width_values = widths.loc[family]
        low = bins.loc[(bins["family"] == family) & (bins["similarity_bin"] == "[0,.4)")]
        low_pivot = low.pivot(index="seed", columns="method", values="marginal_coverage")
        baseline = values["M1_marginal_unweighted"]
        proposed = values["M5_similarity_normalized"]
        summary_rows.append(
            {
                "family": family,
                "seeds": "2,3,4,5",
                "baseline_coverage_mean": baseline.mean(),
                "baseline_coverage_sd": baseline.std(),
                "m5_coverage_mean": proposed.mean(),
                "m5_coverage_sd": proposed.std(),
                "coverage_gain": (proposed - baseline).mean(),
                "absolute_error_reduction": 1
                - (proposed - 0.9).abs().mean() / (baseline - 0.9).abs().mean(),
                "seeds_with_lower_absolute_coverage_error": int(
                    ((proposed - 0.9).abs() < (baseline - 0.9).abs()).sum()
                ),
                "baseline_width_mean": width_values["M1_marginal_unweighted"].mean(),
                "m5_width_mean": width_values["M5_similarity_normalized"].mean(),
                "width_increase": (
                    width_values["M5_similarity_normalized"]
                    / width_values["M1_marginal_unweighted"]
                    - 1
                ).mean(),
                "low_similarity_baseline_coverage_mean": low_pivot[
                    "M1_marginal_unweighted"
                ].mean(),
                "low_similarity_m5_coverage_mean": low_pivot[
                    "M5_similarity_normalized"
                ].mean(),
                "low_similarity_coverage_gain": (
                    low_pivot["M5_similarity_normalized"]
                    - low_pivot["M1_marginal_unweighted"]
                ).mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results / "confirmation_seed2to5_summary.csv", index=False)

    ood = coverage.loc[["scaffold", "leader_cluster"]]
    base_error = (ood["M1_marginal_unweighted"] - 0.9).abs().mean()
    m5_error = (ood["M5_similarity_normalized"] - 0.9).abs().mean()
    ood_width_increase = (
        widths.loc[["scaffold", "leader_cluster"], "M5_similarity_normalized"]
        / widths.loc[["scaffold", "leader_cluster"], "M1_marginal_unweighted"]
        - 1
    ).mean()
    low_ood_gain = summary.loc[
        summary["family"].isin(["scaffold", "leader_cluster"]), "low_similarity_coverage_gain"
    ].mean()
    random_coverage = float(
        summary.loc[summary["family"] == "random", "m5_coverage_mean"].iloc[0]
    )
    gate = {
        "confirmation_seeds": [2, 3, 4, 5],
        "criteria": {
            "ood_absolute_coverage_error_reduction_min": 0.25,
            "ood_low_similarity_coverage_gain_min": 0.015,
            "ood_mean_width_increase_max": 0.20,
            "random_coverage_range": [0.885, 0.915],
            "leader_seeds_with_lower_absolute_error_min": 3,
        },
        "observed": {
            "ood_absolute_coverage_error_reduction": 1 - m5_error / base_error,
            "ood_low_similarity_coverage_gain": float(low_ood_gain),
            "ood_mean_width_increase": float(ood_width_increase),
            "random_coverage": random_coverage,
            "leader_seeds_with_lower_absolute_error": int(
                summary.loc[
                    summary["family"] == "leader_cluster",
                    "seeds_with_lower_absolute_coverage_error",
                ].iloc[0]
            ),
        },
    }
    observed = gate["observed"]
    gate["passed"] = bool(
        observed["ood_absolute_coverage_error_reduction"] >= 0.25
        and observed["ood_low_similarity_coverage_gain"] >= 0.015
        and observed["ood_mean_width_increase"] <= 0.20
        and 0.885 <= observed["random_coverage"] <= 0.915
        and observed["leader_seeds_with_lower_absolute_error"] >= 3
    )
    (results / "confirmation_seed2to5_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()

