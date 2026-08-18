"""Summarize Phase 5 confirmation seeds without changing fitted models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


OOD_FAMILIES = ("scaffold", "leader_cluster")
PROPOSED = "Proposed_shared_monotone"
SCALE_SHARED = "shared_monotone_isotonic"
SCALE_PER_CELL = "per_cell_monotone_isotonic"


def t_interval(values: pd.Series) -> tuple[float, float, float, float]:
    x = values.to_numpy(float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan, np.nan
    mean = float(x.mean())
    if len(x) == 1:
        return mean, np.nan, np.nan, np.nan
    sd = float(x.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)))
    return mean, sd, mean - half, mean + half


def aggregate(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["n_units"] = len(part)
        for metric in metrics:
            mean, sd, low, high = t_interval(part[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_t95_low"] = low
            row[f"{metric}_t95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def low_similarity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    low = frame[frame["bin_order"].isin((0, 1))].copy()
    rows = []
    for keys, part in low.groupby(["family", "seed", "alpha", "method"]):
        weights = part["n_labels"].to_numpy(float)
        coverage = float(np.average(part["coverage"], weights=weights))
        width = float(np.average(part["mean_width"], weights=weights))
        rows.append(
            {
                "family": keys[0],
                "seed": keys[1],
                "alpha": keys[2],
                "method": keys[3],
                "coverage": coverage,
                "coverage_error_abs": abs(coverage - (1.0 - keys[2])),
                "mean_width": width,
                "n_labels": int(weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def scale_cell_units(cell: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in cell.groupby(["family", "seed", "alpha", "method"]):
        signed_gap = part["coverage"] - (1.0 - keys[2])
        rows.append(
            {
                "family": keys[0],
                "seed": keys[1],
                "alpha": keys[2],
                "method": keys[3],
                "macro_cell_ace": float(part["coverage_error_abs"].mean()),
                "q90_cell_ace": float(part["coverage_error_abs"].quantile(0.90)),
                "cell_gap_sd": float(signed_gap.std(ddof=1)),
                "macro_cell_width": float(part["mean_width"].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_scale_contrasts(cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = cell[cell["method"].isin((SCALE_SHARED, SCALE_PER_CELL))]
    wide = selected.pivot(
        index=["family", "seed", "alpha", "cell_index"],
        columns="method",
        values="coverage_error_abs",
    ).dropna()
    wide["difference_shared_minus_per_cell"] = wide[SCALE_SHARED] - wide[SCALE_PER_CELL]
    paired = wide.reset_index()
    clusters = (
        paired.groupby(["family", "seed"])["difference_shared_minus_per_cell"]
        .mean()
        .reset_index()
    )
    rows = []
    for scope, part in (("OOD_combined", clusters[clusters["family"].isin(OOD_FAMILIES)]),):
        mean, sd, low, high = t_interval(part["difference_shared_minus_per_cell"])
        raw = paired[paired["family"].isin(OOD_FAMILIES)]
        rows.append(
            {
                "scope": scope,
                "n_family_seed_clusters": len(part),
                "mean_difference_shared_minus_per_cell": mean,
                "sd_across_clusters": sd,
                "t95_low": low,
                "t95_high": high,
                "descriptive_shared_win_fraction_cell_alpha": float(
                    (raw["difference_shared_minus_per_cell"] < 0).mean()
                ),
            }
        )
    for family in OOD_FAMILIES:
        part = clusters[clusters["family"] == family]
        mean, sd, low, high = t_interval(part["difference_shared_minus_per_cell"])
        raw = paired[paired["family"] == family]
        rows.append(
            {
                "scope": family,
                "n_family_seed_clusters": len(part),
                "mean_difference_shared_minus_per_cell": mean,
                "sd_across_clusters": sd,
                "t95_low": low,
                "t95_high": high,
                "descriptive_shared_win_fraction_cell_alpha": float(
                    (raw["difference_shared_minus_per_cell"] < 0).mean()
                ),
            }
        )
    return paired, pd.DataFrame(rows)


def method_contrasts(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    ood = seed_metrics[seed_metrics["family"].isin(OOD_FAMILIES)]
    wide = ood.pivot(
        index=["family", "seed", "alpha"], columns="method", values="coverage_error_abs"
    )
    rows = []
    for comparator in [name for name in wide.columns if name != PROPOSED]:
        values = (wide[PROPOSED] - wide[comparator]).dropna().reset_index(name="difference")
        clusters = values.groupby(["family", "seed"])["difference"].mean().reset_index()
        mean, sd, low, high = t_interval(clusters["difference"])
        rows.append(
            {
                "contrast": f"{PROPOSED}_minus_{comparator}",
                "n_family_seed_clusters": len(clusters),
                "mean_calibration_error_difference": mean,
                "sd_across_clusters": sd,
                "t95_low": low,
                "t95_high": high,
                "clusters_favouring_proposed": int((clusters["difference"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    source = root / "results" / "phase5"
    output = source / "confirmation"
    output.mkdir(parents=True, exist_ok=True)

    multi = pd.read_csv(source / "multi_alpha_seed_metrics.csv")
    multi_cells = pd.read_csv(source / "multi_alpha_cell_metrics.csv")
    multi_similarity = pd.read_csv(source / "multi_alpha_similarity_metrics.csv")
    ablation = pd.read_csv(source / "scale_ablation_seed_metrics.csv")
    ablation_cells = pd.read_csv(source / "scale_ablation_cell_metrics.csv")

    expected_seeds = {2, 3, 4, 5}
    if set(multi["seed"].unique()) != expected_seeds:
        raise ValueError("Phase 5 confirmation must contain exactly seeds 2--5")
    if multi["finite_interval_fraction"].min() < 0.99:
        # Retain the data, but make the incomplete interval rate explicit in metadata.
        finite_warning = True
    else:
        finite_warning = False

    # Average alpha levels within each frozen family/seed cluster before t summaries.
    multi_clusters = (
        multi.groupby(["family", "seed", "method"], as_index=False)
        .agg(
            calibration_error=("coverage_error_abs", "mean"),
            mean_width=("mean_width", "mean"),
            finite_interval_fraction=("finite_interval_fraction", "min"),
        )
    )
    multi_summary = aggregate(
        multi_clusters,
        ["family", "method"],
        ["calibration_error", "mean_width", "finite_interval_fraction"],
    )
    multi_summary.to_csv(output / "multi_alpha_summary_by_family.csv", index=False)

    ood_clusters = multi_clusters[multi_clusters["family"].isin(OOD_FAMILIES)]
    aggregate(ood_clusters, ["method"], ["calibration_error", "mean_width"]).to_csv(
        output / "multi_alpha_ood_summary.csv", index=False
    )
    method_contrasts(multi).to_csv(output / "multi_alpha_paired_method_contrasts.csv", index=False)

    low = low_similarity_summary(multi_similarity)
    low.to_csv(output / "low_similarity_seed_alpha_metrics.csv", index=False)
    low_clusters = (
        low.groupby(["family", "seed", "method"], as_index=False)
        .agg(calibration_error=("coverage_error_abs", "mean"), mean_width=("mean_width", "mean"))
    )
    aggregate(low_clusters, ["family", "method"], ["calibration_error", "mean_width"]).to_csv(
        output / "low_similarity_summary_by_family.csv", index=False
    )

    scale_units = scale_cell_units(ablation_cells)
    scale_units.to_csv(output / "scale_cell_summary_by_seed_alpha.csv", index=False)
    scale_clusters = (
        scale_units.groupby(["family", "seed", "method"], as_index=False)
        .agg(
            macro_cell_ace=("macro_cell_ace", "mean"),
            q90_cell_ace=("q90_cell_ace", "mean"),
            cell_gap_sd=("cell_gap_sd", "mean"),
            macro_cell_width=("macro_cell_width", "mean"),
        )
    )
    aggregate(
        scale_clusters,
        ["family", "method"],
        ["macro_cell_ace", "q90_cell_ace", "cell_gap_sd", "macro_cell_width"],
    ).to_csv(output / "scale_ablation_summary_by_family.csv", index=False)
    aggregate(
        scale_clusters[scale_clusters["family"].isin(OOD_FAMILIES)],
        ["method"],
        ["macro_cell_ace", "q90_cell_ace", "cell_gap_sd", "macro_cell_width"],
    ).to_csv(output / "scale_ablation_ood_summary.csv", index=False)
    paired, paired_summary = paired_scale_contrasts(ablation_cells)
    paired.to_csv(output / "scale_shared_vs_per_cell_paired_units.csv", index=False)
    paired_summary.to_csv(output / "scale_shared_vs_per_cell_contrast.csv", index=False)

    metadata = {
        "confirmation_seeds": sorted(expected_seeds),
        "alpha_levels": sorted(multi["alpha"].unique().tolist()),
        "primary_calibration_metric": "absolute observed-minus-nominal coverage error",
        "scale_macro_metric": "mean absolute coverage error across 60 cell lines",
        "tail_metric": "90th percentile cell-line absolute coverage error",
        "pairing": "family/seed/alpha/cell; t interval computed after averaging to family-seed clusters",
        "seed_interval_caveat": "descriptive; split seeds reuse compounds and are not independent external cohorts",
        "finite_interval_warning": finite_warning,
        "minimum_finite_interval_fraction": float(multi["finite_interval_fraction"].min()),
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

