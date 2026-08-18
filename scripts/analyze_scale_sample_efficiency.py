"""Evaluate validation-label sample efficiency of shared versus per-cell scales."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.analyze_phase4_statistics import mean_t_interval
    from scripts.analyze_uacqrp_exploratory import extended_interval_metrics
    from scripts.evaluate_phase1a_conformal import EvaluationData
    from scripts.evaluate_phase3_local_and_screening import sim
    from scripts.evaluate_phase5_extensions import (
        cell_metrics,
        fit_per_cell_monotone_scale,
        fit_similarity_scale,
        predict_per_cell_monotone_scale,
        predict_similarity_scale,
        scaled_intervals,
    )
except ModuleNotFoundError:
    from analyze_phase4_statistics import mean_t_interval  # type: ignore[no-redef]
    from analyze_uacqrp_exploratory import extended_interval_metrics  # type: ignore[no-redef]
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]
    from evaluate_phase3_local_and_screening import sim  # type: ignore[no-redef]
    from evaluate_phase5_extensions import (  # type: ignore[no-redef]
        cell_metrics,
        fit_per_cell_monotone_scale,
        fit_similarity_scale,
        predict_per_cell_monotone_scale,
        predict_similarity_scale,
        scaled_intervals,
    )


FAMILIES = ("scaffold", "leader_cluster")
METHODS = ("shared_monotone", "per_cell_monotone")
PRIMARY_ALPHA = 0.10


def stratified_subsample(
    similarity: np.ndarray, n: int, rng: np.random.Generator, strata: int = 5
) -> np.ndarray:
    """Sample validation rows across similarity quantiles without using outcomes."""
    similarity = np.asarray(similarity, dtype=float)
    if n >= len(similarity):
        return np.arange(len(similarity), dtype=int)
    edges = np.unique(np.quantile(similarity, np.linspace(0, 1, strata + 1)))
    groups = np.digitize(similarity, edges[1:-1], right=True)
    selected: list[int] = []
    for group in np.unique(groups):
        candidates = np.flatnonzero(groups == group)
        target = int(np.floor(n * len(candidates) / len(similarity)))
        if target:
            selected.extend(rng.choice(candidates, size=target, replace=False).tolist())
    remaining = n - len(selected)
    if remaining:
        available = np.setdiff1d(np.arange(len(similarity)), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(available, size=remaining, replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def metrics_for_intervals(
    truth: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> dict:
    overall = extended_interval_metrics(truth, lower, upper, alpha)
    per_cell = pd.DataFrame(cell_metrics(truth, lower, upper, alpha))
    return {
        **overall,
        "macro_cell_ace": float(per_cell["coverage_error_abs"].mean()),
        "q90_cell_ace": float(per_cell["coverage_error_abs"].quantile(0.90)),
        "cell_coverage_gap_sd": float((per_cell["coverage"] - (1 - alpha)).std(ddof=1)),
    }


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
    parser.add_argument("--sizes", default="50,75,100,250,500,1000,2524")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    root = Path(args.root)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    sizes = tuple(int(value) for value in args.sizes.split(",") if value)
    output = root / "results" / "phase8_scale_efficiency"
    output.mkdir(parents=True, exist_ok=True)
    data = EvaluationData(root)
    rows: list[dict] = []
    feasibility_rows: list[dict] = []

    for family_index, family in enumerate(FAMILIES):
        for seed in seeds:
            split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
            phase1_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
            phase3_dir = root / "results" / "phase3" / f"{family}_seed_{seed}"
            stored = np.load(phase1_dir / "mlp_predictions.npz")
            names = ("source_cal", "valid", "test")
            ids = {name: stored[f"nsc_{name}"].astype("int64") for name in names}
            prediction = {name: stored[f"pred_{name}"].astype(float) for name in names}
            truth = {name: data.truth(ids[name]) for name in names}
            fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
            similarity = {
                name: sim(phase3_dir, name, data, fit_ids, ids[name]) for name in names
            }
            low = similarity["test"] < 0.4

            for n_requested in sizes:
                n = min(n_requested, len(ids["valid"]))
                repeat_count = 1 if n == len(ids["valid"]) else args.repeats
                for repeat in range(repeat_count):
                    rng = np.random.default_rng(
                        20260722 + family_index * 100_000 + seed * 1_000 + n * 10 + repeat
                    )
                    subset = stratified_subsample(similarity["valid"], n, rng)
                    shared_model, _, shared_diag = fit_similarity_scale(
                        similarity["valid"][subset],
                        truth["valid"][subset],
                        prediction["valid"][subset],
                    )
                    scales = {
                        "shared_monotone": (
                            predict_similarity_scale(shared_model, similarity["source_cal"]),
                            predict_similarity_scale(shared_model, similarity["test"]),
                            shared_diag,
                        ),
                    }
                    label_counts = np.isfinite(truth["valid"][subset]).sum(axis=0)
                    feasibility = {
                        "family": family,
                        "seed": seed,
                        "validation_n": n,
                        "repeat": repeat,
                        "min_cell_labels": int(label_counts.min()),
                        "cells_below_20_labels": int((label_counts < 20).sum()),
                    }
                    try:
                        per_cell_model, per_cell_diag = fit_per_cell_monotone_scale(
                            similarity["valid"][subset],
                            truth["valid"][subset],
                            prediction["valid"][subset],
                        )
                        scales["per_cell_monotone"] = (
                            predict_per_cell_monotone_scale(
                                per_cell_model, similarity["source_cal"]
                            ),
                            predict_per_cell_monotone_scale(per_cell_model, similarity["test"]),
                            per_cell_diag,
                        )
                        feasibility["per_cell_status"] = "estimated"
                        feasibility["per_cell_error"] = ""
                    except ValueError as exc:
                        feasibility["per_cell_status"] = "not_estimable"
                        feasibility["per_cell_error"] = str(exc)
                    feasibility_rows.append(feasibility)
                    for method, (cal_scale, test_scale, diagnostics) in scales.items():
                        lower, upper = scaled_intervals(
                            truth["source_cal"],
                            prediction["source_cal"],
                            prediction["test"],
                            cal_scale,
                            test_scale,
                            PRIMARY_ALPHA,
                        )
                        for scope, mask in (
                            ("all", np.ones(len(ids["test"]), dtype=bool)),
                            ("similarity_lt_0.4", low),
                        ):
                            rows.append(
                                {
                                    "family": family,
                                    "seed": seed,
                                    "validation_n": n,
                                    "validation_fraction": n / len(ids["valid"]),
                                    "repeat": repeat,
                                    "method": method,
                                    "scope": scope,
                                    "scale_min": diagnostics.get(
                                        "scale_min", diagnostics.get("scale_min_across_cells")
                                    ),
                                    "scale_max": diagnostics.get(
                                        "scale_max", diagnostics.get("scale_max_across_cells")
                                    ),
                                    **metrics_for_intervals(
                                        truth["test"][mask], lower[mask], upper[mask], PRIMARY_ALPHA
                                    ),
                                }
                            )
                print(f"sample-efficiency family={family} seed={seed} n={n}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "scale_efficiency_repeat_metrics.csv", index=False)
    feasibility_frame = pd.DataFrame(feasibility_rows)
    feasibility_frame.to_csv(output / "scale_efficiency_feasibility.csv", index=False)
    unit = (
        frame.groupby(
            ["family", "seed", "validation_n", "validation_fraction", "method", "scope"],
            as_index=False,
        )
        .agg(
            coverage=("coverage", "mean"),
            coverage_error_abs=("coverage_error_abs", "mean"),
            mean_width=("mean_width", "mean"),
            macro_cell_ace=("macro_cell_ace", "mean"),
            q90_cell_ace=("q90_cell_ace", "mean"),
            cell_coverage_gap_sd=("cell_coverage_gap_sd", "mean"),
            repeat_ace_sd=("coverage_error_abs", "std"),
            repeat_macro_cell_ace_sd=("macro_cell_ace", "std"),
        )
    )
    unit.to_csv(output / "scale_efficiency_family_seed_units.csv", index=False)

    summary_rows = []
    metrics = (
        "coverage_error_abs",
        "mean_width",
        "macro_cell_ace",
        "q90_cell_ace",
        "cell_coverage_gap_sd",
        "repeat_ace_sd",
    )
    for keys, part in unit.groupby(["validation_n", "validation_fraction", "method", "scope"]):
        row = dict(zip(("validation_n", "validation_fraction", "method", "scope"), keys))
        row["n_family_seed_units"] = int(len(part))
        for metric in metrics:
            mean, sd, low, high = mean_t_interval(part[metric].to_numpy(float))
            row.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_sd": sd,
                    f"{metric}_t95_low": low,
                    f"{metric}_t95_high": high,
                }
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "scale_efficiency_summary_ci.csv", index=False)

    paired = unit.pivot(
        index=["family", "seed", "validation_n", "scope"],
        columns="method",
        values=["coverage_error_abs", "macro_cell_ace", "q90_cell_ace", "mean_width"],
    )
    paired.columns = [f"{metric}_{method}" for metric, method in paired.columns]
    paired = paired.reset_index()
    for metric in ("coverage_error_abs", "macro_cell_ace", "q90_cell_ace", "mean_width"):
        paired[f"shared_minus_per_cell_{metric}"] = (
            paired[f"{metric}_shared_monotone"] - paired[f"{metric}_per_cell_monotone"]
        )
    paired.to_csv(output / "scale_efficiency_paired_units.csv", index=False)

    compact = []
    required = [
        "coverage_error_abs_shared_monotone",
        "coverage_error_abs_per_cell_monotone",
        "macro_cell_ace_shared_monotone",
        "macro_cell_ace_per_cell_monotone",
        "q90_cell_ace_shared_monotone",
        "q90_cell_ace_per_cell_monotone",
        "mean_width_shared_monotone",
        "mean_width_per_cell_monotone",
    ]
    paired_complete = paired.dropna(subset=required)
    for (n, scope), part in paired_complete.groupby(["validation_n", "scope"]):
        item = {"validation_n": int(n), "scope": scope}
        for metric in ("coverage_error_abs", "macro_cell_ace", "q90_cell_ace", "mean_width"):
            item[f"mean_shared_minus_per_cell_{metric}"] = float(
                part[f"shared_minus_per_cell_{metric}"].mean()
            )
        compact.append(item)
    result = {
        "status": "completed_frozen_prediction_sample_efficiency_analysis",
        "alpha": PRIMARY_ALPHA,
        "families": list(FAMILIES),
        "seeds": list(seeds),
        "validation_sizes": list(sizes),
        "subsampling": (
            "Outcome-blind stratified random sampling over validation similarity quintiles; "
            "same subset for shared and per-cell methods."
        ),
        "repeats": args.repeats,
        "per_cell_not_estimable_records": int(
            (feasibility_frame["per_cell_status"] == "not_estimable").sum()
        ),
        "paired_summary": compact,
    }
    (output / "scale_efficiency_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    source_files = (
        root / "scripts" / "analyze_scale_sample_efficiency.py",
        root / "scripts" / "evaluate_phase5_extensions.py",
        root / "data" / "processed" / "cellminer" / "response_matrix_loggi50.csv.gz",
        root / "results" / "final_jbhi" / "final_results_manifest.json",
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            f"python scripts/analyze_scale_sample_efficiency.py --root {root.as_posix()} "
            f"--seeds {','.join(map(str, seeds))} --sizes {','.join(map(str, sizes))} "
            f"--repeats {args.repeats}"
        ),
        "files_sha256": {
            str(path.relative_to(root)): sha256(path) for path in source_files if path.exists()
        },
    }
    (output / "scale_efficiency_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

