"""Select the dAD neighbourhood size using development seed 1 only.

The selection endpoint is mean absolute calibration error across the two
out-of-distribution split families and four nominal coverage levels.  Width is
used only as a deterministic tie-breaker.  Confirmation seeds are never read.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.analyze_peer_review_revisions import ranked_top_k_tanimoto_indices
    from scripts.analyze_uacqrp_exploratory import extended_interval_metrics
    from scripts.evaluate_phase1a_conformal import EvaluationData
    from scripts.evaluate_phase5_extensions import ALPHAS, dad_intervals
except ModuleNotFoundError:
    from analyze_peer_review_revisions import ranked_top_k_tanimoto_indices  # type: ignore[no-redef]
    from analyze_uacqrp_exploratory import extended_interval_metrics  # type: ignore[no-redef]
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]
    from evaluate_phase5_extensions import ALPHAS, dad_intervals  # type: ignore[no-redef]


FAMILIES = ("scaffold", "leader_cluster")
DEVELOPMENT_SEED = 1
K_VALUES = (50, 100, 250, 500)


def evaluate_development_grid(root: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = EvaluationData(root)
    rows: list[dict] = []
    cache = output / "neighbours"
    cache.mkdir(parents=True, exist_ok=True)

    for family in FAMILIES:
        result_dir = root / "results" / "phase1a" / f"{family}_seed_{DEVELOPMENT_SEED}"
        stored = np.load(result_dir / "mlp_predictions.npz")
        cal_ids = stored["nsc_source_cal"].astype("int64")
        test_ids = stored["nsc_test"].astype("int64")
        truth_cal = data.truth(cal_ids)
        truth_test = data.truth(test_ids)
        pred_cal = stored["pred_source_cal"].astype(float)
        pred_test = stored["pred_test"].astype(float)

        neighbour_path = cache / f"{family}_seed_{DEVELOPMENT_SEED}_k{max(K_VALUES)}.npz"
        if neighbour_path.exists():
            neighbours_max = np.load(neighbour_path)["indices"].astype(np.int32)
        else:
            neighbours_max = ranked_top_k_tanimoto_indices(
                data.x[data.rows(cal_ids)], data.x[data.rows(test_ids)], max(K_VALUES)
            )
            np.savez_compressed(neighbour_path, indices=neighbours_max)

        for k in K_VALUES:
            neighbours = neighbours_max[:, :k]
            for alpha in ALPHAS:
                lower, upper = dad_intervals(
                    truth_cal, pred_cal, pred_test, neighbours, alpha
                )
                rows.append(
                    {
                        "family": family,
                        "seed": DEVELOPMENT_SEED,
                        "k": k,
                        "alpha": alpha,
                        **extended_interval_metrics(truth_test, lower, upper, alpha),
                    }
                )
        print(f"development dAD grid complete: {family}", flush=True)

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("k", as_index=False)
        .agg(
            absolute_calibration_error=("coverage_error_abs", "mean"),
            mean_width=("mean_width", "mean"),
            mean_interval_score=("mean_interval_score", "mean"),
            finite_interval_fraction=("finite_interval_fraction", "mean"),
        )
        .sort_values(["absolute_calibration_error", "mean_width", "k"], kind="mergesort")
        .reset_index(drop=True)
    )
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="results/dad_development_selection_20260816",
        help="Output directory, relative to --root unless absolute.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    detail, summary = evaluate_development_grid(root, output)
    detail.to_csv(output / "dad_development_seed_metrics.csv", index=False)
    summary.to_csv(output / "dad_development_selection_summary.csv", index=False)
    selected_k = int(summary.iloc[0]["k"])
    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_only_model_selection",
        "development_seed": DEVELOPMENT_SEED,
        "confirmation_seeds_read": [],
        "split_families": list(FAMILIES),
        "alpha_levels": list(ALPHAS),
        "candidate_k": list(K_VALUES),
        "primary_selection_endpoint": "mean absolute calibration error",
        "tie_breakers": ["mean width", "smaller k"],
        "selected_k": selected_k,
    }
    (output / "dad_locked_selection.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(lock, indent=2), flush=True)


if __name__ == "__main__":
    main()

