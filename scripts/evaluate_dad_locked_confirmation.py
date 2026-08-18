"""Evaluate the development-selected dAD configuration on confirmation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.analyze_peer_review_revisions import ranked_top_k_tanimoto_indices
    from scripts.analyze_uacqrp_exploratory import extended_interval_metrics
    from scripts.evaluate_phase1a_conformal import EvaluationData
    from scripts.evaluate_phase1b_similarity import nearest_fit_similarity
    from scripts.evaluate_phase5_extensions import ALPHAS, dad_intervals
    from scripts.finalize_jbhi_results import summarize_across_seeds
except ModuleNotFoundError:
    from analyze_peer_review_revisions import ranked_top_k_tanimoto_indices  # type: ignore[no-redef]
    from analyze_uacqrp_exploratory import extended_interval_metrics  # type: ignore[no-redef]
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]
    from evaluate_phase1b_similarity import nearest_fit_similarity  # type: ignore[no-redef]
    from evaluate_phase5_extensions import ALPHAS, dad_intervals  # type: ignore[no-redef]
    from finalize_jbhi_results import summarize_across_seeds  # type: ignore[no-redef]


FAMILIES = ("random", "scaffold", "leader_cluster")
CONFIRMATION_SEEDS = (2, 3, 4, 5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    selection_path = (
        root / "results" / "dad_development_selection_20260816" / "dad_locked_selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("confirmation_seeds_read") != []:
        raise ValueError("dAD selection record is not development-only")
    k = int(selection["selected_k"])

    output = root / "results" / "dad_locked_confirmation_20260816"
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "neighbours"
    cache.mkdir(exist_ok=True)
    data = EvaluationData(root)
    rows: list[dict] = []

    for family in FAMILIES:
        for seed in CONFIRMATION_SEEDS:
            split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
            result_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
            stored = np.load(result_dir / "mlp_predictions.npz")
            fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
            cal_ids = stored["nsc_source_cal"].astype("int64")
            test_ids = stored["nsc_test"].astype("int64")
            x_cal = data.x[data.rows(cal_ids)]
            x_test = data.x[data.rows(test_ids)]
            similarity = nearest_fit_similarity(data.x[data.rows(fit_ids)], x_test)
            truth_cal = data.truth(cal_ids)
            truth_test = data.truth(test_ids)
            pred_cal = stored["pred_source_cal"].astype(float)
            pred_test = stored["pred_test"].astype(float)

            neighbour_path = cache / f"{family}_seed_{seed}_k{k}.npz"
            if neighbour_path.exists():
                neighbours = np.load(neighbour_path)["indices"].astype(np.int32)
            else:
                neighbours = ranked_top_k_tanimoto_indices(x_cal, x_test, k)
                np.savez_compressed(neighbour_path, indices=neighbours)

            for alpha in ALPHAS:
                lower, upper = dad_intervals(
                    truth_cal, pred_cal, pred_test, neighbours, alpha
                )
                for scope, mask in (
                    ("all", np.ones(len(test_ids), dtype=bool)),
                    ("similarity_lt_0.4", similarity < 0.4),
                ):
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "k": k,
                            "alpha": alpha,
                            "scope": scope,
                            **extended_interval_metrics(
                                truth_test[mask], lower[mask], upper[mask], alpha
                            ),
                        }
                    )
            print(f"locked dAD confirmation complete: {family} seed {seed}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(output / "dad_locked_seed_metrics.csv", index=False)
    summary = summarize_across_seeds(
        detail,
        ["family", "alpha", "scope", "k"],
        ["coverage", "coverage_error_abs", "mean_width", "mean_interval_score"],
    )
    summary.to_csv(output / "dad_locked_summary_ci.csv", index=False)
    aggregate = (
        detail[detail["family"].isin(("scaffold", "leader_cluster"))]
        .groupby("scope", as_index=False)
        .agg(
            absolute_calibration_error=("coverage_error_abs", "mean"),
            mean_width=("mean_width", "mean"),
            mean_interval_score=("mean_interval_score", "mean"),
            finite_interval_fraction=("finite_interval_fraction", "mean"),
        )
    )
    aggregate.to_csv(output / "dad_locked_ood_aggregate.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "status": "locked_confirmation_evaluation",
                "selection_record": str(selection_path.relative_to(root)),
                "selected_k": k,
                "confirmation_seeds": list(CONFIRMATION_SEEDS),
                "families": list(FAMILIES),
                "alpha_levels": list(ALPHAS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

