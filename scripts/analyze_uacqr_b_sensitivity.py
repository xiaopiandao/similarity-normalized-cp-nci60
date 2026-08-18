"""Evaluate UACQR-P Monte Carlo ensemble-size sensitivity on OOD splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.analyze_uacqrp_exploratory import extended_interval_metrics
    from scripts.evaluate_phase1a_conformal import EvaluationData
except ModuleNotFoundError:
    from analyze_uacqrp_exploratory import extended_interval_metrics  # type: ignore[no-redef]
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]


ALPHAS = (0.05, 0.10, 0.15, 0.20)
FAMILIES = ("scaffold", "leader_cluster")
SEEDS = (2, 3, 4, 5)


def main() -> None:
    root = Path(".")
    output = root / "results" / "revision_peer_review" / "uacqr_b_sensitivity"
    sources = {
        50: output / "B50",
        100: root / "results" / "phase6_uacqrp_exploratory",
        200: output / "B200",
    }
    data = EvaluationData(root)
    rows: list[dict] = []
    for b, source in sources.items():
        for family in FAMILIES:
            for seed in SEEDS:
                path = source / f"{family}_seed_{seed}" / "uacqrp_intervals.npz"
                saved = np.load(path)
                ids = saved["nsc_test"].astype("int64")
                truth = data.truth(ids)
                stored_alphas = saved["alphas"].astype(float)
                for alpha in ALPHAS:
                    index = int(np.flatnonzero(np.isclose(stored_alphas, alpha))[0])
                    rows.append(
                        {
                            "ensemble_size": b,
                            "family": family,
                            "seed": seed,
                            "alpha": alpha,
                            **extended_interval_metrics(
                                truth,
                                saved["lower"][index].astype(float),
                                saved["upper"][index].astype(float),
                                alpha,
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "uacqr_b_sensitivity_seed_metrics.csv", index=False)
    frame.groupby("ensemble_size", as_index=False).agg(
        absolute_calibration_error=("coverage_error_abs", "mean"),
        mean_width=("mean_width", "mean"),
        mean_interval_score=("mean_interval_score", "mean"),
        finite_interval_fraction=("finite_interval_fraction", "mean"),
    ).to_csv(output / "uacqr_b_sensitivity_ood_aggregate.csv", index=False)
    frame.groupby(["family", "ensemble_size"], as_index=False).agg(
        absolute_calibration_error=("coverage_error_abs", "mean"),
        mean_width=("mean_width", "mean"),
        mean_interval_score=("mean_interval_score", "mean"),
        finite_interval_fraction=("finite_interval_fraction", "mean"),
    ).to_csv(output / "uacqr_b_sensitivity_by_family.csv", index=False)


if __name__ == "__main__":
    main()

