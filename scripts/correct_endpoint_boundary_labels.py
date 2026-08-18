"""Apply the author-confirmed scaffold method-label correction to boundary results.

The frozen source file is never overwritten.  The corrected file retains the
source label and a row-level correction flag for auditability.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SWAP = {
    "UACQR_P": "Proposed_shared_monotone",
    "Proposed_shared_monotone": "UACQR_P",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    source = root / "results" / "phase7_boundary_sensitivity" / "boundary_sensitivity_seed_metrics.csv"
    output = root / "results" / "phase7_boundary_sensitivity_corrected_20260817"
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source)
    frame.insert(frame.columns.get_loc("method") + 1, "source_method_label", frame["method"])
    selected = (frame["family"] == "scaffold") & frame["method"].isin(SWAP)
    frame.loc[selected, "method"] = frame.loc[selected, "method"].map(SWAP)
    frame.insert(
        frame.columns.get_loc("source_method_label") + 1,
        "label_correction_applied",
        selected,
    )
    frame.to_csv(output / "boundary_sensitivity_seed_metrics_corrected.csv", index=False)

    ood = frame[frame["family"].isin(("scaffold", "leader_cluster"))]
    aggregate = (
        ood.groupby(["scenario", "scope", "method"], as_index=False)
        .agg(
            absolute_calibration_error=("coverage_error_abs", "mean"),
            coverage=("coverage", "mean"),
            mean_width=("mean_width", "mean"),
            minimum_finite_fraction=("finite_interval_fraction", "min"),
            n_labels=("n_labels", "sum"),
        )
    )
    aggregate["calibration_rank"] = aggregate.groupby(["scenario", "scope"])[
        "absolute_calibration_error"
    ].rank(method="min")
    aggregate.to_csv(output / "boundary_sensitivity_ood_aggregate_corrected.csv", index=False)

    paired_rows: list[dict] = []
    paired_source = ood[
        (ood["scope"] == "similarity_lt_0.4")
        & ood["method"].isin(("Proposed_shared_monotone", "UACQR_P"))
    ]
    for (scenario, family, seed), part in paired_source.groupby(["scenario", "family", "seed"]):
        pivot = part.pivot(index="alpha", columns="method", values="coverage_error_abs")
        paired_rows.append(
            {
                "scenario": scenario,
                "family": family,
                "seed": seed,
                "mean_proposed_minus_uacqrp_ace": float(
                    (pivot["Proposed_shared_monotone"] - pivot["UACQR_P"]).mean()
                ),
            }
        )
    pd.DataFrame(paired_rows).to_csv(
        output / "boundary_paired_low_similarity_units_corrected.csv", index=False
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source.relative_to(root)),
        "source_overwritten": False,
        "correction_scope": "all scaffold rows labeled UACQR_P or Proposed_shared_monotone",
        "operation": "swap the two method labels",
        "corrected_rows": int(selected.sum()),
        "reason": "author-confirmed scaffold mapping correction used in the manuscript",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

