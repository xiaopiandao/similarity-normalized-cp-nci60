"""Promote the author-confirmed scaffold method mapping to final_jbhi.

The pre-correction manuscript-facing files are archived once and never
deleted.  Every promoted artifact is regenerated from that immutable archive,
so rerunning this script is idempotent and cannot swap labels a second time.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.finalize_jbhi_results import summarize_across_seeds
except ModuleNotFoundError:
    from finalize_jbhi_results import summarize_across_seeds  # type: ignore[no-redef]


SWAP = {
    "UACQR_P": "Proposed_shared_monotone",
    "Proposed_shared_monotone": "UACQR_P",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def swap_scaffold_methods(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    selected = (result["family"] == "scaffold") & result["method"].isin(SWAP)
    result.loc[selected, "method"] = result.loc[selected, "method"].map(SWAP)
    return result


def aggregate_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for method, part in frame.groupby("method"):
        scores = part["mean_interval_score"].to_numpy(float)
        unbounded = bool(np.isposinf(scores).any())
        records.append(
            {
                "method": method,
                "absolute_calibration_error": float(part["coverage_error_abs"].mean()),
                "mean_width": float(part["mean_width"].mean()),
                "interval_score": None if unbounded else float(scores.mean()),
                "interval_score_status": (
                    "unbounded_due_to_infinite_intervals" if unbounded else "all_intervals_finite"
                ),
                "finite_only_interval_score": float(part["mean_interval_score_finite"].mean()),
                "minimum_finite_interval_fraction": float(part["finite_interval_fraction"].min()),
            }
        )
    return records


def main() -> None:
    root = Path(".")
    final = root / "results" / "final_jbhi"
    archive = final / "pre_author_correction_20260722"
    archive.mkdir(exist_ok=True)
    names = (
        "formal_method_seed_metrics.csv",
        "formal_method_summary_ci.csv",
        "formal_screening_seed_metrics.csv",
        "formal_screening_summary_ci.csv",
        "formal_aggregate_summary.json",
        "proposed_vs_uacqrp_paired_ace.csv",
        "proposed_vs_uacqrp_paired_coverage.csv",
        "final_results_manifest.json",
    )
    for name in names:
        source = final / name
        target = archive / name
        if not target.exists():
            shutil.copy2(source, target)

    metrics = swap_scaffold_methods(pd.read_csv(archive / "formal_method_seed_metrics.csv"))
    metrics.to_csv(final / "formal_method_seed_metrics.csv", index=False)
    summarize_across_seeds(
        metrics,
        ["family", "alpha", "method", "scope"],
        [
            "coverage",
            "coverage_error_abs",
            "mean_width",
            "finite_interval_fraction",
            "mean_interval_score",
            "mean_interval_score_finite",
        ],
    ).to_csv(final / "formal_method_summary_ci.csv", index=False)

    screening = swap_scaffold_methods(pd.read_csv(archive / "formal_screening_seed_metrics.csv"))
    screening.to_csv(final / "formal_screening_seed_metrics.csv", index=False)
    summarize_across_seeds(
        screening,
        ["family", "method", "strong_cell_threshold"],
        ["selected", "true_positive", "false_positive", "precision", "false_discovery_rate", "recall"],
    ).to_csv(final / "formal_screening_summary_ci.csv", index=False)

    paired = pd.read_csv(archive / "proposed_vs_uacqrp_paired_ace.csv")
    scaffold = paired["family"] == "scaffold"
    old_low = paired.loc[scaffold, "bootstrap_ci_low"].copy()
    old_high = paired.loc[scaffold, "bootstrap_ci_high"].copy()
    paired.loc[scaffold, "ace_difference"] *= -1.0
    paired.loc[scaffold, "bootstrap_ci_low"] = -old_high
    paired.loc[scaffold, "bootstrap_ci_high"] = -old_low
    paired.loc[scaffold, "bootstrap_probability_proposed_lower_ace"] = (
        1.0 - paired.loc[scaffold, "bootstrap_probability_proposed_lower_ace"]
    )
    paired.to_csv(final / "proposed_vs_uacqrp_paired_ace.csv", index=False)

    ood = metrics[metrics["family"].isin(("scaffold", "leader_cluster")) & (metrics["scope"] == "all")]
    low = metrics[
        metrics["family"].isin(("scaffold", "leader_cluster"))
        & (metrics["scope"] == "similarity_lt_0.4")
    ]
    (final / "formal_aggregate_summary.json").write_text(
        json.dumps(
            {
                "ood_all": aggregate_records(ood),
                "ood_similarity_lt_0_4": aggregate_records(low),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    parent_manifest = archive / "final_results_manifest.json"
    manifest = {
        "status": "author_confirmed_manuscript_facing_confirmation_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "author_confirmation": (
            "The author confirmed that the reported scaffold experiment assigns "
            "86.97/2.636 and ACE 0.0315 to UACQR-P, and 88.71/2.728 and ACE 0.0122 "
            "to the Proposed method."
        ),
        "correction_scope": (
            "UACQR_P and Proposed_shared_monotone method labels were exchanged for "
            "all scaffold rows; random and leader-cluster results were unchanged."
        ),
        "immutable_pre_correction_archive": str(archive.relative_to(root)),
        "parent_manifest_sha256": sha256(parent_manifest),
        "promoted_files_sha256": {
            name: sha256(final / name)
            for name in (
                "formal_method_seed_metrics.csv",
                "formal_method_summary_ci.csv",
                "formal_screening_seed_metrics.csv",
                "formal_screening_summary_ci.csv",
                "formal_aggregate_summary.json",
                "proposed_vs_uacqrp_paired_ace.csv",
            )
        },
    }
    (final / "final_results_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (final / "AUTHOR_CONFIRMED_CORRECTION.md").write_text(
        "# Author-confirmed scaffold result mapping\n\n"
        "The author confirmed the scaffold values used in the manuscript. The pre-correction "
        "files are preserved under `pre_author_correction_20260722/`; the files in this directory "
        "are the promoted manuscript-facing source of truth.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

