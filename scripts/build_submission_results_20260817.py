"""Build canonical machine-readable tables for the revised JBHI submission.

This script preserves all historical outputs.  It combines the audited
scaffold label correction with the development-selected dAD k=50 confirmation
results and copies the new external and sensitivity evidence into one
manuscript-facing directory.
"""

from __future__ import annotations

import argparse
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


CORE_COLUMNS = (
    "family",
    "seed",
    "alpha",
    "method",
    "scope",
    "nominal_coverage",
    "coverage",
    "coverage_error_abs",
    "mean_width",
    "median_width",
    "finite_interval_fraction",
    "n_compounds",
    "n_labels",
    "mean_interval_score",
    "mean_interval_score_finite",
    "crossed_interval_fraction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / "results" / "submission_20260817"
    output.mkdir(parents=True, exist_ok=True)

    corrected_path = (
        root
        / "results"
        / "revision_peer_review"
        / "formal_method_seed_metrics_corrected.csv"
    )
    dad_path = (
        root
        / "results"
        / "dad_locked_confirmation_20260816"
        / "dad_locked_seed_metrics.csv"
    )
    corrected = pd.read_csv(corrected_path)
    corrected = corrected[corrected["method"] != "dAD_style_k250"].copy()
    dad = pd.read_csv(dad_path)
    dad.insert(3, "method", "dAD_style_k50")

    missing_corrected = set(CORE_COLUMNS) - set(corrected.columns)
    missing_dad = set(CORE_COLUMNS) - set(dad.columns)
    if missing_corrected or missing_dad:
        raise ValueError(
            f"Missing columns: corrected={sorted(missing_corrected)}, dAD={sorted(missing_dad)}"
        )
    combined = pd.concat(
        [corrected.loc[:, CORE_COLUMNS], dad.loc[:, CORE_COLUMNS]], ignore_index=True
    ).sort_values(["family", "seed", "alpha", "method", "scope"])
    expected_methods = {
        "Global_CP",
        "UACQR_P",
        "dAD_style_k50",
        "Estimated_WCP",
        "Proposed_shared_monotone",
    }
    if set(combined["method"]) != expected_methods:
        raise ValueError("Canonical method set is incomplete")
    combined.to_csv(output / "manuscript_method_seed_metrics.csv", index=False)

    summary = summarize_across_seeds(
        combined,
        ["family", "alpha", "method", "scope"],
        [
            "coverage",
            "coverage_error_abs",
            "mean_width",
            "finite_interval_fraction",
            "mean_interval_score",
            "mean_interval_score_finite",
        ],
    )
    summary.to_csv(output / "manuscript_method_summary_ci.csv", index=False)

    table_ii = (
        combined[
            np.isclose(combined["alpha"], 0.10) & (combined["scope"] == "all")
        ]
        .groupby(["family", "method"], as_index=False)
        .agg(coverage=("coverage", "mean"), mean_width=("mean_width", "mean"))
    )
    table_ii.to_csv(output / "table_ii_90_coverage_width.csv", index=False)

    ood = combined[combined["family"].isin(("scaffold", "leader_cluster"))]
    table_iii = (
        ood.groupby(["method", "scope"], as_index=False)
        .agg(
            absolute_calibration_error=("coverage_error_abs", "mean"),
            mean_width=("mean_width", "mean"),
            mean_interval_score=("mean_interval_score", "mean"),
            mean_interval_score_finite=("mean_interval_score_finite", "mean"),
            minimum_finite_interval_fraction=("finite_interval_fraction", "min"),
        )
    )
    table_iii.to_csv(output / "table_iii_multilevel.csv", index=False)

    companion_files = {
        "dad_development_selection.csv": root
        / "results"
        / "dad_development_selection_20260816"
        / "dad_development_selection_summary.csv",
        "dad_locked_selection.json": root
        / "results"
        / "dad_development_selection_20260816"
        / "dad_locked_selection.json",
        "endpoint_boundary_ood_corrected.csv": root
        / "results"
        / "phase7_boundary_sensitivity_corrected_20260817"
        / "boundary_sensitivity_ood_aggregate_corrected.csv",
        "endpoint_boundary_manifest.json": root
        / "results"
        / "phase7_boundary_sensitivity_corrected_20260817"
        / "manifest.json",
        "hts384_comparator_summary.csv": root
        / "results"
        / "phase9_hts384_comparators_20260816"
        / "hts384_comparator_summary.csv",
        "hts384_compound_bootstrap_90.csv": root
        / "results"
        / "phase9_hts384_comparators_20260816"
        / "hts384_compound_bootstrap_90.csv",
        "hts384_manifest.json": root
        / "results"
        / "phase9_hts384_comparators_20260816"
        / "manifest.json",
        "fixed_budget_summary.csv": root
        / "results"
        / "screening_fixed_budget_20260815_repro"
        / "fixed_budget_summary_ci.csv",
    }
    for target_name, source in companion_files.items():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / target_name)

    sources = [corrected_path, dad_path, *companion_files.values()]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "canonical_revised_manuscript_results",
        "historical_outputs_overwritten": False,
        "scaffold_label_correction": (
            "author-confirmed swap of UACQR_P and Proposed_shared_monotone labels "
            "for scaffold rows; source labels remain in the audited correction file"
        ),
        "dad_configuration": (
            "k=50 selected on development seed 1 by mean OOD multi-level ACE and "
            "frozen for confirmation seeds 2-5"
        ),
        "confirmation_seeds": [2, 3, 4, 5],
        "files_sha256": {str(path.relative_to(root)): sha256(path) for path in sources},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table_ii.to_string(index=False), flush=True)
    print(table_iii.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

