"""Generate auditable analyses requested during JBHI peer-review revision.

This script never overwrites the frozen manuscript-facing results.  It applies
the author-confirmed scaffold method-label correction in a separate revision
directory, expands the interval-score reporting, quantifies similarity-tail
support and leader-cluster sizes, summarizes the Mondrian comparator, and runs
a dAD neighbourhood-size sensitivity analysis from frozen predictions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.analyze_uacqrp_exploratory import extended_interval_metrics
    from scripts.evaluate_phase5_extensions import ALPHAS, dad_intervals
    from scripts.finalize_jbhi_results import FAMILIES, reconstruct_run, summarize_across_seeds
    from scripts.evaluate_phase1a_conformal import EvaluationData
except ModuleNotFoundError:
    from analyze_uacqrp_exploratory import extended_interval_metrics  # type: ignore[no-redef]
    from evaluate_phase5_extensions import ALPHAS, dad_intervals  # type: ignore[no-redef]
    from finalize_jbhi_results import FAMILIES, reconstruct_run, summarize_across_seeds  # type: ignore[no-redef]
    from evaluate_phase1a_conformal import EvaluationData  # type: ignore[no-redef]


SEEDS = (2, 3, 4, 5)
DAD_K_VALUES = (50, 100, 250, 500)
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


def corrected_formal_results(root: Path, output: Path) -> pd.DataFrame:
    source = root / "results" / "final_jbhi" / "formal_method_seed_metrics.csv"
    frame = pd.read_csv(source)
    frame.insert(4, "source_method_label", frame["method"])
    selected = (frame["family"] == "scaffold") & frame["method"].isin(SWAP)
    frame.loc[selected, "method"] = frame.loc[selected, "method"].map(SWAP)
    frame.insert(5, "label_correction_applied", selected)
    frame.to_csv(output / "formal_method_seed_metrics_corrected.csv", index=False)

    metrics = [
        "coverage",
        "coverage_error_abs",
        "mean_width",
        "finite_interval_fraction",
        "mean_interval_score",
        "mean_interval_score_finite",
    ]
    summary = summarize_across_seeds(
        frame,
        ["family", "alpha", "method", "scope"],
        metrics,
    )
    summary.to_csv(output / "formal_method_summary_ci_corrected.csv", index=False)

    ood = frame[(frame["family"].isin(("scaffold", "leader_cluster"))) & (frame["scope"] == "all")]
    aggregate = (
        ood.groupby("method", as_index=False)
        .agg(
            absolute_calibration_error=("coverage_error_abs", "mean"),
            mean_width=("mean_width", "mean"),
            mean_interval_score=("mean_interval_score", "mean"),
            finite_only_interval_score=("mean_interval_score_finite", "mean"),
            minimum_finite_interval_fraction=("finite_interval_fraction", "min"),
        )
    )
    aggregate.to_csv(output / "ood_multilevel_aggregate_corrected.csv", index=False)

    low = frame[
        frame["family"].isin(("scaffold", "leader_cluster"))
        & (frame["scope"] == "similarity_lt_0.4")
    ]
    low.groupby("method", as_index=False).agg(
        absolute_calibration_error=("coverage_error_abs", "mean"),
        mean_width=("mean_width", "mean"),
        mean_interval_score=("mean_interval_score", "mean"),
        finite_only_interval_score=("mean_interval_score_finite", "mean"),
        minimum_finite_interval_fraction=("finite_interval_fraction", "min"),
    ).to_csv(output / "ood_low_similarity_aggregate_corrected.csv", index=False)

    return frame


def ranked_top_k_tanimoto_indices(cal_x, test_x, k: int, block_size: int = 32) -> np.ndarray:
    """Return the top-k calibration indices in descending Tanimoto order."""
    cal_counts = np.asarray(cal_x.sum(axis=1)).ravel()
    test_counts = np.asarray(test_x.sum(axis=1)).ravel()
    result = np.empty((test_x.shape[0], k), dtype=np.int32)
    for start in range(0, test_x.shape[0], block_size):
        stop = min(start + block_size, test_x.shape[0])
        intersection = (test_x[start:stop] @ cal_x.T).toarray()
        union = test_counts[start:stop, None] + cal_counts[None, :] - intersection
        similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union > 0,
        )
        selected = np.argpartition(similarity, -k, axis=1)[:, -k:]
        selected_similarity = np.take_along_axis(similarity, selected, axis=1)
        order = np.argsort(-selected_similarity, axis=1)
        result[start:stop] = np.take_along_axis(selected, order, axis=1)
    return result


def d_ad_sensitivity(root: Path, output: Path, data: EvaluationData) -> pd.DataFrame:
    rows: list[dict] = []
    cache_root = output / "dad_neighbours"
    cache_root.mkdir(exist_ok=True)
    for family in ("scaffold", "leader_cluster"):
        for seed in SEEDS:
            payload = reconstruct_run(root, data, family, seed)
            ids = payload["ids"]
            truth = payload["truth"]
            prediction = payload["prediction"]
            run_cache = cache_root / f"{family}_seed_{seed}_k{max(DAD_K_VALUES)}.npz"
            if run_cache.exists():
                neighbours_max = np.load(run_cache)["indices"].astype(np.int32)
            else:
                neighbours_max = ranked_top_k_tanimoto_indices(
                    data.x[data.rows(ids["source_cal"])],
                    data.x[data.rows(ids["test"])],
                    max(DAD_K_VALUES),
                )
                np.savez_compressed(run_cache, indices=neighbours_max)
            for k in DAD_K_VALUES:
                neighbours = neighbours_max[:, :k]
                for alpha in ALPHAS:
                    lower, upper = dad_intervals(
                        truth["source_cal"],
                        prediction["source_cal"],
                        prediction["test"],
                        neighbours,
                        alpha,
                    )
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "k": k,
                            "alpha": alpha,
                            **extended_interval_metrics(truth["test"], lower, upper, alpha),
                        }
                    )
            print(f"dAD sensitivity complete: {family} seed {seed}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "dad_k_sensitivity_seed_metrics.csv", index=False)
    frame.groupby("k", as_index=False).agg(
        absolute_calibration_error=("coverage_error_abs", "mean"),
        mean_width=("mean_width", "mean"),
        mean_interval_score=("mean_interval_score", "mean"),
        finite_interval_fraction=("finite_interval_fraction", "mean"),
    ).to_csv(output / "dad_k_sensitivity_ood_aggregate.csv", index=False)
    summarize_across_seeds(
        frame,
        ["family", "alpha", "k"],
        ["coverage", "coverage_error_abs", "mean_width", "mean_interval_score"],
    ).to_csv(output / "dad_k_sensitivity_summary_ci.csv", index=False)
    return frame


def support_diagnostics(root: Path, output: Path, data: EvaluationData) -> None:
    tail_rows: list[dict] = []
    for family in FAMILIES:
        for seed in SEEDS:
            payload = reconstruct_run(root, data, family, seed)
            for role in ("valid", "test"):
                similarity = payload["similarity"][role]
                truth = payload["truth"][role]
                for label, mask in (
                    ("S<0.3", similarity < 0.3),
                    ("S<0.4", similarity < 0.4),
                    ("S>=0.8", similarity >= 0.8),
                ):
                    tail_rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "role": role,
                            "similarity_scope": label,
                            "n_compounds": int(mask.sum()),
                            "n_observed_labels": int(np.isfinite(truth[mask]).sum()),
                            "fraction_compounds": float(mask.mean()),
                        }
                    )
    tails = pd.DataFrame(tail_rows)
    tails.to_csv(output / "similarity_tail_support_by_seed.csv", index=False)
    tails.groupby(["family", "role", "similarity_scope"], as_index=False).agg(
        n_compounds_mean=("n_compounds", "mean"),
        n_compounds_min=("n_compounds", "min"),
        n_compounds_max=("n_compounds", "max"),
        n_observed_labels_mean=("n_observed_labels", "mean"),
        fraction_compounds_mean=("fraction_compounds", "mean"),
    ).to_csv(output / "similarity_tail_support_summary.csv", index=False)

    groups = pd.read_csv(root / "data" / "processed" / "cellminer" / "chemical_split_groups.csv")
    sizes = groups.groupby("leader_cluster_id").size().to_numpy()
    cluster_distribution = {
        "n_clusters": int(len(sizes)),
        "n_compounds": int(sizes.sum()),
        "minimum": int(sizes.min()),
        "q25": float(np.quantile(sizes, 0.25)),
        "median": float(np.median(sizes)),
        "q75": float(np.quantile(sizes, 0.75)),
        "q90": float(np.quantile(sizes, 0.90)),
        "q95": float(np.quantile(sizes, 0.95)),
        "q99": float(np.quantile(sizes, 0.99)),
        "maximum": int(sizes.max()),
        "singleton_fraction": float(np.mean(sizes == 1)),
    }
    (output / "leader_cluster_size_distribution.json").write_text(
        json.dumps(cluster_distribution, indent=2), encoding="utf-8"
    )

    high = pd.read_csv(root / "results" / "phase4" / "coverage_similarity_summary_ci.csv")
    high = high[
        (high["family"] == "leader_cluster")
        & (high["similarity_bin"] == "[.8,1]")
        & high["method"].isin(("M1_global", "M5_similarity_normalized"))
    ]
    high.to_csv(output / "leader_cluster_high_similarity_summary.csv", index=False)

    m3 = pd.read_csv(root / "results" / "phase4" / "calibration_seed_metrics.csv")
    m3[m3["method"] == "M3_mondrian"].to_csv(
        output / "mondrian_calibration_seed_metrics.csv", index=False
    )
    m3_screen = pd.read_csv(root / "results" / "phase4" / "screening_threshold_summary_ci.csv")
    m3_screen[m3_screen["method"] == "M3_mondrian"].to_csv(
        output / "mondrian_screening_summary_ci.csv", index=False
    )


def main() -> None:
    root = Path(".")
    output = root / "results" / "revision_peer_review"
    output.mkdir(parents=True, exist_ok=True)
    data = EvaluationData(root)
    corrected_formal_results(root, output)
    support_diagnostics(root, output, data)
    d_ad_sensitivity(root, output, data)

    inputs = (
        root / "results" / "final_jbhi" / "formal_method_seed_metrics.csv",
        root / "results" / "phase4" / "coverage_similarity_summary_ci.csv",
        root / "results" / "phase4" / "calibration_seed_metrics.csv",
        root / "data" / "processed" / "cellminer" / "chemical_split_groups.csv",
        root / "scripts" / "analyze_peer_review_revisions.py",
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "peer_review_revision_analysis",
        "confirmation_seeds": list(SEEDS),
        "alpha_levels": list(ALPHAS),
        "scaffold_label_correction": {
            "scope": "all scaffold rows in the frozen formal interval-metric table",
            "operation": "swap UACQR_P and Proposed_shared_monotone labels",
            "reason": "author-confirmed correction after tracing the Table II scaffold mapping",
            "raw_frozen_results_overwritten": False,
        },
        "dad_k_values": list(DAD_K_VALUES),
        "files_sha256": {str(path.relative_to(root)): sha256(path) for path in inputs},
    }
    (output / "revision_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

