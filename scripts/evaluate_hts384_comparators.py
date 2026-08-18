"""Evaluate all locked interval comparators on the HTS384 external lockbox.

The script uses only classic NCI-60 fit/validation/calibration responses to
construct intervals.  HTS384 fingerprints may be used by transductive
comparators, but HTS384 responses are read only after every interval has been
formed.  dAD uses the neighbourhood size selected on development seed 1.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.evaluate_hts384_external_lockbox import (
        ALPHAS,
        FAMILIES,
        HTS384Data,
        load_split_arrays,
        masked_interval_metrics,
        predict_external_mlp,
    )
    from scripts.evaluate_phase1a_conformal import EvaluationData, estimate_domain_weights
    from scripts.evaluate_phase1b_similarity import (
        fit_similarity_scale,
        nearest_fit_similarity,
        similarity_normalized_intervals,
        standard_intervals,
    )
    from scripts.evaluate_phase5_extensions import (
        dad_intervals,
        top_k_tanimoto_indices,
        weighted_intervals,
    )
    from scripts.run_phase1a_baselines import set_torch_seed
    from scripts.run_uacqrp_exploratory import (
        build_quantile_mlp,
        draw_ensemble,
        uacqrp_test_intervals,
    )
    from scripts.train_phase5_cqr import QUANTILES
except ModuleNotFoundError:
    from evaluate_hts384_external_lockbox import (  # type: ignore[no-redef]
        ALPHAS,
        FAMILIES,
        HTS384Data,
        load_split_arrays,
        masked_interval_metrics,
        predict_external_mlp,
    )
    from evaluate_phase1a_conformal import EvaluationData, estimate_domain_weights  # type: ignore[no-redef]
    from evaluate_phase1b_similarity import (  # type: ignore[no-redef]
        fit_similarity_scale,
        nearest_fit_similarity,
        similarity_normalized_intervals,
        standard_intervals,
    )
    from evaluate_phase5_extensions import (  # type: ignore[no-redef]
        dad_intervals,
        top_k_tanimoto_indices,
        weighted_intervals,
    )
    from run_phase1a_baselines import set_torch_seed  # type: ignore[no-redef]
    from run_uacqrp_exploratory import (  # type: ignore[no-redef]
        build_quantile_mlp,
        draw_ensemble,
        uacqrp_test_intervals,
    )
    from train_phase5_cqr import QUANTILES  # type: ignore[no-redef]


METHODS = (
    "Reference_global_CP",
    "UACQR_P",
    "dAD_style_k50",
    "Estimated_WCP",
    "Proposed_shared_monotone",
)
DAD_K = 50
EXTERNAL_DROPOUT_SEED_BASE = 90_000


def interval_score_metrics(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    label_mask: np.ndarray,
) -> dict:
    observed = np.isfinite(truth) & label_mask
    width = upper - lower
    score = width.copy()
    below = observed & (truth < lower)
    above = observed & (truth > upper)
    score[below] += (2.0 / alpha) * (lower[below] - truth[below])
    score[above] += (2.0 / alpha) * (truth[above] - upper[above])
    selected = score[observed]
    finite = selected[np.isfinite(selected)]
    pooled = float(np.mean(selected)) if len(selected) and np.isfinite(selected).all() else np.inf
    return {
        "mean_interval_score": pooled,
        "mean_interval_score_finite": float(np.mean(finite)) if len(finite) else np.nan,
    }


def predict_external_uacqrp(
    root: Path,
    external: HTS384Data,
    family: str,
    seed: int,
    ensemble_size: int,
    batch_size: int,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    import torch

    checkpoint = torch.load(
        root / "results" / "phase5" / f"{family}_seed_{seed}" / "cqr_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    stored = np.load(
        root
        / "results"
        / "phase6_uacqrp_exploratory"
        / f"{family}_seed_{seed}"
        / "uacqrp_intervals.npz"
    )
    thresholds = stored["thresholds"].astype(np.int16)
    if thresholds.shape != (len(ALPHAS), len(checkpoint["response_mean"])):
        raise ValueError("Stored UACQR-P thresholds have an unexpected shape")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_quantile_mlp(external.x.shape[1], len(checkpoint["response_mean"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    set_torch_seed(EXTERNAL_DROPOUT_SEED_BASE + seed)
    ensemble = draw_ensemble(
        model,
        external.x,
        np.asarray(checkpoint["response_mean"], dtype=np.float32),
        np.asarray(checkpoint["response_scale"], dtype=np.float32),
        device,
        batch_size,
        ensemble_size,
        label=f"{family}_seed_{seed}/HTS384",
    )

    result: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for alpha_index, alpha in enumerate(ALPHAS):
        lower_index = int(np.flatnonzero(np.isclose(QUANTILES, alpha / 2.0))[0])
        upper_index = int(np.flatnonzero(np.isclose(QUANTILES, 1.0 - alpha / 2.0))[0])
        result[alpha] = uacqrp_test_intervals(
            ensemble[:, :, :, lower_index],
            ensemble[:, :, :, upper_index],
            thresholds[alpha_index],
        )
    return result


def evaluate_one(
    root: Path,
    data: EvaluationData,
    external: HTS384Data,
    family: str,
    seed: int,
    ensemble_size: int,
    batch_size: int,
) -> tuple[list[dict], dict, dict[str, tuple[np.ndarray, np.ndarray]]]:
    arrays = load_split_arrays(root, data, family, seed, "mlp")
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")

    x_cal = data.x[data.rows(arrays["ids_source_cal"])]
    x_fit = data.x[data.rows(fit_ids)]
    pred_external = predict_external_mlp(root, external, family, seed)
    sim_cal = nearest_fit_similarity(x_fit, x_cal)
    sim_valid = nearest_fit_similarity(x_fit, data.x[data.rows(arrays["ids_valid"])])
    sim_external = nearest_fit_similarity(x_fit, external.x)
    scale_model, _, scale_diagnostics = fit_similarity_scale(
        sim_valid, arrays["truth_valid"], arrays["pred_valid"]
    )

    neighbours = top_k_tanimoto_indices(x_cal, external.x, DAD_K)
    source_weights, external_weights, weight_diagnostics = estimate_domain_weights(
        x_cal, external.x, seed=50_000 + seed
    )
    uacqr_intervals = predict_external_uacqrp(
        root, external, family, seed, ensemble_size, batch_size
    )

    # Interval construction is complete before the external outcomes are read.
    truth_external = external.truth(data.cell_lines)
    rows: list[dict] = []
    alpha_90_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    label_masks = {
        "all_labels": np.isfinite(truth_external),
        "non_endpoint_4_8": np.isfinite(truth_external)
        & ~np.isclose(truth_external, 4.0)
        & ~np.isclose(truth_external, 8.0),
    }
    row_masks = {
        "all_compounds": np.ones(len(external.ids), dtype=bool),
        "similarity_lt_0.4": sim_external < 0.4,
    }

    for alpha in ALPHAS:
        global_interval = standard_intervals(
            arrays["truth_source_cal"], arrays["pred_source_cal"], pred_external, alpha
        )
        proposed_lower, proposed_upper, _, _ = similarity_normalized_intervals(
            arrays["truth_source_cal"],
            arrays["pred_source_cal"],
            sim_cal,
            sim_external,
            scale_model,
            pred_external,
            alpha,
        )
        intervals = {
            "Reference_global_CP": global_interval,
            "UACQR_P": uacqr_intervals[alpha],
            "dAD_style_k50": dad_intervals(
                arrays["truth_source_cal"],
                arrays["pred_source_cal"],
                pred_external,
                neighbours,
                alpha,
            ),
            "Estimated_WCP": weighted_intervals(
                arrays["truth_source_cal"],
                arrays["pred_source_cal"],
                pred_external,
                source_weights,
                external_weights,
                alpha,
            ),
            "Proposed_shared_monotone": (proposed_lower, proposed_upper),
        }
        if np.isclose(alpha, 0.10):
            alpha_90_intervals = intervals

        for method, (lower, upper) in intervals.items():
            for row_scope, row_mask in row_masks.items():
                for label_scope, label_mask in label_masks.items():
                    combined = label_mask & row_mask[:, None]
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "alpha": alpha,
                            "method": method,
                            "row_scope": row_scope,
                            "label_scope": label_scope,
                            **masked_interval_metrics(
                                truth_external, lower, upper, alpha, combined
                            ),
                            **interval_score_metrics(
                                truth_external, lower, upper, alpha, combined
                            ),
                        }
                    )

    diagnostics = {
        "family": family,
        "seed": seed,
        "dAD_k": DAD_K,
        "uacqr_external_dropout_seed": EXTERNAL_DROPOUT_SEED_BASE + seed,
        "similarity_below_0_4_fraction": float((sim_external < 0.4).mean()),
        "scale_diagnostics": scale_diagnostics,
        "wcp_diagnostics": weight_diagnostics,
    }
    return rows, diagnostics, alpha_90_intervals


def compound_bootstrap(
    truth: np.ndarray,
    intervals_by_run: list[dict[str, tuple[np.ndarray, np.ndarray]]],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Cluster bootstrap HTS384 compounds while preserving all 60 outcomes."""
    observed = np.isfinite(truth)
    covered: dict[str, np.ndarray] = {}
    for method in METHODS:
        method_covered = []
        for intervals in intervals_by_run:
            lower, upper = intervals[method]
            method_covered.append(observed & (truth >= lower) & (truth <= upper))
        covered[method] = np.asarray(method_covered)

    rng = np.random.default_rng(seed)
    n_compounds = truth.shape[0]
    rows: list[dict] = []
    for comparator in METHODS:
        if comparator == "Proposed_shared_monotone":
            continue
        differences = np.empty(replicates, dtype=float)
        ace_differences = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            index = rng.integers(0, n_compounds, n_compounds)
            denominator = observed[index].sum() * len(intervals_by_run)
            proposed_coverage = covered["Proposed_shared_monotone"][:, index].sum() / denominator
            comparator_coverage = covered[comparator][:, index].sum() / denominator
            differences[replicate] = proposed_coverage - comparator_coverage
            ace_differences[replicate] = abs(proposed_coverage - 0.9) - abs(
                comparator_coverage - 0.9
            )
        rows.append(
            {
                "comparator": comparator,
                "comparison": "Proposed minus comparator",
                "coverage_difference_mean": float(differences.mean()),
                "coverage_difference_ci_low": float(np.quantile(differences, 0.025)),
                "coverage_difference_ci_high": float(np.quantile(differences, 0.975)),
                "ace_difference_mean": float(ace_differences.mean()),
                "ace_difference_ci_low": float(np.quantile(ace_differences, 0.025)),
                "ace_difference_ci_high": float(np.quantile(ace_differences, 0.975)),
                "bootstrap_replicates": replicates,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--families", default="scaffold,leader_cluster")
    parser.add_argument("--seeds", default="2,3,4,5")
    parser.add_argument("--ensemble-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()

    root = Path(args.root)
    output = root / "results" / "phase9_hts384_comparators_20260816"
    output.mkdir(parents=True, exist_ok=True)
    data = EvaluationData(root)
    external = HTS384Data(root)
    rows: list[dict] = []
    diagnostics: list[dict] = []
    intervals_by_run: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []

    families = tuple(value for value in args.families.split(",") if value)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for seed in seeds:
            metric_rows, detail, intervals = evaluate_one(
                root,
                data,
                external,
                family,
                seed,
                args.ensemble_size,
                args.batch_size,
            )
            rows.extend(metric_rows)
            diagnostics.append(detail)
            intervals_by_run.append(intervals)
            print(f"completed external comparators: {family} seed {seed}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "hts384_comparator_seed_metrics.csv", index=False)
    summary = (
        metrics.groupby(["alpha", "method", "row_scope", "label_scope"], as_index=False)
        .agg(
            coverage=("coverage", "mean"),
            coverage_error_abs=("coverage_error_abs", "mean"),
            mean_width=("mean_width", "mean"),
            mean_interval_score=("mean_interval_score", "mean"),
            mean_interval_score_finite=("mean_interval_score_finite", "mean"),
            finite_interval_fraction=("finite_interval_fraction", "mean"),
        )
    )
    summary.to_csv(output / "hts384_comparator_summary.csv", index=False)
    bootstrap = compound_bootstrap(
        external.truth(data.cell_lines),
        intervals_by_run,
        args.bootstrap_replicates,
        seed=20260816,
    )
    bootstrap.to_csv(output / "hts384_compound_bootstrap_90.csv", index=False)
    (output / "hts384_comparator_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "external_lockbox_comparator_evaluation",
        "families": list(families),
        "confirmation_seeds": list(seeds),
        "methods": list(METHODS),
        "dAD_k": DAD_K,
        "dAD_selection_file": "results/dad_development_selection_20260816/dad_locked_selection.json",
        "uacqr_dropout_draws": args.ensemble_size,
        "uacqr_external_seed_base": EXTERNAL_DROPOUT_SEED_BASE,
        "bootstrap_replicates": args.bootstrap_replicates,
        "guardrail": (
            "Intervals were formed without HTS384 responses. Estimated WCP used only "
            "unlabeled external fingerprints; responses were read after construction."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        summary[
            (np.isclose(summary["alpha"], 0.10))
            & (summary["row_scope"] == "all_compounds")
            & (summary["label_scope"] == "all_labels")
        ].to_string(index=False),
        flush=True,
    )
    print(bootstrap.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

