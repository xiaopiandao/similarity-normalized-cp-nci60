"""Run an internal UACQR-P experiment from frozen Phase 5 CQR checkpoints.

This script does not retrain or alter the CQR baseline.  It activates dropout
in the frozen multi-quantile MLP to obtain an ensemble of quantile estimates,
then applies the percentile construction from Rossellini et al. (AISTATS 2024).
Only fit-trained model state and source-calibration responses are used; test
responses are never loaded here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.run_phase1a_baselines import FAMILIES, Phase1Data, set_torch_seed
    from scripts.train_phase5_cqr import QUANTILES
except ModuleNotFoundError:
    from run_phase1a_baselines import FAMILIES, Phase1Data, set_torch_seed  # type: ignore[no-redef]
    from train_phase5_cqr import QUANTILES  # type: ignore[no-redef]


ALPHAS = (0.05, 0.10, 0.15, 0.20)


def build_quantile_mlp(input_dim: int, n_cells: int):
    """Recreate the exact Phase 5 CQR network architecture."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(256, n_cells * len(QUANTILES)),
    )


def uacqrp_calibration_scores(
    truth: np.ndarray,
    lower_ensemble: np.ndarray,
    upper_ensemble: np.ndarray,
) -> np.ndarray:
    """Return the minimal UACQR-P expansion rank for every observed response.

    Ensemble arrays have shape [B, compound, cell].  Lower estimates are
    ordered from largest to smallest and upper estimates from smallest to
    largest.  Rank B represents the virtual (-inf, +inf) terminal interval.
    """
    truth = np.asarray(truth, dtype=float)
    lower_ensemble = np.asarray(lower_ensemble, dtype=float)
    upper_ensemble = np.asarray(upper_ensemble, dtype=float)
    if lower_ensemble.shape != upper_ensemble.shape:
        raise ValueError("Lower and upper ensembles must have identical shapes")
    if lower_ensemble.ndim != 3 or lower_ensemble.shape[1:] != truth.shape:
        raise ValueError("Expected ensemble shape [B, compound, cell]")

    b_size = lower_ensemble.shape[0]
    lower_sorted = np.sort(lower_ensemble, axis=0)[::-1]
    upper_sorted = np.sort(upper_ensemble, axis=0)
    lower_ok = lower_sorted <= truth[None, :, :]
    upper_ok = upper_sorted >= truth[None, :, :]
    lower_score = np.where(lower_ok.any(axis=0), lower_ok.argmax(axis=0), b_size)
    upper_score = np.where(upper_ok.any(axis=0), upper_ok.argmax(axis=0), b_size)
    scores = np.maximum(lower_score, upper_score).astype(np.int16)
    scores[~np.isfinite(truth)] = -1
    return scores


def uacqrp_thresholds(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Compute the exact UACQR reference-code expansion rank per cell line."""
    thresholds = np.empty(scores.shape[1], dtype=np.int16)
    for cell in range(scores.shape[1]):
        observed = scores[:, cell] >= 0
        clean = scores[observed, cell]
        if not len(clean):
            raise ValueError(f"No calibration scores for cell {cell}")
        order_index = min(int(np.ceil((1.0 - alpha) * (len(clean) + 1))) - 1, len(clean) - 1)
        thresholds[cell] = int(np.sort(clean)[order_index])
    return thresholds


def uacqrp_test_intervals(
    lower_ensemble: np.ndarray,
    upper_ensemble: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the calibrated UACQR-P percentile interval for each cell."""
    lower_ensemble = np.asarray(lower_ensemble, dtype=float)
    upper_ensemble = np.asarray(upper_ensemble, dtype=float)
    if lower_ensemble.shape != upper_ensemble.shape or lower_ensemble.ndim != 3:
        raise ValueError("Expected matching [B, compound, cell] ensembles")
    b_size, n_compounds, n_cells = lower_ensemble.shape
    if thresholds.shape != (n_cells,):
        raise ValueError("Expected one threshold per cell line")
    if np.any(thresholds < 0) or np.any(thresholds > b_size):
        raise ValueError("UACQR-P threshold lies outside [0, B]")

    lower_sorted = np.sort(lower_ensemble, axis=0)[::-1]
    upper_sorted = np.sort(upper_ensemble, axis=0)
    lower = np.empty((n_compounds, n_cells), dtype=np.float32)
    upper = np.empty((n_compounds, n_cells), dtype=np.float32)
    for cell, threshold in enumerate(thresholds.astype(int)):
        if threshold == b_size:
            lower[:, cell] = -np.inf
            upper[:, cell] = np.inf
        else:
            lower[:, cell] = lower_sorted[threshold, :, cell]
            upper[:, cell] = upper_sorted[threshold, :, cell]
    return lower, upper


def predict_quantiles(
    model,
    x_sparse,
    response_mean: np.ndarray,
    response_scale: np.ndarray,
    device,
    batch_size: int,
    stochastic: bool,
) -> np.ndarray:
    """Predict all quantiles once, preserving Phase 5 monotone rearrangement."""
    import torch

    model.train(stochastic)
    n_cells = len(response_mean)
    result = np.empty((x_sparse.shape[0], n_cells, len(QUANTILES)), dtype=np.float32)
    mean_t = torch.from_numpy(response_mean).to(device)[None, :, None]
    scale_t = torch.from_numpy(response_scale).to(device)[None, :, None]
    with torch.no_grad():
        for start in range(0, x_sparse.shape[0], batch_size):
            stop = min(start + batch_size, x_sparse.shape[0])
            x_batch = torch.from_numpy(x_sparse[start:stop].toarray()).float().to(device)
            prediction = model(x_batch).reshape(-1, n_cells, len(QUANTILES))
            prediction = torch.sort(prediction * scale_t + mean_t, dim=2).values
            result[start:stop] = prediction.cpu().numpy().astype(np.float32)
    return result


def draw_ensemble(
    model,
    x_sparse,
    response_mean: np.ndarray,
    response_scale: np.ndarray,
    device,
    batch_size: int,
    ensemble_size: int,
    label: str,
) -> np.ndarray:
    """Draw a deterministic sequence of Monte Carlo dropout predictions."""
    ensemble = np.empty(
        (ensemble_size, x_sparse.shape[0], len(response_mean), len(QUANTILES)),
        dtype=np.float32,
    )
    for draw in range(ensemble_size):
        ensemble[draw] = predict_quantiles(
            model,
            x_sparse,
            response_mean,
            response_scale,
            device,
            batch_size,
            stochastic=True,
        )
        if draw == 0 or (draw + 1) % 10 == 0:
            print(f"{label}: dropout_draw={draw + 1}/{ensemble_size}", flush=True)
    return ensemble


def run_one(
    root: Path,
    data: Phase1Data,
    family: str,
    seed: int,
    ensemble_size: int,
    batch_size: int,
    output_root: Path | None = None,
) -> dict:
    import torch

    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    cqr_dir = root / "results" / "phase5" / f"{family}_seed_{seed}"
    if output_root is None:
        output_root = root / "results" / "phase6_uacqrp_exploratory"
    output_dir = output_root / f"{family}_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = cqr_dir / "cqr_checkpoint.pt"
    prediction_path = cqr_dir / "cqr_predictions.npz"
    if not checkpoint_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"Missing frozen CQR artifacts in {cqr_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frozen = np.load(prediction_path)
    loaded = {part: data.load_part(split_dir, part) for part in ("source_cal", "test")}
    n_cells = loaded["source_cal"][2].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_quantile_mlp(loaded["source_cal"][1].shape[1], n_cells).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    response_mean = np.asarray(checkpoint["response_mean"], dtype=np.float32)
    response_scale = np.asarray(checkpoint["response_scale"], dtype=np.float32)

    compatibility = {}
    for part in ("source_cal", "test"):
        ids, x_part, _ = loaded[part]
        if not np.array_equal(ids, frozen[f"nsc_{part}"].astype("int64")):
            raise ValueError(f"CQR IDs disagree for {family} seed {seed} {part}")
        deterministic = predict_quantiles(
            model,
            x_part,
            response_mean,
            response_scale,
            device,
            batch_size,
            stochastic=False,
        )
        difference = np.abs(deterministic - frozen[f"pred_{part}"].astype(np.float32))
        compatibility[part] = {
            "max_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
        }
        if difference.max() > 1e-4:
            raise RuntimeError(f"Frozen CQR checkpoint compatibility failed for {part}")

    dropout_seed = 70_000 + seed
    set_torch_seed(dropout_seed)
    cal_ids, cal_x, cal_truth = loaded["source_cal"]
    cal_ensemble = draw_ensemble(
        model,
        cal_x,
        response_mean,
        response_scale,
        device,
        batch_size,
        ensemble_size,
        label=f"{family}_seed_{seed}/source_cal",
    )

    alpha_indices = []
    thresholds = []
    score_diagnostics = []
    for alpha in ALPHAS:
        lower_index = int(np.flatnonzero(np.isclose(QUANTILES, alpha / 2.0))[0])
        upper_index = int(np.flatnonzero(np.isclose(QUANTILES, 1.0 - alpha / 2.0))[0])
        scores = uacqrp_calibration_scores(
            cal_truth,
            cal_ensemble[:, :, :, lower_index],
            cal_ensemble[:, :, :, upper_index],
        )
        threshold = uacqrp_thresholds(scores, alpha)
        alpha_indices.append((lower_index, upper_index))
        thresholds.append(threshold)
        observed_scores = scores[scores >= 0]
        score_diagnostics.append(
            {
                "alpha": alpha,
                "score_median": float(np.median(observed_scores)),
                "score_q90": float(np.quantile(observed_scores, 0.90)),
                "score_max": int(observed_scores.max()),
                "cells_requiring_infinite_terminal_interval": int((threshold == ensemble_size).sum()),
                "threshold_min": int(threshold.min()),
                "threshold_median": float(np.median(threshold)),
                "threshold_max": int(threshold.max()),
            }
        )
    del cal_ensemble

    test_ids, test_x, _ = loaded["test"]
    test_ensemble = draw_ensemble(
        model,
        test_x,
        response_mean,
        response_scale,
        device,
        batch_size,
        ensemble_size,
        label=f"{family}_seed_{seed}/test",
    )
    lower_intervals = []
    upper_intervals = []
    interval_diagnostics = []
    for alpha, (lower_index, upper_index), threshold in zip(ALPHAS, alpha_indices, thresholds):
        lower, upper = uacqrp_test_intervals(
            test_ensemble[:, :, :, lower_index],
            test_ensemble[:, :, :, upper_index],
            threshold,
        )
        lower_intervals.append(lower)
        upper_intervals.append(upper)
        interval_diagnostics.append(
            {
                "alpha": alpha,
                "finite_interval_fraction": float((np.isfinite(lower) & np.isfinite(upper)).mean()),
                "crossed_interval_fraction": float((lower > upper).mean()),
            }
        )
    del test_ensemble

    np.savez_compressed(
        output_dir / "uacqrp_intervals.npz",
        nsc_test=test_ids,
        alphas=np.asarray(ALPHAS, dtype=np.float32),
        lower=np.asarray(lower_intervals, dtype=np.float32),
        upper=np.asarray(upper_intervals, dtype=np.float32),
        thresholds=np.asarray(thresholds, dtype=np.int16),
    )
    report = {
        "status": "internal_exploratory_not_for_manuscript",
        "method": "UACQR-P faithful multitask adaptation",
        "reference": "Rossellini, Barber, and Willett, AISTATS 2024",
        "family": family,
        "seed": seed,
        "device": str(device),
        "ensemble_source": "Monte Carlo dropout from frozen Phase 5 CQR checkpoint",
        "dropout_rate": 0.10,
        "dropout_draws": ensemble_size,
        "dropout_seed": dropout_seed,
        "monotone_rearrangement": "sort all eight quantiles within every dropout draw",
        "checkpoint_compatibility": compatibility,
        "calibration_scores": score_diagnostics,
        "test_intervals": interval_diagnostics,
    }
    (output_dir / "uacqrp_run_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--families", default="random,scaffold,leader_cluster")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--ensemble-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional result directory. When omitted, the historical "
            "results/phase6_uacqrp_exploratory location is used."
        ),
    )
    args = parser.parse_args()
    if args.ensemble_size < 2:
        raise ValueError("UACQR-P requires at least two ensemble members")

    root = Path(args.root)
    output_root = Path(args.output_root) if args.output_root else None
    data = Phase1Data(root)
    families = tuple(value for value in args.families.split(",") if value)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for seed in seeds:
            run_one(
                root,
                data,
                family,
                seed,
                args.ensemble_size,
                args.batch_size,
                output_root,
            )


if __name__ == "__main__":
    main()

