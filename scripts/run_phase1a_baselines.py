"""Train Phase 1A point-prediction baselines and cache split predictions."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


FAMILIES = ("random", "scaffold", "leader_cluster")
PARTS = ("source_cal", "valid", "test")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, pd.DataFrame]:
    records = []
    for column in range(y_true.shape[1]):
        mask = np.isfinite(y_true[:, column])
        truth = y_true[mask, column]
        pred = y_pred[mask, column]
        if not len(truth):
            continue
        error = pred - truth
        correlation = (
            float(np.corrcoef(truth, pred)[0, 1])
            if len(truth) > 1 and np.std(truth) > 0 and np.std(pred) > 0
            else np.nan
        )
        records.append(
            {
                "cell_index": column,
                "n": int(len(truth)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "pearson": correlation,
            }
        )
    per_cell = pd.DataFrame(records)
    mask = np.isfinite(y_true)
    all_error = y_pred[mask] - y_true[mask]
    overall = {
        "observations": int(mask.sum()),
        "mae_micro": float(np.mean(np.abs(all_error))),
        "rmse_micro": float(np.sqrt(np.mean(all_error**2))),
        "mae_macro": float(per_cell["mae"].mean()),
        "rmse_macro": float(per_cell["rmse"].mean()),
        "pearson_macro": float(per_cell["pearson"].mean()),
    }
    return overall, per_cell


class Phase1Data:
    def __init__(self, root: Path) -> None:
        feature_dir = root / "data" / "features" / "cellminer"
        self.x_sparse = sparse.load_npz(feature_dir / "morgan_r2_2048.npz").astype(
            np.float32
        )
        feature_index = pd.read_csv(feature_dir / "morgan_r2_2048_index.csv")
        self.row_by_nsc = {
            int(nsc): row for row, nsc in enumerate(feature_index["nsc"].astype(int))
        }
        response = pd.read_csv(
            root / "data" / "processed" / "cellminer" / "response_matrix_loggi50.csv.gz"
        ).set_index("nsc")
        self.cell_lines = response.columns.tolist()
        self.response = response

    def load_part(self, split_dir: Path, part: str) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray]:
        ids = pd.read_csv(split_dir / f"{part}_nsc.csv")["nsc"].to_numpy("int64")
        rows = np.fromiter((self.row_by_nsc[int(nsc)] for nsc in ids), dtype=np.int64)
        x = self.x_sparse[rows]
        y = self.response.loc[ids].to_numpy(dtype=np.float32)
        return ids, x, y


def train_ridge(
    data: Phase1Data,
    split_dir: Path,
    output_dir: Path,
    alpha_grid: list[float],
) -> dict:
    fit_ids, x_fit, y_fit = data.load_part(split_dir, "fit")
    del fit_ids
    loaded = {part: data.load_part(split_dir, part) for part in PARTS}

    validation_scores = {}
    for alpha in alpha_grid:
        predictions = np.full_like(loaded["valid"][2], np.nan, dtype=np.float32)
        for cell in range(y_fit.shape[1]):
            observed = np.isfinite(y_fit[:, cell])
            model = Ridge(alpha=alpha, solver="lsqr", tol=1e-3, max_iter=500)
            model.fit(x_fit[observed], y_fit[observed, cell])
            predictions[:, cell] = model.predict(loaded["valid"][1]).astype(np.float32)
        validation_scores[str(alpha)] = point_metrics(loaded["valid"][2], predictions)[0]

    best_alpha = min(
        alpha_grid, key=lambda value: validation_scores[str(value)]["rmse_macro"]
    )
    predictions = {
        part: np.full_like(loaded[part][2], np.nan, dtype=np.float32) for part in PARTS
    }
    for cell in range(y_fit.shape[1]):
        observed = np.isfinite(y_fit[:, cell])
        model = Ridge(alpha=best_alpha, solver="lsqr", tol=1e-3, max_iter=500)
        model.fit(x_fit[observed], y_fit[observed, cell])
        for part in PARTS:
            predictions[part][:, cell] = model.predict(loaded[part][1]).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"cell_lines": np.asarray(data.cell_lines, dtype="U")}
    report = {
        "model": "per-cell Ridge on ECFP4",
        "alpha_grid": alpha_grid,
        "validation_grid": validation_scores,
        "best_alpha": best_alpha,
        "parts": {},
    }
    for part in PARTS:
        ids, _, truth = loaded[part]
        overall, per_cell = point_metrics(truth, predictions[part])
        report["parts"][part] = overall
        per_cell.insert(0, "cell_line", data.cell_lines)
        per_cell.to_csv(output_dir / f"ridge_{part}_per_cell.csv", index=False)
        payload[f"nsc_{part}"] = ids
        payload[f"pred_{part}"] = predictions[part]
    np.savez_compressed(output_dir / "ridge_predictions.npz", **payload)
    (output_dir / "ridge_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def set_torch_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_mlp(
    data: Phase1Data,
    split_dir: Path,
    output_dir: Path,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> dict:
    import torch
    from torch import nn

    set_torch_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, x_fit_sparse, y_fit_np = data.load_part(split_dir, "fit")
    loaded = {part: data.load_part(split_dir, part) for part in PARTS}

    mean = np.nanmean(y_fit_np, axis=0).astype(np.float32)
    scale = np.nanstd(y_fit_np, axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    y_fit_normalized = (y_fit_np - mean) / scale

    x_fit = torch.from_numpy(x_fit_sparse.toarray()).float()
    y_fit = torch.from_numpy(np.nan_to_num(y_fit_normalized, nan=0.0)).float()
    m_fit = torch.from_numpy(np.isfinite(y_fit_np)).bool()
    valid_x = torch.from_numpy(loaded["valid"][1].toarray()).float().to(device)
    valid_y = loaded["valid"][2]

    model = nn.Sequential(
        nn.Linear(x_fit.shape[1], 512),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(256, y_fit.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_fit, y_fit, m_fit),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    history = []
    best_state = None
    best_rmse = float("inf")
    stale = 0
    mean_t = torch.from_numpy(mean).to(device)
    scale_t = torch.from_numpy(scale).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for x_batch, y_batch, mask_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = ((prediction - y_batch)[mask_batch] ** 2).mean()
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * int(mask_batch.sum())
            train_count += int(mask_batch.sum())

        model.eval()
        with torch.no_grad():
            valid_pred = (model(valid_x) * scale_t + mean_t).cpu().numpy()
        valid_metrics, _ = point_metrics(valid_y, valid_pred)
        valid_rmse = valid_metrics["rmse_macro"]
        scheduler.step(valid_rmse)
        history.append(
            {
                "epoch": epoch,
                "train_masked_mse_normalized": train_sum / train_count,
                "valid_rmse_macro": valid_rmse,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch={epoch:03d} train_mse={train_sum/train_count:.6f} "
            f"valid_rmse={valid_rmse:.6f} lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        if valid_rmse < best_rmse - 1e-5:
            best_rmse = valid_rmse
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"cell_lines": np.asarray(data.cell_lines, dtype="U")}
    report = {
        "model": "shared 60-output MLP on ECFP4",
        "device": str(device),
        "training_seed": seed,
        "best_valid_rmse_macro": best_rmse,
        "epochs_completed": len(history),
        "parts": {},
    }
    with torch.no_grad():
        for part in PARTS:
            ids, x_part, truth = loaded[part]
            pred = (
                model(torch.from_numpy(x_part.toarray()).float().to(device)) * scale_t + mean_t
            ).cpu().numpy()
            overall, per_cell = point_metrics(truth, pred)
            report["parts"][part] = overall
            per_cell.insert(0, "cell_line", data.cell_lines)
            per_cell.to_csv(output_dir / f"mlp_{part}_per_cell.csv", index=False)
            payload[f"nsc_{part}"] = ids
            payload[f"pred_{part}"] = pred.astype(np.float32)
    np.savez_compressed(output_dir / "mlp_predictions.npz", **payload)
    pd.DataFrame(history).to_csv(output_dir / "mlp_history.csv", index=False)
    torch.save(
        {"state_dict": best_state, "response_mean": mean, "response_scale": scale},
        output_dir / "mlp_checkpoint.pt",
    )
    (output_dir / "mlp_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--families", default="random,scaffold,leader_cluster")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--models", default="ridge,mlp")
    parser.add_argument("--ridge-alphas", default="1,10,100")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    root = Path(args.root)
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    alphas = [float(item) for item in args.ridge_alphas.split(",") if item.strip()]
    data = Phase1Data(root)
    summary = []
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for seed in seeds:
            split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
            output_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
            if "ridge" in models:
                report = train_ridge(data, split_dir, output_dir, alphas)
                summary.append({"family": family, "seed": seed, **report})
                print(json.dumps(summary[-1], indent=2), flush=True)
            if "mlp" in models:
                report = train_mlp(
                    data,
                    split_dir,
                    output_dir,
                    seed=10_000 + seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                )
                summary.append({"family": family, "seed": seed, **report})
                print(json.dumps(summary[-1], indent=2), flush=True)

    result_root = root / "results" / "phase1a"
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "baseline_summary.json"
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    replacement_keys = {(item["family"], item["seed"], item["model"]) for item in summary}
    merged = [
        item
        for item in existing
        if (item["family"], item["seed"], item["model"]) not in replacement_keys
    ] + summary
    summary_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

