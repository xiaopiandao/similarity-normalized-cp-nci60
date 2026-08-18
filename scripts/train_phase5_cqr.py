"""Train a shared multi-task quantile MLP for the Phase 5 CQR baseline.

The split roles remain frozen: fit trains the quantile network, validation is
used only for early stopping, source-calibration is reserved for conformal
correction, and test labels are never read by this script.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.run_phase1a_baselines import FAMILIES, PARTS, Phase1Data, set_torch_seed
except ModuleNotFoundError:
    from run_phase1a_baselines import FAMILIES, PARTS, Phase1Data, set_torch_seed  # type: ignore[no-redef]


QUANTILES = np.asarray((0.025, 0.05, 0.075, 0.10, 0.90, 0.925, 0.95, 0.975), dtype=np.float32)


def pinball_loss(prediction, target, observed, quantiles):
    """Masked mean pinball loss; tensors use [batch, cell, quantile]."""
    error = target[:, :, None] - prediction
    loss = np.maximum(quantiles * error, (quantiles - 1.0) * error)
    mask = observed[:, :, None]
    return float(loss[mask.repeat(loss.shape[2], axis=2)].mean())


def train_cqr(
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
    n_cells = y_fit_np.shape[1]

    response_mean = np.nanmean(y_fit_np, axis=0).astype(np.float32)
    response_scale = np.nanstd(y_fit_np, axis=0).astype(np.float32)
    response_scale[response_scale < 1e-6] = 1.0
    y_fit_normalized = (y_fit_np - response_mean) / response_scale

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
        nn.Linear(256, n_cells * len(QUANTILES)),
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
    q_t = torch.from_numpy(QUANTILES).to(device)[None, None, :]
    mean_t = torch.from_numpy(response_mean).to(device)[None, :, None]
    scale_t = torch.from_numpy(response_scale).to(device)[None, :, None]

    history: list[dict] = []
    best_state = None
    best_valid_loss = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for x_batch, y_batch, mask_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch).reshape(-1, n_cells, len(QUANTILES))
            error = y_batch[:, :, None] - prediction
            all_loss = torch.maximum(q_t * error, (q_t - 1.0) * error)
            mask = mask_batch[:, :, None].expand_as(all_loss)
            loss = all_loss[mask].mean()
            loss.backward()
            optimizer.step()
            count = int(mask.sum())
            train_sum += float(loss.detach()) * count
            train_count += count

        model.eval()
        with torch.no_grad():
            valid_pred = model(valid_x).reshape(-1, n_cells, len(QUANTILES))
            valid_pred = (valid_pred * scale_t + mean_t).cpu().numpy()
        valid_loss = pinball_loss(
            valid_pred,
            valid_y,
            np.isfinite(valid_y),
            QUANTILES[None, None, :],
        )
        scheduler.step(valid_loss)
        history.append(
            {
                "epoch": epoch,
                "train_pinball_normalized": train_sum / train_count,
                "valid_pinball_original_scale": valid_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch={epoch:03d} train_pinball={train_sum/train_count:.6f} "
            f"valid_pinball={valid_loss:.6f} lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        if valid_loss < best_valid_loss - 1e-5:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("CQR training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "cell_lines": np.asarray(data.cell_lines, dtype="U"),
        "quantiles": QUANTILES,
    }
    crossing_rows = []
    with torch.no_grad():
        for part in PARTS:
            ids, x_part, _ = loaded[part]
            raw = model(torch.from_numpy(x_part.toarray()).float().to(device))
            raw = raw.reshape(-1, n_cells, len(QUANTILES))
            prediction = (raw * scale_t + mean_t).cpu().numpy().astype(np.float32)
            crossing_rows.append(
                {
                    "part": part,
                    "adjacent_crossing_fraction": float((np.diff(prediction, axis=2) < 0).mean()),
                }
            )
            # Monotone rearrangement is deterministic and uses no response labels.
            prediction.sort(axis=2)
            payload[f"nsc_{part}"] = ids
            payload[f"pred_{part}"] = prediction
    np.savez_compressed(output_dir / "cqr_predictions.npz", **payload)
    pd.DataFrame(history).to_csv(output_dir / "cqr_history.csv", index=False)
    torch.save(
        {
            "state_dict": best_state,
            "response_mean": response_mean,
            "response_scale": response_scale,
            "quantiles": QUANTILES,
        },
        output_dir / "cqr_checkpoint.pt",
    )
    report = {
        "model": "shared 60-task multi-quantile MLP on ECFP4",
        "method": "Conformalized Quantile Regression baseline",
        "device": str(device),
        "training_seed": seed,
        "quantiles": QUANTILES.tolist(),
        "best_valid_pinball_original_scale": best_valid_loss,
        "epochs_completed": len(history),
        "monotone_rearrangement": "sort predicted quantiles per compound and cell line",
        "crossing_before_rearrangement": crossing_rows,
    }
    (output_dir / "cqr_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--families", default="random,scaffold,leader_cluster")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    root = Path(args.root)
    data = Phase1Data(root)
    families = tuple(value for value in args.families.split(",") if value)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for seed in seeds:
            split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
            output_dir = root / "results" / "phase5" / f"{family}_seed_{seed}"
            report = train_cqr(
                data,
                split_dir,
                output_dir,
                seed=50_000 + seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )
            print(json.dumps({"family": family, "seed": seed, **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()

