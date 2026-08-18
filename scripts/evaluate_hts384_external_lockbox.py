"""Evaluate conformal intervals on the HTS384 assay-external lockbox."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

try:
    from scripts.evaluate_phase1a_conformal import EvaluationData, finite_sample_quantile
    from scripts.evaluate_phase1b_similarity import (
        fit_similarity_scale,
        nearest_fit_similarity,
        predict_similarity_scale,
        standard_intervals,
        similarity_normalized_intervals,
    )
except ModuleNotFoundError:
    from evaluate_phase1a_conformal import EvaluationData, finite_sample_quantile  # type: ignore[no-redef]
    from evaluate_phase1b_similarity import (  # type: ignore[no-redef]
        fit_similarity_scale,
        nearest_fit_similarity,
        predict_similarity_scale,
        standard_intervals,
        similarity_normalized_intervals,
    )


FAMILIES = ("scaffold", "leader_cluster")
MODELS = ("mlp", "ridge")
ALPHAS = (0.05, 0.10, 0.15, 0.20)


class HTS384Data:
    def __init__(self, root: Path) -> None:
        feature_dir = root / "data" / "features" / "cellminer_hts384"
        self.x = sparse.load_npz(feature_dir / "morgan_r2_2048.npz").astype(np.float32)
        self.index = pd.read_csv(feature_dir / "morgan_r2_2048_index.csv")
        self.ids = self.index["nsc"].to_numpy("int64")
        self.responses = pd.read_csv(
            root
            / "data"
            / "processed"
            / "cellminer_hts384"
            / "response_matrix_loggi50.csv.gz"
        ).set_index("nsc")

    def truth(self, cell_lines: list[str]) -> np.ndarray:
        return self.responses.loc[self.ids, cell_lines].to_numpy(dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def masked_interval_metrics(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    label_mask: np.ndarray,
) -> dict:
    observed = np.isfinite(truth) & label_mask
    finite = observed & np.isfinite(lower) & np.isfinite(upper)
    covered = finite & (truth >= lower) & (truth <= upper)
    width = upper - lower
    coverage = float(covered.sum() / observed.sum()) if observed.any() else np.nan
    finite_width = width[finite]
    return {
        "nominal_coverage": 1.0 - alpha,
        "coverage": coverage,
        "coverage_error_abs": abs(coverage - (1.0 - alpha)) if np.isfinite(coverage) else np.nan,
        "mean_width": float(finite_width.mean()) if len(finite_width) else np.nan,
        "median_width": float(np.median(finite_width)) if len(finite_width) else np.nan,
        "finite_interval_fraction": float(finite.sum() / observed.sum()) if observed.any() else np.nan,
        "n_compounds": int(truth.shape[0]),
        "n_labels": int(observed.sum()),
    }


def point_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    observed = np.isfinite(truth)
    error = prediction[observed] - truth[observed]
    per_cell = []
    for cell in range(truth.shape[1]):
        mask = np.isfinite(truth[:, cell])
        if mask.any():
            per_cell.append(float(np.mean(np.abs(prediction[mask, cell] - truth[mask, cell]))))
    return {
        "observations": int(observed.sum()),
        "mae_micro": float(np.mean(np.abs(error))),
        "rmse_micro": float(np.sqrt(np.mean(error**2))),
        "mae_macro": float(np.mean(per_cell)),
    }


def load_split_arrays(root: Path, data: EvaluationData, family: str, seed: int, model: str) -> dict:
    result_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    stored = np.load(result_dir / f"{model}_predictions.npz")
    arrays = {}
    for part in ("source_cal", "valid", "test"):
        ids = stored[f"nsc_{part}"].astype("int64")
        arrays[f"ids_{part}"] = ids
        arrays[f"pred_{part}"] = stored[f"pred_{part}"].astype(float)
        arrays[f"truth_{part}"] = data.truth(ids)
    return arrays


def predict_external_ridge(
    root: Path,
    data: EvaluationData,
    external: HTS384Data,
    family: str,
    seed: int,
) -> np.ndarray:
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    result_dir = root / "results" / "phase1a" / f"{family}_seed_{seed}"
    metrics = json.loads((result_dir / "ridge_metrics.json").read_text(encoding="utf-8"))
    alpha = float(metrics["best_alpha"])
    fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
    x_fit = data.x[data.rows(fit_ids)]
    y_fit = data.truth(fit_ids)
    prediction = np.full((external.x.shape[0], y_fit.shape[1]), np.nan, dtype=np.float32)
    for cell in range(y_fit.shape[1]):
        observed = np.isfinite(y_fit[:, cell])
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-3, max_iter=500)
        model.fit(x_fit[observed], y_fit[observed, cell])
        prediction[:, cell] = model.predict(external.x).astype(np.float32)
    return prediction.astype(float)


def predict_external_mlp(root: Path, external: HTS384Data, family: str, seed: int) -> np.ndarray:
    import torch
    from torch import nn

    checkpoint = torch.load(
        root / "results" / "phase1a" / f"{family}_seed_{seed}" / "mlp_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = nn.Sequential(
        nn.Linear(external.x.shape[1], 512),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(256, len(checkpoint["response_mean"])),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(external.x.toarray()).float()
        mean = torch.from_numpy(checkpoint["response_mean"]).float()
        scale = torch.from_numpy(checkpoint["response_scale"]).float()
        prediction = model(x) * scale + mean
    return prediction.numpy().astype(float)


def evaluate_one(
    root: Path,
    data: EvaluationData,
    external: HTS384Data,
    family: str,
    seed: int,
    model: str,
) -> tuple[list[dict], dict]:
    split_dir = root / "data" / "splits" / "phase1a" / f"{family}_seed_{seed}"
    arrays = load_split_arrays(root, data, family, seed, model)
    fit_ids = pd.read_csv(split_dir / "fit_nsc.csv")["nsc"].to_numpy("int64")
    truth_external = external.truth(data.cell_lines)
    if model == "ridge":
        pred_external = predict_external_ridge(root, data, external, family, seed)
    elif model == "mlp":
        pred_external = predict_external_mlp(root, external, family, seed)
    else:
        raise ValueError(model)

    sim_cal = nearest_fit_similarity(data.x[data.rows(fit_ids)], data.x[data.rows(arrays["ids_source_cal"])])
    sim_valid = nearest_fit_similarity(data.x[data.rows(fit_ids)], data.x[data.rows(arrays["ids_valid"])])
    sim_external = nearest_fit_similarity(data.x[data.rows(fit_ids)], external.x)
    scale_model, _, scale_diag = fit_similarity_scale(
        sim_valid, arrays["truth_valid"], arrays["pred_valid"]
    )

    rows: list[dict] = []
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
        global_lower, global_upper = standard_intervals(
            arrays["truth_source_cal"], arrays["pred_source_cal"], pred_external, alpha
        )
        proposed_lower, proposed_upper, cal_scale, external_scale = similarity_normalized_intervals(
            arrays["truth_source_cal"],
            arrays["pred_source_cal"],
            sim_cal,
            sim_external,
            scale_model,
            pred_external,
            alpha,
        )
        methods = {
            "Reference_global_CP": (global_lower, global_upper),
            "Proposed_shared_monotone": (proposed_lower, proposed_upper),
        }
        for method, (lower, upper) in methods.items():
            for row_scope, row_mask in row_masks.items():
                for label_scope, label_mask in label_masks.items():
                    combined = label_mask & row_mask[:, None]
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "model": model,
                            "alpha": alpha,
                            "method": method,
                            "row_scope": row_scope,
                            "label_scope": label_scope,
                            **masked_interval_metrics(
                                truth_external, lower, upper, alpha, combined
                            ),
                        }
                    )
    detail = {
        "family": family,
        "seed": seed,
        "model": model,
        "external_point_metrics": point_metrics(truth_external, pred_external),
        "external_compounds": int(len(external.ids)),
        "external_labels": int(np.isfinite(truth_external).sum()),
        "external_exact_4_rate": float(np.isclose(truth_external, 4.0, equal_nan=False).sum() / np.isfinite(truth_external).sum()),
        "external_exact_8_rate": float(np.isclose(truth_external, 8.0, equal_nan=False).sum() / np.isfinite(truth_external).sum()),
        "external_similarity": {
            "mean": float(sim_external.mean()),
            "median": float(np.median(sim_external)),
            "q90": float(np.quantile(sim_external, 0.90)),
            "max": float(sim_external.max()),
            "fraction_below_0_4": float((sim_external < 0.4).mean()),
        },
        "scale_diagnostics": scale_diag,
        "cal_scale_reference": {
            "min": float(predict_similarity_scale(scale_model, sim_cal).min()),
            "median": float(np.median(predict_similarity_scale(scale_model, sim_cal))),
            "max": float(predict_similarity_scale(scale_model, sim_cal).max()),
        },
        "external_scale_reference": {
            "min": float(external_scale.min()),
            "median": float(np.median(external_scale)),
            "max": float(external_scale.max()),
        },
    }
    return rows, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--families", default="scaffold,leader_cluster")
    parser.add_argument("--seeds", default="2,3,4,5")
    parser.add_argument("--models", default="mlp,ridge")
    args = parser.parse_args()
    root = Path(args.root)
    data = EvaluationData(root)
    external = HTS384Data(root)
    output = root / "results" / "phase9_hts384_lockbox"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    details: list[dict] = []
    for family in [value for value in args.families.split(",") if value]:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for seed in [int(value) for value in args.seeds.split(",") if value]:
            for model in [value for value in args.models.split(",") if value]:
                if model not in MODELS:
                    raise ValueError(f"Unknown model: {model}")
                metric_rows, detail = evaluate_one(root, data, external, family, seed, model)
                rows.extend(metric_rows)
                details.append(detail)
                print(f"completed family={family} seed={seed} model={model}", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "hts384_external_interval_metrics.csv", index=False)
    pd.DataFrame(details).to_json(
        output / "hts384_external_details.json", orient="records", indent=2
    )
    summary = (
        metrics.groupby(["model", "alpha", "method", "row_scope", "label_scope"], as_index=False)
        .agg(
            coverage=("coverage", "mean"),
            coverage_error_abs=("coverage_error_abs", "mean"),
            mean_width=("mean_width", "mean"),
            finite_interval_fraction=("finite_interval_fraction", "mean"),
            n_labels=("n_labels", "sum"),
        )
    )
    summary.to_csv(output / "hts384_external_summary.csv", index=False)
    manifest_sources = (
        root / "scripts" / "evaluate_hts384_external_lockbox.py",
        root / "scripts" / "prepare_hts384_lockbox.py",
        root / "data" / "processed" / "cellminer_hts384" / "lockbox_preparation_report.json",
        root / "data" / "features" / "cellminer_hts384" / "morgan_r2_2048.npz",
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            f"python scripts/evaluate_hts384_external_lockbox.py --root {root.as_posix()} "
            f"--families {args.families} --seeds {args.seeds} --models {args.models}"
        ),
        "guardrail": (
            "HTS384 is an assay-external chemical lockbox. It is not a temporal "
            "lockbox because per-compound assay dates are not documented in the "
            "official CellMiner workbook."
        ),
        "files_sha256": {
            str(path.relative_to(root)): sha256(path) for path in manifest_sources if path.exists()
        },
    }
    (output / "hts384_external_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

