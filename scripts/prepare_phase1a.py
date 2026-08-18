"""Prepare cached Morgan features and source-domain calibration splits.

The original Phase 0 split remains untouched.  For each split family/seed, the
original 80% training partition is divided into a proper-fit set and a labelled
source-calibration set.  Identical Morgan fingerprints stay together so that a
calibration compound never has an exact fingerprint duplicate in the fit set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, __version__ as rdkit_version
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse


FAMILIES = ("random", "scaffold", "leader_cluster")


def write_ids(path: Path, values: np.ndarray) -> None:
    pd.DataFrame({"nsc": np.sort(values.astype("int64"))}).to_csv(path, index=False)


def select_grouped_calibration(
    frame: pd.DataFrame, target_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select whole Morgan groups while staying close to the target size."""
    groups = [
        group["nsc"].to_numpy(dtype="int64")
        for _, group in frame.groupby("morgan_fp_group_id", sort=True)
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    calibration: list[np.ndarray] = []
    count = 0
    for ids in groups:
        if count >= target_count:
            break
        # Include the next group when it improves the absolute target error.
        if count == 0 or abs(count + len(ids) - target_count) <= abs(count - target_count):
            calibration.append(ids)
            count += len(ids)
    cal = np.sort(np.concatenate(calibration))
    all_ids = frame["nsc"].to_numpy(dtype="int64")
    fit = np.sort(np.setdiff1d(all_ids, cal, assume_unique=False))
    return fit, cal


def cache_morgan_features(
    structures: pd.DataFrame, output_dir: Path, radius: int, fp_size: int
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = structures.loc[structures["structure_valid"]].copy()
    valid = valid.sort_values("nsc").reset_index(drop=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    rows: list[int] = []
    columns: list[int] = []
    for row_index, smiles in enumerate(valid["standardized_smiles"]):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Cached valid SMILES failed to parse at row {row_index}")
        fingerprint = generator.GetFingerprint(molecule)
        on_bits = list(fingerprint.GetOnBits())
        rows.extend([row_index] * len(on_bits))
        columns.extend(on_bits)

    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(valid), fp_size),
        dtype=np.uint8,
    )
    feature_path = output_dir / f"morgan_r{radius}_{fp_size}.npz"
    index_path = output_dir / f"morgan_r{radius}_{fp_size}_index.csv"
    metadata_path = output_dir / f"morgan_r{radius}_{fp_size}_metadata.json"
    sparse.save_npz(feature_path, matrix, compressed=True)
    valid[["nsc", "standardized_smiles"]].to_csv(index_path, index=False)

    metadata = {
        "feature": "Morgan bit fingerprint",
        "radius": radius,
        "fp_size": fp_size,
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "nonzero": int(matrix.nnz),
        "density": float(matrix.nnz / np.prod(matrix.shape)),
        "rdkit_version": rdkit_version,
        "matrix": str(feature_path),
        "index": str(index_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--source-cal-size", type=int, default=2524)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=2048)
    args = parser.parse_args()

    root = Path(args.root)
    processed = root / "data" / "processed" / "cellminer"
    phase0_splits = root / "data" / "splits" / "cellminer"
    phase1_splits = root / "data" / "splits" / "phase1a"
    feature_dir = root / "data" / "features" / "cellminer"
    phase1_splits.mkdir(parents=True, exist_ok=True)

    structures = pd.read_csv(processed / "chemical_structures.csv")
    groups = pd.read_csv(processed / "chemical_split_groups.csv")
    feature_metadata = cache_morgan_features(
        structures, feature_dir, args.radius, args.fp_size
    )

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    summary: dict[str, dict] = {}
    group_index = groups.set_index("nsc")
    for family in FAMILIES:
        for seed in seeds:
            key = f"{family}_seed_{seed}"
            original = phase0_splits / key
            output = phase1_splits / key
            output.mkdir(parents=True, exist_ok=True)
            train_ids = pd.read_csv(original / "train_nsc.csv")["nsc"].to_numpy("int64")
            valid_ids = pd.read_csv(original / "valid_nsc.csv")["nsc"].to_numpy("int64")
            test_ids = pd.read_csv(original / "test_nsc.csv")["nsc"].to_numpy("int64")
            train_frame = group_index.loc[train_ids].reset_index()
            fit_ids, cal_ids = select_grouped_calibration(
                train_frame, args.source_cal_size, seed=20260721 + 101 * seed
            )
            write_ids(output / "fit_nsc.csv", fit_ids)
            write_ids(output / "source_cal_nsc.csv", cal_ids)
            write_ids(output / "valid_nsc.csv", valid_ids)
            write_ids(output / "test_nsc.csv", test_ids)

            fit_groups = set(group_index.loc[fit_ids, "morgan_fp_group_id"].astype(str))
            cal_groups = set(group_index.loc[cal_ids, "morgan_fp_group_id"].astype(str))
            audit = {
                "family": family,
                "seed": seed,
                "counts": {
                    "fit": int(len(fit_ids)),
                    "source_cal": int(len(cal_ids)),
                    "valid": int(len(valid_ids)),
                    "test": int(len(test_ids)),
                },
                "source_cal_target": args.source_cal_size,
                "fit_source_cal_nsc_overlap": int(len(set(fit_ids) & set(cal_ids))),
                "fit_source_cal_morgan_group_overlap": int(len(fit_groups & cal_groups)),
                "test_overlap_with_non_test": int(
                    len(set(test_ids) & (set(fit_ids) | set(cal_ids) | set(valid_ids)))
                ),
                "policy": (
                    "source calibration is carved from original train while keeping "
                    "identical Morgan fingerprints together"
                ),
            }
            (output / "metadata.json").write_text(
                json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            summary[key] = audit

    report = {
        "feature_cache": feature_metadata,
        "splits": summary,
        "transductive_policy": (
            "test structures may be used only for density-ratio estimation; test labels stay locked"
        ),
    }
    (phase1_splits / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

