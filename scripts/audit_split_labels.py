"""Summarize response-label distributions for every approved chemical split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ACTIVITY_CLASSES = [
    "inactive_lt4.1",
    "weak_4.1_5",
    "moderate_5_6",
    "potent_6_7",
    "very_potent_7_8",
    "super_ge8",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    args = parser.parse_args()

    root = Path(args.root)
    processed = root / "data" / "processed" / "cellminer"
    split_root = root / "data" / "splits" / "cellminer"
    response = pd.read_csv(processed / "response_long.csv.gz")
    valid_ids = set(pd.read_csv(split_root / "subsets" / "chemical_valid_nsc.csv")["nsc"].astype(int))
    modeling_response = response.loc[response["nsc"].isin(valid_ids)].copy()
    excluded_response = response.loc[~response["nsc"].isin(valid_ids)]

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    rows = []
    distribution_shift = []
    for family in ("random", "scaffold", "leader_cluster"):
        for seed in seeds:
            split_dir = split_root / f"{family}_seed_{seed}"
            assignment = {}
            compound_counts = {}
            for part in ("train", "valid", "test"):
                ids = pd.read_csv(split_dir / f"{part}_nsc.csv")["nsc"].astype(int)
                compound_counts[part] = int(len(ids))
                assignment.update({int(nsc): part for nsc in ids})
            annotated = modeling_response.assign(
                partition=modeling_response["nsc"].map(assignment)
            )
            if annotated["partition"].isna().any():
                raise AssertionError(f"Unassigned modeling labels in {family} seed {seed}")

            rate_vectors = {}
            for part in ("train", "valid", "test"):
                subset = annotated.loc[annotated["partition"].eq(part)]
                counts = (
                    subset["activity_class"]
                    .value_counts()
                    .reindex(ACTIVITY_CLASSES, fill_value=0)
                    .astype(int)
                )
                rates = counts / len(subset)
                rate_vectors[part] = rates.to_numpy(dtype=float)
                row = {
                    "split_family": family,
                    "seed": seed,
                    "partition": part,
                    "compounds": compound_counts[part],
                    "response_labels": int(len(subset)),
                    "mean_loggi50": float(subset["loggi50"].mean()),
                    "median_loggi50": float(subset["loggi50"].median()),
                    "minimum_loggi50": float(subset["loggi50"].min()),
                    "maximum_loggi50": float(subset["loggi50"].max()),
                    "boundary_candidate_rate": float(
                        subset["standard_boundary_candidate"].mean()
                    ),
                }
                for activity_class in ACTIVITY_CLASSES:
                    row[f"count__{activity_class}"] = int(counts[activity_class])
                    row[f"rate__{activity_class}"] = float(rates[activity_class])
                rows.append(row)

            distribution_shift.append(
                {
                    "split_family": family,
                    "seed": seed,
                    "train_test_total_variation": float(
                        0.5 * np.abs(rate_vectors["train"] - rate_vectors["test"]).sum()
                    ),
                    "train_valid_total_variation": float(
                        0.5 * np.abs(rate_vectors["train"] - rate_vectors["valid"]).sum()
                    ),
                }
            )

    audit = pd.DataFrame(rows)
    audit.to_csv(split_root / "split_label_audit.csv", index=False)
    report = {
        "total_response_labels": int(len(response)),
        "modeling_response_labels": int(len(modeling_response)),
        "excluded_invalid_structure_labels": int(len(excluded_response)),
        "modeling_label_coverage": float(len(modeling_response) / len(response)),
        "distribution_shift": distribution_shift,
        "output": str(split_root / "split_label_audit.csv"),
    }
    (split_root / "split_label_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


