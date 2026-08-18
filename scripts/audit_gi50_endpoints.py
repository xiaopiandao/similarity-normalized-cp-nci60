"""Audit NCI-60 logGI50 endpoint spikes without inventing censor labels.

The CellMiner export contains numeric endpoint values but does not preserve a
per-observation comparison sign or tested concentration bounds. This script
therefore reports boundary candidates and sensitivity subsets; it deliberately
does not assign left/right censoring types.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root)
    processed = root / "data" / "processed" / "cellminer"
    response_path = processed / "response_long.csv.gz"
    response = pd.read_csv(response_path)

    required = {"nsc", "cell_line", "loggi50", "endpoint_audit_flag"}
    missing = sorted(required - set(response.columns))
    if missing:
        raise ValueError(f"response_long is missing endpoint-audit columns: {missing}")

    y = pd.to_numeric(response["loggi50"], errors="raise")
    total = int(len(response))

    exact_frequencies = (
        y.value_counts(dropna=False)
        .rename_axis("loggi50")
        .reset_index(name="count")
        .sort_values("loggi50")
    )
    exact_frequencies["rate"] = exact_frequencies["count"] / total
    exact_frequencies.to_csv(processed / "endpoint_value_frequencies.csv", index=False)

    per_cell = (
        response.assign(
            exact_4=y.eq(4.0),
            exact_8=y.eq(8.0),
            outside_standard=y.lt(4.0) | y.gt(8.0),
        )
        .groupby("cell_line", sort=True)
        .agg(
            valid_labels=("loggi50", "size"),
            minimum=("loggi50", "min"),
            maximum=("loggi50", "max"),
            exact_4_count=("exact_4", "sum"),
            exact_8_count=("exact_8", "sum"),
            outside_standard_count=("outside_standard", "sum"),
        )
        .reset_index()
    )
    for count_column in ["exact_4_count", "exact_8_count", "outside_standard_count"]:
        per_cell[count_column.replace("_count", "_rate")] = (
            per_cell[count_column] / per_cell["valid_labels"]
        )
    per_cell.to_csv(processed / "endpoint_audit_by_cell_line.csv", index=False)

    per_compound = (
        response.assign(
            standard_boundary_candidate=y.isin([4.0, 8.0]),
            outside_standard=y.lt(4.0) | y.gt(8.0),
        )
        .groupby("nsc", sort=True)
        .agg(
            valid_labels=("loggi50", "size"),
            minimum=("loggi50", "min"),
            maximum=("loggi50", "max"),
            standard_boundary_candidate_count=("standard_boundary_candidate", "sum"),
            outside_standard_count=("outside_standard", "sum"),
        )
        .reset_index()
    )
    per_compound["standard_boundary_candidate_rate"] = (
        per_compound["standard_boundary_candidate_count"] / per_compound["valid_labels"]
    )
    per_compound["outside_standard_rate"] = (
        per_compound["outside_standard_count"] / per_compound["valid_labels"]
    )
    per_compound.to_csv(processed / "endpoint_audit_by_compound.csv.gz", index=False, compression="gzip")

    def count_rate(mask: pd.Series) -> dict:
        count = int(mask.sum())
        return {"count": count, "rate": round(count / total, 8)}

    flag_counts = response["endpoint_audit_flag"].value_counts().to_dict()
    report = {
        "source": str(response_path),
        "total_valid_labels": total,
        "censoring_identifiability": {
            "status": "not_identifiable_from_current_cellminer_export",
            "reason": (
                "The numeric CellMiner workbook does not provide a per-observation "
                "comparison sign or the tested lower/upper concentration bounds."
            ),
            "allowed_interpretation": (
                "Values equal to common assay endpoints are boundary candidates only, "
                "not confirmed censored observations."
            ),
        },
        "distribution": {
            "minimum": float(y.min()),
            "maximum": float(y.max()),
            "mean": float(y.mean()),
            "median": float(y.median()),
            "exact_4": count_rate(y.eq(4.0)),
            "exact_8": count_rate(y.eq(8.0)),
            "below_4": count_rate(y.lt(4.0)),
            "above_8": count_rate(y.gt(8.0)),
            "below_0": count_rate(y.lt(0.0)),
            "above_12": count_rate(y.gt(12.0)),
            "endpoint_audit_flag_counts": {key: int(value) for key, value in flag_counts.items()},
        },
        "sensitivity_subsets": {
            "exclude_exact_4_or_8": int((~y.isin([4.0, 8.0])).sum()),
            "inside_open_interval_4_8": int((y.gt(4.0) & y.lt(8.0)).sum()),
            "inside_closed_interval_4_8": int((y.ge(4.0) & y.le(8.0)).sum()),
            "winsorize_4_8_candidate_count": int((y.lt(4.0) | y.gt(8.0)).sum()),
        },
        "outputs": {
            "value_frequencies": str(processed / "endpoint_value_frequencies.csv"),
            "by_cell_line": str(processed / "endpoint_audit_by_cell_line.csv"),
            "by_compound": str(processed / "endpoint_audit_by_compound.csv.gz"),
        },
    }
    write_json(processed / "endpoint_boundary_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


