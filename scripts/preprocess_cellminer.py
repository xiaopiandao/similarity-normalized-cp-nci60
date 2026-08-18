"""Preprocess official CellMiner NCI-60 data for the MAMBA-Net rewrite.

Stage 1 outputs deterministic, auditable tables. The CellMiner RAW workbook
repeats identical exported profiles across experiment rows, so this script
verifies consistency and deduplicates by NSC instead of treating rows as
independent response replicates:
- compounds.csv
- cell_lines.csv
- response_qc_metadata.csv
- response_matrix_loggi50.csv.gz
- response_long.csv.gz
- omics_rna.csv.gz
- omics_cnv.csv.gz
- omics_mutation.csv.gz
- preprocessing_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ACTIVITY_BINS = [-np.inf, 4.1, 5.0, 6.0, 7.0, 8.0, np.inf]
ACTIVITY_LABELS = [
    "inactive_lt4.1",
    "weak_4.1_5",
    "moderate_5_6",
    "potent_6_7",
    "very_potent_7_8",
    "super_ge8",
]
ACTIVITY_TO_ORDINAL = {label: i for i, label in enumerate(ACTIVITY_LABELS)}


def normalize_cellminer_column(column: object) -> str:
    """Remove CellMiner footnote suffixes from headers such as 'NSC # b'."""
    text = str(column).strip()
    return re.sub(r"\s+[a-z](?:\s*)$", "", text)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def find_cell_line_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if isinstance(col, str) and ":" in col]


def cancer_type_from_cell_line(cell_line: str) -> str:
    return cell_line.split(":", 1)[0]


def read_largest_html_table(path: Path) -> pd.DataFrame:
    tables = pd.read_html(path)
    if not tables:
        raise ValueError(f"No HTML tables found in {path}")
    index = max(range(len(tables)), key=lambda i: tables[i].shape[0] * tables[i].shape[1])
    return tables[index]


def clean_omics_table(path: Path, cell_lines: list[str], mode: str) -> pd.DataFrame:
    frame = read_largest_html_table(path)
    frame.columns = [normalize_cellminer_column(col) for col in frame.columns]
    require_columns(frame, ["Gene name", "Entrez gene id"], f"omics {mode}")

    available = [col for col in cell_lines if col in frame.columns]
    if len(available) != len(cell_lines):
        missing = sorted(set(cell_lines) - set(available))
        raise ValueError(f"omics {mode} missing cell-line columns: {missing}")

    cleaned = frame[["Gene name", "Entrez gene id", "Chromosome", "Start", "End", "Cytoband"] + cell_lines].copy()
    cleaned = cleaned.rename(
        columns={
            "Gene name": "gene_name",
            "Entrez gene id": "entrez_gene_id",
            "Chromosome": "chromosome",
            "Start": "start",
            "End": "end",
            "Cytoband": "cytoband",
        }
    )

    if mode == "mutation":
        for col in cell_lines:
            raw = cleaned[col].replace({"-": 0, "na": np.nan, "NA": np.nan, "": np.nan})
            numeric = pd.to_numeric(raw, errors="coerce")
            nonempty_text = raw.notna() & numeric.isna()
            numeric.loc[nonempty_text] = 1
            cleaned[col] = numeric.fillna(0).astype("int8")
    else:
        for col in cell_lines:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    return cleaned


def build_compound_table(zscore: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "NSC #",
        "Drug name",
        "FDA status",
        "Mechanism of action",
        "PubChem SID",
        "SMILES",
    ]
    require_columns(zscore, metadata_columns, "DTP_NCI60_ZSCORE")
    compounds = zscore[metadata_columns].copy()
    compounds = compounds.rename(
        columns={
            "NSC #": "nsc",
            "Drug name": "drug_name",
            "FDA status": "fda_status",
            "Mechanism of action": "mechanism_of_action",
            "PubChem SID": "pubchem_sid",
            "SMILES": "smiles",
        }
    )
    compounds["nsc"] = pd.to_numeric(compounds["nsc"], errors="raise").astype("int64")
    compounds = compounds.dropna(subset=["smiles"]).drop_duplicates("nsc").sort_values("nsc")
    return compounds


def build_response_tables(
    raw: pd.DataFrame,
    compounds: pd.DataFrame,
    cell_lines: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build one response profile per NSC and preserve source-level QC metadata.

    The CellMiner RAW workbook repeats the same exported 60-cell profile on
    multiple rows when an NSC has multiple experiment identifiers. Those rows
    must not be interpreted as independent response replicates. We verify that
    repeated profiles are identical after numeric coercion, then retain one
    profile per NSC. A hard failure is preferable to silently aggregating
    conflicting profiles if a future CellMiner release changes this behavior.
    """
    require_columns(raw, ["NSC #"], "DTP_NCI60_RAW")
    raw_subset = raw[raw["NSC #"].isin(set(compounds["nsc"]))].copy()
    raw_numeric = raw_subset[["NSC #"] + cell_lines].rename(columns={"NSC #": "nsc"})
    raw_numeric["nsc"] = pd.to_numeric(raw_numeric["nsc"], errors="raise").astype("int64")

    for col in cell_lines:
        raw_numeric[col] = pd.to_numeric(raw_numeric[col], errors="coerce")

    profile_hash = pd.util.hash_pandas_object(
        raw_numeric[cell_lines].fillna(np.finfo("float64").min),
        index=False,
    )
    profile_audit = pd.DataFrame(
        {"nsc": raw_numeric["nsc"].to_numpy(), "profile_hash": profile_hash.to_numpy()}
    )
    distinct_profiles = profile_audit.groupby("nsc", sort=True)["profile_hash"].nunique()
    conflicting_nsc = distinct_profiles[distinct_profiles > 1]
    if not conflicting_nsc.empty:
        examples = conflicting_nsc.index[:10].tolist()
        raise ValueError(
            "CellMiner RAW contains conflicting repeated profiles for "
            f"{len(conflicting_nsc)} NSCs; examples: {examples}"
        )

    unique_profiles = raw_numeric.drop_duplicates("nsc", keep="first")
    matrix = unique_profiles.set_index("nsc")[cell_lines].sort_index()
    matrix = matrix.reindex(compounds["nsc"].to_numpy())
    matrix.index.name = "nsc"

    long = matrix.reset_index().melt(id_vars="nsc", var_name="cell_line", value_name="loggi50")
    long = long.dropna(subset=["loggi50"]).reset_index(drop=True)
    long["cancer_type"] = long["cell_line"].map(cancer_type_from_cell_line)
    long["activity_class"] = pd.cut(
        long["loggi50"],
        bins=ACTIVITY_BINS,
        labels=ACTIVITY_LABELS,
        right=False,
    ).astype(str)
    long["ordinal_label"] = long["activity_class"].map(ACTIVITY_TO_ORDINAL).astype("int8")
    long["endpoint_audit_flag"] = np.select(
        [
            long["loggi50"].eq(4.0),
            long["loggi50"].eq(8.0),
            long["loggi50"].lt(4.0),
            long["loggi50"].gt(8.0),
        ],
        ["exact_standard_lower_4", "exact_standard_upper_8", "below_4", "above_8"],
        default="interior_4_8",
    )
    long["standard_boundary_candidate"] = long["loggi50"].isin([4.0, 8.0])
    long = long[
        [
            "nsc",
            "cell_line",
            "cancer_type",
            "loggi50",
            "activity_class",
            "ordinal_label",
            "endpoint_audit_flag",
            "standard_boundary_candidate",
        ]
    ]

    qc_columns = [
        "NSC #",
        "Total probes",
        "Total after quality control",
        "Failure reason",
        "Experiment name",
    ]
    require_columns(raw_subset, qc_columns, "DTP_NCI60_RAW")
    qc_source = raw_subset[qc_columns].copy()
    qc_source["NSC #"] = pd.to_numeric(qc_source["NSC #"], errors="raise").astype("int64")
    qc_first = qc_source.groupby("NSC #", sort=True).first()
    qc = pd.DataFrame(index=qc_first.index)
    qc.index.name = "nsc"
    qc["raw_row_count"] = qc_source.groupby("NSC #", sort=True).size().astype("int64")
    qc["experiment_name_count"] = (
        qc_source.groupby("NSC #", sort=True)["Experiment name"].nunique(dropna=True).astype("int64")
    )
    qc["total_probes"] = pd.to_numeric(qc_first["Total probes"], errors="coerce").astype("Int64")
    qc["total_after_quality_control"] = pd.to_numeric(
        qc_first["Total after quality control"], errors="coerce"
    ).astype("Int64")
    qc["failure_reason"] = qc_first["Failure reason"].fillna("").astype(str)
    qc["profile_rows_identical"] = distinct_profiles.reindex(qc.index).eq(1)
    qc["passed_cellminer_qc"] = qc["total_after_quality_control"].fillna(0).gt(0)
    qc = qc.reindex(compounds["nsc"].to_numpy()).reset_index()

    repeated_counts = raw_numeric.groupby("nsc", sort=True).size()
    response_audit = {
        "raw_rows_in_smiles_subset": int(len(raw_numeric)),
        "unique_nsc_in_smiles_subset": int(raw_numeric["nsc"].nunique()),
        "nsc_with_multiple_raw_rows": int((repeated_counts > 1).sum()),
        "nsc_with_identical_repeated_profiles": int(
            ((repeated_counts > 1) & distinct_profiles.eq(1)).sum()
        ),
        "nsc_with_conflicting_repeated_profiles": int(len(conflicting_nsc)),
        "interpretation": (
            "Repeated RAW rows carry identical exported 60-cell profiles and are "
            "deduplicated by NSC; they are not treated as independent response replicates."
        ),
    }
    return matrix, long, qc, response_audit


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root.")
    args = parser.parse_args()

    root = Path(args.root)
    raw_dir = root / "data" / "raw" / "cellminer" / "extracted"
    out_dir = root / "data" / "processed" / "cellminer"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "DTP_NCI60_RAW" / "output" / "DTP_NCI60_RAW.xlsx"
    zscore_path = raw_dir / "DTP_NCI60_ZSCORE" / "output" / "DTP_NCI60_ZSCORE.xlsx"
    rna_path = raw_dir / "nci60_RNA__RNA_seq_composite_expression" / "output" / "html" / "RNA__RNA_seq_composite_expression.html"
    cnv_path = raw_dir / "nci60_DNA__Combined_aCGH_gene_summary" / "output" / "html" / "DNA__Combined_aCGH_gene_summary.html"
    mutation_path = raw_dir / "nci60_DNA__Exome_Seq_Protein_function_affecting" / "output" / "html" / "DNA__Exome_Seq_Protein_function_affecting.html"

    raw = pd.read_excel(raw_path, sheet_name="all", header=9)
    zscore = pd.read_excel(zscore_path, sheet_name="all", header=8)
    zscore.columns = [normalize_cellminer_column(col) for col in zscore.columns]

    cell_lines = find_cell_line_columns(raw)
    if len(cell_lines) != 60:
        raise ValueError(f"Expected 60 NCI-60 cell lines, found {len(cell_lines)}")

    compounds = build_compound_table(zscore)
    response_matrix, response_long, response_qc, response_audit = build_response_tables(
        raw, compounds, cell_lines
    )

    cell_line_table = pd.DataFrame(
        {
            "cell_line": cell_lines,
            "cancer_type": [cancer_type_from_cell_line(cell) for cell in cell_lines],
            "valid_loggi50_labels": [int(response_matrix[cell].notna().sum()) for cell in cell_lines],
        }
    )

    rna = clean_omics_table(rna_path, cell_lines, mode="rna")
    cnv = clean_omics_table(cnv_path, cell_lines, mode="cnv")
    mutation = clean_omics_table(mutation_path, cell_lines, mode="mutation")

    compounds.to_csv(out_dir / "compounds.csv", index=False, encoding="utf-8")
    cell_line_table.to_csv(out_dir / "cell_lines.csv", index=False, encoding="utf-8")
    response_qc.to_csv(out_dir / "response_qc_metadata.csv", index=False, encoding="utf-8")
    response_matrix.to_csv(out_dir / "response_matrix_loggi50.csv.gz", compression="gzip")
    response_long.to_csv(out_dir / "response_long.csv.gz", index=False, compression="gzip")
    rna.to_csv(out_dir / "omics_rna.csv.gz", index=False, compression="gzip")
    cnv.to_csv(out_dir / "omics_cnv.csv.gz", index=False, compression="gzip")
    mutation.to_csv(out_dir / "omics_mutation.csv.gz", index=False, compression="gzip")

    class_counts = response_long["activity_class"].value_counts().reindex(ACTIVITY_LABELS).fillna(0).astype(int)
    valid_cells_per_compound = response_matrix.notna().sum(axis=1)
    report = {
        "source_files": {
            "raw": str(raw_path),
            "zscore": str(zscore_path),
            "rna": str(rna_path),
            "cnv": str(cnv_path),
            "mutation": str(mutation_path),
        },
        "raw": {
            "rows": int(raw.shape[0]),
            "columns": int(raw.shape[1]),
            "unique_nsc": int(raw["NSC #"].nunique()),
            "cell_lines": len(cell_lines),
        },
        "compounds": {
            "rows_with_smiles": int(compounds.shape[0]),
        },
        "responses": {
            "matrix_shape": [int(response_matrix.shape[0]), int(response_matrix.shape[1])],
            "valid_compound_cell_labels": int(response_matrix.notna().sum().sum()),
            "density": round(float(response_matrix.notna().sum().sum() / response_matrix.size), 6),
            "compounds_with_all_60_cell_labels": int((valid_cells_per_compound == 60).sum()),
            "compounds_with_at_least_56_cell_labels": int((valid_cells_per_compound >= 56).sum()),
            "compounds_with_at_least_30_cell_labels": int((valid_cells_per_compound >= 30).sum()),
            "median_valid_cells_per_compound": float(valid_cells_per_compound.median()),
            "activity_class_counts": class_counts.to_dict(),
            "activity_class_rates": {
                label: round(float(count / len(response_long)), 6)
                for label, count in class_counts.items()
            },
            "endpoint_audit_flag_counts": {
                key: int(value)
                for key, value in response_long["endpoint_audit_flag"].value_counts().items()
            },
            "standard_boundary_candidate_count": int(
                response_long["standard_boundary_candidate"].sum()
            ),
            "source_row_audit": response_audit,
        },
        "omics": {
            "rna_shape": [int(rna.shape[0]), int(rna.shape[1])],
            "cnv_shape": [int(cnv.shape[0]), int(cnv.shape[1])],
            "mutation_shape": [int(mutation.shape[0]), int(mutation.shape[1])],
            "cell_line_columns_aligned": True,
        },
        "outputs": {
            "compounds": str(out_dir / "compounds.csv"),
            "cell_lines": str(out_dir / "cell_lines.csv"),
            "response_qc_metadata": str(out_dir / "response_qc_metadata.csv"),
            "response_matrix_loggi50": str(out_dir / "response_matrix_loggi50.csv.gz"),
            "response_long": str(out_dir / "response_long.csv.gz"),
            "omics_rna": str(out_dir / "omics_rna.csv.gz"),
            "omics_cnv": str(out_dir / "omics_cnv.csv.gz"),
            "omics_mutation": str(out_dir / "omics_mutation.csv.gz"),
        },
    }
    write_json(out_dir / "preprocessing_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

