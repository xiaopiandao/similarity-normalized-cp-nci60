"""Prepare an assay-external HTS384 NCI-60 chemical lockbox.

Only HTS384 compounds absent from the classic DTP data by NSC, standardized
structure, and ECFP4 bit vector are retained.  Structures are resolved from the
PubChem SIDs supplied by the official CellMiner workbook and cached locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from rdkit import Chem, __version__ as rdkit_version
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

try:
    from scripts.make_chemical_splits import stable_id, structure_record
except ModuleNotFoundError:
    from make_chemical_splits import stable_id, structure_record  # type: ignore[no-redef]


def normalize_column(column: object) -> str:
    text = str(column).strip()
    for suffix in (" b", " c"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def chunks(values: list[int], size: int = 50):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_pubchem_json(url: str) -> dict:
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_pubchem_structures(frame: pd.DataFrame, cache: Path) -> pd.DataFrame:
    if cache.exists():
        cached = pd.read_csv(cache)
        required = set(frame["pubchem_sid"].astype(int))
        cached_required = cached.loc[cached["pubchem_sid"].astype(int).isin(required)]
        has_required = required.issubset(set(cached["pubchem_sid"].astype(int)))
        has_smiles = cached_required["smiles"].fillna("").astype(str).ne("").all()
        if has_required and has_smiles:
            return cached

    sids = sorted(frame["pubchem_sid"].astype(int).unique().tolist())
    sid_to_cid: dict[int, int] = {}
    for batch in chunks(sids):
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/"
            + ",".join(map(str, batch))
            + "/cids/JSON"
        )
        information = load_pubchem_json(url)["InformationList"]["Information"]
        for item in information:
            cids = item.get("CID", [])
            if cids:
                sid_to_cid[int(item["SID"])] = int(cids[0])

    cid_to_smiles: dict[int, str] = {}
    for batch in chunks(sorted(set(sid_to_cid.values()))):
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            + ",".join(map(str, batch))
            + "/property/CanonicalSMILES,IsomericSMILES/JSON"
        )
        properties = load_pubchem_json(url)["PropertyTable"]["Properties"]
        for item in properties:
            smiles = (
                item.get("IsomericSMILES")
                or item.get("CanonicalSMILES")
                or item.get("SMILES")
                or item.get("ConnectivitySMILES")
            )
            if smiles:
                cid_to_smiles[int(item["CID"])] = str(smiles)

    rows = []
    nsc_by_sid = frame.set_index("pubchem_sid")["nsc"].to_dict()
    for sid in sids:
        cid = sid_to_cid.get(sid)
        rows.append(
            {
                "nsc": int(nsc_by_sid[sid]),
                "pubchem_sid": sid,
                "pubchem_cid": cid,
                "smiles": cid_to_smiles.get(cid, "") if cid is not None else "",
                "retrieval_status": (
                    "resolved" if cid is not None and cid in cid_to_smiles else "unresolved"
                ),
            }
        )
    result = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache, index=False)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    raw_root = root / "data" / "raw" / "cellminer" / "hts384"
    extracted = raw_root / "extracted"
    processed = root / "data" / "processed" / "cellminer_hts384"
    feature_dir = root / "data" / "features" / "cellminer_hts384"
    processed.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    raw_path = (
        extracted
        / "DTP_HTS384_NCI60_RAW"
        / "output"
        / "DTP_HTS384_NCI60_RAW.xlsx"
    )
    z_path = (
        extracted
        / "DTP_HTS384_NCI60_ZSCORE"
        / "output"
        / "DTP_HTS384_NCI60_ZSCORE.xlsx"
    )
    raw = pd.read_excel(raw_path, header=9, na_values=["na", "NA", "-"])
    zscore = pd.read_excel(z_path, header=8, na_values=["na", "NA", "-"])
    raw.columns = [normalize_column(value) for value in raw.columns]
    zscore.columns = [normalize_column(value) for value in zscore.columns]
    raw = raw.rename(columns={"NSC #": "nsc", "PubChem SID": "pubchem_sid"})
    zscore = zscore.rename(columns={"NSC #": "nsc", "PubChem SID": "pubchem_sid"})
    raw["nsc"] = pd.to_numeric(raw["nsc"], errors="raise").astype("int64")
    zscore["nsc"] = pd.to_numeric(zscore["nsc"], errors="raise").astype("int64")
    zscore["pubchem_sid"] = pd.to_numeric(zscore["pubchem_sid"], errors="raise").astype("int64")

    classic_structures = pd.read_csv(
        root / "data" / "processed" / "cellminer" / "chemical_structures.csv"
    )
    classic_groups = pd.read_csv(
        root / "data" / "processed" / "cellminer" / "chemical_split_groups.csv"
    )
    classic_ids = set(classic_structures["nsc"].astype(int))
    new = zscore.loc[~zscore["nsc"].isin(classic_ids), ["nsc", "pubchem_sid"]].copy()
    cache = processed / "pubchem_structure_cache.csv"
    pubchem = fetch_pubchem_structures(new, cache)
    records = [
        structure_record(int(row.nsc), row.smiles)
        for row in pubchem[["nsc", "smiles"]].itertuples(index=False)
    ]
    structures = pd.DataFrame(records)
    structures = structures.merge(
        pubchem[["nsc", "pubchem_sid", "pubchem_cid", "retrieval_status"]],
        on="nsc",
        how="left",
        validate="one_to_one",
    )

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_strings = []
    for row in structures.itertuples(index=False):
        if not row.structure_valid:
            fp_strings.append("")
            continue
        molecule = Chem.MolFromSmiles(row.standardized_smiles)
        fp_strings.append(generator.GetFingerprint(molecule).ToBitString())
    structures["morgan_fp_group_id"] = [
        stable_id("morgan", value) if value else "" for value in fp_strings
    ]
    classic_standardized = set(
        classic_structures.loc[
            classic_structures["structure_valid"], "standardized_smiles"
        ].astype(str)
    )
    classic_fingerprints = set(classic_groups["morgan_fp_group_id"].astype(str))
    structures["classic_structure_overlap"] = structures["standardized_smiles"].isin(
        classic_standardized
    )
    structures["classic_fingerprint_overlap"] = structures["morgan_fp_group_id"].isin(
        classic_fingerprints
    )
    structures["lockbox_eligible"] = (
        structures["structure_valid"]
        & ~structures["classic_structure_overlap"]
        & ~structures["classic_fingerprint_overlap"]
    )
    structures.to_csv(processed / "hts384_structure_audit.csv", index=False)

    eligible = structures.loc[structures["lockbox_eligible"]].copy()
    eligible = eligible.sort_values("nsc").drop_duplicates("standardized_smiles", keep="first")
    cell_lines = [column for column in raw.columns if isinstance(column, str) and ":" in column]
    classic_cell_lines = pd.read_csv(
        root / "data" / "processed" / "cellminer" / "cell_lines.csv"
    )["cell_line"].tolist()
    if cell_lines != classic_cell_lines:
        raise ValueError("HTS384 and classic NCI-60 cell-line columns do not match exactly")
    response = raw.set_index("nsc").loc[eligible["nsc"], cell_lines].apply(
        pd.to_numeric, errors="coerce"
    )
    response.index.name = "nsc"
    response.to_csv(processed / "response_matrix_loggi50.csv.gz", compression="gzip")
    eligible.to_csv(processed / "lockbox_compounds.csv", index=False)

    rows, columns = [], []
    for row_index, smiles in enumerate(eligible["standardized_smiles"]):
        molecule = Chem.MolFromSmiles(smiles)
        fingerprint = generator.GetFingerprint(molecule)
        bits = list(fingerprint.GetOnBits())
        rows.extend([row_index] * len(bits))
        columns.extend(bits)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(eligible), 2048),
        dtype=np.uint8,
    )
    sparse.save_npz(feature_dir / "morgan_r2_2048.npz", matrix, compressed=True)
    eligible[["nsc", "standardized_smiles"]].to_csv(
        feature_dir / "morgan_r2_2048_index.csv", index=False
    )

    observed = np.isfinite(response.to_numpy(float))
    if len(eligible) == 0 or observed.sum() == 0:
        raise ValueError("HTS384 lockbox is empty after NSC, structure, and fingerprint exclusion")
    report = {
        "status": "prepared_assay_external_chemical_lockbox",
        "source": "NCI CellMiner DTP HTS384 NCI-60, database v2.15, 2025-09-17",
        "source_assay": "3-day 384-well luminescence GI50",
        "training_assay": "classic 48-hour sulforhodamine B NCI-60 GI50",
        "raw_compounds": int(raw["nsc"].nunique()),
        "qc_passed_compounds": int(zscore["nsc"].nunique()),
        "qc_passed_absent_by_nsc_from_classic": int(len(new)),
        "pubchem_resolved": int((pubchem["retrieval_status"] == "resolved").sum()),
        "valid_structures": int(structures["structure_valid"].sum()),
        "excluded_classic_structure_overlap": int(
            structures["classic_structure_overlap"].sum()
        ),
        "excluded_classic_fingerprint_overlap": int(
            structures["classic_fingerprint_overlap"].sum()
        ),
        "final_unique_lockbox_compounds": int(len(eligible)),
        "observed_labels": int(observed.sum()),
        "label_fraction": float(observed.mean()),
        "exact_4_rate": float(np.isclose(response.to_numpy(float), 4.0, equal_nan=False).sum() / observed.sum()),
        "exact_8_rate": float(np.isclose(response.to_numpy(float), 8.0, equal_nan=False).sum() / observed.sum()),
        "rdkit_version": rdkit_version,
        "guardrail": (
            "This is an assay-external lockbox, not a temporal lockbox. Cross-platform "
            "endpoint differences are part of the evaluated shift."
        ),
    }
    (processed / "lockbox_preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    source_files = (
        raw_root / "DTP_HTS384_NCI60_RAW.zip",
        raw_root / "DTP_HTS384_NCI60_ZSCORE.zip",
        root / "scripts" / "prepare_hts384_lockbox.py",
        cache,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_urls": [
            "https://discover.nci.nih.gov/cellminer/download/rawdataset/DTP_HTS384_NCI60_RAW.zip",
            "https://discover.nci.nih.gov/cellminer/download/processeddataset/DTP_HTS384_NCI60_ZSCORE.zip",
        ],
        "files_sha256": {
            str(path.relative_to(root)): sha256(path) for path in source_files if path.exists()
        },
    }
    (processed / "lockbox_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

