"""Canonicalize CellMiner structures and create leakage-controlled splits.

Outputs random, Bemis-Murcko scaffold, and Tanimoto leader-cluster splits. All
split types keep identical standardized structures in one partition. The
leader-cluster method is a scalable sphere-exclusion alternative to the
quadratic-memory Butina implementation for ~25K compounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, __version__ as rdkit_version
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters.rdSimDivPickers import LeaderPicker


RDLogger.DisableLog("rdApp.error")


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def structure_record(nsc: int, smiles: object) -> dict:
    original = "" if pd.isna(smiles) else str(smiles).strip()
    base = {
        "nsc": int(nsc),
        "original_smiles": original,
        "structure_valid": False,
        "structure_error": "",
        "canonical_smiles": "",
        "standardized_smiles": "",
        "structure_group_id": "",
        "num_fragments": 0,
        "heavy_atom_count": np.nan,
        "molecular_weight": np.nan,
        "formal_charge": np.nan,
        "murcko_scaffold_smiles": "",
        "generic_scaffold_smiles": "",
        "generic_scaffold_error": "",
        "scaffold_split_key": "",
        "scaffold_fallback_used": False,
    }
    if not original:
        base["structure_error"] = "empty_smiles"
        return base

    mol = Chem.MolFromSmiles(original)
    if mol is None:
        if original in {"-", "na", "NA"}:
            base["structure_error"] = "missing_structure_placeholder"
        elif "[R]" in original or "[RH]" in original:
            base["structure_error"] = "unresolved_r_group"
        else:
            base["structure_error"] = "rdkit_parse_failed"
        return base

    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if not fragments:
        base["structure_error"] = "no_sanitized_fragments"
        return base

    largest = max(
        fragments,
        key=lambda fragment: (fragment.GetNumHeavyAtoms(), Descriptors.MolWt(fragment)),
    )
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    standardized = Chem.MolToSmiles(largest, canonical=True, isomericSmiles=True)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(largest)
    scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=True)
    if scaffold:
        generic_error = ""
        try:
            generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
            generic_scaffold = Chem.MolToSmiles(
                generic_mol, canonical=True, isomericSmiles=False
            )
        except Exception as exc:  # RDKit may reject genericized organometallic valence.
            generic_scaffold = ""
            generic_error = f"{type(exc).__name__}: {exc}"
        scaffold_key = f"MURCKO::{scaffold}"
        fallback = False
    else:
        generic_scaffold = ""
        generic_error = ""
        # Acyclic molecules have an empty Bemis-Murcko scaffold. Grouping all
        # of them together creates one pathological giant group, so identical
        # standardized acyclic structures are grouped while distinct acyclic
        # structures receive distinct fallback keys. This policy is recorded.
        scaffold_key = f"ACYCLIC_STRUCTURE::{standardized}"
        fallback = True

    base.update(
        {
            "structure_valid": True,
            "canonical_smiles": canonical,
            "standardized_smiles": standardized,
            "structure_group_id": stable_id("structure", standardized),
            "num_fragments": len(fragments),
            "heavy_atom_count": int(largest.GetNumHeavyAtoms()),
            "molecular_weight": float(Descriptors.MolWt(largest)),
            "formal_charge": int(Chem.GetFormalCharge(largest)),
            "murcko_scaffold_smiles": scaffold,
            "generic_scaffold_smiles": generic_scaffold,
            "generic_scaffold_error": generic_error,
            "scaffold_split_key": scaffold_key,
            "scaffold_fallback_used": fallback,
        }
    )
    return base


def grouped_partition(
    frame: pd.DataFrame,
    group_column: str,
    seed: int,
    train_frac: float,
    valid_frac: float,
) -> dict[str, np.ndarray]:
    fractions = {
        "train": train_frac,
        "valid": valid_frac,
        "test": 1.0 - train_frac - valid_frac,
    }
    if min(fractions.values()) <= 0:
        raise ValueError(f"Invalid split fractions: {fractions}")

    grouped = [group["nsc"].to_numpy(dtype="int64") for _, group in frame.groupby(group_column)]
    rng = np.random.default_rng(seed)
    tie_breakers = rng.random(len(grouped))
    ordered = [
        grouped[index]
        for index in sorted(
            range(len(grouped)),
            key=lambda index: (-len(grouped[index]), tie_breakers[index]),
        )
    ]

    target = {name: len(frame) * fraction for name, fraction in fractions.items()}
    assigned: dict[str, list[np.ndarray]] = {name: [] for name in fractions}
    counts = {name: 0 for name in fractions}
    for ids in ordered:
        destination = min(
            fractions,
            key=lambda name: ((counts[name] + len(ids)) / target[name], counts[name]),
        )
        assigned[destination].append(ids)
        counts[destination] += len(ids)

    return {
        name: np.sort(np.concatenate(groups) if groups else np.array([], dtype="int64"))
        for name, groups in assigned.items()
    }


def write_id_list(path: Path, ids: np.ndarray) -> None:
    pd.DataFrame({"nsc": ids.astype("int64")}).to_csv(path, index=False)


def assert_partition(parts: dict[str, np.ndarray], expected: set[int]) -> None:
    sets = {name: set(ids.tolist()) for name, ids in parts.items()}
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(f"NSC leakage between {left} and {right}: {len(overlap)}")
    combined = set().union(*sets.values())
    if combined != expected:
        raise AssertionError(
            f"Partition coverage mismatch: expected {len(expected)}, observed {len(combined)}"
        )


def group_overlap_audit(
    frame: pd.DataFrame,
    parts: dict[str, np.ndarray],
    group_column: str,
) -> dict[str, int]:
    group_sets = {}
    indexed = frame.set_index("nsc")
    for part, ids in parts.items():
        group_sets[part] = set(indexed.loc[ids, group_column].astype(str).tolist())
    return {
        "train_valid": len(group_sets["train"] & group_sets["valid"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "valid_test": len(group_sets["valid"] & group_sets["test"]),
    }


def assign_leader_clusters(
    frame: pd.DataFrame,
    distance_threshold: float,
    radius: int,
    fp_size: int,
) -> tuple[pd.Series, list, list[int]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    molecules = [Chem.MolFromSmiles(smiles) for smiles in frame["standardized_smiles"]]
    fingerprints = [generator.GetFingerprint(mol) for mol in molecules]

    picker = LeaderPicker()
    leader_indices = list(
        picker.LazyBitVectorPick(
            fingerprints,
            len(fingerprints),
            distance_threshold,
        )
    )
    leader_fingerprints = [fingerprints[index] for index in leader_indices]
    leader_nsc = frame.iloc[leader_indices]["nsc"].astype("int64").tolist()

    cluster_ids = []
    for fingerprint in fingerprints:
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, leader_fingerprints)
        best = int(np.argmax(similarities))
        cluster_ids.append(f"leader_{leader_nsc[best]}")
    return pd.Series(cluster_ids, index=frame.index), fingerprints, leader_indices


def merge_primary_groups_by_constraint(
    frame: pd.DataFrame,
    primary_column: str,
    constraint_column: str,
    output_prefix: str,
) -> pd.Series:
    """Merge primary groups connected by any shared leakage-constraint group."""
    primary_values = frame[primary_column].astype(str).tolist()
    parent = {value: value for value in set(primary_values)}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

    for _, group in frame.groupby(constraint_column, sort=False):
        keys = sorted(set(group[primary_column].astype(str)))
        for key in keys[1:]:
            union(keys[0], key)

    roots = frame[primary_column].astype(str).map(find)
    return roots.map(lambda value: stable_id(output_prefix, value))


def nearest_neighbor_audit(
    frame: pd.DataFrame,
    fingerprints: list,
    parts: dict[str, np.ndarray],
    sample_size: int,
    seed: int,
) -> dict:
    positions = {int(nsc): index for index, nsc in enumerate(frame["nsc"].tolist())}
    train_fps = [fingerprints[positions[int(nsc)]] for nsc in parts["train"]]
    test_ids = parts["test"].copy()
    rng = np.random.default_rng(seed)
    if len(test_ids) > sample_size:
        test_ids = np.sort(rng.choice(test_ids, size=sample_size, replace=False))
    maxima = []
    for nsc in test_ids:
        fp = fingerprints[positions[int(nsc)]]
        maxima.append(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    values = np.asarray(maxima, dtype=float)
    return {
        "sample_size": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def write_split_family(
    split_root: Path,
    prefix: str,
    frame: pd.DataFrame,
    group_column: str,
    seeds: list[int],
    train_frac: float,
    valid_frac: float,
    metadata_base: dict,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict]]:
    expected = set(frame["nsc"].astype(int).tolist())
    partitions = {}
    audits = {}
    for seed in seeds:
        parts = grouped_partition(frame, group_column, seed, train_frac, valid_frac)
        assert_partition(parts, expected)
        structure_overlap = group_overlap_audit(frame, parts, "structure_group_id")
        fingerprint_overlap = group_overlap_audit(frame, parts, "morgan_fp_group_id")
        grouping_overlap = group_overlap_audit(frame, parts, group_column)
        if any(grouping_overlap.values()):
            raise AssertionError(f"{prefix} group leakage for seed {seed}: {grouping_overlap}")
        if any(structure_overlap.values()):
            raise AssertionError(f"Identical-structure leakage for seed {seed}: {structure_overlap}")
        if any(fingerprint_overlap.values()):
            raise AssertionError(
                f"Identical-Morgan-fingerprint leakage for seed {seed}: {fingerprint_overlap}"
            )

        split_dir = split_root / f"{prefix}_seed_{seed}"
        split_dir.mkdir(parents=True, exist_ok=True)
        for part, ids in parts.items():
            write_id_list(split_dir / f"{part}_nsc.csv", ids)
        write_id_list(split_dir / "all_nsc.csv", np.sort(frame["nsc"].to_numpy(dtype="int64")))
        audit = {
            "counts": {part: int(len(ids)) for part, ids in parts.items()},
            "fractions": {part: float(len(ids) / len(frame)) for part, ids in parts.items()},
            "structure_group_overlap": structure_overlap,
            "morgan_fingerprint_group_overlap": fingerprint_overlap,
            "split_group_overlap": grouping_overlap,
        }
        metadata = {**metadata_base, "seed": seed, **audit}
        (split_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        partitions[seed] = parts
        audits[seed] = audit
    return partitions, audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument(
        "--leader-distance-threshold",
        type=float,
        default=0.6,
        help="Minimum Tanimoto distance between leaders; 0.6 corresponds to similarity 0.4.",
    )
    parser.add_argument("--nn-audit-sample-size", type=int, default=500)
    args = parser.parse_args()

    root = Path(args.root)
    processed = root / "data" / "processed" / "cellminer"
    split_root = root / "data" / "splits" / "cellminer"
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "subsets").mkdir(parents=True, exist_ok=True)
    compounds = pd.read_csv(processed / "compounds.csv")

    records = [
        structure_record(int(row.nsc), row.smiles)
        for row in compounds[["nsc", "smiles"]].itertuples(index=False)
    ]
    structures = pd.DataFrame.from_records(records).sort_values("nsc").reset_index(drop=True)
    structures.to_csv(processed / "chemical_structures.csv", index=False)
    invalid = structures.loc[~structures["structure_valid"]].copy()
    invalid.to_csv(processed / "invalid_chemical_structures.csv", index=False)
    valid = structures.loc[structures["structure_valid"]].copy().reset_index(drop=True)
    if valid.empty:
        raise ValueError("No valid RDKit structures were found")

    valid["leader_cluster_id"], fingerprints, leader_indices = assign_leader_clusters(
        valid,
        args.leader_distance_threshold,
        args.radius,
        args.fp_size,
    )
    valid["morgan_fp_group_id"] = [
        stable_id("morgan", fingerprint.ToBitString()) for fingerprint in fingerprints
    ]
    valid["scaffold_component_id"] = merge_primary_groups_by_constraint(
        valid,
        "scaffold_split_key",
        "morgan_fp_group_id",
        "scaffold_component",
    )
    valid[
        [
            "nsc",
            "structure_group_id",
            "morgan_fp_group_id",
            "scaffold_split_key",
            "scaffold_component_id",
            "leader_cluster_id",
        ]
    ].to_csv(
        processed / "chemical_split_groups.csv", index=False
    )
    write_id_list(
        split_root / "subsets" / "chemical_valid_nsc.csv",
        np.sort(valid["nsc"].to_numpy(dtype="int64")),
    )

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    common = {
        "train_frac": args.train_frac,
        "valid_frac": args.valid_frac,
        "test_frac": 1.0 - args.train_frac - args.valid_frac,
        "valid_structures": int(len(valid)),
        "excluded_invalid_structures": int(len(invalid)),
        "identical_standardized_structures_are_grouped": True,
        "identical_morgan_fingerprints_are_grouped": True,
        "rdkit_version": rdkit_version,
    }

    random_parts, random_audits = write_split_family(
        split_root,
        "random",
        valid,
        "morgan_fp_group_id",
        seeds,
        args.train_frac,
        args.valid_frac,
        {
            **common,
            "split_type": "random_grouped_by_identical_morgan_fingerprint",
            "morgan_radius": args.radius,
            "fingerprint_size": args.fp_size,
        },
    )
    scaffold_parts, scaffold_audits = write_split_family(
        split_root,
        "scaffold",
        valid,
        "scaffold_component_id",
        seeds,
        args.train_frac,
        args.valid_frac,
        {
            **common,
            "split_type": "bemis_murcko_scaffold",
            "acyclic_policy": "group_identical_standardized_structures_only",
            "fingerprint_constraint": (
                "Scaffold groups connected by an identical Morgan fingerprint are merged."
            ),
        },
    )
    cluster_parts, cluster_audits = write_split_family(
        split_root,
        "leader_cluster",
        valid,
        "leader_cluster_id",
        seeds,
        args.train_frac,
        args.valid_frac,
        {
            **common,
            "split_type": "tanimoto_leader_sphere_exclusion",
            "morgan_radius": args.radius,
            "fingerprint_size": args.fp_size,
            "leader_distance_threshold": args.leader_distance_threshold,
            "leader_similarity_radius": 1.0 - args.leader_distance_threshold,
            "cluster_count": int(valid["leader_cluster_id"].nunique()),
        },
    )

    nn_audits = {
        "random_seed_1": nearest_neighbor_audit(
            valid, fingerprints, random_parts[seeds[0]], args.nn_audit_sample_size, 20260720
        ),
        "scaffold_seed_1": nearest_neighbor_audit(
            valid, fingerprints, scaffold_parts[seeds[0]], args.nn_audit_sample_size, 20260720
        ),
        "leader_cluster_seed_1": nearest_neighbor_audit(
            valid, fingerprints, cluster_parts[seeds[0]], args.nn_audit_sample_size, 20260720
        ),
    }

    duplicate_sizes = valid.groupby("structure_group_id").size()
    fingerprint_duplicate_sizes = valid.groupby("morgan_fp_group_id").size()
    scaffold_sizes = valid.groupby("scaffold_split_key").size()
    scaffold_component_sizes = valid.groupby("scaffold_component_id").size()
    cluster_sizes = valid.groupby("leader_cluster_id").size()
    report = {
        "source_compounds": int(len(compounds)),
        "valid_structures": int(len(valid)),
        "invalid_structures": int(len(invalid)),
        "invalid_structure_reasons": {
            key: int(value) for key, value in invalid["structure_error"].value_counts().items()
        },
        "structure_standardization": {
            "canonical_smiles": "full RDKit canonical isomeric SMILES",
            "standardized_smiles": "largest sanitized fragment by heavy atoms, then molecular weight",
            "identical_structure_groups": int(valid["structure_group_id"].nunique()),
            "groups_with_multiple_nsc": int((duplicate_sizes > 1).sum()),
            "largest_identical_structure_group": int(duplicate_sizes.max()),
            "identical_morgan_fingerprint_groups": int(
                valid["morgan_fp_group_id"].nunique()
            ),
            "morgan_fingerprint_groups_with_multiple_nsc": int(
                (fingerprint_duplicate_sizes > 1).sum()
            ),
            "largest_identical_morgan_fingerprint_group": int(
                fingerprint_duplicate_sizes.max()
            ),
            "multi_fragment_compounds": int(valid["num_fragments"].gt(1).sum()),
        },
        "scaffolds": {
            "split_groups": int(valid["scaffold_split_key"].nunique()),
            "constraint_merged_split_groups": int(
                valid["scaffold_component_id"].nunique()
            ),
            "acyclic_fallback_compounds": int(valid["scaffold_fallback_used"].sum()),
            "largest_group": int(scaffold_sizes.max()),
            "largest_constraint_merged_group": int(scaffold_component_sizes.max()),
            "generic_scaffold_failures": int(valid["generic_scaffold_error"].ne("").sum()),
        },
        "leader_clustering": {
            "method": "RDKit LeaderPicker sphere exclusion + nearest-leader assignment",
            "morgan_radius": args.radius,
            "fingerprint_size": args.fp_size,
            "distance_threshold": args.leader_distance_threshold,
            "similarity_radius": 1.0 - args.leader_distance_threshold,
            "leader_count": int(len(leader_indices)),
            "cluster_count": int(valid["leader_cluster_id"].nunique()),
            "largest_cluster": int(cluster_sizes.max()),
        },
        "split_audits": {
            "random": random_audits,
            "scaffold": scaffold_audits,
            "leader_cluster": cluster_audits,
        },
        "sampled_test_to_train_nearest_neighbor_tanimoto": nn_audits,
        "outputs": {
            "chemical_structures": str(processed / "chemical_structures.csv"),
            "invalid_structures": str(processed / "invalid_chemical_structures.csv"),
            "split_groups": str(processed / "chemical_split_groups.csv"),
            "split_root": str(split_root),
        },
    }
    (processed / "chemical_structure_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    split_root_metadata = {
        "approved_split_families": ["random", "scaffold", "leader_cluster"],
        "seeds": seeds,
        "valid_structures": int(len(valid)),
        "excluded_invalid_structures": int(len(invalid)),
        "primary_ood_split": "leader_cluster",
        "random_split_policy": "group_identical_morgan_fingerprints",
        "scaffold_split_policy": (
            "Bemis-Murcko groups with acyclic structure fallback; groups connected "
            "by identical Morgan fingerprints are merged."
        ),
        "leakage_tests": "passed during split generation; rerun tests/test_phase0_outputs.py",
        "details": str(processed / "chemical_structure_report.json"),
    }
    (split_root / "metadata.json").write_text(
        json.dumps(split_root_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

