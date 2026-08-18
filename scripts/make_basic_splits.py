"""Create dependency-light completeness subsets for CellMiner preprocessing.

This script intentionally avoids RDKit/sklearn. Its legacy NSC-only random
split can leak identical structures or identical fingerprints and is disabled
by default. Use make_chemical_splits.py for modeling splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def split_ids(ids: np.ndarray, seed: int, train_frac: float, valid_frac: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(round(n_total * train_frac))
    n_valid = int(round(n_total * valid_frac))
    return {
        "train": np.sort(shuffled[:n_train]),
        "valid": np.sort(shuffled[n_train : n_train + n_valid]),
        "test": np.sort(shuffled[n_train + n_valid :]),
    }


def write_id_list(path: Path, ids: np.ndarray) -> None:
    pd.DataFrame({"nsc": ids.astype("int64")}).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument(
        "--allow-legacy-nsc-only-random",
        action="store_true",
        help=(
            "Write legacy NSC-only random splits under random_nsc_only_seed_*. "
            "These are not approved for primary modeling because identical "
            "structures/fingerprints can cross partitions."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root)
    processed = root / "data" / "processed" / "cellminer"
    split_root = root / "data" / "splits" / "cellminer"
    split_root.mkdir(parents=True, exist_ok=True)

    compounds = pd.read_csv(processed / "compounds.csv")
    response_matrix = pd.read_csv(processed / "response_matrix_loggi50.csv.gz")
    response_matrix = response_matrix.rename(columns={response_matrix.columns[0]: "nsc"})
    response_matrix["nsc"] = response_matrix["nsc"].astype("int64")

    valid_counts = response_matrix.drop(columns=["nsc"]).notna().sum(axis=1)
    subsets = {
        "main_all_smiles": response_matrix["nsc"].to_numpy(dtype="int64"),
        "high_complete_ge56": response_matrix.loc[valid_counts >= 56, "nsc"].to_numpy(dtype="int64"),
        "complete_profile_60": response_matrix.loc[valid_counts == 60, "nsc"].to_numpy(dtype="int64"),
    }

    subsets_dir = split_root / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in subsets.items():
        write_id_list(subsets_dir / f"{name}_nsc.csv", np.sort(ids))

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    split_metadata = {
        "split_type": "completeness_subsets",
        "train_frac": args.train_frac,
        "valid_frac": args.valid_frac,
        "test_frac": round(1.0 - args.train_frac - args.valid_frac, 6),
        "seeds": seeds,
        "subsets": {name: int(len(ids)) for name, ids in subsets.items()},
        "note": (
            "Use make_chemical_splits.py for modeling splits. Legacy NSC-only "
            "random partitions are disabled by default."
        ),
    }

    if args.allow_legacy_nsc_only_random:
        all_ids = np.sort(compounds["nsc"].to_numpy(dtype="int64"))
        for seed in seeds:
            split_dir = split_root / f"random_nsc_only_seed_{seed}"
            split_dir.mkdir(parents=True, exist_ok=True)
            parts = split_ids(all_ids, seed, args.train_frac, args.valid_frac)
            for part, ids in parts.items():
                write_id_list(split_dir / f"{part}_nsc.csv", ids)
            write_id_list(split_dir / "all_nsc.csv", all_ids)
            with (split_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        **split_metadata,
                        "split_type": "legacy_nsc_only_random",
                        "approved_for_primary_modeling": False,
                        "seed": seed,
                        "counts": {part: int(len(ids)) for part, ids in parts.items()},
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

    print(json.dumps(split_metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

