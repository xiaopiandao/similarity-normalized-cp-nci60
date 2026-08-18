from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "splits" / "phase1a"
FEATURES = ROOT / "data" / "features" / "cellminer"
GROUPS_PATH = ROOT / "data" / "processed" / "cellminer" / "chemical_split_groups.csv"
GROUPS = pd.read_csv(GROUPS_PATH) if GROUPS_PATH.exists() else None


class Phase1APreparationTests(unittest.TestCase):
    @unittest.skipUnless(
        (FEATURES / "morgan_r2_2048.npz").exists()
        and (FEATURES / "morgan_r2_2048_index.csv").exists(),
        "processed Morgan feature cache is not included in the code-only archive",
    )
    def test_feature_cache_alignment(self) -> None:
        matrix = sparse.load_npz(FEATURES / "morgan_r2_2048.npz")
        index = pd.read_csv(FEATURES / "morgan_r2_2048_index.csv")
        self.assertEqual(matrix.shape, (25_245, 2048))
        self.assertEqual(len(index), matrix.shape[0])
        self.assertTrue(index["nsc"].is_unique)
        self.assertGreater(matrix.nnz, 0)

    @unittest.skipUnless(
        GROUPS_PATH.exists(),
        "processed chemical group table is not included in the code-only archive",
    )
    def test_all_source_calibration_splits(self) -> None:
        assert GROUPS is not None
        group_index = GROUPS.set_index("nsc")
        for family in ("random", "scaffold", "leader_cluster"):
            for seed in range(1, 6):
                folder = PHASE1 / f"{family}_seed_{seed}"
                parts = {
                    name: set(pd.read_csv(folder / f"{name}_nsc.csv")["nsc"].astype(int))
                    for name in ("fit", "source_cal", "valid", "test")
                }
                names = list(parts)
                for i, left in enumerate(names):
                    for right in names[i + 1 :]:
                        self.assertFalse(parts[left] & parts[right])
                self.assertEqual(len(set().union(*parts.values())), 25_245)
                fit_groups = set(
                    group_index.loc[sorted(parts["fit"]), "morgan_fp_group_id"].astype(str)
                )
                cal_groups = set(
                    group_index.loc[sorted(parts["source_cal"]), "morgan_fp_group_id"].astype(str)
                )
                self.assertFalse(fit_groups & cal_groups)

    def test_metadata_policy(self) -> None:
        metadata = json.loads((PHASE1 / "metadata.json").read_text(encoding="utf-8"))
        self.assertIn("test labels stay locked", metadata["transductive_policy"])


if __name__ == "__main__":
    unittest.main()
