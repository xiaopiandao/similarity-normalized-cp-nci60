from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "cellminer"
SPLITS = ROOT / "data" / "splits" / "cellminer"


@unittest.skipUnless(
    (PROCESSED / "response_long.csv.gz").exists(),
    "processed CellMiner outputs are not included in the code-only archive",
)
class Phase0OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.response = pd.read_csv(PROCESSED / "response_long.csv.gz")
        cls.qc = pd.read_csv(PROCESSED / "response_qc_metadata.csv")
        cls.structures = pd.read_csv(PROCESSED / "chemical_structures.csv")
        cls.groups = pd.read_csv(PROCESSED / "chemical_split_groups.csv")

    def test_response_shape_and_uniqueness(self) -> None:
        self.assertEqual(len(self.response), 1_430_158)
        self.assertEqual(self.response["nsc"].nunique(), 25_731)
        self.assertEqual(self.response["cell_line"].nunique(), 60)
        self.assertFalse(self.response.duplicated(["nsc", "cell_line"]).any())

    def test_source_profile_qc(self) -> None:
        self.assertEqual(len(self.qc), 25_731)
        self.assertTrue(self.qc["profile_rows_identical"].all())
        self.assertTrue(self.qc["passed_cellminer_qc"].all())
        self.assertEqual(int((self.qc["raw_row_count"] > 1).sum()), 11_366)

    def test_endpoint_status_is_not_overclaimed(self) -> None:
        report = json.loads((PROCESSED / "endpoint_boundary_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(
            report["censoring_identifiability"]["status"],
            "not_identifiable_from_current_cellminer_export",
        )
        self.assertEqual(report["distribution"]["exact_4"]["count"], 213_634)
        self.assertEqual(report["distribution"]["exact_8"]["count"], 25_745)

    def test_structure_accounting(self) -> None:
        self.assertEqual(len(self.structures), 25_731)
        self.assertEqual(int(self.structures["structure_valid"].sum()), 25_245)
        self.assertEqual(len(self.groups), 25_245)
        self.assertTrue(self.groups["nsc"].is_unique)

    def test_split_families_have_no_entity_or_representation_leakage(self) -> None:
        group_index = self.groups.set_index("nsc")
        expected = set(self.groups["nsc"].astype(int))
        family_group_columns = {
            "random": "morgan_fp_group_id",
            "scaffold": "scaffold_component_id",
            "leader_cluster": "leader_cluster_id",
        }
        for family, group_column in family_group_columns.items():
            for seed in range(1, 6):
                split_dir = SPLITS / f"{family}_seed_{seed}"
                parts = {
                    name: set(
                        pd.read_csv(split_dir / f"{name}_nsc.csv")["nsc"].astype(int)
                    )
                    for name in ("train", "valid", "test")
                }
                self.assertFalse(parts["train"] & parts["valid"])
                self.assertFalse(parts["train"] & parts["test"])
                self.assertFalse(parts["valid"] & parts["test"])
                self.assertEqual(set().union(*parts.values()), expected)

                for leakage_column in (
                    "structure_group_id",
                    "morgan_fp_group_id",
                    group_column,
                ):
                    values = {
                        name: set(group_index.loc[sorted(ids), leakage_column].astype(str))
                        for name, ids in parts.items()
                    }
                    self.assertFalse(values["train"] & values["valid"])
                    self.assertFalse(values["train"] & values["test"])
                    self.assertFalse(values["valid"] & values["test"])


if __name__ == "__main__":
    unittest.main()
