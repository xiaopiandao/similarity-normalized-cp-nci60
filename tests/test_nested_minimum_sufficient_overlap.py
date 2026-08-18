import numpy as np
import pandas as pd

from scripts.evaluate_overlap_shrunk_factorized_wcp import (
    CURRENT_METHOD,
    SHRUNK_METHOD,
)
from scripts.nested_validate_minimum_sufficient_overlap import heldout_unit_rows


def test_heldout_rows_use_requested_kappa_and_pair_with_current():
    rows = []
    for family in ("scaffold", "leader_cluster"):
        for scenario in (
            "reference_all",
            "exclude_exact_4_or_8",
            "one_sided_compatible",
        ):
            rows.append(
                {
                    "domain": "internal_ood",
                    "family": family,
                    "seed": 3,
                    "scenario": scenario,
                    "method": CURRENT_METHOD,
                    "kappa": np.nan,
                    "coverage": 0.88,
                    "coverage_error_abs": 0.02,
                    "mean_width": 2.0,
                    "finite_interval_fraction": 1.0,
                }
            )
            for kappa, ace in ((0.5, 0.03), (0.75, 0.01)):
                rows.append(
                    {
                        "domain": "internal_ood",
                        "family": family,
                        "seed": 3,
                        "scenario": scenario,
                        "method": SHRUNK_METHOD,
                        "kappa": kappa,
                        "coverage": 0.89,
                        "coverage_error_abs": ace,
                        "mean_width": 1.9,
                        "finite_interval_fraction": 1.0,
                    }
                )
    result = pd.DataFrame(heldout_unit_rows(pd.DataFrame(rows), 3, 0.75))
    assert len(result) == 6
    assert np.allclose(result["selected_kappa"], 0.75)
    assert np.allclose(result["ace_difference"], -0.01)
