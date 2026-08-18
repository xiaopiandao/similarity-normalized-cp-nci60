import numpy as np
import pandas as pd

from scripts.evaluate_overlap_shrunk_factorized_wcp import (
    CURRENT_METHOD,
    SHRUNK_METHOD,
)
from scripts.select_minimum_sufficient_overlap import (
    select_minimum_sufficient_kappa,
)


def test_selects_smallest_kappa_that_meets_all_internal_conditions():
    rows = []
    for family in ("scaffold", "leader_cluster"):
        current_ace = 0.02 if family == "scaffold" else 0.04
        for scenario in (
            "reference_all",
            "exclude_exact_4_or_8",
            "one_sided_compatible",
        ):
            rows.append(
                {
                    "domain": "internal_ood",
                    "family": family,
                    "seed": 1,
                    "method": CURRENT_METHOD,
                    "kappa": np.nan,
                    "scenario": scenario,
                    "coverage_error_abs": current_ace,
                    "mean_width": 2.0,
                    "finite_interval_fraction": 1.0,
                }
            )
            for kappa, ace in ((0.0, 0.033), (0.5, 0.031), (1.0, 0.025)):
                rows.append(
                    {
                        "domain": "internal_ood",
                        "family": family,
                        "seed": 1,
                        "method": SHRUNK_METHOD,
                        "kappa": kappa,
                        "scenario": scenario,
                        "coverage_error_abs": ace,
                        "mean_width": 2.0,
                        "finite_interval_fraction": 1.0,
                    }
                )
    diagnostics = pd.DataFrame(
        [
            {
                "domain": "internal_ood",
                "kappa": kappa,
                "ess_fraction": 0.6,
                "support_failure_fraction": 0.0,
                "feasible": True,
            }
            for kappa in (0.0, 0.5, 1.0)
        ]
    )
    selected, selection = select_minimum_sufficient_kappa(
        pd.DataFrame(rows), diagnostics
    )
    assert selected == 0.5
    assert not selection.loc[np.isclose(selection["kappa"], 0.0), "eligible"].item()
    assert selection.loc[np.isclose(selection["kappa"], 0.5), "eligible"].item()
    assert selection.loc[
        selection["selected_minimum_sufficient"], "kappa"
    ].item() == 0.5
