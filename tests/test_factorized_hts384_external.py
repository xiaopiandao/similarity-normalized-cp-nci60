import numpy as np

from scripts.evaluate_factorized_hts384_external import summarize


def test_external_summary_preserves_scope_and_method() -> None:
    import pandas as pd

    rows = []
    for seed, value in ((2, 0.88), (3, 0.90)):
        rows.append(
            {
                "alpha": 0.1,
                "method": "candidate",
                "row_scope": "all_compounds",
                "label_scope": "all_labels",
                "seed": seed,
                "coverage": value,
                "coverage_error_abs": abs(value - 0.9),
                "mean_width": 2.5,
                "finite_interval_fraction": 1.0,
            }
        )
    summary = summarize(pd.DataFrame(rows))
    assert len(summary) == 1
    assert summary.loc[0, "method"] == "candidate"
    assert np.isclose(summary.loc[0, "coverage_mean"], 0.89)
    assert summary.loc[0, "units"] == 2

