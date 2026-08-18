from pathlib import Path

from scripts.select_dad_k_development import (
    DEVELOPMENT_SEED,
    FAMILIES,
    K_VALUES,
)


def test_development_selection_contract() -> None:
    assert DEVELOPMENT_SEED == 1
    assert FAMILIES == ("scaffold", "leader_cluster")
    assert K_VALUES == (50, 100, 250, 500)


def test_script_does_not_name_confirmation_seed_outputs() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "select_dad_k_development.py"
    ).read_text(encoding="utf-8")
    assert "confirmation_seeds_read\": []" in source
    assert "SEEDS = (2" not in source
