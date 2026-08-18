import pytest

pytest.importorskip("rdkit")

from scripts.prepare_hts384_lockbox import normalize_column


def test_normalize_column_removes_cellminer_footnote_suffixes():
    assert normalize_column("NSC # b") == "NSC #"
    assert normalize_column("Mechanism of action c") == "Mechanism of action"
    assert normalize_column("BR:MCF7") == "BR:MCF7"
