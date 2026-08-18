import numpy as np

from scripts.analyze_scale_sample_efficiency import stratified_subsample


def test_stratified_subsample_is_reproducible_and_unique():
    similarity = np.linspace(0.1, 0.9, 100)
    first = stratified_subsample(similarity, 25, np.random.default_rng(7))
    second = stratified_subsample(similarity, 25, np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert len(first) == 25
    assert len(np.unique(first)) == 25


def test_stratified_subsample_returns_all_rows_when_requested_size_is_large():
    similarity = np.asarray([0.2, 0.4, 0.6])
    selected = stratified_subsample(similarity, 10, np.random.default_rng(1))
    assert np.array_equal(selected, np.arange(3))
