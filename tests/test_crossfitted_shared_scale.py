import numpy as np

from scripts.analyze_crossfitted_shared_scale import (
    fit_crossfitted_scale,
    group_stratified_twofold,
    predict_crossfitted_scale,
    predict_external_scale,
)


def test_group_stratified_twofold_keeps_groups_intact_and_balanced():
    similarity = np.linspace(0.1, 0.9, 40)
    groups = np.repeat(np.arange(20), 2)
    folds = group_stratified_twofold(
        similarity, groups, np.random.default_rng(5)
    )
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1
    assert abs(int((folds == 0).sum()) - int((folds == 1).sum())) <= 4


def test_group_stratified_twofold_is_deterministic():
    similarity = np.linspace(0.1, 0.9, 50)
    groups = np.arange(50)
    first = group_stratified_twofold(
        similarity, groups, np.random.default_rng(12)
    )
    second = group_stratified_twofold(
        similarity, groups, np.random.default_rng(12)
    )
    assert np.array_equal(first, second)


def test_crossfitted_scale_is_positive_and_decreasing():
    rng = np.random.default_rng(8)
    similarity = np.linspace(0.05, 0.95, 100)
    tasks = 12
    task_levels = np.exp(rng.normal(0.0, 0.2, tasks))
    residual_scale = np.exp(1.1 * (0.55 - similarity[:, None])) * task_levels
    truth = residual_scale * rng.normal(size=(len(similarity), tasks))
    prediction = np.zeros_like(truth)
    groups = np.arange(len(similarity))
    folds = group_stratified_twofold(
        similarity, groups, np.random.default_rng(3)
    )
    model, single, diagnostics = fit_crossfitted_scale(
        similarity, truth, prediction, folds
    )
    evaluation = np.linspace(0.05, 0.95, 200)
    crossfit = predict_crossfitted_scale(model, evaluation)
    single_scale = predict_external_scale(single, evaluation)
    assert np.all(crossfit > 0)
    assert np.all(single_scale > 0)
    assert np.all(np.diff(crossfit) <= 1e-12)
    assert np.all(np.diff(single_scale) <= 1e-12)
    assert diagnostics["fold_0_scale_valid_compounds"] >= 20
    assert diagnostics["fold_1_scale_valid_compounds"] >= 20
