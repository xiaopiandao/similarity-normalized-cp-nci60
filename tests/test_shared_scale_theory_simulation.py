import numpy as np

from experiments.shared_scale_theory.simulate_shared_scale import (
    Scenario,
    SimulationConfig,
    draw_similarity,
    make_task_parameters,
    relative_shape,
    run_one_replicate,
    scenario_grid,
)


def test_relative_shape_is_decreasing_without_task_heterogeneity():
    scenario = Scenario(50, 8, 0.0, 0.0, 0.0)
    parameters = make_task_parameters(np.random.default_rng(1), scenario)
    similarity = np.asarray([0.1, 0.5, 0.9])
    scale = relative_shape(similarity, parameters, base_scale_slope=1.2)
    assert np.all(scale[0] > scale[1])
    assert np.all(scale[1] > scale[2])


def test_scenario_grid_has_cartesian_product_size():
    scenarios = scenario_grid([50, 100], [5], [0.0, 0.5], [0.0], [0.0, 0.3])
    assert len(scenarios) == 8


def test_one_replicate_is_deterministic_and_returns_all_methods():
    scenario = Scenario(80, 10, 0.25, 0.0, 0.0)
    config = SimulationConfig(n_calibration=120, n_test=240)
    first = run_one_replicate(scenario, config, seed=77, replicate=0)
    second = run_one_replicate(scenario, config, seed=77, replicate=0)
    assert [item["method"] for item in first] == [
        "global",
        "shared_monotone",
        "per_cell_monotone",
        "partial_blend_025",
    ]
    assert [item["status"] for item in first] == [item["status"] for item in second]
    for left, right in zip(first, second):
        assert np.isclose(left["coverage"], right["coverage"], equal_nan=True)
        assert np.isclose(
            left["scale_log_mae"], right["scale_log_mae"], equal_nan=True
        )


def test_target_similarity_is_lower_than_source_on_average():
    rng = np.random.default_rng(9)
    source = draw_similarity(rng, 5000, "source_calibration")
    target = draw_similarity(rng, 5000, "validation")
    assert target.mean() < source.mean()
