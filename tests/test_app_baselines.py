"""Unit tests for shared capacity scaling, comparison helpers, and header formatting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import distances_l1, model_main, preprocessing
from src.app_baselines import (
    DEFAULT_CAPACITY_SCALE,
    build_baseline_header_line,
    compute_common_metrics,
    count_over_capacity,
    format_scale_tag,
    reference_capacity_scale,
)


def _tiny_problem(cap_scale: float = 1.0) -> tuple[preprocessing.ProblemData, np.ndarray, np.ndarray]:
    demand_df = pd.DataFrame({
        "id": [0, 1],
        "lat": [21.03, 21.03],
        "lon": [105.79, 105.7905],
        "population": [300.0, 400.0],
    })
    demand_df["daily_demand"] = 2.0 * demand_df["population"]  # 600 + 800 = 1400
    cand_df = pd.DataFrame({
        "id": [10, 11],
        "lat": [21.03, 21.03],
        "lon": [105.79, 105.82],
        "type": ["amenity=parking", "shop=convenience"],
        "feasible": [True, True],
    })
    params_df = pd.DataFrame({
        "type": ["amenity=parking", "shop=convenience"],
        "capacity_demand_day": [1000.0, 500.0],
        "fixed_cost": [850_000_000.0, 300_000_000.0],
        "enabled": [1, 1],
    })
    problem = preprocessing.build_problem_data(
        demand_df, cand_df, params_df, radius=300.0, alpha=0.5, cap_scale=cap_scale
    )
    dist = np.array([[10.0, 5000.0], [50.0, 5000.0]], dtype=float)
    pop = demand_df["population"].to_numpy(float)
    return problem, dist, pop


def test_over_capacity_differs_between_x1_and_x3():
    problem, dist, pop = _tiny_problem(cap_scale=1.0)
    selected = np.array([0], dtype=int)

    # Total demand = 1400, base capacity of station 0 = 1000:
    # exceeds at x1 (1400 > 1000), fits at x3 (1400 <= 3000)
    metrics_x1 = compute_common_metrics(dist, pop, selected, radius=300.0, problem=problem, capacity_scale=1.0)
    metrics_x3 = compute_common_metrics(dist, pop, selected, radius=300.0, problem=problem, capacity_scale=3.0)

    assert metrics_x1["over_capacity_count"] == 1
    assert metrics_x3["over_capacity_count"] == 0
    assert count_over_capacity(selected, metrics_x1["loads"], metrics_x1["base_capacities"], 1.0) == 1
    assert count_over_capacity(selected, metrics_x1["loads"], metrics_x1["base_capacities"], 3.0) == 0


def test_main_model_always_has_zero_over_capacity_at_own_scale(make_inputs):
    data_dir = make_inputs()
    demand_df = preprocessing.load_demand(data_dir / "demand.csv")
    cand_df = preprocessing.load_candidates(data_dir / "candidates_feasible.csv")
    params_df = preprocessing.load_station_parameters(data_dir / "station_parameters.csv")
    dist, _manifest = distances_l1.compute_matrix(demand_df, cand_df, params_df)
    pop = demand_df["population"].to_numpy(float)

    for scale in (1.0, 3.0):
        problem = preprocessing.build_problem_data(
            demand_df, cand_df, params_df, radius=5000.0, alpha=0.8, cap_scale=scale
        )
        res = model_main.solve(problem, dist, time_limit=30.0)
        assert res["status"] == "optimal"
        common = compute_common_metrics(
            dist, pop, np.asarray(res["selected"], dtype=int), radius=5000.0, problem=problem, capacity_scale=scale
        )
        assert common["over_capacity_count"] == 0


def test_reference_capacity_scale_and_header_lines():
    assert reference_capacity_scale({}) == DEFAULT_CAPACITY_SCALE
    assert reference_capacity_scale({"main": {"kind": "precheck_error", "params": {"capacity_scale": 1.5}}}) == DEFAULT_CAPACITY_SCALE
    assert reference_capacity_scale({"main": {"kind": "solved", "params": {"capacity_scale": 2.0}}}) == 2.0
    assert format_scale_tag(3.0) == "×3"
    assert format_scale_tag(2.5) == "×2,5"

    pm_line = build_baseline_header_line("p_median", {"radius": 300, "p": 31})
    assert "p-median · p = 31 · R = 300 m chỉ dùng để tính độ phủ · không xét sức chứa." in pm_line
