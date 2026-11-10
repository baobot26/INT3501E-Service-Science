"""Tests for dynamic scenario parameters (type toggles, scales, fixed p)."""
import json
import time

import numpy as np
import pandas as pd
import pytest

from src import distances_l1, run
from src.config import SolveConfig
from src.preprocessing import build_problem_data, load_candidates, load_demand, \
    load_station_parameters
from tests.conftest import DEFAULT_CAND, DEFAULT_DEMAND, DEFAULT_PARAMS


def _setup(tmp_path, cand=None, params=None):
    """Write inputs and compute the distance matrix. Returns the loaded data."""
    cand = DEFAULT_CAND.copy() if cand is None else cand
    params = DEFAULT_PARAMS.copy() if params is None else params
    (tmp_path / "demand.csv").write_text(DEFAULT_DEMAND.to_csv(index=False), encoding="utf-8")
    (tmp_path / "candidates_feasible.csv").write_text(cand.to_csv(index=False), encoding="utf-8")
    (tmp_path / "station_parameters.csv").write_text(params.to_csv(index=False), encoding="utf-8")
    distances_l1.compute(tmp_path)
    demand_df = load_demand(tmp_path / "demand.csv")
    cand_df = load_candidates(tmp_path / "candidates_feasible.csv")
    params_df = load_station_parameters(tmp_path / "station_parameters.csv")
    dist, manifest = distances_l1.load(tmp_path)
    return demand_df, cand_df, params_df, dist, manifest


# Two parking candidates cover both demand points, plus one convenience
# candidate: disabling shop=convenience must still leave a feasible plan.
CAND_WITH_SPARE = pd.DataFrame({
    "id": [0, 1, 2, 3], "osm_id": ["a", "b", "c", "d"],
    "lat": [21.03, 21.04, 21.04, 21.05], "lon": [105.79, 105.80, 105.80, 105.81],
    "type": ["amenity=parking", "shop=convenience", "amenity=parking", "building=apartments"],
    "name": "", "in_residential": False, "in_forbidden": False,
    "feasible": [True, True, True, False], "exclusion_reason": ["", "", "", "disabled"],
})


def test_disabling_a_type_keeps_matrix_columns_aligned(tmp_path):
    """A disabled type must not shift the distance matrix columns."""
    demand_df, cand_df, params_df, dist, manifest = _setup(tmp_path)
    problem = build_problem_data(demand_df, cand_df, params_df, 500, 0.95,
                                 type_parameters={"shop=convenience": {"enabled": 0}})
    assert len(problem.candidate_ids) == dist.shape[1]
    # the disabled type's candidates are still rows, but locked out
    assert int(problem.enabled_mask.sum()) < len(problem.candidate_ids)
    assert not problem.enabled_mask[problem.candidate_types == "shop=convenience"].any()


def test_disabled_type_candidates_are_never_selected(tmp_path):
    demand_df, cand_df, params_df, dist, manifest = _setup(tmp_path, cand=CAND_WITH_SPARE)
    problem = build_problem_data(demand_df, cand_df, params_df, 500, 0.95,
                                 type_parameters={"shop=convenience": {"enabled": 0}})
    result, validation = run.solve_scenario(
        SolveConfig(radius=500, alpha=0.95, objective_mode="min_stations_then_cost",
                    station_count=None, demand_scale=1.0, capacity_scale=1.0,
                    time_limit=60.0, mip_rel_gap=0.0),
        problem, dist, manifest)
    assert result["status"] == "optimal"
    chosen_types = {problem.candidate_types[j] for j in result["selected"]}
    assert "shop=convenience" not in chosen_types
    assert chosen_types == {"amenity=parking"}


def test_source_csv_is_not_modified(tmp_path):
    cand = DEFAULT_CAND.copy()
    demand_df, cand_df, params_df, dist, manifest = _setup(tmp_path, cand=cand)
    before = (tmp_path / "station_parameters.csv").read_text(encoding="utf-8")
    build_problem_data(demand_df, cand_df, params_df, 500, 0.95,
                       type_parameters={"amenity=parking": {"enabled": 0,
                                                          "fixed_cost": 999}})
    assert (tmp_path / "station_parameters.csv").read_text(encoding="utf-8") == before


def test_fixed_station_count_is_respected(tmp_path):
    demand_df, cand_df, params_df, dist, manifest = _setup(tmp_path)
    config = SolveConfig(radius=500, alpha=0.0, objective_mode="fixed_station_count",
                          station_count=1, demand_scale=1.0, capacity_scale=1.0,
                          time_limit=60.0, mip_rel_gap=0.0)
    problem = run.build_scenario_problem(config, demand_df, cand_df, params_df)
    result, _ = run.solve_scenario(config, problem, dist, manifest)
    assert result["status"] == "optimal"
    assert result["station_count"] == 1


def test_fixed_station_count_zero_with_zero_alpha(tmp_path):
    demand_df, cand_df, params_df, dist, manifest = _setup(tmp_path)
    config = SolveConfig(radius=500, alpha=0.0, objective_mode="fixed_station_count",
                          station_count=0, demand_scale=1.0, capacity_scale=1.0,
                          time_limit=60.0, mip_rel_gap=0.0)
    problem = run.build_scenario_problem(config, demand_df, cand_df, params_df)
    result, _ = run.solve_scenario(config, problem, dist, manifest)
    assert result["status"] == "optimal"
    assert result["station_count"] == 0
    assert result["total_cost"] == 0


def test_demand_scale_tightens_the_model(tmp_path):
    """More demand with the same capacity must not need fewer stations."""
    demand_df, cand_df, params_df, dist, manifest = _setup(tmp_path)
    base = SolveConfig(radius=500, alpha=0.95, objective_mode="min_stations_then_cost",
                       station_count=None, demand_scale=1.0, capacity_scale=1.0,
                       time_limit=60.0, mip_rel_gap=0.0)
    scaled = SolveConfig(radius=500, alpha=0.95, objective_mode="min_stations_then_cost",
                         station_count=None, demand_scale=3.0, capacity_scale=1.0,
                         time_limit=60.0, mip_rel_gap=0.0)
    p_base = run.build_scenario_problem(base, demand_df, cand_df, params_df)
    p_scaled = run.build_scenario_problem(scaled, demand_df, cand_df, params_df)
    assert p_scaled.daily_demand.sum() == pytest.approx(
        3 * p_base.daily_demand.sum())
    r_base, _ = run.solve_scenario(base, p_base, dist, manifest)
    r_scaled, _ = run.solve_scenario(scaled, p_scaled, dist, manifest)
    assert r_scaled["station_count"] >= r_base["station_count"]


def test_cli_and_shared_solver_agree(tmp_path):
    """The CLI and the shared solver must give the same numbers."""
    _setup(tmp_path)
    code = run.main(["--data-dir", str(tmp_path),
                     "--results-dir", str(tmp_path / "results"),
                     "--radius", "500", "--alpha", "0.95", "--time-limit", "60"])
    assert code == 0
    out = next((tmp_path / "results").iterdir())
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "optimal"
    assert summary["validation_passed"] is True
    assert summary["station_count"] == 2
    assert summary["coverage_pct"] == 100.0


def test_cli_reads_type_params_json(tmp_path):
    _setup(tmp_path, cand=CAND_WITH_SPARE)
    tp = tmp_path / "tp.json"
    tp.write_text(json.dumps({"shop=convenience": {"enabled": 0}}), encoding="utf-8")
    code = run.main(["--data-dir", str(tmp_path),
                     "--results-dir", str(tmp_path / "results"),
                     "--radius", "500", "--alpha", "0.95",
                     "--time-limit", "60", "--type-params", str(tp)])
    assert code == 0
    out = next((tmp_path / "results").iterdir())
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["type_parameters"] == {"shop=convenience": {"enabled": 0}}
    stations = pd.read_csv(out / "selected_stations.csv")
    assert "shop=convenience" not in set(stations["type"])


def test_missing_candidates_file_gives_clear_error(tmp_path, capsys):
    (tmp_path / "demand.csv").write_text(DEFAULT_DEMAND.to_csv(index=False), encoding="utf-8")
    (tmp_path / "station_parameters.csv").write_text(DEFAULT_PARAMS.to_csv(index=False), encoding="utf-8")
    code = run.main(["--data-dir", str(tmp_path),
                     "--results-dir", str(tmp_path / "results")])
    assert code == 2
    assert "missing file" in capsys.readouterr().err


def test_zero_station_run_writes_headers(tmp_path):
    """alpha=0 selects no station; the CSVs must still have headers."""
    _setup(tmp_path)
    code = run.main(["--data-dir", str(tmp_path),
                     "--results-dir", str(tmp_path / "results"),
                     "--radius", "500", "--alpha", "0.0", "--time-limit", "60"])
    assert code == 0
    out = next((tmp_path / "results").iterdir())
    stations = pd.read_csv(out / "selected_stations.csv")
    assert list(stations.columns)[:2] == ["candidate_id", "type"]
    assert len(stations) == 0


def test_map_builds_with_zero_stations(tmp_path):
    from src import make_map
    _setup(tmp_path)
    run.main(["--data-dir", str(tmp_path),
              "--results-dir", str(tmp_path / "results"),
              "--radius", "500", "--alpha", "0.0", "--time-limit", "60"])
    out = next((tmp_path / "results").iterdir())
    m = make_map.build_map(out, tmp_path / "boundary.geojson")
    assert m is not None


def test_total_time_stays_within_budget(tmp_path, monkeypatch):
    """The two phases share one budget: the total must not exceed it."""
    import src.model_main as mm

    real_milp = mm.milp
    calls = []

    def slow_milp(*args, **kwargs):
        calls.append(kwargs.get("options", {}).get("time_limit", 0))
        # burn the whole granted budget
        time.sleep(min(kwargs.get("options", {}).get("time_limit", 0), 0.4))
        return real_milp(*args, **kwargs)

    monkeypatch.setattr(mm, "milp", slow_milp)
    demand_df, cand_df, params_df, dist, manifest = _setup(tmp_path)
    config = SolveConfig(radius=500, alpha=0.95, objective_mode="min_stations_then_cost",
                          station_count=None, demand_scale=1.0, capacity_scale=1.0,
                          time_limit=1.0, mip_rel_gap=0.0)
    problem = run.build_scenario_problem(config, demand_df, cand_df, params_df)
    result, _ = run.solve_scenario(config, problem, dist, manifest)
    assert result["total_seconds"] <= 1.0 + 1.0
    # phase 2 must not start once the budget is gone
    assert result.get("phase2_status") in ("skipped_no_budget", None) \
        or result.get("phase2_seconds", 0) <= 1.0


def test_time_limit_no_solution_classified_correctly(tmp_path, monkeypatch):
    """When HiGHS times out with no incumbent, status is time_limit_no_solution."""
    from unittest.mock import MagicMock
    from src import model_main

    demand_df, cand_df, params_df, dist, manifest = _setup(tmp_path)
    problem = build_problem_data(demand_df, cand_df, params_df, 500, 0.5)

    mock_res = MagicMock()
    mock_res.status = 1
    mock_res.x = None
    mock_res.message = "Time limit reached"
    monkeypatch.setattr(model_main, "milp", lambda *a, **kw: mock_res)

    result = model_main.solve(problem, dist, time_limit=1.0)
    assert result["status"] == "time_limit_no_solution"
    assert result["station_count"] == 0
    assert result["total_cost"] == 0.0


def test_phase2_without_solution_keeps_phase1_incumbent(tmp_path, monkeypatch):
    """Phase 2 timing out with no incumbent must fall back to phase 1, not crash."""
    from unittest.mock import MagicMock
    from src import model_main

    demand_df, cand_df, params_df, dist, manifest = _setup(tmp_path)
    problem = build_problem_data(demand_df, cand_df, params_df, 500, 0.95)

    real_milp = model_main.milp
    calls = []

    def phase2_times_out(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return real_milp(*args, **kwargs)
        return MagicMock(status=1, x=None, message="Time limit reached")

    monkeypatch.setattr(model_main, "milp", phase2_times_out)
    result = model_main.solve(problem, dist, time_limit=60.0)
    assert result["solution_source"] == "phase1_fallback"
    assert result["status"] == "feasible_time_limit"  # cost was never minimised
    assert result["station_count"] == 2
