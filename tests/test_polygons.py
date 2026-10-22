"""Offline tests for the candidate filter and the Manhattan upper bound."""
import numpy as np
import pytest
from shapely import box

from src.polygons import flag_candidates, manhattan_reachable


def test_flag_candidates(small_cand, h_geom, c_geom, station_params):
    cand, xy = small_cand
    out = flag_candidates(cand, xy, h_geom, c_geom, station_params)
    assert len(out) == len(cand)  # no row is lost
    assert out["in_residential"].tolist() == [True, False, False, False, False]
    assert out["in_forbidden"].tolist() == [False, True, True, False, False]
    assert out["feasible"].tolist() == [False, False, False, True, False]
    assert out.loc[3, "exclusion_reason"] == ""
    assert out.loc[0, "exclusion_reason"] == "residential"
    assert out.loc[1, "exclusion_reason"] == "forbidden"
    assert out.loc[4, "exclusion_reason"] == "disabled"
    assert "feasible" not in cand.columns  # input is not mutated


def test_point_on_boundary_counts_inside(small_cand, c_geom, station_params):
    cand, xy = small_cand
    h = box(0, 0, 100, 100)  # candidate 0 at (100, 100) sits on the corner
    out = flag_candidates(cand, xy, h, c_geom, station_params)
    assert out.loc[0, "in_residential"]


def test_missing_param_type_raises(small_cand, h_geom, c_geom, station_params):
    cand, xy = small_cand
    params = station_params[station_params["type"] != "shop=convenience"]
    with pytest.raises(AssertionError, match="missing parameters"):
        flag_candidates(cand, xy, h_geom, c_geom, params)


def test_no_polygons_keeps_enabled(small_cand, station_params):
    cand, xy = small_cand
    out = flag_candidates(cand, xy, None, None, station_params)
    assert out["feasible"].tolist() == [True, True, True, True, False]


def test_manhattan_reachable():
    demand = np.array([[0.0, 0.0], [300.0, 400.0], [1000.0, 0.0]])
    cand = np.array([[100.0, 100.0]])
    # L1 distances to the candidate: 200, 500, 1000
    assert manhattan_reachable(demand, cand, 200).tolist() == [True, False, False]
    assert manhattan_reachable(demand, cand, 500).tolist() == [True, True, False]
    assert manhattan_reachable(demand, np.zeros((0, 2)), 10000).tolist() == [False] * 3
