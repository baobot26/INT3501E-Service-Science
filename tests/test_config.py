"""Tests for SolveConfig validation."""
import pytest

from src.config import SolveConfig


def test_valid_config():
    cfg = SolveConfig(
        radius=800, alpha=0.95,
        objective_mode="min_stations_then_cost",
        station_count=None,
        demand_scale=1.0, capacity_scale=3.0,
        time_limit=120, mip_rel_gap=0.0,
    )
    assert cfg.radius == 800


def test_invalid_radius():
    with pytest.raises(ValueError, match="radius"):
        SolveConfig(radius=-1, alpha=0.95, objective_mode="min_stations_then_cost",
                    station_count=None, demand_scale=1.0, capacity_scale=1.0,
                    time_limit=120, mip_rel_gap=0.0)


def test_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        SolveConfig(radius=800, alpha=1.5, objective_mode="min_stations_then_cost",
                    station_count=None, demand_scale=1.0, capacity_scale=1.0,
                    time_limit=120, mip_rel_gap=0.0)


def test_fixed_count_requires_station_count():
    with pytest.raises(ValueError, match="station_count"):
        SolveConfig(radius=800, alpha=0.95, objective_mode="fixed_station_count",
                    station_count=None, demand_scale=1.0, capacity_scale=1.0,
                    time_limit=120, mip_rel_gap=0.0)


def test_negative_station_count():
    with pytest.raises(ValueError, match="station_count"):
        SolveConfig(radius=800, alpha=0.95, objective_mode="fixed_station_count",
                    station_count=-1, demand_scale=1.0, capacity_scale=1.0,
                    time_limit=120, mip_rel_gap=0.0)


def test_invalid_demand_scale():
    with pytest.raises(ValueError, match="demand_scale"):
        SolveConfig(radius=800, alpha=0.95, objective_mode="min_stations_then_cost",
                    station_count=None, demand_scale=0.0, capacity_scale=1.0,
                    time_limit=120, mip_rel_gap=0.0)


def test_invalid_mip_gap():
    with pytest.raises(ValueError, match="mip_rel_gap"):
        SolveConfig(radius=800, alpha=0.95, objective_mode="min_stations_then_cost",
                    station_count=None, demand_scale=1.0, capacity_scale=1.0,
                    time_limit=120, mip_rel_gap=1.5)