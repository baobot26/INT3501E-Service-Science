"""Shared fixtures: tiny offline instances, no OSM/network access."""
import numpy as np
import pandas as pd
import pytest
from shapely import box


@pytest.fixture
def station_params():
    return pd.DataFrame({
        "type": ["amenity=parking", "shop=convenience", "building=apartments"],
        "capacity_demand_day": [1000, 500, 800],
        "fixed_cost": [10, 20, 30],
        "enabled": [1, 1, 0],
    })


@pytest.fixture
def small_cand():
    """Candidates (inside H, inside C, inside C, outside both, outside but disabled)."""
    cand = pd.DataFrame({
        "id": [0, 1, 2, 3, 4],
        "type": ["amenity=parking", "shop=convenience", "amenity=parking",
                 "shop=convenience", "building=apartments"],
    })
    xy = np.array([
        [100.0, 100.0],
        [500.0, 500.0],
        [520.0, 520.0],
        [900.0, 900.0],
        [900.0, 100.0],
    ])
    return cand, xy


@pytest.fixture
def h_geom():
    return box(0, 0, 200, 200)


@pytest.fixture
def c_geom():
    return box(400, 400, 600, 600)


# --- main-model pipeline fixtures -----------------------------------------
DEFAULT_DEMAND = pd.DataFrame({
    "id": [0, 1], "lat": [21.03, 21.04], "lon": [105.79, 105.80],
    "population": [10.0, 20.0],
})
DEFAULT_CAND = pd.DataFrame({
    "id": [0, 1, 2], "osm_id": ["a", "b", "c"],
    "lat": [21.03, 21.04, 21.05], "lon": [105.79, 105.80, 105.81],
    "type": ["amenity=parking", "shop=convenience", "building=apartments"],
    "name": "", "in_residential": False, "in_forbidden": False,
    "feasible": [True, True, False], "exclusion_reason": ["", "", "disabled"],
})
DEFAULT_PARAMS = pd.DataFrame({
    "type": ["amenity=parking", "shop=convenience", "building=apartments"],
    "capacity_demand_day": [1000, 500, 800],
    "fixed_cost": [10, 20, 30], "enabled": [1, 1, 0],
})


@pytest.fixture
def make_inputs(tmp_path):
    """Write pipeline input CSVs (override any of the three defaults)."""
    def make(demand=None, cand=None, params=None):
        (demand if demand is not None else DEFAULT_DEMAND).to_csv(
            tmp_path / "demand.csv", index=False)
        (cand if cand is not None else DEFAULT_CAND).to_csv(
            tmp_path / "candidates_feasible.csv", index=False)
        (params if params is not None else DEFAULT_PARAMS).to_csv(
            tmp_path / "station_parameters.csv", index=False)
        return tmp_path
    return make
