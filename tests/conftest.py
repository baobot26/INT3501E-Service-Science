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
