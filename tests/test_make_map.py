"""Unit tests for L1 diamond geometry and shared map helpers in src/make_map.py."""
import numpy as np
import pytest

from src import distances_l1, make_map


@pytest.mark.parametrize("radius_m", [200.0, 300.0, 500.0, 800.0])
def test_l1_diamond_vertices_lie_at_exact_l1_radius_in_utm(radius_m):
    # Central Cau Giay coordinates
    center_lat, center_lon = 21.0315, 105.7920
    vertices = make_map.l1_diamond(center_lat, center_lon, radius_m)

    assert len(vertices) == 4
    center_xy = distances_l1._project_to_32648(
        np.array([center_lat]), np.array([center_lon])
    )[0]
    vert_lats = np.array([v[0] for v in vertices])
    vert_lons = np.array([v[1] for v in vertices])
    vert_xy = distances_l1._project_to_32648(vert_lats, vert_lons)

    # Every vertex must be at L1 distance R from center (error < 0.5 m)
    dx = np.abs(vert_xy[:, 0] - center_xy[0])
    dy = np.abs(vert_xy[:, 1] - center_xy[1])
    l1_dists = dx + dy
    assert np.all(np.abs(l1_dists - radius_m) < 0.5), l1_dists

    # Vertices are N, E, S, W in UTM: alternating pure dy and pure dx axes
    assert np.isclose(vert_xy[0, 1] - center_xy[1], radius_m, atol=0.5)
    assert np.isclose(vert_xy[1, 0] - center_xy[0], radius_m, atol=0.5)
    assert np.isclose(center_xy[1] - vert_xy[2, 1], radius_m, atol=0.5)
    assert np.isclose(center_xy[0] - vert_xy[3, 0], radius_m, atol=0.5)


def test_l1_diamond_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        make_map.l1_diamond(21.03, 105.79, 0.0)
    with pytest.raises(ValueError):
        make_map.l1_diamond(21.03, 105.79, -100.0)


def test_colors_and_labels_cover_all_station_types():
    expected = {
        "amenity=parking",
        "shop=supermarket",
        "amenity=post_office",
        "shop=convenience",
        "building=apartments",
    }
    assert expected.issubset(make_map.COLORS.keys())
    assert expected.issubset(make_map.STATION_TYPE_LABELS.keys())
    # All 5 data colors must be distinct and separate from UI action ink #165D4A
    palette = [make_map.COLORS[k].upper() for k in expected]
    assert len(set(palette)) == 5
    assert "#165D4A" not in palette


def test_build_map_reads_app_summary_config_radius(tmp_path):
    import json
    import pandas as pd

    (tmp_path / "summary.json").write_text(
        json.dumps({
            "status": "optimal",
            "station_count": 1,
            "total_cost": 300000000,
            "coverage_pct": 100.0,
            "config": {
                "radius": 300,
                "alpha": 0.6,
                "objective_mode": "min_stations_then_cost",
                "demand_scale": 1.0,
                "capacity_scale": 3.0,
            },
        }),
        encoding="utf-8",
    )
    pd.DataFrame([{
        "candidate_id": 10,
        "type": "shop=convenience",
        "lat": 21.03,
        "lon": 105.79,
        "capacity": 24000,
        "load": 1000,
        "utilization": 1000 / 24000,
        "fixed_cost": 300000000,
    }]).to_csv(tmp_path / "selected_stations.csv", index=False)
    pd.DataFrame([{
        "demand_id": 1,
        "lat": 21.0305,
        "lon": 105.7902,
        "daily_demand": 1000.0,
        "station_id": 10,
        "distance_m": 80.0,
        "served": True,
    }]).to_csv(tmp_path / "assigned_demand.csv", index=False)

    m = make_map.build_map(tmp_path, tmp_path / "missing_boundary.geojson")
    html = m.get_root().render()
    assert "R = 300 m" in html
    assert "service-diamond" in html


def test_build_solution_map_dark_mode_uses_dark_tokens(tmp_path):
    from types import SimpleNamespace
    from src.design_tokens import MAP_DARK

    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},'
        '"geometry":{"type":"Polygon","coordinates":[[[105.78,21.01],[105.80,21.01],'
        '[105.80,21.04],[105.78,21.04],[105.78,21.01]]]}}]}',
        encoding="utf-8",
    )
    problem = SimpleNamespace(
        demand_ids=np.array([1, 2]),
        demand_lat=np.array([21.02, 21.03]),
        demand_lon=np.array([105.79, 105.795]),
        candidate_ids=np.array([10]),
        candidate_types=["shop=convenience"],
        cand_lat=np.array([21.021]),
        cand_lon=np.array([105.791]),
        capacities=np.array([24000.0]),
        fixed_costs=np.array([300_000_000.0]),
        radius=300.0,
    )
    dist = np.array([[100.0], [900.0]])
    m = make_map.build_solution_map(
        problem=problem,
        dist=dist,
        selected=[0],
        assignment=np.array([0, -1]),
        loads={0: 1200.0},
        boundary_path=boundary,
        mode="dark",
    )
    html = m.get_root().render()
    assert str(MAP_DARK["boundary"]) in html
    assert str(MAP_DARK["diamond_stroke"]) in html
    assert "invert(1)" in html
    assert str(MAP_DARK["legend_bg"]) in html


