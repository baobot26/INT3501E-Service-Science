"""Check residential-area (H) and forbidden-area (C) polygons and filter candidates.

Reads from data/: boundary.geojson, candidates.csv, demand.csv, station_parameters.csv
Fetches H and C from OSM and writes to data/:
    residential_areas.geojson   residential polygons
    forbidden_areas.geojson     forbidden-area polygons
    candidates_feasible.csv     candidates with in_residential/in_forbidden/feasible flags
    polygon_report.json         check report and the two upper bounds
    polygon_check.html          check map (if folium is installed)

A point on a polygon boundary counts as inside (uses covers).

Usage:
    python src/polygons.py                          # option A: landuse=residential
    python src/polygons.py --mode B                 # option B: residential buildings
    python src/polygons.py --radius 500 --alpha 0.95 --cap-scale 1

Requires: pip install osmnx geopandas shapely pandas numpy
"""

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely import covers, make_valid, points

DATA = Path("data")
METRIC_CRS = "EPSG:32648"  # UTM 48N, meters
WGS84 = "EPSG:4326"

# option A: residential landuse; option B: residential buildings
H_TAGS = {
    "A": {"landuse": ["residential"]},
    "B": {"building": ["residential", "house", "apartments", "dormitory"]},
}
C_TAGS = {"landuse": ["military", "railway", "cemetery"], "natural": ["water"]}


def fetch(area_poly, tags, name):
    for attempt in range(3):
        try:
            g = ox.features_from_polygon(area_poly, tags=tags)
            print(f"{name}: fetched {len(g)} features from OSM")
            return g
        except Exception as e:
            wait = 30 * 2**attempt
            print(f"overpass busy, retrying in {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError("overpass still busy after 3 attempts, try again in a few minutes")


def clean(g, boundary_geom_m):
    """Fix invalid polygons, drop empty/non-areal shapes, clip to the boundary (metric CRS)."""
    g = g.to_crs(METRIC_CRS)
    g["geometry"] = make_valid(g.geometry)
    g["geometry"] = g.geometry.intersection(boundary_geom_m)
    g = g.explode(index_parts=False)
    g = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    g = g[g.geometry.area > 0]
    return g


def flag_candidates(cand, cand_xy, h_geom, c_geom, params):
    """Flag each candidate with in_residential/in_forbidden/feasible/exclusion_reason.

    cand_xy is an (n, 2) metric coordinate array matching the cand rows;
    h_geom/c_geom are metric-CRS polygons (or None). covers() counts points
    on a polygon boundary as inside. Returns a copy; disabled types are also
    excluded from the candidate set.
    """
    cand = cand.copy()
    pt_arr = points(cand_xy[:, 0], cand_xy[:, 1])
    cand["in_residential"] = covers(h_geom, pt_arr) if h_geom is not None else False
    cand["in_forbidden"] = covers(c_geom, pt_arr) if c_geom is not None else False
    missing = set(cand["type"]) - set(params["type"])
    assert not missing, f"missing parameters for candidate types: {missing}"
    par = params.set_index("type")
    enabled = cand["type"].map(par["enabled"]) == 1
    cand["feasible"] = enabled & ~cand["in_residential"] & ~cand["in_forbidden"]
    flags = pd.DataFrame({
        "disabled": ~enabled,
        "residential": cand["in_residential"].to_numpy(),
        "forbidden": cand["in_forbidden"].to_numpy(),
    })
    cand["exclusion_reason"] = flags.apply(lambda r: ";".join(r.index[r]), axis=1)
    return cand


def manhattan_reachable(demand_xy, cand_xy, radius):
    """Per-demand flag: within Manhattan distance `radius` of any candidate.

    demand_xy/cand_xy are (n, 2) metric coordinate arrays; empty cand -> all False.
    """
    if len(cand_xy) == 0:
        return np.zeros(len(demand_xy), dtype=bool)
    dist = (np.abs(demand_xy[:, None, 0] - cand_xy[None, :, 0])
            + np.abs(demand_xy[:, None, 1] - cand_xy[None, :, 1]))
    return (dist <= radius).any(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], default="A",
                    help="method for defining residential area H")
    ap.add_argument("--radius", type=float, default=500, help="service radius R (m)")
    ap.add_argument("--alpha", type=float, default=0.95, help="minimum served demand fraction")
    ap.add_argument("--cap-scale", type=float, default=1.0, help="station capacity scale factor")
    args = ap.parse_args()

    ox.settings.use_cache = True
    ox.settings.log_console = False

    boundary = gpd.read_file(DATA / "boundary.geojson")
    boundary_m = boundary.to_crs(METRIC_CRS)
    area_km2 = boundary_m.area.sum() / 1e6
    boundary_geom_m = boundary_m.union_all()

    # 1. fetch and clean H, C polygons
    H = clean(fetch(boundary.geometry.iloc[0], H_TAGS[args.mode], "residential H"), boundary_geom_m)
    C = clean(fetch(boundary.geometry.iloc[0], C_TAGS, "forbidden C"), boundary_geom_m)
    h_geom = H.union_all() if len(H) else None
    c_geom = C.union_all() if len(C) else None
    h_pct = (h_geom.area if h_geom is not None else 0) / boundary_m.area.sum() * 100
    c_pct = (c_geom.area if c_geom is not None else 0) / boundary_m.area.sum() * 100
    print(f"area {area_km2:.2f} km2: H covers {h_pct:.1f}%, C covers {c_pct:.1f}%")

    H.to_crs(WGS84).to_file(DATA / "residential_areas.geojson", driver="GeoJSON")
    C.to_crs(WGS84).to_file(DATA / "forbidden_areas.geojson", driver="GeoJSON")

    # 2. flag candidates — covers counts points on the boundary too
    cand = pd.read_csv(DATA / "candidates.csv")
    assert cand["id"].is_unique and cand["lat"].notna().all() and cand["lon"].notna().all()
    cpts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(cand["lon"], cand["lat"]), crs=WGS84
    ).to_crs(METRIC_CRS)
    cand_xy = np.column_stack([cpts.geometry.x.to_numpy(), cpts.geometry.y.to_numpy()])

    # 3. capacity/cost parameters; disabled types are also excluded from the candidate set
    params = pd.read_csv(DATA / "station_parameters.csv")
    assert (params["capacity_demand_day"] > 0).all() and (params["fixed_cost"] >= 0).all()
    cand = flag_candidates(cand, cand_xy, h_geom, c_geom, params)

    print(f"\ncandidates by type (feasible / total):")
    for t, g in cand.groupby("type"):
        print(f"  {t}: {int(g['feasible'].sum())} / {len(g)}")
    n_feas = int(cand["feasible"].sum())
    print(f"total feasible candidates: {n_feas} / {len(cand)}")

    # 4. coverage upper bound: open ALL feasible candidates, Manhattan distance on UTM
    demand = pd.read_csv(DATA / "demand.csv")
    assert (demand["population"] >= 0).all()
    daily = 2 * demand["population"].to_numpy()  # D_i = 2 * P_i
    total = daily.sum()
    dpts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(demand["lon"], demand["lat"]), crs=WGS84
    ).to_crs(METRIC_CRS)
    demand_xy = np.column_stack([dpts.geometry.x.to_numpy(), dpts.geometry.y.to_numpy()])
    reachable = manhattan_reachable(demand_xy, cand_xy[cand["feasible"].to_numpy()], args.radius)
    reach_pct = daily[reachable].sum() / total * 100

    # 5. capacity upper bound
    par = params.set_index("type")
    capacity = float((cand.loc[cand["feasible"], "type"].map(par["capacity_demand_day"])
                      * args.cap_scale).sum())
    need = args.alpha * total

    ok_cov = reach_pct >= args.alpha * 100
    ok_cap = capacity >= need
    print(f"\ncoverage upper bound @R={args.radius:.0f} m: {reach_pct:.2f}% of demand "
          f"(need >= {args.alpha * 100:.0f}%) -> {'OK' if ok_cov else 'FAIL'}")
    print(f"capacity upper bound: {capacity:,.0f} / need {need:,.0f} "
          f"-> {'OK' if ok_cap else 'FAIL'}")
    if not (ok_cov and ok_cap):
        print("=> problem is certainly infeasible with the current parameters; "
              "suggestions: raise R, lower alpha, raise cap_scale or switch H option")

    # 6. save results
    cand.to_csv(DATA / "candidates_feasible.csv", index=False, encoding="utf-8-sig")
    report = {
        "polygon_mode": args.mode,
        "h_tags": H_TAGS[args.mode],
        "c_tags": C_TAGS,
        "radius_m": args.radius,
        "alpha": args.alpha,
        "cap_scale": args.cap_scale,
        "boundary_km2": round(area_km2, 2),
        "residential_area_pct": round(h_pct, 1),
        "forbidden_area_pct": round(c_pct, 1),
        "candidates_total": len(cand),
        "candidates_feasible": n_feas,
        "feasible_by_type": {t: int(g["feasible"].sum()) for t, g in cand.groupby("type")},
        "exclusion_counts": cand["exclusion_reason"].value_counts().to_dict(),
        "total_daily_demand": float(total),
        "coverage_upper_bound_pct": round(reach_pct, 2),
        "capacity_upper_bound": capacity,
        "required_served_demand": float(need),
        "coverage_ok": bool(ok_cov),
        "capacity_ok": bool(ok_cap),
    }
    (DATA / "polygon_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nsaved data/residential_areas.geojson, data/forbidden_areas.geojson, "
          "data/candidates_feasible.csv, data/polygon_report.json")

    # 7. check map
    try:
        import folium
        m = folium.Map(location=[cand["lat"].mean(), cand["lon"].mean()],
                       zoom_start=13)
        folium.GeoJson(boundary, name="boundary").add_to(m)
        folium.GeoJson(H.to_crs(WGS84), name="residential H",
                       style_function=lambda _: {"color": "red", "weight": 0.5,
                                                 "fillOpacity": 0.15}).add_to(m)
        folium.GeoJson(C.to_crs(WGS84), name="forbidden C",
                       style_function=lambda _: {"color": "gray", "weight": 0.5,
                                                 "fillOpacity": 0.3}).add_to(m)
        for _, r in cand.iterrows():
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=3,
                color="green" if r["feasible"] else "red", fill=True,
                popup=f"{r['type']} ({r['exclusion_reason'] or 'feasible'})",
            ).add_to(m)
        folium.LayerControl().add_to(m)
        m.save(DATA / "polygon_check.html")
        print("saved map data/polygon_check.html")
    except ImportError:
        print("skipping map: folium not installed")


if __name__ == "__main__":
    main()