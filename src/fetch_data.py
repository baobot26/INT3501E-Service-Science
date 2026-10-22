"""Fetch input data for the locker-placement problem in Cầu Giấy.

Outputs are saved in data/:
    boundary.geojson      area boundary
    candidates.csv        candidate locker sites (from OSM)
    walk_network.graphml  walking road network
    demand.csv            demand points and population (from WorldPop)

Requires: pip install osmnx geopandas rasterio shapely
"""

from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
import time

WARDS = [
    "Phường Cầu Giấy, Hà Nội, Việt Nam",
    "Phường Nghĩa Đô, Hà Nội, Việt Nam",
    "Phường Yên Hòa, Hà Nội, Việt Nam",
]
# if OSM does not return ward boundaries, point this to another file (geojson or shapefile)
BOUNDARY_FALLBACK = None

WORLDPOP_TIF = "data/vnm_pop_100m.tif"

CANDIDATE_TAGS = {
    "shop": ["convenience", "supermarket"],
    "amenity": ["post_office", "parking"],
    "building": ["apartments"],
}

METRIC_CRS = "EPSG:32648"  # UTM 48N, meters
WGS84 = "EPSG:4326"
NETWORK_BUFFER_M = 500  # fetch roads beyond the boundary so edge points still get distances

OUT = Path("data")
OUT.mkdir(exist_ok=True)


def get_boundary():
    try:
        wards = ox.geocode_to_gdf(WARDS)
        print(f"boundary: fetched {len(wards)} wards from OSM")
    except Exception as e:
        if BOUNDARY_FALLBACK is None:
            raise RuntimeError(
                "could not fetch ward boundaries from OSM, set BOUNDARY_FALLBACK"
            ) from e
        wards = gpd.read_file(BOUNDARY_FALLBACK)
        print("boundary: using fallback file")
    area = gpd.GeoDataFrame(geometry=[wards.to_crs(WGS84).union_all()], crs=WGS84)
    km2 = area.to_crs(METRIC_CRS).area.iloc[0] / 1e6
    print(f"area about {km2:.2f} km2 (old Cầu Giấy district is 12.44 km2)")
    area.to_file(OUT / "boundary.geojson", driver="GeoJSON")
    return area


def get_candidates(area):
    poly = area.geometry.iloc[0]
    feats = ox.features_from_polygon(poly, tags=CANDIDATE_TAGS).reset_index()

    # polygon features (buildings, parking lots): use the centroid
    pts = feats.to_crs(METRIC_CRS)
    pts["geometry"] = pts.geometry.centroid
    pts = pts.to_crs(WGS84)

    def site_type(row):
        for key in CANDIDATE_TAGS:
            if key in row and isinstance(row[key], str) and row[key] in CANDIDATE_TAGS[key]:
                return f"{key}={row[key]}"
        return "other"

    df = pd.DataFrame({
        "osm_id": pts["id"].astype(str) if "id" in pts else pts.index.astype(str),
        "lat": pts.geometry.y,
        "lon": pts.geometry.x,
        "type": pts.apply(site_type, axis=1),
        "name": pts["name"] if "name" in pts else "",
    })
    df = df[df["type"] != "other"].reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    df.to_csv(OUT / "candidates.csv", index=False, encoding="utf-8-sig")
    print(f"candidate points: {len(df)}")
    print(df["type"].value_counts().to_string())
    return df


def get_network(area):
    buffered = area.to_crs(METRIC_CRS).buffer(NETWORK_BUFFER_M).to_crs(WGS84).iloc[0]
    for attempt in range(5):
        try:
            G = ox.graph_from_polygon(buffered, network_type="walk")
            break
        except Exception as e:
            wait = 30 * 2 ** attempt
            print(f"overpass busy, retrying in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError("overpass still busy after 5 attempts, try again in a few minutes")
    ox.save_graphml(G, OUT / "walk_network.graphml")
    print(f"walking network: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


def get_demand(area):
    import numpy as np
    import rasterio
    from rasterio.mask import mask
    from rasterio.transform import xy

    if not Path(WORLDPOP_TIF).exists():
        print(f"skipping demand points: missing {WORLDPOP_TIF}")
        return pd.DataFrame()

    with rasterio.open(WORLDPOP_TIF) as src:
        shapes = area.to_crs(src.crs).geometry
        img, transform = mask(src, shapes, crop=True, nodata=0)
        band = img[0]
        rows, cols = np.where(band > 0)  # drop cells with no population
        xs, ys = xy(transform, rows, cols)
        pts = gpd.GeoDataFrame(
            {"population": band[rows, cols]},
            geometry=gpd.points_from_xy(xs, ys),
            crs=src.crs,
        ).to_crs(WGS84)

    df = pd.DataFrame({
        "id": range(len(pts)),
        "lat": pts.geometry.y,
        "lon": pts.geometry.x,
        "population": pts["population"].round(1),
    })
    df.to_csv(OUT / "demand.csv", index=False)
    print(f"demand points: {len(df)}, total population about {df['population'].sum():,.0f}")
    return df


if __name__ == "__main__":
    ox.settings.use_cache = True
    ox.settings.log_console = False
    area = get_boundary()
    get_candidates(area)
    get_demand(area)
    get_network(area)
    print("done, data is in data/")