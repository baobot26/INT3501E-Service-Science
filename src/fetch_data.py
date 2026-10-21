"""Tải dữ liệu đầu vào cho bài chọn vị trí tủ nhận hàng ở Cầu Giấy.

Kết quả lưu trong thư mục data/:
    boundary.geojson      ranh giới khu vực
    candidates.csv        các điểm có thể đặt tủ (từ OSM)
    walk_network.graphml  mạng đường đi bộ
    demand.csv            điểm nhu cầu và dân số (từ WorldPop)

Cần cài: pip install osmnx geopandas rasterio shapely
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
# nếu OSM không trả về ranh giới phường thì dùng file khác (geojson hoặc shapefile)
BOUNDARY_FALLBACK = None

WORLDPOP_TIF = "data/vnm_pop_100m.tif"

CANDIDATE_TAGS = {
    "shop": ["convenience", "supermarket"],
    "amenity": ["post_office", "parking"],
    "building": ["apartments"],
}

METRIC_CRS = "EPSG:32648"  # UTM 48N, đơn vị mét
WGS84 = "EPSG:4326"
NETWORK_BUFFER_M = 500  # lấy đường rộng ra ngoài ranh giới để điểm ở mép vẫn tính được

OUT = Path("data")
OUT.mkdir(exist_ok=True)


def get_boundary():
    try:
        wards = ox.geocode_to_gdf(WARDS)
        print(f"ranh giới: lấy được {len(wards)} phường từ OSM")
    except Exception as e:
        if BOUNDARY_FALLBACK is None:
            raise RuntimeError(
                "không lấy được ranh giới phường từ OSM, hãy điền BOUNDARY_FALLBACK"
            ) from e
        wards = gpd.read_file(BOUNDARY_FALLBACK)
        print("ranh giới: dùng file dự phòng")
    area = gpd.GeoDataFrame(geometry=[wards.to_crs(WGS84).union_all()], crs=WGS84)
    km2 = area.to_crs(METRIC_CRS).area.iloc[0] / 1e6
    print(f"diện tích khoảng {km2:.2f} km2 (quận Cầu Giấy cũ là 12,44 km2)")
    area.to_file(OUT / "boundary.geojson", driver="GeoJSON")
    return area


def get_candidates(area):
    poly = area.geometry.iloc[0]
    feats = ox.features_from_polygon(poly, tags=CANDIDATE_TAGS).reset_index()

    # địa điểm dạng đa giác (tòa nhà, bãi xe) thì lấy tâm
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
    print(f"điểm ứng viên: {len(df)}")
    print(df["type"].value_counts().to_string())
    return df


def get_network(area):
    buffered = area.to_crs(METRIC_CRS).buffer(NETWORK_BUFFER_M).to_crs(WGS84).iloc[0]
    for lan in range(5):
        try:
            G = ox.graph_from_polygon(buffered, network_type="walk")
            break
        except Exception as e:
            cho = 30 * 2 ** lan
            print(f"overpass đang bận, thử lại sau {cho} giây")
            time.sleep(cho)
    else:
        raise RuntimeError("overpass vẫn bận sau 5 lần thử, chạy lại sau ít phút")
    ox.save_graphml(G, OUT / "walk_network.graphml")
    print(f"mạng đường đi bộ: {len(G.nodes)} nút, {len(G.edges)} đoạn")
    return G


def get_demand(area):
    import numpy as np
    import rasterio
    from rasterio.mask import mask
    from rasterio.transform import xy

    if not Path(WORLDPOP_TIF).exists():
        print(f"bỏ qua điểm nhu cầu: chưa có file {WORLDPOP_TIF}")
        return pd.DataFrame()

    with rasterio.open(WORLDPOP_TIF) as src:
        shapes = area.to_crs(src.crs).geometry
        img, transform = mask(src, shapes, crop=True, nodata=0)
        band = img[0]
        rows, cols = np.where(band > 0)  # bỏ ô không có dân
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
    print(f"điểm nhu cầu: {len(df)}, tổng dân khoảng {df['population'].sum():,.0f}")
    return df


if __name__ == "__main__":
    ox.settings.use_cache = True
    ox.settings.log_console = False
    area = get_boundary()
    get_candidates(area)
    get_demand(area)
    get_network(area)
    print("xong, dữ liệu nằm trong thư mục data/")
