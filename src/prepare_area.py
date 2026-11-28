"""Prepare a standalone area data directory from a set of boundary vertices and local national store.

Reads from `store/vn/` (offline):
    vietnam-latest.osm.pbf   Geofabrik Vietnam OSM extract
    population.tif           WorldPop Vietnam 100 m population GeoTIFF

Writes to `data/areas/<name>/` in the exact format expected by `src.run` and `app.py`.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from shapely import make_valid
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from src import distances_l1, osm_store, polygons
from src.fetch_data import (
    CANDIDATE_TAGS,
    METRIC_CRS,
    WGS84,
    get_candidates,
    get_demand,
)
from src.polygons import C_TAGS, H_TAGS
from src.preprocessing import InputError

DEFAULT_STORE = Path("store/vn")
DEFAULT_AREAS_DIR = Path("data/areas")
DEFAULT_STATION_PARAMS = (
    Path(__file__).resolve().parent.parent / "data" / "station_parameters.csv"
)

PBF_FILENAME = "vietnam-latest.osm.pbf"
TIF_FILENAME = "population.tif"
NAME_RE = re.compile(r"^[a-z0-9_-]+$")

# shortcut: Vietnam store bounds; generalize when adding other country stores.
VN_LON_BOUNDS = (102.0, 110.0)
VN_LAT_BOUNDS = (8.0, 24.0)


def validate_name(name: str) -> str:
    """Ensure area name is a safe single path segment matching `[a-z0-9_-]+`."""
    if not name or not NAME_RE.fullmatch(name):
        raise InputError(
            f"tên vùng {name!r} không hợp lệ: chỉ dùng chữ thường không dấu, chữ số, '_' và '-' "
            "(invalid area name)"
        )
    return name


def _parse_valid_point(item) -> tuple[float, float] | None:
    """Return `(lat, lon)` as finite floats or `None` if missing / non-numeric."""
    if item is None:
        return None
    try:
        raw_lat, raw_lon = item[0], item[1]
    except (TypeError, IndexError, KeyError):
        return None
    if raw_lat is None or raw_lon is None:
        return None
    try:
        lat = float(raw_lat)
        lon = float(raw_lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    return (lat, lon)


def order_around_centroid(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sort `(lat, lon)` points counter-clockwise by polar angle around their centroid.

    Deterministic even if a point coincides with the centroid (`atan2(0, 0)`).
    """
    valid = [pt for item in points if (pt := _parse_valid_point(item)) is not None]
    if len(valid) < 3:
        return valid
    cy = math.fsum(lat for lat, _ in valid) / len(valid)
    cx = math.fsum(lon for _, lon in valid) / len(valid)
    return sorted(
        valid,
        key=lambda pt: (
            math.atan2(pt[0] - cy, pt[1] - cx),
            (pt[0] - cy) ** 2 + (pt[1] - cx) ** 2,
            pt[0],
            pt[1],
        ),
    )


def boundary_from_points(points: Sequence[tuple[float, float]]) -> gpd.GeoDataFrame:
    """Build a valid WGS84 Polygon GeoDataFrame from a set of `(lat, lon)` points.

    Input order does not matter: duplicates are removed (keeping first occurrence) and
    points are connected around their centroid so concave shapes are preserved without
    falling back to a convex hull.
    """
    seen: set[tuple[float, float]] = set()
    cleaned: list[tuple[float, float]] = []
    for row_idx, item in enumerate(points, start=1):
        parsed = _parse_valid_point(item)
        if parsed is None:
            continue
        lat, lon = parsed
        if not (-90.0 <= lat <= 90.0):
            raise InputError(f"dòng {row_idx}: vĩ độ {lat} ngoài khoảng [-90, 90]")
        if not (-180.0 <= lon <= 180.0):
            raise InputError(f"dòng {row_idx}: kinh độ {lon} ngoài khoảng [-180, 180]")
        if parsed in seen:
            continue
        seen.add(parsed)
        cleaned.append(parsed)

    if len(cleaned) < 3:
        raise InputError("cần ít nhất 3 điểm phân biệt để tạo đa giác")

    raw_coords = [(lon, lat) for lat, lon in cleaned]
    hull = MultiPoint(raw_coords).convex_hull
    if hull.geom_type != "Polygon" or hull.area <= 0:
        raise InputError("các điểm thẳng hàng, đa giác có diện tích bằng 0")

    ordered = order_around_centroid(cleaned)
    poly = Polygon([(lon, lat) for lat, lon in ordered])
    if not poly.is_valid:
        raise InputError("các điểm không tạo được đa giác đơn")
    if poly.is_empty or poly.area <= 0:
        raise InputError("đa giác có diện tích bằng 0")

    return gpd.GeoDataFrame(geometry=[poly], crs=WGS84)


def clean_boundary(gdf: gpd.GeoDataFrame, max_area_km2: float = 50.0) -> gpd.GeoDataFrame:
    """Validate, clean, and dissolve a boundary GeoDataFrame into a single WGS84 polygon GDF."""
    if gdf.empty:
        raise InputError("ranh giới trống (boundary is empty)")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise InputError("ranh giới không có hình học hợp lệ (boundary has no non-empty geometries)")

    if gdf.crs is None:
        raise InputError("ranh giới thiếu hệ tọa độ CRS (boundary has no CRS)")

    gdf = gdf.to_crs(WGS84)
    geom_types = set(gdf.geometry.geom_type)
    if not geom_types.issubset({"Polygon", "MultiPolygon"}):
        raise InputError(
            "ranh giới chỉ được chứa hình Polygon hoặc MultiPolygon "
            f"(boundary must contain only Polygon or MultiPolygon, got {sorted(geom_types)})"
        )

    valid_geoms = make_valid(gdf.geometry)
    merged = valid_geoms.union_all()
    if merged is not None and merged.geom_type == "GeometryCollection":
        poly_parts = [
            g for g in merged.geoms if g.geom_type in ("Polygon", "MultiPolygon")
        ]
        merged = unary_union(poly_parts) if poly_parts else None

    if (
        merged is None
        or merged.is_empty
        or merged.geom_type not in ("Polygon", "MultiPolygon")
    ):
        raise InputError("ranh giới không tạo thành đa giác hợp lệ (boundary does not produce a valid polygon)")

    boundary = gpd.GeoDataFrame(geometry=[merged], crs=WGS84)
    area_km2 = float(boundary.to_crs(METRIC_CRS).area.iloc[0]) / 1e6
    if area_km2 <= 0:
        raise InputError("ranh giới có diện tích bằng 0 (boundary has zero area)")
    if max_area_km2 is not None and area_km2 > max_area_km2:
        raise InputError(
            f"diện tích vùng vẽ {area_km2:.2f} km² vượt giới hạn {max_area_km2:g} km² "
            "(boundary area exceeds limit); hãy vẽ vùng nhỏ hơn"
        )

    # shortcut: Vietnam store bounds; generalize when adding other country stores.
    centroid = boundary.geometry.iloc[0].centroid
    lon, lat = float(centroid.x), float(centroid.y)
    if not (
        VN_LON_BOUNDS[0] <= lon <= VN_LON_BOUNDS[1]
        and VN_LAT_BOUNDS[0] <= lat <= VN_LAT_BOUNDS[1]
    ):
        raise InputError(
            f"tâm vùng vẽ ({lat:.4f}°N, {lon:.4f}°E) nằm ngoài phạm vi kho Việt Nam "
            f"(outside the Vietnam store bounds: lat {VN_LAT_BOUNDS[0]}–{VN_LAT_BOUNDS[1]}, "
            f"lon {VN_LON_BOUNDS[0]}–{VN_LON_BOUNDS[1]})"
        )

    return boundary


def prepare(
    boundary_gdf: gpd.GeoDataFrame,
    out_dir: str | Path,
    osm_xml: str | Path,
    pop_tif: str | Path,
    mode: str = "B",
    radius: float = 500.0,
    alpha: float = 0.95,
    cap_scale: float = 1.0,
    *,
    name: str | None = None,
    pbf_path: str | Path | None = None,
    station_params_path: str | Path | None = None,
) -> dict:
    """Build all pipeline input files in `out_dir` from a pre-extracted OSM XML and raster."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    area_name = name or out_path.name

    boundary_wgs = boundary_gdf.to_crs(WGS84)
    boundary_wgs.to_file(out_path / "boundary.geojson", driver="GeoJSON")

    params_src = (
        Path(station_params_path)
        if station_params_path is not None
        else DEFAULT_STATION_PARAMS
    )
    if not params_src.is_file():
        raise FileNotFoundError(f"missing station parameters file: {params_src}")
    shutil.copyfile(params_src, out_path / "station_parameters.csv")

    poly = boundary_wgs.geometry.iloc[0]
    cand_feats = osm_store.features(osm_xml, poly, CANDIDATE_TAGS)
    cand_df = get_candidates(boundary_wgs, out_dir=out_path, feats=cand_feats)
    if cand_df.empty:
        raise InputError("no candidate sites found inside boundary")

    try:
        demand_df = get_demand(boundary_wgs, tif=pop_tif, out_dir=out_path)
    except ValueError as exc:
        raise InputError(
            f"boundary does not overlap population raster {pop_tif}: {exc}"
        ) from exc
    if demand_df.empty:
        raise InputError(
            f"no positive-population demand points found inside boundary in {pop_tif}"
        )

    n_demand, n_cand = len(demand_df), len(cand_df)
    cells = n_demand * n_cand
    if cells > distances_l1.MAX_CELLS:
        raise InputError(
            f"area grid too large ({n_demand} demand × {n_cand} candidates = "
            f"{cells:,} > MAX_CELLS={distances_l1.MAX_CELLS:,}); "
            "use a smaller boundary"
        )

    report = polygons.check(
        out_path,
        mode=mode,
        radius=radius,
        alpha=alpha,
        cap_scale=cap_scale,
        fetch_fn=lambda p, tags, _label: osm_store.features(osm_xml, p, tags),
    )

    distances_l1.compute(out_path)

    boundary_km2 = round(float(boundary_wgs.to_crs(METRIC_CRS).area.iloc[0]) / 1e6, 2)
    pbf_str = None
    pbf_mtime = None
    if pbf_path is not None:
        pbf_p = Path(pbf_path)
        pbf_str = str(pbf_p)
        if pbf_p.is_file():
            pbf_mtime = datetime.fromtimestamp(
                pbf_p.stat().st_mtime, tz=timezone.utc
            ).isoformat()

    meta = {
        "name": area_name,
        "boundary_km2": boundary_km2,
        "pbf": pbf_str,
        "pbf_mtime": pbf_mtime,
        "tif": str(pop_tif),
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_path / "area.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        **meta,
        "demand_points": len(demand_df),
        "candidates_total": report["candidates_total"],
        "candidates_feasible": report["candidates_feasible"],
        "coverage_upper_bound_pct": report["coverage_upper_bound_pct"],
        "report": report,
    }


def _check_store_files(store_dir: Path) -> tuple[Path, Path]:
    store_root = Path(store_dir)
    pbf_path = store_root / PBF_FILENAME
    tif_path = store_root / TIF_FILENAME
    missing = []
    if not pbf_path.is_file():
        missing.append(
            f"missing OSM PBF: {pbf_path}\n"
            f"  Download (PowerShell): curl.exe -L -R -o {pbf_path} "
            "https://download.geofabrik.de/asia/vietnam-latest.osm.pbf"
        )
    if not tif_path.is_file():
        missing.append(
            f"missing WorldPop raster: {tif_path}\n"
            "  Download vnm_pop_2025_CN_100m_R2025A_v1.tif (WorldPop 2025, constrained, 100 m) "
            f"from WorldPop Hub (https://hub.worldpop.org/) and place it at {tif_path}"
        )
    if missing:
        raise InputError("\n".join(missing))
    return pbf_path, tif_path


def build_area(
    boundary_gdf: gpd.GeoDataFrame,
    name: str,
    *,
    store_dir: str | Path = DEFAULT_STORE,
    areas_dir: str | Path = DEFAULT_AREAS_DIR,
    mode: str = "B",
    max_area_km2: float = 50.0,
) -> dict:
    """Validate inputs, extract OSM data, and atomically build `areas_dir / name`."""
    area_name = validate_name(name)
    areas_root = Path(areas_dir)
    out_dir = areas_root / area_name
    if out_dir.exists():
        raise InputError(
            f"vùng '{area_name}' đã tồn tại tại {out_dir}; "
            "hãy chọn tên khác hoặc xoá thư mục cũ trước khi dựng lại"
        )

    pbf_path, tif_path = _check_store_files(Path(store_dir))
    boundary = clean_boundary(boundary_gdf, max_area_km2=max_area_km2)

    partial: Path | None = out_dir.with_name(f"{area_name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        osm_xml = partial / "area.osm"
        osm_store.extract_xml(
            pbf_path,
            boundary,
            osm_xml,
            [CANDIDATE_TAGS, *H_TAGS.values(), C_TAGS],
        )
        result = prepare(
            boundary,
            partial,
            osm_xml,
            tif_path,
            mode=mode,
            name=area_name,
            pbf_path=pbf_path,
        )
        partial.rename(out_dir)
        partial = None
    finally:
        if partial is not None:
            shutil.rmtree(partial, ignore_errors=True)

    return {**result, "data_dir": out_dir}


def list_areas(areas_dir: str | Path = DEFAULT_AREAS_DIR) -> list[str]:
    """Return sorted prepared area names under `areas_dir` matching `NAME_RE` with `area.json`."""
    root = Path(areas_dir)
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir()
        and NAME_RE.fullmatch(entry.name)
        and (entry / "area.json").is_file()
    )
