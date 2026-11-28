"""Prepare a standalone area data directory from a boundary polygon and local national store.

Reads from `store/vn/` (offline):
    vietnam-latest.osm.pbf   Geofabrik Vietnam OSM extract
    population.tif           WorldPop Vietnam 100 m population GeoTIFF

Writes to `data/areas/<name>/` in the exact format expected by `src.run` and `app.py`.

Exit codes: 0 prepared, 2 input/validation error (including missing store files), 3 other error.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from shapely import make_valid
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
            f"invalid area name {name!r}: only lowercase letters, digits, '_' and '-' are allowed"
        )
    return name


def load_boundary(path: str | Path, max_area_km2: float = 50.0) -> gpd.GeoDataFrame:
    """Read, validate, clean, and dissolve a boundary file into a single WGS84 polygon GDF."""
    file_path = Path(path)
    if not file_path.is_file():
        raise InputError(f"boundary file not found: {file_path}")

    try:
        gdf = gpd.read_file(file_path)
    except Exception as exc:
        raise InputError(f"cannot read boundary file {file_path}: {exc}") from exc

    if gdf.empty:
        raise InputError(f"boundary file is empty: {file_path}")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise InputError(f"boundary file has no non-empty geometries: {file_path}")

    if gdf.crs is None:
        raise InputError(f"boundary file has no CRS: {file_path}")

    gdf = gdf.to_crs(WGS84)
    geom_types = set(gdf.geometry.geom_type)
    if not geom_types.issubset({"Polygon", "MultiPolygon"}):
        raise InputError(
            "boundary must contain only Polygon or MultiPolygon geometries, "
            f"got {sorted(geom_types)}"
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
        raise InputError("boundary does not produce a valid polygon")

    boundary = gpd.GeoDataFrame(geometry=[merged], crs=WGS84)
    area_km2 = float(boundary.to_crs(METRIC_CRS).area.iloc[0]) / 1e6
    if area_km2 <= 0:
        raise InputError("boundary has zero area")
    if max_area_km2 is not None and area_km2 > max_area_km2:
        raise InputError(
            f"boundary area {area_km2:.2f} km² exceeds limit {max_area_km2:g} km²; "
            "use a smaller boundary or raise --max-area-km2"
        )

    # shortcut: Vietnam store bounds; generalize when adding other country stores.
    centroid = boundary.geometry.iloc[0].centroid
    lon, lat = float(centroid.x), float(centroid.y)
    if not (
        VN_LON_BOUNDS[0] <= lon <= VN_LON_BOUNDS[1]
        and VN_LAT_BOUNDS[0] <= lat <= VN_LAT_BOUNDS[1]
    ):
        raise InputError(
            f"boundary centroid ({lat:.4f}°N, {lon:.4f}°E) is outside the Vietnam store "
            f"bounds (lat {VN_LAT_BOUNDS[0]}–{VN_LAT_BOUNDS[1]}, "
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prepare area input files from a boundary polygon and local national store."
    )
    ap.add_argument("--boundary", type=Path, required=True, help="boundary file (GeoJSON/GPKG/SHP)")
    ap.add_argument("--name", type=str, required=True, help="area folder name under data/areas/")
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE, help="national store directory")
    ap.add_argument("--mode", choices=["A", "B"], default="B", help="residential area H method")
    ap.add_argument("--radius", type=float, default=500.0, help="service radius R (m)")
    ap.add_argument("--alpha", type=float, default=0.95, help="minimum served demand fraction")
    ap.add_argument("--cap-scale", type=float, default=1.0, help="station capacity scale factor")
    ap.add_argument(
        "--max-area-km2",
        type=float,
        default=50.0,
        help="maximum allowed boundary area in km²",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing data/areas/<name>/ directory",
    )
    return ap.parse_args(argv)


def _check_store_files(store_dir: Path) -> tuple[Path, Path]:
    pbf_path = store_dir / PBF_FILENAME
    tif_path = store_dir / TIF_FILENAME
    missing = []
    if not pbf_path.is_file():
        missing.append(
            f"missing OSM PBF: {pbf_path}\n"
            f"  Download (PowerShell): curl.exe -L -o {pbf_path} "
            "https://download.geofabrik.de/asia/vietnam-latest.osm.pbf"
        )
    if not tif_path.is_file():
        missing.append(
            f"missing WorldPop raster: {tif_path}\n"
            "  Download Vietnam 100 m population GeoTIFF from WorldPop Hub "
            f"(https://hub.worldpop.org/) and place it at {tif_path}"
        )
    if missing:
        raise InputError("\n".join(missing))
    return pbf_path, tif_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    partial: Path | None = None

    try:
        if not (0.0 <= args.alpha <= 1.0):
            raise InputError("alpha must be in [0, 1]")
        if args.radius <= 0 or args.cap_scale <= 0 or args.max_area_km2 <= 0:
            raise InputError("radius, cap-scale and max-area-km2 must be positive")

        area_name = validate_name(args.name)
        out_dir = DEFAULT_AREAS_DIR / area_name
        if out_dir.exists() and not args.overwrite:
            raise InputError(
                f"output directory already exists: {out_dir} (pass --overwrite to replace)"
            )

        pbf_path, tif_path = _check_store_files(args.store)
        boundary = load_boundary(args.boundary, max_area_km2=args.max_area_km2)

        partial = out_dir.with_name(f"{area_name}.partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)

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
            mode=args.mode,
            radius=args.radius,
            alpha=args.alpha,
            cap_scale=args.cap_scale,
            name=area_name,
            pbf_path=pbf_path,
        )

        if out_dir.exists():
            shutil.rmtree(out_dir)
        partial.rename(out_dir)
        partial = None
    except (InputError, FileNotFoundError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    finally:
        if partial is not None:
            shutil.rmtree(partial, ignore_errors=True)

    print(
        f"\nPrepared area '{area_name}' in {out_dir} "
        f"({result['boundary_km2']} km², {result['demand_points']} demand points, "
        f"{result['candidates_feasible']}/{result['candidates_total']} feasible candidates, "
        f"coverage upper bound {result['coverage_upper_bound_pct']:.2f}%)."
    )
    print(f"Next step:\n  python -m src.run --data-dir {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
