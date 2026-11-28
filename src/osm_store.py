"""Read OSM features from a local national PBF store via osmium + osmnx."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from osmnx._errors import InsufficientResponseError

METRIC_CRS = "EPSG:32648"  # UTM 48N, meters
WGS84 = "EPSG:4326"


def osm_filters(*tag_dicts: dict) -> list[str]:
    """Convert OSMnx tag dicts into `osmium tags-filter` expressions.

    Example:
        {"shop": ["convenience", "supermarket"]} -> ["nwr/shop=convenience,supermarket"]
    """
    merged: dict[str, list[str] | None] = {}
    for tags in tag_dicts:
        for key, raw_val in tags.items():
            if raw_val is True:
                merged[key] = None  # match any value for this key
                continue
            if raw_val is False or raw_val is None:
                continue
            vals = [raw_val] if isinstance(raw_val, str) else list(raw_val)
            if not vals:
                continue
            if key in merged and merged[key] is None:
                continue
            existing = merged.setdefault(key, [])
            assert existing is not None
            for v in vals:
                s = str(v)
                if s not in existing:
                    existing.append(s)

    exprs: list[str] = []
    for key, vals in merged.items():
        if vals is None:
            exprs.append(f"nwr/{key}")
        elif vals:
            exprs.append(f"nwr/{key}={','.join(vals)}")
    return exprs


def extract_xml(
    pbf: str | Path,
    boundary_gdf: gpd.GeoDataFrame,
    out_xml: str | Path,
    tag_dicts: Iterable[dict] | dict,
    buffer_m: float = 200.0,
) -> Path:
    """Clip `pbf` to buffered boundary bbox and tag-filter to an OSM XML file."""
    osmium_bin = shutil.which("osmium")
    if osmium_bin is None:
        raise RuntimeError(
            "osmium binary not found on PATH; install osmium-tool "
            "(e.g. `apt-get install osmium-tool` or run via Docker)."
        )

    pbf_path = Path(pbf)
    if not pbf_path.is_file():
        raise FileNotFoundError(f"missing OSM PBF file: {pbf_path}")

    dicts = [tag_dicts] if isinstance(tag_dicts, dict) else list(tag_dicts)
    filters = osm_filters(*dicts)
    if not filters:
        raise ValueError("tag_dicts produced no osmium tag filters")

    buffered = boundary_gdf.to_crs(METRIC_CRS).buffer(buffer_m).to_crs(WGS84)
    minlon, minlat, maxlon, maxlat = buffered.total_bounds
    bbox = f"{minlon:.6f},{minlat:.6f},{maxlon:.6f},{maxlat:.6f}"

    out_path = Path(out_xml)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pbf = Path(tmpdir) / "bbox.osm.pbf"
        subprocess.run(
            [
                osmium_bin,
                "extract",
                "-b",
                bbox,
                "-s",
                "smart",
                "--overwrite",
                "-o",
                str(tmp_pbf),
                str(pbf_path),
            ],
            check=True,
        )
        # Do not pass -R: osmnx needs the referenced nodes of matched ways.
        subprocess.run(
            [
                osmium_bin,
                "tags-filter",
                str(tmp_pbf),
                *filters,
                "--overwrite",
                "-o",
                str(out_path),
            ],
            check=True,
        )

    return out_path


def features(xml: str | Path, polygon, tags: dict) -> gpd.GeoDataFrame:
    """Read OSM features matching `tags` inside `polygon` from an OSM XML file."""
    try:
        return ox.features_from_xml(xml, polygon=polygon, tags=tags)
    except InsufficientResponseError:
        return gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs=WGS84),
            crs=WGS84,
        )
