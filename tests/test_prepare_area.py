"""Offline tests for national OSM store extraction and boundary-driven area preparation."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src import distances_l1, osm_store, prepare_area, preprocessing, run
from src.fetch_data import CANDIDATE_TAGS, candidates_table, get_demand
from src.polygons import C_TAGS, H_TAGS
from src.preprocessing import InputError
from tests.conftest import DEFAULT_CAND, DEFAULT_DEMAND, DEFAULT_PARAMS

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MINI_OSM = FIXTURES / "mini.osm"
FULL_BOX = box(105.7980, 21.0285, 105.8040, 21.0335)
NO_WATER_BOX = box(105.7985, 21.0290, 105.8022, 21.0328)


def _make_boundary_gdf(geom=FULL_BOX) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")


def _make_pop_tif(path: Path) -> Path:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    # 6 x 6 grid at 0.001 deg (~100 m) covering lon [105.798, 105.804], lat [21.028, 21.034]
    data = np.zeros((6, 6), dtype=np.float32)
    # Place positive population near convenience store (21.0300, 105.8000) and parking (21.0320, 105.7995)
    data[1, 1] = 120.0  # lat ~21.0325, lon ~105.7995
    data[2, 1] = 150.0  # lat ~21.0315, lon ~105.7995
    data[3, 2] = 180.0  # lat ~21.0305, lon ~105.8005
    data[4, 2] = 100.0  # lat ~21.0295, lon ~105.8005

    transform = from_origin(105.798, 21.034, 0.001, 0.001)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=0.0,
    ) as dst:
        dst.write(data, 1)
    return path


def test_osm_filters_formatting_and_merge():
    assert osm_store.osm_filters({"shop": ["convenience", "supermarket"]}) == [
        "nwr/shop=convenience,supermarket"
    ]
    merged = osm_store.osm_filters(CANDIDATE_TAGS, *H_TAGS.values(), C_TAGS)
    assert merged == [
        "nwr/shop=convenience,supermarket",
        "nwr/amenity=post_office,parking",
        "nwr/building=apartments,residential,house,dormitory",
        "nwr/landuse=residential,military,railway,cemetery",
        "nwr/natural=water",
    ]


def test_candidates_table_from_mini_osm():
    feats = osm_store.features(MINI_OSM, FULL_BOX, CANDIDATE_TAGS)
    df = candidates_table(feats)
    assert len(df) == 4
    assert set(df["type"]) == {
        "shop=convenience",
        "shop=supermarket",
        "amenity=parking",
        "amenity=post_office",
    }
    parking = df.loc[df["type"] == "amenity=parking"].iloc[0]
    assert parking["lat"] == pytest.approx(21.0320, abs=1e-4)
    assert parking["lon"] == pytest.approx(105.7995, abs=1e-4)


def test_prepare_and_run_end_to_end(tmp_path):
    tif_path = _make_pop_tif(tmp_path / "population.tif")
    area_dir = tmp_path / "mini_area"
    results_dir = tmp_path / "results"

    summary = prepare_area.prepare(
        _make_boundary_gdf(FULL_BOX),
        area_dir,
        MINI_OSM,
        tif_path,
        mode="B",
        radius=500.0,
        alpha=0.95,
        cap_scale=1.0,
    )

    for fname in (
        "boundary.geojson",
        "station_parameters.csv",
        "candidates.csv",
        "demand.csv",
        "residential_areas.geojson",
        "forbidden_areas.geojson",
        "candidates_feasible.csv",
        "polygon_report.json",
        "polygon_check.html",
        "dist_l1.npy",
        "dist_l1_manifest.json",
        "area.json",
    ):
        assert (area_dir / fname).is_file(), f"missing {fname}"

    cand = pd.read_csv(area_dir / "candidates_feasible.csv", keep_default_na=False)
    by_type = cand.set_index("type")
    assert bool(by_type.loc["shop=convenience", "feasible"]) is True
    assert bool(by_type.loc["amenity=parking", "feasible"]) is True
    assert bool(by_type.loc["shop=supermarket", "feasible"]) is False
    assert by_type.loc["shop=supermarket", "exclusion_reason"] == "residential"
    assert bool(by_type.loc["amenity=post_office", "feasible"]) is False
    assert by_type.loc["amenity=post_office", "exclusion_reason"] == "forbidden"

    demand_df = preprocessing.load_demand(area_dir / "demand.csv")
    cand_df = preprocessing.load_candidates(area_dir / "candidates_feasible.csv")
    dist, manifest = distances_l1.load(area_dir)
    preprocessing.validate_distance_manifest(
        dist,
        manifest,
        demand_df["id"].values,
        cand_df.loc[cand_df["feasible"], "id"].values,
    )
    assert dist.shape == (len(demand_df), 2)

    area_meta = json.loads((area_dir / "area.json").read_text(encoding="utf-8"))
    for key in ("name", "boundary_km2", "pbf", "pbf_mtime", "tif", "mode", "created_at"):
        assert key in area_meta
    assert area_meta["name"] == "mini_area"
    assert area_meta["mode"] == "B"
    assert summary["candidates_feasible"] == 2

    code = run.main([
        "--data-dir", str(area_dir),
        "--results-dir", str(results_dir),
        "--radius", "500",
        "--alpha", "0.95",
        "--cap-scale", "1.0",
    ])
    assert code == 0
    run_out = next(results_dir.iterdir())
    cfg = json.loads((run_out / "config.json").read_text(encoding="utf-8"))
    assert cfg["data_dir"] == str(area_dir)


def test_prepare_with_empty_forbidden_c(tmp_path):
    tif_path = _make_pop_tif(tmp_path / "population.tif")
    area_dir = tmp_path / "no_water"

    summary = prepare_area.prepare(
        _make_boundary_gdf(NO_WATER_BOX),
        area_dir,
        MINI_OSM,
        tif_path,
        mode="B",
        radius=500.0,
        alpha=0.95,
        cap_scale=1.0,
    )
    assert summary["report"]["forbidden_area_pct"] == 0.0
    c_gdf = gpd.read_file(area_dir / "forbidden_areas.geojson")
    assert len(c_gdf) == 0


def test_load_boundary_and_name_validation(tmp_path):
    # Area > max_area_km2
    big_path = tmp_path / "big.geojson"
    gpd.GeoDataFrame(
        geometry=[box(105.7, 20.9, 106.0, 21.2)], crs="EPSG:4326"
    ).to_file(big_path, driver="GeoJSON")
    with pytest.raises(InputError, match="exceeds limit"):
        prepare_area.load_boundary(big_path, max_area_km2=50.0)

    # Centroid outside Vietnam store bounds
    outside_path = tmp_path / "tokyo.geojson"
    gpd.GeoDataFrame(
        geometry=[box(139.70, 35.68, 139.71, 35.69)], crs="EPSG:4326"
    ).to_file(outside_path, driver="GeoJSON")
    with pytest.raises(InputError, match="outside the Vietnam store"):
        prepare_area.load_boundary(outside_path, max_area_km2=50.0)

    # Non-polygon geometry
    pt_path = tmp_path / "point.geojson"
    gpd.GeoDataFrame(
        geometry=[Point(105.80, 21.03)], crs="EPSG:4326"
    ).to_file(pt_path, driver="GeoJSON")
    with pytest.raises(InputError, match="Polygon"):
        prepare_area.load_boundary(pt_path, max_area_km2=50.0)

    # Path traversal / invalid area names
    for bad_name in ("../x", "a/b", "CauGiay", "bad name", ""):
        with pytest.raises(InputError, match="invalid area name"):
            prepare_area.validate_name(bad_name)
    assert prepare_area.validate_name("cau_giay-01") == "cau_giay-01"

    # CLI rejects invalid name, existing dir without --overwrite, and missing store with exit code 2
    valid_path = tmp_path / "valid.geojson"
    _make_boundary_gdf(FULL_BOX).to_file(valid_path, driver="GeoJSON")
    assert prepare_area.main(["--boundary", str(valid_path), "--name", "../x"]) == 2
    assert (
        prepare_area.main([
            "--boundary", str(valid_path),
            "--name", "missing_store_test",
            "--store", str(tmp_path / "empty_store"),
        ])
        == 2
    )


def test_cli_overwrite_guard_and_missing_osmium(tmp_path, monkeypatch):
    areas_root = tmp_path / "areas"
    existing = areas_root / "dup_area"
    existing.mkdir(parents=True)
    monkeypatch.setattr(prepare_area, "DEFAULT_AREAS_DIR", areas_root)

    valid_path = tmp_path / "valid.geojson"
    _make_boundary_gdf(FULL_BOX).to_file(valid_path, driver="GeoJSON")
    assert (
        prepare_area.main([
            "--boundary", str(valid_path),
            "--name", "dup_area",
            "--store", str(tmp_path / "empty_store"),
        ])
        == 2
    )

    monkeypatch.setattr(osm_store.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="osmium-tool"):
        osm_store.extract_xml(
            tmp_path / "dummy.pbf",
            _make_boundary_gdf(FULL_BOX),
            tmp_path / "out.osm",
            [CANDIDATE_TAGS],
        )


def test_compute_matrix_rejects_oversized_grid(monkeypatch):
    monkeypatch.setattr(distances_l1, "MAX_CELLS", 1)
    with pytest.raises(InputError, match="too large"):
        distances_l1.compute_matrix(DEFAULT_DEMAND, DEFAULT_CAND, DEFAULT_PARAMS)


def test_get_demand_missing_tif_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_demand(_make_boundary_gdf(), tif=tmp_path / "missing.tif", out_dir=tmp_path)


@pytest.mark.skipif(shutil.which("osmium") is None, reason="osmium-tool not installed")
def test_extract_xml_matches_direct_mini_osm(tmp_path):
    pbf_path = tmp_path / "mini.osm.pbf"
    subprocess.run(
        ["osmium", "cat", str(MINI_OSM), "-o", str(pbf_path)],
        check=True,
    )
    out_xml = tmp_path / "extracted.osm"
    boundary_gdf = _make_boundary_gdf(FULL_BOX)

    osm_store.extract_xml(
        pbf_path,
        boundary_gdf,
        out_xml,
        [CANDIDATE_TAGS, *H_TAGS.values(), C_TAGS],
    )
    assert out_xml.is_file()

    poly = boundary_gdf.geometry.iloc[0]
    direct_cand = candidates_table(
        osm_store.features(MINI_OSM, poly, CANDIDATE_TAGS)
    ).sort_values("osm_id").reset_index(drop=True)
    extracted_cand = candidates_table(
        osm_store.features(out_xml, poly, CANDIDATE_TAGS)
    ).sort_values("osm_id").reset_index(drop=True)

    assert len(extracted_cand) == len(direct_cand) == 4
    assert extracted_cand["osm_id"].tolist() == direct_cand["osm_id"].tolist()
    assert extracted_cand["type"].tolist() == direct_cand["type"].tolist()
    assert np.allclose(extracted_cand["lat"], direct_cand["lat"])
    assert np.allclose(extracted_cand["lon"], direct_cand["lon"])

    for tags in (H_TAGS["B"], C_TAGS):
        f_direct = osm_store.features(MINI_OSM, poly, tags)
        f_extracted = osm_store.features(out_xml, poly, tags)
        assert len(f_extracted) == len(f_direct)


def test_app_loads_with_app_data_dir(tmp_path, monkeypatch):
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    if not app_path.is_file():
        pytest.skip("app.py not present")

    tif_path = _make_pop_tif(tmp_path / "population.tif")
    area_dir = tmp_path / "area_for_app"
    prepare_area.prepare(
        _make_boundary_gdf(FULL_BOX),
        area_dir,
        MINI_OSM,
        tif_path,
        mode="B",
        radius=500.0,
        alpha=0.95,
        cap_scale=1.0,
    )

    monkeypatch.setenv("APP_DATA_DIR", str(area_dir))
    streamlit.cache_data.clear()
    try:
        at = AppTest.from_file(str(app_path), default_timeout=120)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        enabled_metrics = [m.value for m in at.metric if m.label == "Ứng viên đang bật"]
        assert enabled_metrics == ["2 điểm"]
    finally:
        streamlit.cache_data.clear()


EXPECTED_AREA_FILES = (
    "boundary.geojson",
    "station_parameters.csv",
    "candidates.csv",
    "demand.csv",
    "residential_areas.geojson",
    "forbidden_areas.geojson",
    "candidates_feasible.csv",
    "polygon_report.json",
    "polygon_check.html",
    "dist_l1.npy",
    "dist_l1_manifest.json",
    "area.json",
    "area.osm",
)


def _setup_main_env(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    areas_dir = tmp_path / "areas"
    tif_path = _make_pop_tif(tmp_path / "population.tif")
    fake_pbf = tmp_path / "fake.pbf"
    fake_pbf.write_bytes(b"")
    boundary_path = tmp_path / "boundary.geojson"
    _make_boundary_gdf(FULL_BOX).to_file(boundary_path, driver="GeoJSON")

    monkeypatch.setattr(prepare_area, "DEFAULT_AREAS_DIR", areas_dir)
    monkeypatch.setattr(prepare_area, "_check_store_files", lambda _store: (fake_pbf, tif_path))
    monkeypatch.setattr(
        osm_store,
        "extract_xml",
        lambda _pbf, _bdf, out_xml, _tag_dicts, buffer_m=200.0: Path(
            shutil.copyfile(MINI_OSM, out_xml)
        ),
    )
    return areas_dir, boundary_path


def test_main_atomic_success(tmp_path, monkeypatch):
    areas_dir, boundary_path = _setup_main_env(tmp_path, monkeypatch)

    code = prepare_area.main(["--boundary", str(boundary_path), "--name", "demo", "--mode", "B"])
    assert code == 0

    out_dir = areas_dir / "demo"
    for fname in EXPECTED_AREA_FILES:
        assert (out_dir / fname).is_file(), f"missing {fname}"
    assert not (areas_dir / "demo.partial").exists()
    meta = json.loads((out_dir / "area.json").read_text(encoding="utf-8"))
    assert meta["name"] == "demo"


def test_main_overwrite_replaces_old_files(tmp_path, monkeypatch):
    areas_dir, boundary_path = _setup_main_env(tmp_path, monkeypatch)
    out_dir = areas_dir / "demo"
    out_dir.mkdir(parents=True)
    sentinel = out_dir / "sentinel.txt"
    sentinel.write_text("old", encoding="utf-8")

    code = prepare_area.main([
        "--boundary", str(boundary_path),
        "--name", "demo",
        "--overwrite",
    ])
    assert code == 0
    assert not sentinel.exists()
    for fname in EXPECTED_AREA_FILES:
        assert (out_dir / fname).is_file(), f"missing {fname}"
    assert not (areas_dir / "demo.partial").exists()


def test_main_overwrite_preserves_existing_dir_on_input_error(tmp_path, monkeypatch):
    areas_dir, boundary_path = _setup_main_env(tmp_path, monkeypatch)
    out_dir = areas_dir / "demo"
    out_dir.mkdir(parents=True)
    sentinel = out_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise InputError("simulated failure")

    monkeypatch.setattr(prepare_area, "prepare", boom)
    code = prepare_area.main([
        "--boundary", str(boundary_path),
        "--name", "demo",
        "--overwrite",
    ])
    assert code == 2
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(out_dir.iterdir()) == [sentinel]
    assert not (areas_dir / "demo.partial").exists()


def test_main_cleans_up_on_runtime_error_and_allows_retry(tmp_path, monkeypatch):
    areas_dir, boundary_path = _setup_main_env(tmp_path, monkeypatch)
    real_prepare = prepare_area.prepare

    def crash(*_args, **_kwargs):
        raise RuntimeError("transient error")

    monkeypatch.setattr(prepare_area, "prepare", crash)
    code = prepare_area.main(["--boundary", str(boundary_path), "--name", "retry_area"])
    assert code == 3
    assert not (areas_dir / "retry_area").exists()
    assert not (areas_dir / "retry_area.partial").exists()

    monkeypatch.setattr(prepare_area, "prepare", real_prepare)
    retry_code = prepare_area.main(["--boundary", str(boundary_path), "--name", "retry_area"])
    assert retry_code == 0
    assert (areas_dir / "retry_area" / "area.json").is_file()
    assert not (areas_dir / "retry_area.partial").exists()


def test_main_cleans_stale_partial_dir(tmp_path, monkeypatch):
    areas_dir, boundary_path = _setup_main_env(tmp_path, monkeypatch)
    stale_partial = areas_dir / "interrupted.partial"
    stale_partial.mkdir(parents=True)
    (stale_partial / "leftover.txt").write_text("stale", encoding="utf-8")

    code = prepare_area.main(["--boundary", str(boundary_path), "--name", "interrupted"])
    assert code == 0
    assert not stale_partial.exists()
    out_dir = areas_dir / "interrupted"
    assert (out_dir / "area.json").is_file()
    assert not (out_dir / "leftover.txt").exists()


def test_prepare_rejects_oversized_grid_before_polygons(tmp_path, monkeypatch):
    tif_path = _make_pop_tif(tmp_path / "population.tif")
    area_dir = tmp_path / "oversized"

    monkeypatch.setattr(distances_l1, "MAX_CELLS", 1)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("polygons.check must not be called when grid exceeds MAX_CELLS")

    monkeypatch.setattr(prepare_area.polygons, "check", must_not_run)

    with pytest.raises(InputError, match="MAX_CELLS"):
        prepare_area.prepare(
            _make_boundary_gdf(FULL_BOX),
            area_dir,
            MINI_OSM,
            tif_path,
            mode="B",
            radius=500.0,
            alpha=0.95,
            cap_scale=1.0,
        )
    assert not (area_dir / "candidates_feasible.csv").exists()
