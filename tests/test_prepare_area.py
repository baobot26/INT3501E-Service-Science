"""Offline tests for national OSM store extraction and drawn-boundary area preparation."""
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


def _setup_build_env(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    areas_dir = tmp_path / "areas"
    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True)
    tif_path = _make_pop_tif(tmp_path / "population.tif")
    fake_pbf = tmp_path / "fake.pbf"
    fake_pbf.write_bytes(b"")

    monkeypatch.setattr(prepare_area, "_check_store_files", lambda _store: (fake_pbf, tif_path))
    monkeypatch.setattr(
        osm_store,
        "extract_xml",
        lambda _pbf, _bdf, out_xml, _tag_dicts, buffer_m=200.0: Path(
            shutil.copyfile(MINI_OSM, out_xml)
        ),
    )
    return store_dir, areas_dir


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


def test_clean_boundary_validation():
    # Area > 50 km² (10 km x 10 km square ~ 100 km² in Hanoi)
    big_gdf = prepare_area.boundary_from_points([
        (21.00, 105.75),
        (21.00, 105.85),
        (21.09, 105.85),
        (21.09, 105.75),
    ])
    with pytest.raises(InputError, match="exceeds limit|vượt giới hạn"):
        prepare_area.clean_boundary(big_gdf, max_area_km2=50.0)

    # Centroid outside Vietnam store bounds
    outside_gdf = gpd.GeoDataFrame(geometry=[box(139.70, 35.68, 139.71, 35.69)], crs="EPSG:4326")
    with pytest.raises(InputError, match="Vietnam|Việt Nam"):
        prepare_area.clean_boundary(outside_gdf, max_area_km2=50.0)

    # Non-polygon geometry
    pt_gdf = gpd.GeoDataFrame(geometry=[Point(105.80, 21.03)], crs="EPSG:4326")
    with pytest.raises(InputError, match="Polygon"):
        prepare_area.clean_boundary(pt_gdf, max_area_km2=50.0)

    # Two overlapping polygons dissolve into a single Polygon
    poly_a = box(105.798, 21.028, 105.802, 21.032)
    poly_b = box(105.800, 21.030, 105.804, 21.034)
    merged_gdf = prepare_area.clean_boundary(
        gpd.GeoDataFrame(geometry=[poly_a, poly_b], crs="EPSG:4326"),
        max_area_km2=50.0,
    )
    assert len(merged_gdf) == 1
    assert merged_gdf.geometry.iloc[0].geom_type == "Polygon"


def test_boundary_from_points_valid_and_cleaning():
    square = [
        (21.0285, 105.7980),
        (21.0285, 105.8040),
        (21.0335, 105.8040),
        (21.0335, 105.7980),
    ]
    gdf = prepare_area.boundary_from_points(square)
    assert len(gdf) == 1
    assert gdf.crs == prepare_area.WGS84
    geom = gdf.geometry.iloc[0]
    assert geom.geom_type == "Polygon"
    assert geom.bounds == pytest.approx((105.7980, 21.0285, 105.8040, 21.0335))

    # Closing duplicate, consecutive duplicate, None/NaN/non-numeric rows are ignored
    noisy = [
        (None, 105.80),
        (21.0285, 105.7980),
        (21.0285, 105.7980),
        (float("nan"), 105.80),
        (21.0285, 105.8040),
        ("bad", 105.80),
        (21.0335, 105.8040),
        (21.0335, 105.7980),
        (21.0285, 105.7980),
    ]
    gdf_noisy = prepare_area.boundary_from_points(noisy)  # type: ignore[arg-type]
    assert gdf_noisy.geometry.iloc[0].equals(geom)


def test_boundary_from_points_order_free_concave_and_dedup():
    import itertools

    corners = [
        (21.0285, 105.7980),
        (21.0285, 105.8040),
        (21.0335, 105.8040),
        (21.0335, 105.7980),
    ]
    ref_poly = prepare_area.boundary_from_points(corners).geometry.iloc[0]

    # Diagonal (bowtie) order produces the exact same rectangle
    bowtie = [corners[0], corners[2], corners[1], corners[3]]
    bowtie_poly = prepare_area.boundary_from_points(bowtie).geometry.iloc[0]
    assert bowtie_poly.is_valid
    assert bowtie_poly.area == pytest.approx(ref_poly.area, rel=1e-9)
    assert bowtie_poly.bounds == pytest.approx((105.7980, 21.0285, 105.8040, 21.0335))

    # All 24 permutations of the 4 corners yield the exact same polygon
    for perm in itertools.permutations(corners):
        perm_poly = prepare_area.boundary_from_points(list(perm)).geometry.iloc[0]
        assert perm_poly.is_valid
        assert perm_poly.equals(ref_poly)

    # 5 points with one indented inward -> valid 5-vertex concave polygon (not convex hull)
    concave_pts = [
        (21.0285, 105.7980),
        (21.0285, 105.8040),
        (21.0310, 105.8020),  # indented inward on east side
        (21.0335, 105.8040),
        (21.0335, 105.7980),
    ]
    concave_poly = prepare_area.boundary_from_points(concave_pts).geometry.iloc[0]
    assert concave_poly.is_valid
    assert len(concave_poly.exterior.coords) - 1 == 5
    assert concave_poly.area < concave_poly.convex_hull.area

    # Non-consecutive duplicate (A, B, A, C, D) is deduplicated to 4 vertices
    with_non_consec_dup = [corners[0], corners[1], corners[0], corners[2], corners[3]]
    dedup_poly = prepare_area.boundary_from_points(with_non_consec_dup).geometry.iloc[0]
    assert len(dedup_poly.exterior.coords) - 1 == 4
    assert dedup_poly.equals(ref_poly)


def test_boundary_from_points_errors():
    with pytest.raises(InputError, match="ít nhất 3 điểm"):
        prepare_area.boundary_from_points([])

    with pytest.raises(InputError, match="ít nhất 3 điểm"):
        prepare_area.boundary_from_points([(21.03, 105.80), (21.04, 105.81)])

    with pytest.raises(InputError, match="thẳng hàng|diện tích bằng 0"):
        prepare_area.boundary_from_points([
            (21.03, 105.80),
            (21.03, 105.81),
            (21.03, 105.82),
        ])

    with pytest.raises(InputError, match="dòng 2"):
        prepare_area.boundary_from_points([
            (21.03, 105.80),
            (95.0, 105.81),
            (21.04, 105.81),
        ])

    with pytest.raises(InputError, match="dòng 1"):
        prepare_area.boundary_from_points([
            (21.03, 200.0),
            (21.03, 105.81),
            (21.04, 105.81),
        ])


def test_extract_xml_missing_osmium_raises(tmp_path, monkeypatch):
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


def test_build_area_atomic_success(tmp_path, monkeypatch):
    store_dir, areas_dir = _setup_build_env(tmp_path, monkeypatch)

    result = prepare_area.build_area(
        _make_boundary_gdf(FULL_BOX),
        "demo",
        store_dir=store_dir,
        areas_dir=areas_dir,
        mode="B",
    )

    out_dir = areas_dir / "demo"
    assert result["data_dir"] == out_dir
    for fname in EXPECTED_AREA_FILES:
        assert (out_dir / fname).is_file(), f"missing {fname}"
    assert not (areas_dir / "demo.partial").exists()
    meta = json.loads((out_dir / "area.json").read_text(encoding="utf-8"))
    assert meta["name"] == "demo"


def test_build_area_rejects_existing_and_invalid_names(tmp_path, monkeypatch):
    store_dir, areas_dir = _setup_build_env(tmp_path, monkeypatch)
    existing = areas_dir / "dup_area"
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(InputError, match="đã tồn tại|already exists"):
        prepare_area.build_area(
            _make_boundary_gdf(FULL_BOX),
            "dup_area",
            store_dir=store_dir,
            areas_dir=areas_dir,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(existing.iterdir()) == [sentinel]
    assert not (areas_dir / "dup_area.partial").exists()

    for bad_name in ("../x", "Cầu Giấy", "a/b", "CauGiay", "bad name", ""):
        with pytest.raises(InputError):
            prepare_area.build_area(
                _make_boundary_gdf(FULL_BOX),
                bad_name,
                store_dir=store_dir,
                areas_dir=areas_dir,
            )


def test_build_area_cleans_up_on_runtime_error_and_allows_retry(tmp_path, monkeypatch):
    store_dir, areas_dir = _setup_build_env(tmp_path, monkeypatch)
    real_prepare = prepare_area.prepare

    def crash(*_args, **_kwargs):
        raise RuntimeError("transient error")

    monkeypatch.setattr(prepare_area, "prepare", crash)
    with pytest.raises(RuntimeError, match="transient error"):
        prepare_area.build_area(
            _make_boundary_gdf(FULL_BOX),
            "retry_area",
            store_dir=store_dir,
            areas_dir=areas_dir,
        )
    assert not (areas_dir / "retry_area").exists()
    assert not (areas_dir / "retry_area.partial").exists()

    monkeypatch.setattr(prepare_area, "prepare", real_prepare)
    result = prepare_area.build_area(
        _make_boundary_gdf(FULL_BOX),
        "retry_area",
        store_dir=store_dir,
        areas_dir=areas_dir,
    )
    assert result["data_dir"] == areas_dir / "retry_area"
    assert (areas_dir / "retry_area" / "area.json").is_file()
    assert not (areas_dir / "retry_area.partial").exists()


def test_build_area_cleans_stale_partial_dir(tmp_path, monkeypatch):
    store_dir, areas_dir = _setup_build_env(tmp_path, monkeypatch)
    stale_partial = areas_dir / "interrupted.partial"
    stale_partial.mkdir(parents=True)
    (stale_partial / "leftover.txt").write_text("stale", encoding="utf-8")

    result = prepare_area.build_area(
        _make_boundary_gdf(FULL_BOX),
        "interrupted",
        store_dir=store_dir,
        areas_dir=areas_dir,
    )
    out_dir = areas_dir / "interrupted"
    assert result["data_dir"] == out_dir
    assert not stale_partial.exists()
    assert (out_dir / "area.json").is_file()
    assert not (out_dir / "leftover.txt").exists()


def test_list_areas_filters_invalid_and_missing_dirs(tmp_path):
    assert prepare_area.list_areas(tmp_path / "does_not_exist") == []

    areas_dir = tmp_path / "areas"
    areas_dir.mkdir()
    for name in ("z_area", "a-area_1"):
        d = areas_dir / name
        d.mkdir()
        (d / "area.json").write_text("{}", encoding="utf-8")

    # Ignored entries: .partial, missing area.json, invalid name, regular file
    partial_dir = areas_dir / "tmp.partial"
    partial_dir.mkdir()
    (partial_dir / "area.json").write_text("{}", encoding="utf-8")

    (areas_dir / "incomplete_area").mkdir()

    bad_name_dir = areas_dir / "Invalid Name"
    bad_name_dir.mkdir()
    (bad_name_dir / "area.json").write_text("{}", encoding="utf-8")

    (areas_dir / "not_a_dir").write_text("x", encoding="utf-8")

    assert prepare_area.list_areas(areas_dir) == ["a-area_1", "z_area"]


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
