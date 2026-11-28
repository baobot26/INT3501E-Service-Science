"""Offline tests for point-based area creation, map click logic, and AppTest integration."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src import app_areas, osm_store, prepare_area
from src.preprocessing import InputError
from tests.test_prepare_area import (
    EXPECTED_AREA_FILES,
    FULL_BOX,
    MINI_OSM,
    _make_boundary_gdf,
    _make_pop_tif,
)

APP = Path(__file__).resolve().parent.parent / "app.py"

FULL_BOX_POINTS = [
    (21.0285, 105.7980),
    (21.0285, 105.8040),
    (21.0335, 105.8040),
    (21.0335, 105.7980),
]


def test_create_area_from_points_builds_area(tmp_path, monkeypatch):
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

    result = app_areas.create_area_from_points(
        FULL_BOX_POINTS,
        "picked_demo",
        store_dir=store_dir,
        areas_dir=areas_dir,
    )
    out_dir = areas_dir / "picked_demo"
    assert result["data_dir"] == out_dir
    for fname in EXPECTED_AREA_FILES:
        assert (out_dir / fname).is_file(), f"missing {fname}"
    assert not (areas_dir / "picked_demo.partial").exists()

    with pytest.raises(InputError, match="ít nhất 3 điểm"):
        app_areas.create_area_from_points(
            [],
            "empty_points",
            store_dir=store_dir,
            areas_dir=areas_dir,
        )


def test_apply_click_appends_once_and_ignores_repeats():
    pts: list[dict[str, float | None]] = []

    # None click -> unchanged
    pts1, handled1 = app_areas.apply_click(pts, None, None)
    assert pts1 == []
    assert handled1 is None

    # First click -> appended (rounded to 6 decimals)
    click_a = {"lat": 16.06781234, "lng": 108.22081234}
    pts2, handled2 = app_areas.apply_click(pts1, click_a, handled1)
    assert pts2 == [{"lat": 16.067812, "lon": 108.220812}]
    assert handled2 == (16.067812, 108.220812)

    # Same click repeated across reruns -> not appended again
    pts3, handled3 = app_areas.apply_click(pts2, click_a, handled2)
    assert pts3 == pts2
    assert handled3 == handled2

    # Distinct second click -> appended
    click_b = {"lat": 16.0710, "lng": 108.2250}
    pts4, handled4 = app_areas.apply_click(pts3, click_b, handled3)
    assert len(pts4) == 2
    assert pts4[-1] == {"lat": 16.071, "lon": 108.225}
    assert handled4 == (16.071, 108.225)


def test_app_area_selector_switches_area_and_clears_runs(tmp_path, monkeypatch):
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    if not APP.is_file() or not (APP.parent / "data" / "demand.csv").is_file():
        pytest.skip("app.py or default data/ not present")

    areas_dir = tmp_path / "areas"
    tif_path = _make_pop_tif(tmp_path / "population.tif")
    mini_dir = areas_dir / "mini_demo"
    prepare_area.prepare(
        _make_boundary_gdf(FULL_BOX),
        mini_dir,
        MINI_OSM,
        tif_path,
        mode="B",
        radius=500.0,
        alpha=0.95,
        cap_scale=1.0,
        name="mini_demo",
    )

    monkeypatch.setattr(app_areas, "AREAS_DIR", areas_dir)
    monkeypatch.setattr(app_areas, "STORE_DIR", tmp_path / "empty_store")
    streamlit.cache_data.clear()
    try:
        at = AppTest.from_file(str(APP), default_timeout=240)
        at.run()
        # (a) Default area opens cleanly with DEFAULT_LABEL
        assert not at.exception, [str(e.value) for e in at.exception]
        area_box = next(s for s in at.sidebar.selectbox if s.label == "Khu vực")
        assert area_box.value == app_areas.DEFAULT_LABEL
        assert "mini_demo" in area_box.options
        assert app_areas.DRAW_NEW_LABEL in area_box.options

        # (b) Switching to custom area updates candidate count to 2
        area_box.select("mini_demo").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        enabled_metrics = [m.value for m in at.metric if m.label == "Ứng viên đang bật"]
        assert enabled_metrics == ["2 điểm"]

        # (c) Run a scenario on default area, then switch area -> runs cleared
        area_box = next(s for s in at.sidebar.selectbox if s.label == "Khu vực")
        area_box.select(app_areas.DEFAULT_LABEL).run()
        assert not at.exception, [str(e.value) for e in at.exception]

        at.sidebar.radio[0].set_value("Set covering").run()
        at.button[0].click()
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "set_covering" in at.session_state["runs"]

        area_box = next(s for s in at.sidebar.selectbox if s.label == "Khu vực")
        area_box.select("mini_demo").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert at.session_state["runs"] == {}

        # Selecting create-area mode without store shows info guidance without exception
        area_box = next(s for s in at.sidebar.selectbox if s.label == "Khu vực")
        area_box.select(app_areas.DRAW_NEW_LABEL).run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert len(at.info) >= 1
    finally:
        streamlit.cache_data.clear()


def test_parse_pasted_coords_delimiters_and_errors():
    text = (
        "16.065, 108.215\n"
        "\n"
        "16.066;108.216\n"
        "16.067\t108.217\n"
        "   \n"
        "16.068 108.218\n"
        "abc\n"
        "16.0\n"
        "1, 2, 3\n"
    )
    valid, bad_lines = app_areas.parse_pasted_coords(text)
    assert valid == [
        (16.065, 108.215),
        (16.066, 108.216),
        (16.067, 108.217),
        (16.068, 108.218),
    ]
    assert bad_lines == [7, 8, 9]


def test_app_create_area_view_order_free_and_paste(tmp_path, monkeypatch):
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    if not APP.is_file():
        pytest.skip("app.py not present")

    fake_store = tmp_path / "fake_store"
    fake_store.mkdir()
    (fake_store / prepare_area.PBF_FILENAME).write_bytes(b"")
    (fake_store / prepare_area.TIF_FILENAME).write_bytes(b"")

    monkeypatch.setattr(app_areas, "STORE_DIR", fake_store)
    monkeypatch.setattr(app_areas, "AREAS_DIR", tmp_path / "areas")
    streamlit.cache_data.clear()
    try:
        at = AppTest.from_file(str(APP), default_timeout=120)
        at.session_state["area"] = app_areas.DRAW_NEW_LABEL
        # Diagonal (bowtie) order is still a valid rectangle without manual sorting
        bowtie_points = [
            {"lat": 21.0285, "lon": 105.7980},
            {"lat": 21.0335, "lon": 105.8040},
            {"lat": 21.0285, "lon": 105.8040},
            {"lat": 21.0335, "lon": 105.7980},
        ]
        at.session_state["area_points"] = bowtie_points
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]

        editors = at.get("dataframe")
        assert len(editors) >= 1
        assert len(editors[0].value) == 4
        create_btn = next(b for b in at.button if b.label == "Tạo vùng")
        assert create_btn.disabled is False
        assert any("Hợp lệ" in s.value for s in at.success)
        assert len(at.warning) == 0
        assert all("Sắp xếp" not in b.label for b in at.button)

        # Paste 3 valid lines + 1 malformed line via quick-paste textarea
        paste_text = (
            "21.0300, 105.7990\n"
            "bad_line\n"
            "21.0310, 105.8000\n"
            "21.0320, 105.8010\n"
        )
        at.text_area[0].set_value(paste_text)
        add_btn = next(b for b in at.button if b.label == "Thêm vào danh sách")
        add_btn.click().run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert len(at.session_state["area_points"]) == 7
        assert any("2" in w.value for w in at.warning)
    finally:
        streamlit.cache_data.clear()
