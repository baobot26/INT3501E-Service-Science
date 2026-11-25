"""Smoke test for the Streamlit app: it must load and run a scenario.

Skipped in Docker CI where app.py and data/ are not present.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"
pytestmark = pytest.mark.skipif(
    not APP.is_file() or not (APP.parent / "data" / "demand.csv").is_file(),
    reason="app.py or data/ not available (Docker CI)",
)


def test_app_loads_without_error():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any(b.label == "Chạy kịch bản" for b in at.button)
    # larger radii overrun the solver's time limit and return nothing
    assert at.sidebar.slider[0].max <= 800
    # the p input must be there before fixed-p is chosen: a form does not redraw
    assert at.sidebar.number_input[0].label.startswith("Số trạm p")


def test_app_runs_the_default_scenario():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    # no slider touched: the defaults are what a demo runs first, so they must solve
    at.button[0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Số trạm" in m.label for m in at.metric)
    # a download click must not rerun the script: a rerun clears the results
    downloads = at.get("download_button")
    assert len(downloads) == 3 and all(b.proto.ignore_rerun for b in downloads)
    station_df = at.dataframe[0].value
    assert "Tải/ngày" in station_df.columns and "Hiệu suất %" in station_df.columns
    demand_df = at.dataframe[1].value
    served_stations = demand_df.loc[demand_df["Phục vụ"] == "Được phục vụ", "Trạm"]
    unserved_stations = demand_df.loc[demand_df["Phục vụ"] == "Chưa phục vụ", "Trạm"]
    assert len(served_stations) > 0 and all(str(s).isdigit() for s in served_stations)
    assert len(unserved_stations) > 0 and all(s == "—" for s in unserved_stations)


def test_app_runs_fixed_p_with_its_default():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    at.sidebar.selectbox[0].select_index(1)  # fixed p, p left at its default
    at.button[0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert [m.value for m in at.metric if m.label == "Số trạm"] == ["31"]


def test_app_recovers_from_a_stale_distance_matrix(tmp_path):
    # polygons.py rewrote the candidates but distances_l1 was not rerun
    shutil.copy(APP, tmp_path / "app.py")
    shutil.copytree(APP.parent / "data", tmp_path / "data")
    np.save(tmp_path / "data" / "dist_l1.npy", np.zeros((1, 1), dtype=np.float32))
    streamlit.cache_data.clear()  # the loader is cached across tests
    at = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=600)
    at.run()
    at.button[0].click()
    at.run()
    streamlit.cache_data.clear()
    assert not at.exception, [str(e.value) for e in at.exception]

