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
    objective_box = next(s for s in at.sidebar.selectbox if s.label == "Mục tiêu")
    objective_box.select_index(1)  # fixed p, p left at its default
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


def test_model_selector_appears_before_form_and_switches_inputs():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.sidebar.radio) == 1
    radio = at.sidebar.radio[0]
    assert radio.label == "Mô hình"
    assert radio.options == [
        "Mô hình chính · MILP hai pha",
        "Set covering",
        "Max covering",
        "p-median",
    ]
    # Selector sits above the parameter form in the sidebar (below the area selector)
    sidebar_types = [at.sidebar.children[idx].type for idx in sorted(at.sidebar.children)]
    assert sidebar_types.index("selectbox") < sidebar_types.index("radio") < sidebar_types.index("form")

    radio.set_value("Set covering").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    slider_labels = [s.label for s in at.sidebar.slider]
    assert "Bán kính phục vụ R (m)" in slider_labels
    assert "Tỷ lệ phủ tối thiểu α" not in slider_labels
    assert len(at.sidebar.number_input) == 0

    at.sidebar.radio[0].set_value("Max covering").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert [s.label for s in at.sidebar.slider] == ["Bán kính phục vụ R (m)"]
    assert [n.label for n in at.sidebar.number_input] == ["Số trạm p"]

    at.sidebar.radio[0].set_value("p-median").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert [s.label for s in at.sidebar.slider] == ["Bán kính đánh giá R (m)"]
    assert [n.label for n in at.sidebar.number_input] == ["Số trạm p"]


def test_set_covering_default_matches_models_and_preserves_sessions():
    from src import models

    D_l1, _pop, _ids = models.load_inputs("l1", APP.parent / "data")
    expected_stations, _coverable, _status = models.set_covering(D_l1, 300)

    at = AppTest.from_file(str(APP), default_timeout=240)
    at.run()
    at.sidebar.radio[0].set_value("Set covering").run()
    at.button[0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    station_metric = [m.value for m in at.metric if m.label == "Số trạm"]
    assert station_metric == [str(len(expected_stations))]
    assert any("Mô hình này không xét sức chứa" in c.value and "×3" in c.value for c in at.caption)
    # 2 result tabs + 1 comparison table
    assert len(at.dataframe) == 3
    comp_df = at.dataframe[2].value
    assert list(comp_df["Mô hình"]) == ["Set covering"]
    assert "Trạm vượt sức chứa (×3)" in comp_df.columns

    # Switching back to the main model leaves its own idle state intact
    at.sidebar.radio[0].set_value("Mô hình chính · MILP hai pha").run()
    assert [m.value for m in at.metric if m.label == "Trạng thái"] == ["Chưa chạy"]
    # Switching back to Set covering still shows its solved result
    at.sidebar.radio[0].set_value("Set covering").run()
    assert [m.value for m in at.metric if m.label == "Số trạm"] == [str(len(expected_stations))]


def test_comparison_uses_main_capacity_scale_x3():
    from src import distances_l1, models, preprocessing
    from src.app_baselines import compute_nearest_assignment, count_over_capacity

    data_dir = APP.parent / "data"
    demand_df = preprocessing.load_demand(data_dir / "demand.csv")
    cand_df = preprocessing.load_candidates(data_dir / "candidates_feasible.csv")
    params_df = preprocessing.load_station_parameters(data_dir / "station_parameters.csv")
    dist, _manifest = distances_l1.load(data_dir)
    problem_x1 = preprocessing.build_problem_data(demand_df, cand_df, params_df, 300.0, 0.0, cap_scale=1.0)
    col_mask = models.enabled_columns(problem_x1.candidate_ids, cand_df, params_df)
    D_l1, pop, _ids = models.load_inputs("l1", data_dir)
    sc_res = models.run_baseline("set_covering", D_l1, pop, R=300.0)
    selected_full = np.flatnonzero(col_mask)[sc_res["selected"]]
    assign = compute_nearest_assignment(dist, selected_full, problem_x1.candidate_ids, 300.0)
    loads = np.zeros(len(problem_x1.candidate_ids), dtype=float)
    for i, j in enumerate(assign):
        if j >= 0:
            loads[j] += float(problem_x1.daily_demand[i])
    expected_over_x3 = count_over_capacity(selected_full, loads, problem_x1.capacities, capacity_scale=3.0)

    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    # Run main model at default x3
    at.button[0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # Run Set covering at default R=300
    at.sidebar.radio[0].set_value("Set covering").run()
    at.button[0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    assert any("×3" in c.value for c in at.caption)
    comp_df = at.dataframe[2].value
    assert "Trạm vượt sức chứa (×3)" in comp_df.columns
    assert list(comp_df["Mô hình"]) == ["Mô hình chính", "Set covering"]
    assert list(comp_df["Trạm vượt sức chứa (×3)"]) == [0, expected_over_x3]

    # Station costs are integer millions of VND (no decimal commas/localized formatting)
    station_df = at.dataframe[0].value
    assert np.issubdtype(station_df["Chi phí"].dtype, np.integer)
    assert np.issubdtype(station_df["Sức chứa"].dtype, np.integer)
    assert np.issubdtype(station_df["Tải/ngày"].dtype, np.integer)



