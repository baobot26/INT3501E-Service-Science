"""Thành phần giao diện dùng chung cho 3 mô hình baseline và bảng so sánh phiên."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import models, preprocessing, run
from src.config import SolveConfig
from src.make_map import (
    build_preview_map,
    build_solution_map,
    fmt_vi_decimal,
    fmt_vi_int,
    station_type_label,
)
from src.preprocessing import InputError

MAP_HEIGHT = 560
DEFAULT_CAPACITY_SCALE = 3.0
BASELINE_TIME_LIMIT = 30
PMEDIAN_SEED = 0

MODEL_OPTIONS = [
    "Mô hình chính · MILP hai pha",
    "Set covering",
    "Max covering",
    "p-median",
]
MODEL_CAPTIONS = [
    "Ít trạm nhất đạt α, có sức chứa, rồi rẻ nhất",
    "Ít trạm nhất phủ mọi điểm phủ được trong R · không xét sức chứa",
    "p trạm phủ nhiều dân nhất trong R · không xét sức chứa",
    "p trạm, quãng đường × dân nhỏ nhất · không xét R, sức chứa",
]
MODEL_KEY_BY_LABEL = {
    "Mô hình chính · MILP hai pha": "main",
    "Set covering": "set_covering",
    "Max covering": "max_covering",
    "p-median": "p_median",
}
MODEL_SHORT_NAME = {
    "main": "Mô hình chính",
    "set_covering": "Set covering",
    "max_covering": "Max covering",
    "p_median": "p-median",
}
BASELINE_STATUS_LABELS = {
    "optimal": "Tối ưu",
    "time_limit": "Hết thời gian",
    "heuristic": "Heuristic, chưa chứng minh tối ưu",
}
COMPARISON_STATUS_LABELS = {
    "optimal": "Tối ưu",
    "time_limit": "Hết thời gian",
    "heuristic": "Heuristic",
}


def fmt_vi_pct_clean(pct: float) -> str:
    rounded = round(float(pct), 1)
    if abs(rounded - round(rounded)) < 1e-9:
        return f"{round(rounded)}%"
    return f"{fmt_vi_decimal(rounded, 1)}%"


def fmt_vi_cost_ty(cost_vnd: float) -> str:
    ty = float(cost_vnd) / 1e9
    text = f"{ty:.2f}"
    if text.endswith("0"):
        text = text[:-1]
    return f"{text.replace('.', ',')} tỷ đồng"


def format_scale_tag(scale: float) -> str:
    """Định dạng nhãn hệ số sức chứa (vd. 3.0 -> '×3', 2.5 -> '×2,5')."""
    val = float(scale)
    if abs(val - round(val)) < 1e-9:
        return f"×{int(round(val))}"
    return f"×{fmt_vi_decimal(val, 1)}"


def reference_capacity_scale(runs: dict[str, dict]) -> float:
    """Lấy hệ số sức chứa tham chiếu từ lần chạy mô hình chính gần nhất, mặc định ×3."""
    main_entry = runs.get("main")
    if main_entry and main_entry.get("kind") == "solved":
        params = main_entry.get("params") or {}
        if "capacity_scale" in params:
            return float(params["capacity_scale"])
    return DEFAULT_CAPACITY_SCALE


def count_over_capacity(
    selected_idx: np.ndarray | list[int],
    loads: np.ndarray | list[float],
    base_capacities: np.ndarray | list[float],
    capacity_scale: float = DEFAULT_CAPACITY_SCALE,
) -> int:
    """Đếm số trạm được chọn có tải vượt quá sức chứa gốc × capacity_scale."""
    scale = float(capacity_scale)
    return int(
        sum(
            1
            for j in selected_idx
            if float(loads[int(j)]) > float(base_capacities[int(j)]) * scale + 1e-6
        )
    )


@st.cache_data(show_spinner=False)
def render_preview_map_html(
    demand_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    params_df: pd.DataFrame,
    boundary_str: str,
    enabled_flags: tuple[tuple[str, bool], ...],
    theme_mode: str,
) -> str:
    m = build_preview_map(
        demand_df,
        cand_df,
        params_df,
        boundary_path=boundary_str,
        enabled_override=dict(enabled_flags),
        mode=theme_mode,
    )
    return m.get_root().render()


def render_app_header(subtitle: str) -> None:
    st.html(
        f'<div class="app-header">'
        f"<h1>Quy hoạch vị trí đặt tủ đồ thông minh</h1>"
        f"<p>{subtitle}</p>"
        f"</div>"
    )


def render_title_block(cells: list[tuple[str, str, str | None]]) -> None:
    """Hiển thị khung tên dạng dải ô ngăn viền mảnh bằng st.metric."""
    with st.container(key="title_block"):
        cols = st.columns(len(cells), gap=None)
        for col, (label, value, help_text) in zip(cols, cells):
            col.metric(label, value, help=help_text)


def stop_with_state(
    cells: list[tuple[str, str, str | None]],
    message: str,
    *,
    preview_html: str | None = None,
) -> None:
    """Hiển thị khung tên, thông báo lỗi, bản đồ xem trước (nếu có) rồi dừng script."""
    render_title_block(cells)
    st.error(message)
    if preview_html is not None:
        st.iframe(
            preview_html,
            height=MAP_HEIGHT,
            width="stretch",
            alt="Bản đồ xem trước quận Cầu Giấy",
        )
    st.stop()


def compute_nearest_assignment(
    dist: np.ndarray,
    selected_idx: np.ndarray,
    candidate_ids: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Gán mỗi điểm cầu về trạm được chọn gần nhất trong R (đồng hạng chọn ID nhỏ hơn)."""
    n = dist.shape[0]
    assignment = np.full(n, -1, dtype=int)
    if selected_idx.size == 0:
        return assignment
    sub = dist[:, selected_idx]
    for i in range(n):
        d = sub[i]
        within = np.where(d <= radius)[0]
        if within.size:
            ids = candidate_ids[selected_idx[within]]
            order = np.lexsort((ids, d[within]))
            assignment[i] = int(selected_idx[within[order[0]]])
    return assignment


def compute_common_metrics(
    dist: np.ndarray,
    pop: np.ndarray,
    selected_idx: np.ndarray,
    radius: float,
    problem: preprocessing.ProblemData,
    capacity_scale: float = DEFAULT_CAPACITY_SCALE,
) -> dict:
    """Tính tập chỉ số thống nhất (kèm base_capacities ×1) cho baseline và bảng so sánh."""
    assignment = compute_nearest_assignment(dist, selected_idx, problem.candidate_ids, radius)
    loads = np.zeros(len(problem.candidate_ids), dtype=float)
    pop_loads = np.zeros(len(problem.candidate_ids), dtype=float)
    for i, j in enumerate(assignment):
        if j >= 0:
            loads[j] += float(problem.daily_demand[i])
            pop_loads[j] += float(pop[i])

    cap_scale_val = float(getattr(problem, "cap_scale", 1.0) or 1.0)
    base_capacities = np.asarray(problem.capacities, dtype=float) / cap_scale_val
    over_capacity_count = count_over_capacity(
        selected_idx, loads, base_capacities, capacity_scale=capacity_scale
    )
    total_cost = float(problem.fixed_costs[selected_idx].sum()) if selected_idx.size else 0.0

    if selected_idx.size:
        r_int = int(round(radius))
        ev = models.evaluate(dist, pop, selected_idx, radii=[r_int])
        coverage_pct = float(ev[f"cov_{r_int}"])
        avg_dist_m = ev["avg_dist_m"]
        max_dist_m = ev["max_dist_m"]
    else:
        coverage_pct = 0.0
        avg_dist_m = None
        max_dist_m = None

    return {
        "station_count": int(len(selected_idx)),
        "total_cost": total_cost,
        "coverage_pct": coverage_pct,
        "avg_dist_m": avg_dist_m,
        "max_dist_m": max_dist_m,
        "over_capacity_count": over_capacity_count,
        "assignment": assignment,
        "loads": loads,
        "pop_loads": pop_loads,
        "base_capacities": base_capacities,
    }


def render_baseline_type_checkboxes(
    params_df: pd.DataFrame,
    prefix: str,
) -> dict[str, bool]:
    """Hiển thị nhóm checkbox Bật/tắt theo loại trạm cho các mô hình baseline."""
    st.html('<div class="sidebar-group-heading">Loại trạm</div>')
    flags: dict[str, bool] = {}
    for _, row in params_df.iterrows():
        t = str(row["type"])
        flags[t] = st.checkbox(
            station_type_label(t),
            value=bool(row["enabled"]),
            key=f"{prefix}_en_{t}",
        )
    return flags


def current_enabled_candidate_count(
    cand_df: pd.DataFrame,
    params_df: pd.DataFrame,
    prefix: str,
) -> int:
    """Đếm ứng viên khả dụng thuộc loại trạm đang bật theo trạng thái checkbox hiện tại."""
    enabled_types = [
        str(row["type"])
        for _, row in params_df.iterrows()
        if st.session_state.get(f"{prefix}_en_{row['type']}", bool(row["enabled"]))
    ]
    return int(cand_df[cand_df["feasible"] & cand_df["type"].isin(enabled_types)].shape[0])


def clamp_p_state(key: str, max_p: int) -> int:
    """Đảm bảo giá trị p lưu trong session_state nằm trong [1, max_p] trước khi vẽ number_input."""
    default_p = min(31, max_p)
    if key in st.session_state:
        cur = int(st.session_state[key])
        if cur < 1:
            st.session_state[key] = 1
        elif cur > max_p:
            st.session_state[key] = max_p
    return default_p


def format_comparison_params(model_key: str, params: dict) -> str:
    """Tóm tắt tham số ngắn gọn cho bảng so sánh các mô hình."""
    r_txt = f"R={fmt_vi_int(params['radius'])}"
    if model_key == "main":
        a_txt = f"α={fmt_vi_pct_clean(params['alpha'] * 100)}"
        if params["objective_mode"] == "fixed_station_count":
            return f"{r_txt}, {a_txt}, p={params['p_input']}"
        return f"{r_txt}, {a_txt}"
    if model_key == "set_covering":
        return r_txt
    if model_key == "max_covering":
        return f"{r_txt}, p={params['p']}"
    return f"p={params['p']} ({r_txt})"


def build_baseline_header_line(
    model_key: str,
    params: dict,
    area_label: str = "Cầu Giấy (mặc định)",
) -> str:
    """Dòng tóm tắt cấu hình dưới tiêu đề chính cho 3 mô hình baseline."""
    r_txt = fmt_vi_int(params["radius"])
    prefix = f"Khu vực: {area_label}"
    if model_key == "set_covering":
        return f"{prefix} · Set covering · R = {r_txt} m · không xét sức chứa."
    if model_key == "max_covering":
        return (
            f"{prefix} · Max covering · "
            f"R = {r_txt} m · p = {params['p']} · không xét sức chứa."
        )
    return (
        f"{prefix} · p-median · "
        f"p = {params['p']} · R = {r_txt} m chỉ dùng để tính độ phủ · không xét sức chứa."
    )


def render_downloads(
    problem: preprocessing.ProblemData,
    dist: np.ndarray,
    selected: np.ndarray | list[int],
    assignment: np.ndarray,
    loads: np.ndarray | list[float],
    summary_payload: dict,
    *,
    pop: np.ndarray | None = None,
    pop_loads: np.ndarray | None = None,
) -> None:
    """Hiển thị cụm 3 nút tải tệp kết quả (CSV/JSON định dạng máy chuẩn)."""
    if pop_loads is not None:
        station_export_rows = [{
            "candidate_id": int(problem.candidate_ids[j]),
            "type": problem.candidate_types[j],
            "lat": float(problem.cand_lat[j]),
            "lon": float(problem.cand_lon[j]),
            "pop_in_r": float(pop_loads[j]),
            "load": float(loads[j]),
            "capacity": float(problem.capacities[j]),
            "over_capacity": bool(loads[j] > float(problem.capacities[j]) + 1e-6),
            "fixed_cost": float(problem.fixed_costs[j]),
        } for j in selected]
        station_export_cols = [
            "candidate_id", "type", "lat", "lon", "pop_in_r",
            "load", "capacity", "over_capacity", "fixed_cost",
        ]
    else:
        station_export_rows = [{
            "candidate_id": int(problem.candidate_ids[j]),
            "type": problem.candidate_types[j],
            "lat": float(problem.cand_lat[j]),
            "lon": float(problem.cand_lon[j]),
            "capacity": float(problem.capacities[j]),
            "load": float(loads[j]),
            "utilization": float(loads[j]) / float(problem.capacities[j])
            if problem.capacities[j] > 0 else 0.0,
            "fixed_cost": float(problem.fixed_costs[j]),
        } for j in selected]
        station_export_cols = [
            "candidate_id", "type", "lat", "lon", "capacity", "load", "utilization", "fixed_cost",
        ]

    assigned_export_rows = []
    for i, j in enumerate(assignment):
        row: dict[str, object] = {
            "demand_id": int(problem.demand_ids[i]),
            "lat": float(problem.demand_lat[i]),
            "lon": float(problem.demand_lon[i]),
        }
        if pop is not None:
            row["population"] = float(pop[i])
        row.update({
            "daily_demand": float(problem.daily_demand[i]),
            "station_id": int(problem.candidate_ids[j]) if j >= 0 else "",
            "distance_m": float(dist[i, j]) if j >= 0 else float("nan"),
            "served": j >= 0,
        })
        assigned_export_rows.append(row)

    st.html('<h2 class="section-heading">Tải kết quả</h2>')
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button(
            "selected_stations.csv",
            data=pd.DataFrame(station_export_rows, columns=station_export_cols).to_csv(index=False),
            file_name="selected_stations.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
            on_click="ignore",
        )
    with col_d2:
        st.download_button(
            "assigned_demand.csv",
            data=pd.DataFrame(assigned_export_rows).to_csv(index=False),
            file_name="assigned_demand.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
            on_click="ignore",
        )
    with col_d3:
        st.download_button(
            "summary.json",
            data=json.dumps(summary_payload, indent=2, ensure_ascii=False),
            file_name="summary.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
            on_click="ignore",
        )


def handle_baseline_sidebar(
    model_key: str,
    demand_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    params_df: pd.DataFrame,
    dist: np.ndarray,
    runs: dict[str, dict],
) -> dict:
    """Vẽ form thanh bên cho baseline đang chọn và chạy solver khi bấm nút."""
    prefix_map = {"set_covering": "sc", "max_covering": "mc", "p_median": "pm"}
    prefix = prefix_map[model_key]
    enabled_count_now = current_enabled_candidate_count(cand_df, params_df, prefix)
    max_p = max(1, enabled_count_now)
    p_val: int | None = None

    with st.sidebar.form(f"form_{model_key}"):
        st.html('<div class="sidebar-group-heading">Phạm vi</div>')
        if model_key == "set_covering":
            radius = st.slider("Bán kính phục vụ R (m)", 200, 800, 300, 50, key="sc_radius")
        elif model_key == "max_covering":
            radius = st.slider("Bán kính phục vụ R (m)", 200, 800, 300, 50, key="mc_radius")
            default_p = clamp_p_state("mc_p", max_p)
            p_val = int(
                st.number_input(
                    "Số trạm p", min_value=1, max_value=max_p, value=default_p, step=1, key="mc_p"
                )
            )
        else:
            default_p = clamp_p_state("pm_p", max_p)
            p_val = int(
                st.number_input(
                    "Số trạm p", min_value=1, max_value=max_p, value=default_p, step=1, key="pm_p"
                )
            )
            radius = st.slider(
                "Bán kính đánh giá R (m)",
                200,
                800,
                300,
                50,
                key="pm_radius",
                help=(
                    "Chỉ dùng để tính độ phủ trong R và vẽ vùng phục vụ hình thoi L1 trên bản đồ; "
                    "thuật toán p-median không dùng R."
                ),
            )

        type_flags = render_baseline_type_checkboxes(params_df, prefix)
        submitted = st.form_submit_button(
            "Chạy kịch bản",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )

    st.sidebar.caption(
        "Thiết lập cố định: giới hạn MILP 30 giây; p-median 10 lần khởi động lại, seed 0."
    )

    enabled_flags = tuple(sorted((t, bool(on)) for t, on in type_flags.items()))
    form_params = {
        "radius": int(radius),
        "p": p_val,
        "enabled_flags": enabled_flags,
    }

    if submitted:
        type_override = dict(enabled_flags)
        ref_scale = reference_capacity_scale(runs)
        try:
            config = SolveConfig(
                radius=float(radius),
                alpha=0.0,
                objective_mode="min_stations_then_cost",
                station_count=None,
                demand_scale=1.0,
                capacity_scale=1.0,
                time_limit=float(BASELINE_TIME_LIMIT),
                mip_rel_gap=0.0,
                type_parameters={t: {"enabled": on} for t, on in type_override.items()},
            )
            problem = run.build_scenario_problem(config, demand_df, cand_df, params_df)
            col_mask = models.enabled_columns(
                problem.candidate_ids, cand_df, params_df, enabled_override=type_override
            )
            D_filtered = dist[:, col_mask]
            pop = demand_df["population"].to_numpy(float)
            spinner_msg = (
                "Đang chạy heuristic p-median (10 lần khởi động lại)..."
                if model_key == "p_median"
                else f"Đang giải mô hình {MODEL_SHORT_NAME[model_key]} (giới hạn {BASELINE_TIME_LIMIT} giây)..."
            )
            with st.spinner(spinner_msg):
                res = models.run_baseline(
                    model_key,
                    D_filtered,
                    pop,
                    R=float(radius),
                    p=p_val,
                    seed=PMEDIAN_SEED,
                )
            full_selected = np.sort(np.flatnonzero(col_mask)[res["selected"]]).astype(int)
            common = compute_common_metrics(
                dist, pop, full_selected, float(radius), problem, capacity_scale=ref_scale
            )
            runs[model_key] = {
                "kind": "solved",
                "params": form_params,
                "problem": problem,
                "result": res,
                "selected_full": full_selected,
                "common": common,
                "pop": pop,
            }
        except (ValueError, InputError) as exc:
            runs[model_key] = {"kind": "input_error", "params": form_params, "error": str(exc)}
        except RuntimeError as exc:
            runs[model_key] = {"kind": "solver_error", "params": form_params, "error": str(exc)}

    return form_params


def render_baseline_view(
    model_key: str,
    run_state: dict | None,
    active_params: dict,
    runs: dict[str, dict],
    demand_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    params_df: pd.DataFrame,
    dist: np.ndarray,
    preview_html: str,
    theme_mode: str,
    boundary_path: Path,
    *,
    area_label: str = "Cầu Giấy (mặc định)",
    data_dir_rel: str = "data",
) -> None:
    """Hiển thị toàn bộ kết quả (hoặc trạng thái chờ/lỗi) cho mô hình baseline."""
    render_app_header(build_baseline_header_line(model_key, active_params, area_label=area_label))
    base_r_txt = f"{fmt_vi_int(active_params['radius'])} m"

    if run_state is None:
        enabled_types = [t for t, is_on in active_params["enabled_flags"] if is_on]
        enabled_candidates = int(
            cand_df[cand_df["feasible"] & cand_df["type"].isin(enabled_types)].shape[0]
        )
        if model_key == "set_covering":
            idle_cells = [
                ("Số trạm", "—", "Chưa chạy mô hình"),
                ("Ứng viên đang bật", f"{enabled_candidates} điểm", None),
                ("Bán kính R", base_r_txt, "Khoảng cách Manhattan L1 trên EPSG:32648"),
                ("Sức chứa", "Không xét", None),
                ("Trạng thái", "Chưa chạy", f"Giới hạn MILP: {BASELINE_TIME_LIMIT} giây"),
            ]
        elif model_key == "max_covering":
            idle_cells = [
                ("Số trạm", "—", "Chưa chạy mô hình"),
                ("Ứng viên đang bật", f"{enabled_candidates} điểm", None),
                ("Bán kính R", base_r_txt, "Khoảng cách Manhattan L1 trên EPSG:32648"),
                ("Số trạm p", str(active_params["p"]), None),
                ("Trạng thái", "Chưa chạy", f"Giới hạn MILP: {BASELINE_TIME_LIMIT} giây"),
            ]
        else:
            idle_cells = [
                ("Số trạm", "—", "Chưa chạy mô hình"),
                ("Ứng viên đang bật", f"{enabled_candidates} điểm", None),
                ("Số trạm p", str(active_params["p"]), None),
                ("Bán kính đánh giá R", base_r_txt, "Chỉ dùng để tính độ phủ và vẽ vùng phục vụ"),
                ("Trạng thái", "Chưa chạy", "Teitz–Bart · 10 lần khởi động lại, seed 0"),
            ]
        render_title_block(idle_cells)
        st.iframe(
            preview_html,
            height=MAP_HEIGHT,
            width="stretch",
            alt=f"Bản đồ xem trước khu vực {area_label} gồm ranh giới, các điểm ứng viên và phân bố nhu cầu",
        )
        st.html(
            '<p class="map-hint">'
            f"Chỉnh tham số ở thanh bên rồi bấm <strong>Chạy kịch bản</strong> để chạy mô hình {MODEL_SHORT_NAME[model_key]}."
            "</p>"
        )
        st.stop()

    err_preview_html = render_preview_map_html(
        demand_df, cand_df, params_df, str(boundary_path), active_params["enabled_flags"], theme_mode
    )

    if run_state["kind"] == "input_error":
        param_cell = (
            ("Số trạm p", str(active_params["p"]), None)
            if active_params.get("p") is not None
            else ("Sức chứa", "Không xét", None)
        )
        stop_with_state(
            [
                ("Số trạm", "—", None),
                ("Bán kính R", base_r_txt, None),
                param_cell,
                ("Trạng thái", "Tham số không hợp lệ", None),
                ("Thời gian giải", "—", None),
            ],
            f"Không chạy được mô hình {MODEL_SHORT_NAME[model_key]}: {run_state['error']} "
            "Kiểm tra lại loại trạm đang bật và giá trị p ở thanh bên.",
            preview_html=err_preview_html,
        )

    if run_state["kind"] == "solver_error":
        param_cell = (
            ("Số trạm p", str(active_params["p"]), None)
            if active_params.get("p") is not None
            else ("Sức chứa", "Không xét", None)
        )
        stop_with_state(
            [
                ("Số trạm", "—", None),
                ("Bán kính R", base_r_txt, None),
                param_cell,
                ("Trạng thái", "Lỗi solver", None),
                ("Thời gian giải", "—", None),
            ],
            f"Bộ giải gặp lỗi khi chạy {MODEL_SHORT_NAME[model_key]}: {run_state['error']}",
            preview_html=err_preview_html,
        )

    problem = run_state["problem"]
    res = run_state["result"]
    selected_full = run_state["selected_full"]
    common = run_state["common"]
    pop = run_state["pop"]
    metrics = res["metrics"]
    assignment = common["assignment"]
    loads = common["loads"]
    pop_loads = common["pop_loads"]
    base_capacities = common["base_capacities"]

    ref_scale = reference_capacity_scale(runs)
    scale_tag = format_scale_tag(ref_scale)
    over_cap_count = count_over_capacity(selected_full, loads, base_capacities, ref_scale)

    status_display = BASELINE_STATUS_LABELS.get(res["status_label"], res["status_label"])
    avg_dist_txt = (
        f"{fmt_vi_decimal(metrics['avg_dist_m'], 1)} m"
        if metrics["avg_dist_m"] is not None
        else "—"
    )
    elapsed_txt = f"{fmt_vi_decimal(res['seconds'], 1)} giây"
    status_help = (
        f"Mô hình {MODEL_SHORT_NAME[model_key]} · thời gian chạy {elapsed_txt}"
        + (f" · mã MILP {res['status']}" if res["status"] is not None else " · seed 0")
    )

    render_title_block([
        ("Số trạm", str(common["station_count"]), None),
        ("Chi phí tham chiếu", fmt_vi_cost_ty(common["total_cost"]), None),
        ("Độ phủ trong R", f"{fmt_vi_decimal(metrics['coverage_pct'], 1)}%", None),
        ("KC trung bình", avg_dist_txt, None),
        ("Trạng thái", status_display, status_help),
    ])

    note = (
        f"Mô hình này không xét sức chứa. Khi gán mỗi điểm cầu về trạm gần nhất trong R = {base_r_txt}, "
        f"có {over_cap_count} / {common['station_count']} trạm vượt sức chứa {scale_tag} "
        f"(cùng hệ số với mô hình chính)."
    )
    if model_key == "set_covering" and metrics.get("coverable_pct") is not None:
        note += f" Độ phủ tối đa có thể ở R = {base_r_txt} là {fmt_vi_decimal(metrics['coverable_pct'], 2)}%."
    st.caption(note)

    if res["status_label"] == "time_limit":
        st.warning(
            f"Bộ giải chạm giới hạn {BASELINE_TIME_LIMIT} giây trước khi chứng minh tối ưu; "
            "đang hiển thị nghiệm tốt nhất tìm được."
        )

    # Dùng sức chứa theo hệ số tham chiếu để tooltip/popup bản đồ khớp bảng trạm
    scaled_problem = preprocessing.ProblemData(
        demand_ids=problem.demand_ids,
        candidate_ids=problem.candidate_ids,
        demand_xy=problem.demand_xy,
        candidate_xy=problem.candidate_xy,
        daily_demand=problem.daily_demand,
        capacities=base_capacities * ref_scale,
        fixed_costs=problem.fixed_costs,
        candidate_types=problem.candidate_types,
        demand_lat=problem.demand_lat,
        demand_lon=problem.demand_lon,
        cand_lat=problem.cand_lat,
        cand_lon=problem.cand_lon,
        radius=problem.radius,
        alpha=problem.alpha,
        cap_scale=ref_scale,
        enabled_mask=problem.enabled_mask,
    )
    solution_map = build_solution_map(
        problem=scaled_problem,
        dist=dist,
        selected=selected_full,
        assignment=assignment,
        loads=loads,
        boundary_path=boundary_path,
        mode=theme_mode,
    )
    st.iframe(
        solution_map.get_root().render(),
        height=MAP_HEIGHT,
        width="stretch",
        alt="Bản đồ nghiệm baseline gồm trạm được chọn, vùng phục vụ hình thoi L1 và các điểm nhu cầu",
    )

    tab1, tab2 = st.tabs(["Trạm được chọn", "Phân bổ nhu cầu"])

    with tab1:
        rows = []
        for j in selected_full:
            scaled_cap = float(base_capacities[j]) * ref_scale
            rows.append({
                "ID": int(problem.candidate_ids[j]),
                "Loại": station_type_label(problem.candidate_types[j]),
                "Lat": float(problem.cand_lat[j]),
                "Lon": float(problem.cand_lon[j]),
                "Dân trong R": int(round(float(pop_loads[j]))),
                "Tải/ngày": int(round(float(loads[j]))),
                "Sức chứa": int(round(scaled_cap)),
                "Vượt sức chứa": "Có" if float(loads[j]) > scaled_cap + 1e-6 else "—",
                "Chi phí": int(round(float(problem.fixed_costs[j]) / 1e6)),
            })
        station_cols = [
            "ID", "Loại", "Lat", "Lon", "Dân trong R", "Tải/ngày", "Sức chứa", "Vượt sức chứa", "Chi phí",
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=station_cols),
            width="stretch",
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", format="%d"),
                "Loại": st.column_config.TextColumn("Loại trạm"),
                "Lat": st.column_config.NumberColumn("Vĩ độ", format="%.5f"),
                "Lon": st.column_config.NumberColumn("Kinh độ", format="%.5f"),
                "Dân trong R": st.column_config.NumberColumn("Dân trong R", format="%d"),
                "Tải/ngày": st.column_config.NumberColumn("Tải/ngày", format="%d"),
                "Sức chứa": st.column_config.NumberColumn("Sức chứa/ngày", format="%d"),
                "Vượt sức chứa": st.column_config.TextColumn("Vượt sức chứa"),
                "Chi phí": st.column_config.NumberColumn("Chi phí (triệu đồng)", format="%d"),
            },
        )

    with tab2:
        demand_rows = [{
            "ID": int(problem.demand_ids[i]),
            "Dân số": int(round(float(pop[i]))),
            "Trạm": str(int(problem.candidate_ids[j])) if j >= 0 else "—",
            "Khoảng cách (m)": f"{round(float(dist[i, j])):.0f} m" if j >= 0 else "—",
            "Trong R": "Có" if j >= 0 else "—",
        } for i, j in enumerate(assignment)]
        st.dataframe(
            pd.DataFrame(demand_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID điểm cầu", format="%d"),
                "Dân số": st.column_config.NumberColumn("Dân số", format="%d"),
                "Trạm": st.column_config.TextColumn("Trạm gần nhất"),
                "Khoảng cách (m)": st.column_config.TextColumn("KC L1 (m)"),
                "Trong R": st.column_config.TextColumn("Trong R"),
            },
        )

    render_downloads(
        problem,
        dist,
        selected_full,
        assignment,
        loads,
        {
            "model": model_key,
            "data_dir": data_dir_rel,
            "status": res["status_label"],
            "milp_status": res["status"],
            "station_count": common["station_count"],
            "total_cost": common["total_cost"],
            "coverage_pct": metrics["coverage_pct"],
            "avg_dist_m": metrics["avg_dist_m"],
            "max_dist_m": metrics["max_dist_m"],
            "coverable_pct": metrics["coverable_pct"],
            "capacity_scale_ref": ref_scale,
            "over_capacity_count": over_cap_count,
            "seconds": res["seconds"],
            "config": {
                "radius": active_params["radius"],
                "p": active_params.get("p"),
                "enabled_types": [t for t, on in active_params["enabled_flags"] if on],
                "time_limit": BASELINE_TIME_LIMIT,
                "seed": PMEDIAN_SEED if model_key == "p_median" else None,
            },
        },
        pop=pop,
        pop_loads=pop_loads,
    )


def render_comparison_table(
    runs: dict[str, dict],
    demand_df: pd.DataFrame,
    dist: np.ndarray,
) -> None:
    """Hiển thị bảng so sánh các mô hình đã có nghiệm trong phiên làm việc."""
    solved_entries = [
        (key, runs[key])
        for key in ("main", "set_covering", "max_covering", "p_median")
        if key in runs and runs[key].get("kind") == "solved"
    ]
    if not solved_entries:
        return

    ref_scale = reference_capacity_scale(runs)
    scale_tag = format_scale_tag(ref_scale)
    over_cap_col = f"Trạm vượt sức chứa ({scale_tag})"

    pop = demand_df["population"].to_numpy(float)
    rows: list[dict] = []
    for model_key, entry in solved_entries:
        params = entry["params"]
        problem = entry["problem"]
        if model_key == "main":
            res = entry["result"]
            selected_idx = np.asarray(res["selected"], dtype=int)
            common = compute_common_metrics(
                dist, pop, selected_idx, params["radius"], problem, capacity_scale=ref_scale
            )
            over_cap = count_over_capacity(
                selected_idx, common["loads"], common["base_capacities"], ref_scale
            )
            total_cost = float(res["total_cost"])
            status_txt = "Tối ưu" if res["status"] == "optimal" else "Khả thi"
            seconds = float(res.get("total_seconds", 0.0))
        else:
            res = entry["result"]
            common = entry["common"]
            selected_idx = entry["selected_full"]
            over_cap = count_over_capacity(
                selected_idx, common["loads"], common["base_capacities"], ref_scale
            )
            total_cost = float(common["total_cost"])
            status_txt = COMPARISON_STATUS_LABELS.get(res["status_label"], res["status_label"])
            seconds = float(res.get("seconds", 0.0))

        avg_d = common["avg_dist_m"]
        max_d = common["max_dist_m"]
        dist_txt = (
            f"{fmt_vi_decimal(avg_d, 1)} / {fmt_vi_decimal(max_d, 1)}"
            if avg_d is not None and max_d is not None
            else "—"
        )
        rows.append({
            "Mô hình": MODEL_SHORT_NAME[model_key],
            "Tham số": format_comparison_params(model_key, params),
            "Số trạm": common["station_count"],
            "Chi phí": fmt_vi_cost_ty(total_cost).removesuffix(" đồng"),
            "Độ phủ trong R": f"{fmt_vi_decimal(common['coverage_pct'], 1)}%",
            "KC TB · lớn nhất (m)": dist_txt,
            over_cap_col: over_cap,
            "Trạng thái": status_txt,
            "Thời gian": f"{fmt_vi_decimal(seconds, 1)} s",
        })

    st.html('<h2 class="section-heading">So sánh các mô hình đã chạy</h2>')
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Mô hình": st.column_config.TextColumn("Mô hình"),
            "Tham số": st.column_config.TextColumn("Tham số"),
            "Số trạm": st.column_config.NumberColumn("Số trạm", format="%d", width="small"),
            "Chi phí": st.column_config.TextColumn("Chi phí", width="small"),
            "Độ phủ trong R": st.column_config.TextColumn("Độ phủ trong R"),
            "KC TB · lớn nhất (m)": st.column_config.TextColumn("KC TB · lớn nhất (m)"),
            over_cap_col: st.column_config.NumberColumn(over_cap_col, format="%d"),
            "Trạng thái": st.column_config.TextColumn("Trạng thái", width="small"),
            "Thời gian": st.column_config.TextColumn("Thời gian", width="small"),
        },
    )
    st.caption(
        f"Độ phủ trong R: % dân số có ít nhất một trạm được chọn trong bán kính L1 ≤ R. "
        f"Trạm vượt sức chứa tính thống nhất theo sức chứa gốc ×{format_scale_tag(ref_scale).lstrip('×')}."
    )
