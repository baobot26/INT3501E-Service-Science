"""Giao diện Streamlit cho bài toán tối ưu vị trí đặt tủ đồ thông minh.

Chạy:  streamlit run app.py
Dùng chung SolveConfig và hàm giải với CLI (python -m src.run) để hai đường
cho kết quả khớp nhau.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import distances_l1, preprocessing, run
from src.config import SolveConfig
from src.design_tokens import normalize_mode, ui_tokens
from src.make_map import (
    build_preview_map,
    build_solution_map,
    fmt_vi_decimal,
    fmt_vi_int,
    station_type_label,
)
from src.preprocessing import InputError

DATA_DIR = Path(__file__).resolve().parent / "data"
BOUNDARY_PATH = DATA_DIR / "boundary.geojson"
MAP_HEIGHT = 560

SOLUTION_SOURCE_LABELS = {
    "phase2": "Pha 2 (tối thiểu chi phí với số trạm tối ưu)",
    "phase1_fallback": "Pha 1 dự phòng (chưa kịp tối ưu chi phí ở pha 2)",
    "fixed_station_count": "Chế độ cố định số trạm p",
    "phase1": "Pha 1 (tối thiểu số trạm)",
}


def current_theme_mode() -> str:
    """Đọc biến thể sáng/tối hiện hành từ st.context.theme (mặc định 'light')."""
    theme_ctx = getattr(st.context, "theme", None)
    return normalize_mode(getattr(theme_ctx, "type", None))


def build_custom_css(mode: str) -> str:
    """Tạo CSS dùng màu kế thừa cho chrome để đổi tức thì theo theme Streamlit."""
    tokens = ui_tokens(mode)
    return f"""
<style>
  :root {{
    --plan-ground: {tokens["ground"]};
    --plan-panel: {tokens["panel"]};
    --plan-ink: {tokens["ink"]};
    --plan-ink-muted: {tokens["ink_muted"]};
    --plan-line: {tokens["line"]};
    --plan-accent: {tokens["accent"]};
    --plan-accent-hover: {tokens["accent_hover"]};
    --plan-on-accent: {tokens["on_accent"]};
    --plan-selection: {tokens["selection_bg"]};
    --plan-focus: {tokens["focus_ring"]};
    --plan-inherited-muted: color-mix(in srgb, currentColor 78%, transparent);
    --plan-inherited-line: color-mix(in srgb, currentColor 22%, transparent);
  }}

  ::selection {{
    background: var(--plan-selection);
    color: var(--plan-ink);
  }}

  button:focus-visible,
  input:focus-visible,
  select:focus-visible,
  textarea:focus-visible,
  summary:focus-visible,
  a:focus-visible,
  [role="slider"]:focus-visible,
  [role="tab"]:focus-visible,
  [role="combobox"]:focus-visible,
  [tabindex]:focus-visible {{
    outline: 2px solid var(--plan-accent) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 4px var(--plan-focus) !important;
  }}

  .app-header {{
    margin: 0 0 0.65rem 0;
  }}
  .app-header h1 {{
    margin: 0 0 0.2rem 0;
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.25;
    letter-spacing: -0.015em;
    color: inherit;
  }}
  .app-header p {{
    margin: 0;
    font-size: 0.94rem;
    line-height: 1.45;
    color: var(--plan-inherited-muted);
  }}

  /* Khung tên: một dải ô liên tục ngăn bằng viền mảnh 1px, kế thừa nền/chữ Streamlit */
  .st-key-title_block {{
    border: 1px solid var(--plan-inherited-line);
    border-radius: 4px;
    background: transparent;
    margin: 0 0 0.75rem 0;
    overflow: hidden;
  }}
  .st-key-title_block [data-testid="stHorizontalBlock"] {{
    gap: 0 !important;
  }}
  .st-key-title_block [data-testid="stColumn"] {{
    padding: 0 !important;
    border-right: 1px solid var(--plan-inherited-line);
    min-width: 0;
  }}
  .st-key-title_block [data-testid="stColumn"]:last-child {{
    border-right: none;
  }}
  .st-key-title_block [data-testid="stMetric"] {{
    padding: 0.55rem 0.85rem;
    background: transparent;
  }}
  .st-key-title_block [data-testid="stMetricLabel"] {{
    color: var(--plan-inherited-muted);
    font-size: 0.8125rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    margin-bottom: 0.15rem;
  }}
  .st-key-title_block [data-testid="stMetricLabel"] p {{
    font-size: 0.8125rem;
    font-weight: 600;
    color: var(--plan-inherited-muted);
  }}
  .st-key-title_block [data-testid="stMetricValue"] {{
    font-variant-numeric: tabular-nums;
    color: inherit;
    line-height: 1.25;
    white-space: normal;
    overflow-wrap: anywhere;
  }}
  .st-key-title_block [data-testid="stMetricValue"] > div {{
    white-space: normal;
    overflow: visible;
    text-overflow: unset;
  }}

  [data-testid="stMetricValue"],
  [data-testid="stDataFrame"] {{
    font-variant-numeric: tabular-nums;
  }}

  /* Khung bản đồ liền mạch */
  [data-testid="stIFrame"] {{
    border: 1px solid var(--plan-inherited-line);
    border-radius: 4px;
    overflow: hidden;
    display: block;
  }}

  .map-hint {{
    margin: 0.45rem 0 0 0;
    font-size: 0.92rem;
    color: var(--plan-inherited-muted);
  }}

  /* Nhóm tham số trong thanh bên */
  [data-testid="stSidebar"] [data-testid="stForm"] {{
    border: none;
    padding: 0;
    overflow: visible !important;
  }}
  .sidebar-group-heading {{
    font-size: 0.8125rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--plan-inherited-muted);
    margin: 0.95rem 0 0.4rem 0;
    padding-top: 0.65rem;
    border-top: 1px solid var(--plan-inherited-line);
  }}
  .sidebar-group-heading.first {{
    margin-top: 0.2rem;
    padding-top: 0;
    border-top: none;
  }}

  /* Vùng dính đáy thanh bên cần nền đặc theo biến thể theme; nút chính dùng primaryColor của Streamlit */
  [data-testid="stSidebar"] .stElementContainer:has([data-testid="stFormSubmitButton"]),
  [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] {{
    position: sticky;
    bottom: 0;
    z-index: 20;
    background: var(--plan-panel);
  }}
  [data-testid="stSidebar"] .stElementContainer:has([data-testid="stFormSubmitButton"]) {{
    padding: 0.65rem 0 0.5rem 0;
    margin-top: 0.5rem;
    border-top: 1px solid var(--plan-inherited-line);
  }}
  [data-testid="stSidebar"] [data-testid="stBaseButton-primaryFormSubmit"] {{
    font-weight: 600;
  }}

  /* Khoảng trên tiêu đề mục lớn hơn khoảng dưới */
  .section-heading {{
    margin: 1.35rem 0 0.5rem 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: inherit;
  }}

  @media (max-width: 640px) {{
    .app-header {{
      margin: 0 0 0.45rem 0;
    }}
    .app-header h1 {{
      font-size: 1.25rem;
    }}
    .app-header p {{
      font-size: 0.85rem;
    }}
    .st-key-title_block {{
      margin: 0 0 0.5rem 0;
    }}
    .st-key-title_block [data-testid="stHorizontalBlock"] {{
      display: grid !important;
      grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
      gap: 0 !important;
    }}
    .st-key-title_block [data-testid="stColumn"] {{
      width: 100% !important;
      min-width: 0 !important;
      flex: none !important;
      border-right: 1px solid var(--plan-inherited-line);
      border-bottom: 1px solid var(--plan-inherited-line);
    }}
    .st-key-title_block [data-testid="stColumn"]:nth-child(2n) {{
      border-right: none;
    }}
    .st-key-title_block [data-testid="stColumn"]:last-child {{
      grid-column: 1 / -1;
      border-right: none;
      border-bottom: none;
    }}
    .st-key-title_block [data-testid="stMetric"] {{
      padding: 0.35rem 0.65rem;
    }}
    .st-key-title_block [data-testid="stMetricLabel"],
    .st-key-title_block [data-testid="stMetricLabel"] p {{
      font-size: 0.75rem;
    }}
    .st-key-title_block [data-testid="stMetricValue"] {{
      font-size: 1.1rem !important;
    }}
    [data-testid="stIFrame"] {{
      height: 410px !important;
    }}
  }}
</style>
"""


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


@st.cache_data(show_spinner=False)
def load_static_data():
    """Đọc dữ liệu gốc và ma trận khoảng cách Manhattan L1 một lần duy nhất."""
    demand_df = preprocessing.load_demand(DATA_DIR / "demand.csv")
    cand_df = preprocessing.load_candidates(DATA_DIR / "candidates_feasible.csv")
    params_df = preprocessing.load_station_parameters(DATA_DIR / "station_parameters.csv")
    try:
        dist, manifest = distances_l1.load(DATA_DIR)
        preprocessing.validate_distance_manifest(
            dist, manifest, demand_df["id"].values,
            cand_df.loc[cand_df["feasible"], "id"].values,
        )
    except InputError:
        dist, manifest = distances_l1.compute_matrix(demand_df, cand_df, params_df)
    return demand_df, cand_df, params_df, dist, manifest


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


st.set_page_config(
    page_title="Quy hoạch vị trí đặt tủ đồ thông minh — Quận Cầu Giấy",
    page_icon=":material/map:",
    layout="wide",
    initial_sidebar_state="auto",
)
theme_mode = current_theme_mode()
st.html(build_custom_css(theme_mode))

try:
    demand_df, cand_df, params_df, dist, manifest = load_static_data()
except (InputError, FileNotFoundError) as exc:
    st.html(
        '<div class="app-header">'
        "<h1>Quy hoạch vị trí đặt tủ đồ thông minh</h1>"
        "<p>Quận Cầu Giấy cũ (3 phường) · Bài toán tối ưu vị trí trạm dịch vụ theo khoảng cách Manhattan L1.</p>"
        "</div>"
    )
    st.error(
        f"Thiếu hoặc lỗi dữ liệu đầu vào: {exc}. "
        "Chạy lệnh `python -m src.polygons --mode B` để tái tạo dữ liệu trước khi khởi chạy ứng dụng."
    )
    st.stop()

# --- Thanh bên: form tham số kịch bản ---
st.sidebar.header("Tham số kịch bản")

with st.sidebar.form("scenario"):
    st.html('<div class="sidebar-group-heading first">Phạm vi và mục tiêu</div>')
    # Giới hạn R <= 800 m để solver HiGHS giải ổn định trong giới hạn thời gian tương tác
    radius = st.slider("Bán kính phục vụ R (m)", 200, 800, 300, 50)
    alpha = st.slider("Tỷ lệ phủ tối thiểu α", 0.0, 1.0, 0.6, 0.05)

    objective_label = st.selectbox(
        "Mục tiêu",
        [
            "Ít trạm nhất, rồi rẻ nhất",
            "Cố định số trạm p, tối thiểu chi phí (thử nghiệm)",
        ],
    )
    objective_mode = (
        "min_stations_then_cost"
        if objective_label.startswith("Ít")
        else "fixed_station_count"
    )

    p_input = st.number_input(
        "Số trạm p (chỉ dùng khi cố định p)",
        min_value=0,
        max_value=int(dist.shape[1]),
        value=min(31, int(dist.shape[1])),
        step=1,
        help="Chế độ cố định số trạm p có thể chạm giới hạn thời gian trên dữ liệu thực; dùng để thử nghiệm.",
    )
    station_count = p_input if objective_mode == "fixed_station_count" else None

    st.html('<div class="sidebar-group-heading">Hệ số</div>')
    demand_scale = st.slider("Hệ số nhu cầu", 0.1, 5.0, 1.0, 0.1)
    capacity_scale = st.slider("Hệ số sức chứa", 0.5, 10.0, 3.0, 0.5)

    st.html('<div class="sidebar-group-heading">Loại trạm</div>')
    type_params = {}
    for _, row in params_df.iterrows():
        t = str(row["type"])
        with st.expander(station_type_label(t), expanded=False):
            type_params[t] = {
                "enabled": st.checkbox("Bật", value=bool(row["enabled"]), key=f"en_{t}"),
                "capacity": st.number_input(
                    "Sức chứa/ngày",
                    min_value=1,
                    value=int(row["capacity_demand_day"]),
                    key=f"cap_{t}",
                ),
                "fixed_cost": st.number_input(
                    "Chi phí cố định",
                    min_value=0,
                    value=int(row["fixed_cost"]),
                    key=f"cost_{t}",
                ),
            }

    st.html('<div class="sidebar-group-heading">Solver</div>')
    # Mặc định 30 giây: kịch bản mặc định mất ~15 giây trong container Docker
    time_limit = st.slider("Giới hạn thời gian (giây)", 10, 300, 30, 10)
    mip_gap = st.slider("MIP gap", 0.0, 0.2, 0.0, 0.01)

    submitted = st.form_submit_button(
        "Chạy kịch bản",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
    )

enabled_flags = tuple(sorted((t, bool(cfg["enabled"])) for t, cfg in type_params.items()))
preview_html = render_preview_map_html(
    demand_df, cand_df, params_df, str(BOUNDARY_PATH), enabled_flags, theme_mode
)

if submitted:
    st.session_state["last_run"] = None
    snapshot_params = {
        "radius": radius,
        "alpha": alpha,
        "objective_mode": objective_mode,
        "station_count": station_count,
        "p_input": int(p_input),
        "demand_scale": demand_scale,
        "capacity_scale": capacity_scale,
        "type_params": type_params,
        "time_limit": time_limit,
        "mip_gap": mip_gap,
        "enabled_flags": enabled_flags,
    }
    try:
        config = SolveConfig(
            radius=radius,
            alpha=alpha,
            objective_mode=objective_mode,
            station_count=station_count,
            demand_scale=demand_scale,
            capacity_scale=capacity_scale,
            type_parameters=type_params,
            time_limit=time_limit,
            mip_rel_gap=mip_gap,
        )
    except ValueError as exc:
        st.session_state["last_run"] = {"kind": "config_error", "params": snapshot_params, "error": str(exc)}
    else:
        try:
            problem = run.build_scenario_problem(config, demand_df, cand_df, params_df)
        except InputError as exc:
            st.session_state["last_run"] = {"kind": "data_error", "params": snapshot_params, "error": str(exc)}
        else:
            pre = run.precheck(dist, problem)
            if pre["status"] != "ok":
                st.session_state["last_run"] = {"kind": "precheck_error", "params": snapshot_params, "pre": pre}
            else:
                with st.spinner(f"Đang giải mô hình MILP hai pha (giới hạn {time_limit} giây)..."):
                    result, validation = run.solve_scenario(config, problem, dist, manifest)
                st.session_state["last_run"] = {
                    "kind": "solved",
                    "params": snapshot_params,
                    "problem": problem,
                    "result": result,
                    "validation": validation,
                }

run_state = st.session_state.get("last_run")
active_params = run_state["params"] if run_state is not None else {
    "radius": radius,
    "alpha": alpha,
    "objective_mode": objective_mode,
    "p_input": int(p_input),
    "demand_scale": demand_scale,
    "capacity_scale": capacity_scale,
    "type_params": type_params,
    "time_limit": time_limit,
    "mip_gap": mip_gap,
    "station_count": station_count,
    "enabled_flags": enabled_flags,
}

objective_short = (
    "Ít trạm nhất, rồi tối thiểu chi phí"
    if active_params["objective_mode"] == "min_stations_then_cost"
    else f"Cố định p = {active_params['p_input']} trạm, tối thiểu chi phí"
)
scenario_line = (
    f"Quận Cầu Giấy cũ (3 phường) · R = {fmt_vi_int(active_params['radius'])} m, "
    f"α = {fmt_vi_pct_clean(active_params['alpha'] * 100)}, "
    f"sức chứa ×{fmt_vi_decimal(active_params['capacity_scale'], 1)}, "
    f"nhu cầu ×{fmt_vi_decimal(active_params['demand_scale'], 1)} · {objective_short}."
)
st.html(
    f'<div class="app-header">'
    f"<h1>Quy hoạch vị trí đặt tủ đồ thông minh</h1>"
    f"<p>{scenario_line}</p>"
    f"</div>"
)

if run_state is None:
    enabled_types = [t for t, is_on in enabled_flags if is_on]
    enabled_candidates = int(
        cand_df[cand_df["feasible"] & cand_df["type"].isin(enabled_types)].shape[0]
    )
    render_title_block([
        ("Số trạm", "—", "Chưa chạy tối ưu"),
        ("Ứng viên đang bật", f"{enabled_candidates} điểm", None),
        ("Bán kính R", f"{fmt_vi_int(radius)} m", "Khoảng cách Manhattan L1 trên EPSG:32648"),
        ("Độ phủ mục tiêu α", fmt_vi_pct_clean(alpha * 100), None),
        ("Trạng thái", "Chưa chạy", f"Giới hạn giải hiện tại: {time_limit} giây"),
    ])
    st.iframe(
        preview_html,
        height=MAP_HEIGHT,
        width="stretch",
        alt="Bản đồ xem trước quận Cầu Giấy gồm ranh giới 3 phường, các điểm ứng viên và phân bố nhu cầu",
    )
    st.html(
        '<p class="map-hint">'
        "Chỉnh tham số ở thanh bên rồi bấm <strong>Chạy kịch bản</strong> để giải mô hình MILP hai pha."
        "</p>"
    )
    st.stop()

err_preview_html = render_preview_map_html(
    demand_df, cand_df, params_df, str(BOUNDARY_PATH), active_params["enabled_flags"], theme_mode
)
base_r_txt = f"{fmt_vi_int(active_params['radius'])} m"
base_alpha_txt = fmt_vi_pct_clean(active_params["alpha"] * 100)

if run_state["kind"] == "config_error":
    stop_with_state(
        [
            ("Số trạm", "—", None),
            ("Bán kính R", base_r_txt, None),
            ("Độ phủ mục tiêu α", base_alpha_txt, None),
            ("Trạng thái", "Cấu hình không hợp lệ", None),
            ("Thời gian giải", "—", None),
        ],
        f"Tham số kịch bản không hợp lệ: {run_state['error']}. "
        "Kiểm tra lại bán kính R, tỷ lệ phủ α, số trạm p và các hệ số ở thanh bên.",
        preview_html=err_preview_html,
    )

if run_state["kind"] == "data_error":
    stop_with_state(
        [
            ("Số trạm", "—", None),
            ("Bán kính R", base_r_txt, None),
            ("Độ phủ mục tiêu α", base_alpha_txt, None),
            ("Trạng thái", "Lỗi dữ liệu", None),
            ("Thời gian giải", "—", None),
        ],
        f"Không dựng được bài toán từ dữ liệu hiện tại: {run_state['error']}. "
        "Đảm bảo có ít nhất một loại trạm được bật và dữ liệu đầu vào hợp lệ.",
        preview_html=err_preview_html,
    )

if run_state["kind"] == "precheck_error":
    pre = run_state["pre"]
    if pre["status"] == "precheck_infeasible_coverage":
        pre_msg = (
            f"Không khả thi: với R = {base_r_txt} độ phủ tối đa chỉ "
            f"{fmt_vi_decimal(pre['coverage_upper_bound_pct'], 1)}%, thấp hơn "
            f"α = {fmt_vi_pct_clean(pre['required_coverage_pct'])}. "
            "Tăng R, bật thêm loại trạm, hoặc giảm α."
        )
    else:
        pre_msg = (
            f"Không khả thi: tổng sức chứa các ứng viên đang bật chỉ đạt "
            f"{fmt_vi_int(pre['capacity_upper_bound'])} đơn vị/ngày, thấp hơn mức cần "
            f"{fmt_vi_int(pre['required_capacity'])} đơn vị/ngày "
            f"(α = {fmt_vi_pct_clean(pre['required_coverage_pct'])}). "
            "Tăng hệ số sức chứa, bật thêm loại trạm, hoặc giảm α."
        )
    stop_with_state(
        [
            ("Số trạm", "—", None),
            ("Bán kính R", base_r_txt, None),
            ("Độ phủ tối đa", f"{fmt_vi_decimal(pre['coverage_upper_bound_pct'], 1)}%", None),
            ("Trạng thái", "Không khả thi", f"Bị loại ở bước kiểm tra sơ bộ ({pre['status']})"),
            ("Thời gian giải", "0,0 giây", None),
        ],
        pre_msg,
        preview_html=err_preview_html,
    )

problem = run_state["problem"]
result = run_state["result"]
validation = run_state["validation"]
status = result["status"]
solution_source = result.get("solution_source", "N/A")
source_desc = SOLUTION_SOURCE_LABELS.get(solution_source, str(solution_source))
elapsed_txt = f"{fmt_vi_decimal(result.get('total_seconds', 0), 1)} giây"

if status == "time_limit_no_solution":
    stop_with_state(
        [
            ("Số trạm", "—", None),
            ("Bán kính R", base_r_txt, None),
            ("Độ phủ mục tiêu α", base_alpha_txt, None),
            ("Trạng thái", "Hết thời gian", f"Nguồn: {source_desc}"),
            ("Thời gian giải", elapsed_txt, None),
        ],
        f"Hết thời gian ({active_params['time_limit']} giây) mà không tìm được nghiệm khả thi. "
        "Tăng giới hạn thời gian, tăng MIP gap, giảm α, hoặc giảm bán kính R.",
        preview_html=err_preview_html,
    )

if status == "infeasible":
    if active_params["objective_mode"] == "fixed_station_count":
        infeasible_msg = (
            f"Solver chứng minh vô nghiệm khi cố định p = {active_params['p_input']} trạm "
            f"(R = {base_r_txt}, α = {base_alpha_txt}, "
            f"sức chứa ×{fmt_vi_decimal(active_params['capacity_scale'], 1)}). "
            "Số trạm p không đủ để đạt độ phủ hoặc gây quá tải cục bộ khi gán trạm gần nhất; "
            "hãy tăng p, tăng hệ số sức chứa, hoặc giảm α."
        )
    else:
        infeasible_msg = (
            f"Solver chứng minh vô nghiệm với R = {base_r_txt}, α = {base_alpha_txt} "
            f"và sức chứa ×{fmt_vi_decimal(active_params['capacity_scale'], 1)}: thường do quá tải cục bộ "
            "khi gán trọn từng điểm cầu cho trạm gần nhất. "
            "Tăng hệ số sức chứa, bật thêm loại trạm, hoặc giảm α."
        )
    stop_with_state(
        [
            ("Số trạm", "—", None),
            ("Bán kính R", base_r_txt, None),
            ("Độ phủ mục tiêu α", base_alpha_txt, None),
            ("Trạng thái", "Vô nghiệm", f"Nguồn: {source_desc}"),
            ("Thời gian giải", elapsed_txt, None),
        ],
        infeasible_msg,
        preview_html=err_preview_html,
    )

if status == "solver_error":
    stop_with_state(
        [
            ("Số trạm", "—", None),
            ("Bán kính R", base_r_txt, None),
            ("Độ phủ mục tiêu α", base_alpha_txt, None),
            ("Trạng thái", "Lỗi solver", f"Nguồn: {source_desc}"),
            ("Thời gian giải", elapsed_txt, None),
        ],
        "Bộ giải HiGHS gặp lỗi kỹ thuật khi chạy kịch bản này. "
        "Thử điều chỉnh giới hạn thời gian hoặc kiểm tra lại tham số đầu vào.",
        preview_html=err_preview_html,
    )

if not validation["valid"]:
    stop_with_state(
        [
            ("Số trạm", str(result["station_count"]), None),
            ("Tổng chi phí", fmt_vi_cost_ty(result["total_cost"]), None),
            ("Độ phủ", f"{fmt_vi_decimal(validation['metrics']['coverage_pct'], 1)}%", None),
            ("Trạng thái", "Lỗi kiểm định", f"Nguồn: {source_desc}"),
            ("Thời gian giải", elapsed_txt, None),
        ],
        "Kiểm định độc lập nghiệm thất bại: một hoặc nhiều ràng buộc phủ / sức chứa không khớp. "
        "Thử giảm MIP gap về 0 hoặc chạy lại với giới hạn thời gian lớn hơn.",
    )

selected, assignment = result["selected"], result["assignment"]
loads = validation["loads"]

status_display = (
    "Tối ưu"
    if status == "optimal"
    else "Khả thi, chưa chứng minh tối ưu"
)
status_help = (
    f"Mã trạng thái: {status}. Nguồn nghiệm: {source_desc}. "
    f"R = {base_r_txt}, α = {base_alpha_txt}."
)

render_title_block([
    ("Số trạm", str(result["station_count"]), None),
    ("Tổng chi phí", fmt_vi_cost_ty(result["total_cost"]), None),
    ("Độ phủ", f"{fmt_vi_decimal(validation['metrics']['coverage_pct'], 1)}%", None),
    ("Trạng thái", status_display, status_help),
    ("Thời gian giải", elapsed_txt, None),
])

if status == "feasible_time_limit":
    st.warning(
        f"Đã tìm được nghiệm khả thi nhưng hết thời gian ({active_params['time_limit']} giây) "
        "trước khi chứng minh tối ưu. Tăng giới hạn thời gian hoặc đặt MIP gap nhỏ nếu cần chứng minh tối ưu."
    )

solution_map = build_solution_map(
    problem=problem,
    dist=dist,
    selected=selected,
    assignment=assignment,
    loads=loads,
    boundary_path=BOUNDARY_PATH,
    mode=theme_mode,
)
st.iframe(
    solution_map.get_root().render(),
    height=MAP_HEIGHT,
    width="stretch",
    alt="Bản đồ nghiệm gồm trạm được chọn, vùng phục vụ hình thoi L1 và các điểm nhu cầu",
)

# --- Bảng chi tiết dưới bản đồ ---
tab1, tab2 = st.tabs(["Trạm được chọn", "Phân bổ nhu cầu"])

with tab1:
    rows = [{
        "ID": int(problem.candidate_ids[j]),
        "Loại": station_type_label(problem.candidate_types[j]),
        "Lat": float(problem.cand_lat[j]),
        "Lon": float(problem.cand_lon[j]),
        "Sức chứa": float(problem.capacities[j]),
        "Tải/ngày": float(loads[j]),
        "Hiệu suất %": round(loads[j] / problem.capacities[j] * 100, 1)
        if problem.capacities[j] > 0 else 0.0,
        "Chi phí": float(problem.fixed_costs[j]),
    } for j in selected]
    station_table = pd.DataFrame(rows)
    st.dataframe(
        station_table,
        width="stretch",
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID", format="%d"),
            "Loại": st.column_config.TextColumn("Loại trạm"),
            "Lat": st.column_config.NumberColumn("Vĩ độ", format="%.5f"),
            "Lon": st.column_config.NumberColumn("Kinh độ", format="%.5f"),
            "Sức chứa": st.column_config.NumberColumn("Sức chứa/ngày", format="localized"),
            "Tải/ngày": st.column_config.NumberColumn("Tải/ngày", format="%.1f"),
            "Hiệu suất %": st.column_config.NumberColumn("Hiệu suất (%)", format="%.1f%%"),
            "Chi phí": st.column_config.NumberColumn("Chi phí cố định (đồng)", format="localized"),
        },
    )

with tab2:
    demand_rows = [{
        "ID": int(problem.demand_ids[i]),
        "Nhu cầu/ngày": float(problem.daily_demand[i]),
        "Trạm": str(int(problem.candidate_ids[j])) if j >= 0 else "—",
        "Khoảng cách (m)": f"{round(float(dist[i, j])):.0f} m" if j >= 0 else "—",
        "Phục vụ": "Được phục vụ" if j >= 0 else "Chưa phục vụ",
    } for i, j in enumerate(assignment)]
    demand_table = pd.DataFrame(demand_rows)
    st.dataframe(
        demand_table,
        width="stretch",
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID điểm cầu", format="%d"),
            "Nhu cầu/ngày": st.column_config.NumberColumn("Nhu cầu/ngày", format="%.1f"),
            "Trạm": st.column_config.TextColumn("Trạm phục vụ"),
            "Khoảng cách (m)": st.column_config.TextColumn("Khoảng cách L1 (m)"),
            "Phục vụ": st.column_config.TextColumn("Trạng thái"),
        },
    )

# --- Tải kết quả (giữ nguyên định dạng máy CSV/JSON dấu chấm thập phân) ---
st.html('<h2 class="section-heading">Tải kết quả</h2>')
station_export_rows = [{
    "candidate_id": int(problem.candidate_ids[j]),
    "type": problem.candidate_types[j],
    "lat": problem.cand_lat[j],
    "lon": problem.cand_lon[j],
    "capacity": problem.capacities[j],
    "load": loads[j],
    "utilization": loads[j] / problem.capacities[j] if problem.capacities[j] > 0 else 0.0,
    "fixed_cost": problem.fixed_costs[j],
} for j in selected]
station_export_cols = [
    "candidate_id", "type", "lat", "lon", "capacity", "load", "utilization", "fixed_cost",
]
assigned_export_rows = [{
    "demand_id": int(problem.demand_ids[i]),
    "lat": problem.demand_lat[i],
    "lon": problem.demand_lon[i],
    "daily_demand": problem.daily_demand[i],
    "station_id": int(problem.candidate_ids[j]) if j >= 0 else "",
    "distance_m": float(dist[i, j]) if j >= 0 else float("nan"),
    "served": j >= 0,
} for i, j in enumerate(assignment)]

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
        data=json.dumps({
            "status": status,
            "station_count": result["station_count"],
            "total_cost": result["total_cost"],
            "coverage_pct": validation["metrics"]["coverage_pct"],
            "solution_source": result.get("solution_source"),
            "config": {
                "radius": active_params["radius"],
                "alpha": active_params["alpha"],
                "objective_mode": active_params["objective_mode"],
                "station_count": active_params["station_count"],
                "demand_scale": active_params["demand_scale"],
                "capacity_scale": active_params["capacity_scale"],
                "type_parameters": active_params["type_params"],
                "time_limit": active_params["time_limit"],
                "mip_rel_gap": active_params["mip_gap"],
            },
        }, indent=2, ensure_ascii=False),
        file_name="summary.json",
        mime="application/json",
        icon=":material/download:",
        width="stretch",
        on_click="ignore",
    )
