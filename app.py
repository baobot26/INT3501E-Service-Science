"""Streamlit app for the service-science locker placement model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import distances_l1, evaluate, model_main, preprocessing
from src.config import SolveConfig
from src.preprocessing import InputError

DATA_DIR = Path("data")


@st.cache_data(show_spinner=False)
def load_static_data():
    """Load raw inputs and distance matrix once."""
    demand_df = preprocessing.load_demand(DATA_DIR / "demand.csv")
    cand_df = preprocessing.load_candidates(DATA_DIR / "candidates_feasible.csv")
    params_df = preprocessing.load_station_parameters(DATA_DIR / "station_parameters.csv")
    dist, manifest = distances_l1.load(DATA_DIR)
    return demand_df, cand_df, params_df, dist, manifest


@st.cache_data(show_spinner=False)
def get_station_types(params_df_hash: str) -> list[str]:
    """Return list of station types from params CSV."""
    _, _, params_df, _, _ = load_static_data()
    return params_df["type"].tolist()


st.set_page_config(page_title="Locker Placement Optimizer", layout="wide")
st.title("🔒 Tối ưu vị trí đặt tủ đồ")

try:
    demand_df, cand_df, params_df, dist, manifest = load_static_data()
except (InputError, FileNotFoundError) as e:
    st.error(f"Thiếu dữ liệu đầu vào: {e}. Chạy preprocessing trước.")
    st.stop()

# --- Sidebar: scenario form ---
st.sidebar.header("⚙️ Tham số kịch bản")

with st.sidebar.form("scenario"):
    radius = st.slider("Bán kính phục vụ R (m)", 200, 2000, 800, 50)
    alpha = st.slider("Tỷ lệ phủ tối thiểu α", 0.0, 1.0, 0.95, 0.05)

    objective = st.selectbox("Mục tiêu", [
        "Ít trạm nhất rồi rẻ nhất",
        "Cố định số trạm, tối thiểu chi phí",
    ])
    objective_mode = "min_stations_then_cost" if objective.startswith("Ít") else "fixed_station_count"

    station_count = None
    if objective_mode == "fixed_station_count":
        station_count = st.number_input("Số trạm p", min_value=0, max_value=len(cand_df),
                                        value=10, step=1)

    st.subheader("Hệ số điều chỉnh")
    demand_scale = st.slider("Hệ số nhu cầu", 0.1, 5.0, 1.0, 0.1)
    capacity_scale = st.slider("Hệ số sức chứa", 0.5, 10.0, 3.0, 0.5)

    st.subheader("Tham số loại trạm")
    type_params = {}
    for _, row in params_df.iterrows():
        t = row["type"]
        with st.expander(t, expanded=False):
            enabled = st.checkbox("Bật", value=bool(row["enabled"]), key=f"en_{t}")
            cap = st.number_input("Sức chứa/ngày", min_value=1,
                                  value=int(row["capacity_demand_day"]),
                                  key=f"cap_{t}")
            cost = st.number_input("Chi phí cố định", min_value=0,
                                   value=int(row["fixed_cost"]),
                                   key=f"cost_{t}")
            type_params[t] = {"enabled": enabled, "capacity": cap, "fixed_cost": cost}

    st.subheader("Solver")
    time_limit = st.slider("Giới hạn thời gian (giây)", 10, 300, 120, 10)
    mip_gap = st.slider("MIP gap", 0.0, 0.2, 0.0, 0.01)

    submitted = st.form_submit_button("🚀 Chạy kịch bản", type="primary")

# --- Main area: results ---
if submitted:
    try:
        config = SolveConfig(
            radius=radius, alpha=alpha,
            objective_mode=objective_mode,
            station_count=station_count,
            demand_scale=demand_scale,
            capacity_scale=capacity_scale,
            type_parameters=type_params,
            time_limit=time_limit,
            mip_rel_gap=mip_gap,
        )
    except ValueError as e:
        st.error(f"Cấu hình không hợp lệ: {e}")
        st.stop()

    try:
        problem = preprocessing.build_problem_data(
            demand_df, cand_df, params_df,
            radius, alpha,
            cap_scale=capacity_scale,
            demand_scale=demand_scale,
            type_parameters=type_params,
        )
        preprocessing.validate_distance_manifest(
            dist, manifest, problem.demand_ids, problem.candidate_ids)
    except InputError as e:
        st.error(f"Lỗi dữ liệu: {e}")
        st.stop()

    with st.spinner("Đang giải MILP..."):
        result = model_main.solve(
            problem, dist,
            time_limit=config.time_limit,
            mip_rel_gap=config.mip_rel_gap,
            objective_mode=config.objective_mode,
            fixed_station_count=config.station_count,
        )

    status = result["status"]
    if status in ("infeasible", "solver_error"):
        st.error(f"Không tìm được nghiệm: {status}")
        st.stop()

    if status == "feasible_time_limit":
        st.warning("⏱️ Nghiệm khả thi nhưng chưa chứng minh tối ưu (hết thời gian).")

    selected, assignment = result["selected"], result["assignment"]
    validation = evaluate.validate_solution(
        problem, dist, selected, assignment,
        result["station_count"], result["total_cost"])

    # --- Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Số trạm", result["station_count"])
    col2.metric("Tổng chi phí", f"{result['total_cost'] / 1e9:.2f} tỷ")
    col3.metric("Độ phủ", f"{validation['metrics']['coverage_pct']:.1f}%")
    col4.metric("Thời gian", f"{result.get('total_seconds', 0):.1f}s")

    st.caption(f"Trạng thái: **{status}** | Nguồn: {result.get('solution_source', 'N/A')}")

    # --- Tables ---
    tab1, tab2, tab3 = st.tabs(["🏪 Trạm được chọn", "📊 Phân bổ nhu cầu", "🗺️ Bản đồ"])

    with tab1:
        rows = []
        for j in selected:
            rows.append({
                "ID": problem.candidate_ids[j],
                "Loại": problem.candidate_types[j],
                "Lat": problem.cand_lat[j],
                "Lon": problem.cand_lon[j],
                "Sức chứa": problem.capacities[j],
                "Chi phí": problem.fixed_costs[j],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab2:
        demand_rows = []
        for i in range(len(problem.demand_ids)):
            j = assignment[i]
            demand_rows.append({
                "ID": problem.demand_ids[i],
                "Nhu cầu/ngày": problem.daily_demand[i],
                "Trạm": int(problem.candidate_ids[j]) if j >= 0 else "—",
                "Khoảng cách (m)": round(dist[i, j], 0) if j >= 0 else "—",
                "Phục vụ": "✓" if j >= 0 else "✗",
            })
        st.dataframe(pd.DataFrame(demand_rows), use_container_width=True, hide_index=True)

    with tab3:
        # Build map inline using folium
        import folium
        from src.make_map import COLORS

        m = folium.Map(location=[problem.demand_lat.mean(), problem.demand_lon.mean()],
                       zoom_start=14, tiles="OpenStreetMap")

        # demand points
        for i in range(len(problem.demand_ids)):
            served = assignment[i] >= 0
            folium.CircleMarker(
                [problem.demand_lat[i], problem.demand_lon[i]],
                radius=2, weight=0, fill=True, fill_opacity=0.6,
                color="#4c9f70" if served else "#e45756",
            ).add_to(m)

        # selected stations
        for j in selected:
            color = COLORS.get(problem.candidate_types[j], "#333")
            folium.CircleMarker(
                [problem.cand_lat[j], problem.cand_lon[j]],
                radius=7, color="#000", weight=1,
                fill=True, fill_color=color, fill_opacity=0.95,
            ).add_to(m)
            folium.Circle(
                [problem.cand_lat[j], problem.cand_lon[j]],
                radius=radius, color=color, weight=1,
                fill=True, fill_opacity=0.05,
            ).add_to(m)

        st.components.v1.html(m._repr_html_(), height=500, scrolling=True)

    # --- Downloads ---
    st.subheader("📥 Tải kết quả")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "selected_stations.csv",
            data=pd.DataFrame([{
                "candidate_id": int(problem.candidate_ids[j]),
                "type": problem.candidate_types[j],
                "lat": problem.cand_lat[j],
                "lon": problem.cand_lon[j],
                "fixed_cost": problem.fixed_costs[j],
            } for j in selected]).to_csv(index=False),
            file_name="selected_stations.csv",
            mime="text/csv",
        )
    with col_d2:
        st.download_button(
            "summary.json",
            data=json.dumps({
                "status": status,
                "station_count": result["station_count"],
                "total_cost": result["total_cost"],
                "coverage_pct": validation["metrics"]["coverage_pct"],
                "solution_source": result.get("solution_source"),
                "config": {
                    "radius": radius, "alpha": alpha,
                    "objective_mode": objective_mode,
                    "station_count": station_count,
                    "demand_scale": demand_scale,
                    "capacity_scale": capacity_scale,
                },
            }, indent=2, ensure_ascii=False),
            file_name="summary.json",
            mime="application/json",
        )

else:
    st.info("👈 Điều chỉnh tham số ở sidebar rồi bấm **Chạy kịch bản**.")