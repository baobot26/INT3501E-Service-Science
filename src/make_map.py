"""Vẽ bản đồ HTML cho một lần chạy của mô hình chính.

Dùng: python -m src.make_map              (lấy lần chạy mới nhất trong results/)
      python -m src.make_map --run results/<tên_thư_mục>

Đọc selected_stations.csv, assigned_demand.csv, summary.json trong thư mục chạy,
ghi ra map.html cùng thư mục. Mở bằng trình duyệt là xem được, không cần server.
"""
import argparse
import json
from pathlib import Path

import folium
import pandas as pd

COLORS = {
    "shop=convenience": "#1f77b4",
    "shop=supermarket": "#2ca02c",
    "amenity=post_office": "#d62728",
    "amenity=parking": "#9467bd",
    "building=apartments": "#ff7f0e",
}


def latest_run(results_dir):
    runs = [p for p in Path(results_dir).iterdir() if (p / "selected_stations.csv").exists()]
    if not runs:
        raise SystemExit(f"không có lần chạy nào trong {results_dir}, hãy chạy src.run trước")
    return max(runs, key=lambda p: p.stat().st_mtime)


def build_map(run_dir, boundary_path):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    stations = pd.read_csv(run_dir / "selected_stations.csv")
    demand = pd.read_csv(run_dir / "assigned_demand.csv")
    radius = summary.get("radius_m", 500)

    m = folium.Map(location=[demand["lat"].mean(), demand["lon"].mean()],
                   zoom_start=14, tiles="OpenStreetMap")

    if Path(boundary_path).exists():
        folium.GeoJson(str(boundary_path), name="Ranh giới",
                       style_function=lambda _: {"color": "#444", "weight": 2, "fill": False}
                       ).add_to(m)

    # điểm nhu cầu: xanh là được phục vụ, đỏ là không
    g_served = folium.FeatureGroup(name="Dân được phục vụ", show=True)
    g_missed = folium.FeatureGroup(name="Dân không được phục vụ", show=True)
    for r in demand.itertuples():
        served = bool(r.served)
        folium.CircleMarker(
            [r.lat, r.lon], radius=2, weight=0, fill=True, fill_opacity=0.6,
            color="#4c9f70" if served else "#e45756",
            tooltip=f"nhu cầu {r.daily_demand:.1f}/ngày"
                    + (f", đi {r.distance_m:.0f} m" if served else ""),
        ).add_to(g_served if served else g_missed)
    g_served.add_to(m)
    g_missed.add_to(m)

    # trạm được chọn, vòng tròn là bán kính R (khoảng cách thật là Manhattan nên vòng chỉ để tham khảo)
    g_st = folium.FeatureGroup(name="Trạm được chọn", show=True)
    g_circle = folium.FeatureGroup(name=f"Bán kính {radius:g} m", show=False)
    for r in stations.itertuples():
        color = COLORS.get(r.type, "#333")
        popup = (f"<b>{r.type}</b> (id {r.candidate_id})<br>"
                 f"tải {r.load:.0f} / sức chứa {r.capacity:.0f} ({r.utilization * 100:.0f}%)<br>"
                 f"chi phí cố định {r.fixed_cost:,.0f}")
        folium.CircleMarker([r.lat, r.lon], radius=7, color="#000", weight=1,
                            fill=True, fill_color=color, fill_opacity=0.95,
                            popup=folium.Popup(popup, max_width=260)).add_to(g_st)
        folium.Circle([r.lat, r.lon], radius=radius, color=color, weight=1,
                      fill=True, fill_opacity=0.05).add_to(g_circle)
    g_st.add_to(m)
    g_circle.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    total = demand["daily_demand"].sum()
    served_total = demand.loc[demand["served"].astype(bool), "daily_demand"].sum()
    cost = stations["fixed_cost"].sum()
    legend = "".join(
        f'<div><span style="background:{c};width:10px;height:10px;display:inline-block;'
        f'border:1px solid #000;border-radius:50%"></span> {t}</div>' for t, c in COLORS.items())
    box = f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;
                padding:10px 14px;border:1px solid #999;border-radius:6px;font:13px sans-serif">
      <b>Lần chạy:</b> {summary.get('run_id', run_dir.name)}<br>
      R = {radius:g} m, alpha = {summary.get('alpha')}, cap_scale = {summary.get('cap_scale')}<br>
      Số trạm: {len(stations)}, tổng chi phí cố định: {cost:,.0f}<br>
      Phủ: {served_total / total * 100:.1f}% nhu cầu<br><hr style="margin:6px 0">{legend}
    </div>"""
    m.get_root().html.add_child(folium.Element(box))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--boundary", type=Path, default=Path("data/boundary.geojson"))
    args = p.parse_args()

    run_dir = args.run or latest_run(args.results_dir)
    m = build_map(run_dir, args.boundary)
    out = Path(run_dir) / "map.html"
    m.save(str(out))
    print(f"đã lưu {out}, mở bằng trình duyệt")


if __name__ == "__main__":
    main()
