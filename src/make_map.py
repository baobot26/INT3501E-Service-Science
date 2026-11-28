"""Vẽ bản đồ HTML cho bài toán tối ưu vị trí đặt tủ đồ thông minh.

Dùng CLI:
  python -m src.make_map              (lấy lần chạy mới nhất trong results/)
  python -m src.make_map --run results/<tên_thư_mục>

Cung cấp bảng màu dữ liệu dùng chung (COLORS cho chế độ sáng), hàm dựng hình thoi
khoảng cách Manhattan L1 trên hệ chiếu EPSG:32648 (l1_diamond), hàm định dạng số
tiếng Việt và các hàm dựng bản đồ xem trước / bản đồ nghiệm hỗ trợ mode="light"|"dark".
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from pyproj import Transformer

from src.design_tokens import MAP_LIGHT, map_tokens

# Alias giữ bảng màu loại trạm sáng cho CLI và các chỗ import COLORS hiện có.
COLORS: dict[str, str] = MAP_LIGHT["station_types"]  # type: ignore[assignment]

STATION_TYPE_LABELS: dict[str, str] = {
    "amenity=parking": "Bãi đỗ xe",
    "shop=supermarket": "Siêu thị",
    "amenity=post_office": "Bưu điện",
    "shop=convenience": "Cửa hàng tiện lợi",
    "building=apartments": "Tòa chung cư",
}

OBJECTIVE_LABELS: dict[str, str] = {
    "min_stations_then_cost": "Ít trạm nhất, rồi rẻ nhất",
    "fixed_station_count": "Cố định số trạm p, tối thiểu chi phí",
}

MAP_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
MAP_TILE_ATTR = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)

REQUIRED_FILES = ("summary.json", "selected_stations.csv", "assigned_demand.csv")


def station_type_label(osm_type: str) -> str:
    """Trả tên tiếng Việt hiển thị cho loại trạm OSM."""
    return STATION_TYPE_LABELS.get(osm_type, osm_type)


def fmt_vi_int(value: float) -> str:
    """Định dạng số nguyên theo chuẩn Việt Nam (dấu chấm hàng nghìn)."""
    return f"{round(float(value)):,}".replace(",", ".")


def fmt_vi_decimal(value: float, digits: int = 1) -> str:
    """Định dạng số thập phân theo chuẩn Việt Nam (dấu phẩy thập phân)."""
    return f"{float(value):,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


@lru_cache(maxsize=1)
def _utm_transformers() -> tuple[Transformer, Transformer]:
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    to_wgs = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)
    return to_utm, to_wgs


def l1_diamond(lat: float, lon: float, radius_m: float) -> list[tuple[float, float]]:
    """Trả 4 đỉnh (lat, lon) của hình thoi L1 (|dx| + |dy| <= R) trên EPSG:32648.

    Thứ tự đỉnh theo chiều kim đồng hồ: Bắc -> Đông -> Nam -> Tây.
    """
    if radius_m <= 0:
        raise ValueError(f"radius_m must be positive, got {radius_m}")
    to_utm, to_wgs = _utm_transformers()
    cx, cy = to_utm.transform(float(lon), float(lat))
    utm_vertices = (
        (cx, cy + radius_m),
        (cx + radius_m, cy),
        (cx, cy - radius_m),
        (cx - radius_m, cy),
    )
    vertices: list[tuple[float, float]] = []
    for vx, vy in utm_vertices:
        v_lon, v_lat = to_wgs.transform(vx, vy)
        vertices.append((float(v_lat), float(v_lon)))
    return vertices


def _iter_geojson_coords(coords):
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        yield float(coords[0]), float(coords[1])
        return
    for item in coords:
        yield from _iter_geojson_coords(item)


def _boundary_bounds(
    boundary_path: Path | str | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if boundary_path is None:
        return None
    p = Path(boundary_path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    lons: list[float] = []
    lats: list[float] = []
    features = data.get("features") if data.get("type") == "FeatureCollection" else [data]
    for feat in features or []:
        geom = feat.get("geometry") if isinstance(feat, dict) and "geometry" in feat else feat
        if not isinstance(geom, dict):
            continue
        for lon, lat in _iter_geojson_coords(geom.get("coordinates")):
            lons.append(lon)
            lats.append(lat)
    if not lats:
        return None
    return (min(lats), min(lons)), (max(lats), max(lons))


def _bounds_from_coords(
    lats: np.ndarray, lons: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(np.min(lats)), float(np.min(lons))),
        (float(np.max(lats)), float(np.max(lons))),
    )


def _base_map(
    center: tuple[float, float],
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> folium.Map:
    m = folium.Map(
        location=[center[0], center[1]],
        zoom_start=14,
        zoom_snap=0.25,
        zoom_delta=0.5,
        tiles=MAP_TILE_URL,
        attr=MAP_TILE_ATTR,
        control_scale=True,
    )
    if bounds is not None:
        m.fit_bounds([list(bounds[0]), list(bounds[1])], padding=(10, 10))
        map_var = m.get_name()
        b_json = json.dumps([list(bounds[0]), list(bounds[1])])
        refit_js = f"""
        <script>
          window.addEventListener("resize", function () {{
            var mapObj = window["{map_var}"];
            if (!mapObj) return;
            mapObj.invalidateSize();
            mapObj.fitBounds({b_json}, {{padding: [10, 10]}});
          }});
        </script>
        """
        m.get_root().html.add_child(folium.Element(refit_js))
    return m


def _add_boundary(
    m: folium.Map,
    boundary_path: Path | str | None,
    tokens: dict[str, object],
) -> None:
    if boundary_path is None:
        return
    p = Path(boundary_path)
    if not p.exists():
        return
    boundary_color = str(tokens["boundary"])
    folium.GeoJson(
        str(p),
        name="Ranh giới quận Cầu Giấy cũ",
        style_function=lambda _: {
            "color": boundary_color,
            "weight": 1.6,
            "opacity": 0.85,
            "fill": False,
            "className": "district-boundary",
        },
        interactive=False,
    ).add_to(m)


def _inject_map_chrome(
    m: folium.Map,
    legend_body_html: str,
    legend_title: str,
    *,
    tokens: dict[str, object],
    legend_position: str = "top-right",
    aria_label: str = "Chú giải bản đồ",
    collapsible_mobile: bool = True,
) -> None:
    pos_css = (
        "top: 12px; right: 12px;"
        if legend_position == "top-right"
        else "bottom: 34px; left: 12px;"
    )
    css = f"""
    <style>
      :root {{
        --map-canvas: {tokens["canvas_bg"]};
        --map-tile-filter: {tokens["tile_filter"]};
        --map-legend-bg: {tokens["legend_bg"]};
        --map-legend-ink: {tokens["legend_ink"]};
        --map-legend-muted: {tokens["legend_muted"]};
        --map-legend-line: {tokens["legend_line"]};
        --map-legend-sep: {tokens["legend_sep"]};
        --map-legend-focus: {tokens["legend_focus"]};
        --map-marker-border: {tokens["marker_border"]};
        --map-swatch-shadow: {tokens["swatch_ring_shadow"]};
        --map-unserved: {tokens["unserved"]};
        --map-station-fallback: {tokens["station_fallback"]};
        --map-diamond-stroke: {tokens["diamond_stroke"]};
        --map-diamond-bg: {tokens["diamond_swatch_bg"]};
        --map-boundary: {tokens["boundary"]};
        --map-attribution-bg: {tokens["attribution_bg"]};
      }}
      @font-face {{
        font-family: 'Be Vietnam Pro';
        src: url('/app/static/fonts/BeVietnamPro-Regular.ttf') format('truetype');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
      }}
      @font-face {{
        font-family: 'Be Vietnam Pro';
        src: url('/app/static/fonts/BeVietnamPro-SemiBold.ttf') format('truetype');
        font-weight: 600;
        font-style: normal;
        font-display: swap;
      }}
      html, body {{
        margin: 0;
        padding: 0;
        background: var(--map-canvas);
        font-family: 'Be Vietnam Pro', system-ui, -apple-system, 'Segoe UI', sans-serif;
      }}
      .leaflet-container {{
        background: var(--map-canvas);
        font-family: 'Be Vietnam Pro', system-ui, -apple-system, 'Segoe UI', sans-serif;
      }}
      .leaflet-tile-pane {{
        filter: var(--map-tile-filter);
      }}
      .leaflet-control-zoom a {{
        background-color: var(--map-legend-bg) !important;
        color: var(--map-legend-ink) !important;
        border-color: var(--map-legend-line) !important;
      }}
      .leaflet-control-scale-line {{
        background: var(--map-legend-bg) !important;
        color: var(--map-legend-ink) !important;
        border-color: var(--map-legend-muted) !important;
      }}
      .leaflet-popup-content-wrapper,
      .leaflet-popup-tip,
      .leaflet-tooltip {{
        background: var(--map-legend-bg) !important;
        color: var(--map-legend-ink) !important;
        border: 1px solid var(--map-legend-line);
      }}
      .plan-legend {{
        position: fixed;
        {pos_css}
        z-index: 1000;
        background: var(--map-legend-bg);
        color: var(--map-legend-ink);
        border: 1px solid var(--map-legend-line);
        border-radius: 4px;
        font-size: 13px;
        line-height: 1.45;
        max-width: 276px;
      }}
      .plan-legend-summary {{
        list-style: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 8px 12px;
        font-weight: 600;
        font-size: 12px;
        letter-spacing: 0.03em;
        color: var(--map-legend-muted);
        user-select: none;
      }}
      .plan-legend-summary::-webkit-details-marker {{
        display: none;
      }}
      .plan-legend-summary::after {{
        content: "+";
        font-size: 14px;
        font-weight: 600;
        line-height: 1;
        color: var(--map-legend-ink);
      }}
      .plan-legend[open] .plan-legend-summary::after {{
        content: "−";
      }}
      .plan-legend-summary:focus-visible {{
        outline: 2px solid var(--map-legend-focus);
        outline-offset: 2px;
        border-radius: 3px;
      }}
      .plan-legend-body {{
        padding: 0 12px 10px 12px;
      }}
      .plan-legend-title {{
        font-weight: 600;
        font-size: 12px;
        letter-spacing: 0.03em;
        color: var(--map-legend-muted);
        margin-bottom: 6px;
      }}
      .plan-legend-grid {{
        display: flex;
        flex-direction: column;
        gap: 4px;
      }}
      .plan-legend-row {{
        display: flex;
        align-items: center;
        gap: 8px;
      }}
      .plan-legend-sep {{
        border: 0;
        border-top: 1px solid var(--map-legend-sep);
        margin: 7px 0;
      }}
      .swatch-dot {{
        width: 11px;
        height: 11px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
        border: 1.5px solid var(--map-marker-border);
        box-shadow: 0 0 0 1px var(--map-swatch-shadow);
      }}
      .swatch-dot-small {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
        margin: 0 1.5px;
      }}
      .swatch-ring {{
        width: 9px;
        height: 9px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
        background: transparent;
        border: 1.5px solid var(--map-unserved);
        box-sizing: border-box;
        margin: 0 1px;
      }}
      .swatch-ring-muted {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
        background: transparent;
        border: 1.5px solid var(--map-station-fallback);
        box-sizing: border-box;
        margin: 0 0.5px;
      }}
      .swatch-diamond {{
        width: 10px;
        height: 10px;
        display: inline-block;
        flex-shrink: 0;
        transform: rotate(45deg) scale(0.84);
        border: 1.6px solid var(--map-diamond-stroke);
        background: var(--map-diamond-bg);
        box-sizing: border-box;
        margin: 0 0.5px;
      }}
      .swatch-line {{
        width: 12px;
        height: 0;
        border-top: 2px solid var(--map-boundary);
        display: inline-block;
        flex-shrink: 0;
      }}
      .leaflet-control-attribution {{
        font-size: 11px !important;
        background: var(--map-attribution-bg) !important;
        color: var(--map-legend-muted) !important;
      }}
      .leaflet-control-attribution a {{
        color: var(--map-legend-focus) !important;
      }}
      @media (max-width: 640px) {{
        .plan-legend {{
          top: 10px;
          right: 10px;
          left: auto;
          bottom: auto;
          font-size: 12px;
          max-width: calc(100vw - 20px);
        }}
        .plan-legend-summary {{
          padding: 6px 10px;
          font-size: 12px;
        }}
        .plan-legend-body {{
          padding: 2px 10px 8px 10px;
        }}
      }}
    </style>
    """
    collapse_script = ""
    if collapsible_mobile:
        collapse_script = """
        <script>
          (function () {
            var el = document.getElementById("plan-legend");
            if (!el) return;
            var mq = window.matchMedia("(max-width: 640px)");
            var sync = function () {
              if (mq.matches) {
                el.removeAttribute("open");
              } else {
                el.setAttribute("open", "");
              }
            };
            sync();
            if (mq.addEventListener) {
              mq.addEventListener("change", sync);
            }
          })();
        </script>
        """
    legend_html = f"""
    <details class="plan-legend" id="plan-legend" role="region" aria-label="{aria_label}" open>
      <summary class="plan-legend-summary" aria-label="Bật hoặc tắt chú giải bản đồ">Chú giải</summary>
      <div class="plan-legend-body">
        <div class="plan-legend-title">{legend_title}</div>
        {legend_body_html}
      </div>
    </details>
    {collapse_script}
    """
    m.get_root().header.add_child(folium.Element(css))
    m.get_root().html.add_child(folium.Element(legend_html))


def build_preview_map(
    demand_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    params_df: pd.DataFrame,
    boundary_path: Path | str | None = None,
    enabled_override: dict[str, bool] | None = None,
    mode: str = "light",
) -> folium.Map:
    """Bản đồ xem trước quận Cầu Giấy: ranh giới, mật độ nhu cầu và toàn bộ ứng viên hợp lệ."""
    tokens = map_tokens(mode)
    station_colors: dict[str, str] = tokens["station_types"]  # type: ignore[assignment]
    feasible = cand_df[cand_df["feasible"]].copy()
    enabled_map = dict(zip(params_df["type"], params_df["enabled"].astype(bool)))
    if enabled_override:
        enabled_map.update({str(k): bool(v) for k, v in enabled_override.items()})

    bounds = _boundary_bounds(boundary_path)
    if bounds is None:
        all_lats = np.concatenate([demand_df["lat"].to_numpy(float), feasible["lat"].to_numpy(float)])
        all_lons = np.concatenate([demand_df["lon"].to_numpy(float), feasible["lon"].to_numpy(float)])
        bounds = _bounds_from_coords(all_lats, all_lons)
    center = (float(demand_df["lat"].mean()), float(demand_df["lon"].mean()))
    m = _base_map(center=center, bounds=bounds)
    _add_boundary(m, boundary_path, tokens)

    preview_demand_color = str(tokens["preview_demand"])
    marker_border = str(tokens["marker_border"])
    fallback_color = str(tokens["station_fallback"])

    for lat, lon in zip(demand_df["lat"].to_numpy(float), demand_df["lon"].to_numpy(float)):
        folium.CircleMarker(
            [lat, lon],
            radius=2.1,
            weight=0,
            fill=True,
            fill_color=preview_demand_color,
            fill_opacity=0.45,
            interactive=False,
            class_name="demand-preview",
        ).add_to(m)

    counts = feasible["type"].value_counts().to_dict()
    for row in feasible.itertuples(index=False):
        t = str(row.type)
        color = station_colors.get(t, fallback_color)
        is_enabled = enabled_map.get(t, True)
        label = station_type_label(t)
        tip = f"{label} (ID {int(row.id)})" + ("" if is_enabled else " — đang tắt")
        if is_enabled:
            folium.CircleMarker(
                [float(row.lat), float(row.lon)],
                radius=4.6,
                color=marker_border,
                weight=1.3,
                fill=True,
                fill_color=color,
                fill_opacity=0.94,
                tooltip=tip,
                class_name="candidate-marker",
            ).add_to(m)
        else:
            folium.CircleMarker(
                [float(row.lat), float(row.lon)],
                radius=3.8,
                color=color,
                weight=1.4,
                fill=False,
                opacity=0.75,
                tooltip=tip,
                class_name="candidate-marker-disabled",
            ).add_to(m)

    type_rows = []
    enabled_total = 0
    for _, prow in params_df.iterrows():
        t = str(prow["type"])
        c = station_colors.get(t, fallback_color)
        n_t = int(counts.get(t, 0))
        is_enabled = enabled_map.get(t, bool(prow["enabled"]))
        if is_enabled:
            enabled_total += n_t
        swatch = (
            f'<span class="swatch-dot" style="background:{c}"></span>'
            if is_enabled
            else f'<span class="swatch-ring-muted" style="border-color:{c}"></span>'
        )
        suffix = f" ({n_t})" if is_enabled else f" ({n_t}, tắt)"
        type_rows.append(
            f'<div class="plan-legend-row">{swatch}<span>{station_type_label(t)}{suffix}</span></div>'
        )

    body_html = f"""
      <div class="plan-legend-grid">
        {"".join(type_rows)}
      </div>
      <hr class="plan-legend-sep">
      <div class="plan-legend-grid">
        <div class="plan-legend-row">
          <span class="swatch-dot-small" style="background:{preview_demand_color};opacity:0.75"></span>
          <span>Điểm nhu cầu ({fmt_vi_int(len(demand_df))})</span>
        </div>
        <div class="plan-legend-row">
          <span class="swatch-line"></span>
          <span>Ranh giới 3 phường</span>
        </div>
      </div>
    """
    _inject_map_chrome(
        m,
        body_html,
        legend_title=f"Ứng viên ({enabled_total}/{len(feasible)} đang bật)",
        tokens=tokens,
        legend_position="top-right",
        aria_label="Chú giải bản đồ xem trước",
    )
    return m


def build_solution_map(
    problem,
    dist: np.ndarray,
    selected: list[int] | np.ndarray,
    assignment: np.ndarray,
    loads: dict[int, float] | np.ndarray,
    boundary_path: Path | str | None = None,
    mode: str = "light",
) -> folium.Map:
    """Bản đồ nghiệm: ranh giới, vùng phục vụ hình thoi L1, nhu cầu (chấm đặc / vòng rỗng), trạm được chọn."""
    tokens = map_tokens(mode)
    station_colors: dict[str, str] = tokens["station_types"]  # type: ignore[assignment]
    selected_idx = [int(j) for j in selected]
    diamonds: dict[int, list[tuple[float, float]]] = {
        j: l1_diamond(float(problem.cand_lat[j]), float(problem.cand_lon[j]), float(problem.radius))
        for j in selected_idx
    }

    bounds = _boundary_bounds(boundary_path)
    if bounds is None:
        lat_parts = [problem.demand_lat]
        lon_parts = [problem.demand_lon]
        if selected_idx:
            lat_parts.append(problem.cand_lat[selected_idx])
            lon_parts.append(problem.cand_lon[selected_idx])
            d_verts = [v for verts in diamonds.values() for v in verts]
            lat_parts.append(np.array([v[0] for v in d_verts], dtype=float))
            lon_parts.append(np.array([v[1] for v in d_verts], dtype=float))
        bounds = _bounds_from_coords(np.concatenate(lat_parts), np.concatenate(lon_parts))

    center = (float(problem.demand_lat.mean()), float(problem.demand_lon.mean()))
    m = _base_map(center=center, bounds=bounds)
    _add_boundary(m, boundary_path, tokens)

    served_color = str(tokens["served"])
    unserved_color = str(tokens["unserved"])
    diamond_stroke = str(tokens["diamond_stroke"])
    diamond_fill = str(tokens["diamond_fill"])
    marker_border = str(tokens["marker_border"])
    fallback_color = str(tokens["station_fallback"])
    popup_ink = str(tokens["popup_ink"])

    # 1. Điểm nhu cầu (lớp nền): chưa phục vụ = vòng rỗng trung tính, phục vụ = chấm đặc
    for i in range(len(problem.demand_ids)):
        served = int(assignment[i]) >= 0
        if served:
            folium.CircleMarker(
                [float(problem.demand_lat[i]), float(problem.demand_lon[i])],
                radius=2.2,
                weight=0,
                fill=True,
                fill_color=served_color,
                fill_opacity=0.72,
                interactive=False,
                class_name="demand-served",
            ).add_to(m)
        else:
            folium.CircleMarker(
                [float(problem.demand_lat[i]), float(problem.demand_lon[i])],
                radius=1.8,
                color=unserved_color,
                weight=1.0,
                opacity=0.65,
                fill=False,
                interactive=False,
                class_name="demand-unserved",
            ).add_to(m)

    # 2. Vùng phục vụ hình thoi Manhattan L1 trên EPSG:32648 (viền mực rõ, nền nhạt)
    for j in selected_idx:
        folium.Polygon(
            locations=diamonds[j],
            color=diamond_stroke,
            weight=1.6,
            opacity=0.82,
            fill=True,
            fill_color=diamond_fill,
            fill_opacity=0.12,
            interactive=False,
            class_name="service-diamond",
        ).add_to(m)

    # 3. Trạm được chọn (trên cùng, gọn để không che hình thoi)
    selected_counts: dict[str, int] = {}
    for j in selected_idx:
        t = problem.candidate_types[j]
        selected_counts[t] = selected_counts.get(t, 0) + 1
        color = station_colors.get(t, fallback_color)
        cap = float(problem.capacities[j])
        load = float(loads[j])
        util_pct = (load / cap * 100.0) if cap > 0 else 0.0
        label = station_type_label(t)
        cid = int(problem.candidate_ids[j])
        cost_ty = float(problem.fixed_costs[j]) / 1e9
        popup_html = (
            f"<div style='font-family:Be Vietnam Pro,sans-serif;font-size:13px;color:{popup_ink}'>"
            f"<b>{label}</b> (ID {cid})<br>"
            f"Tải: {fmt_vi_int(load)} / {fmt_vi_int(cap)} đơn vị/ngày ({fmt_vi_decimal(util_pct, 1)}%)<br>"
            f"Chi phí cố định: {fmt_vi_decimal(cost_ty, 2)} tỷ đồng"
            f"</div>"
        )
        folium.CircleMarker(
            [float(problem.cand_lat[j]), float(problem.cand_lon[j])],
            radius=5.2,
            color=marker_border,
            weight=2.0,
            fill=True,
            fill_color=color,
            fill_opacity=0.98,
            tooltip=f"{label} (ID {cid}) · hiệu suất {fmt_vi_decimal(util_pct, 1)}%",
            popup=folium.Popup(popup_html, max_width=270),
            class_name="station-marker",
        ).add_to(m)

    # Ẩn các loại trạm có 0 trạm được chọn trong chú giải nghiệm
    type_rows = [
        f'<div class="plan-legend-row">'
        f'<span class="swatch-dot" style="background:{c}"></span>'
        f"<span>{station_type_label(t)} ({selected_counts[t]})</span>"
        f"</div>"
        for t, c in station_colors.items()
        if selected_counts.get(t, 0) > 0
    ]

    radius_txt = fmt_vi_int(problem.radius)
    body_html = f"""
      <div class="plan-legend-grid">
        {"".join(type_rows)}
      </div>
      <hr class="plan-legend-sep">
      <div class="plan-legend-grid">
        <div class="plan-legend-row">
          <span class="swatch-diamond"></span>
          <span>Vùng phục vụ L1 (R = {radius_txt} m)</span>
        </div>
        <div class="plan-legend-row">
          <span class="swatch-dot-small" style="background:{served_color}"></span>
          <span>Nhu cầu được phục vụ</span>
        </div>
        <div class="plan-legend-row">
          <span class="swatch-ring"></span>
          <span>Nhu cầu chưa phục vụ</span>
        </div>
      </div>
    """
    _inject_map_chrome(
        m,
        body_html,
        legend_title=f"Trạm được chọn ({len(selected_idx)})",
        tokens=tokens,
        legend_position="top-right",
        aria_label="Chú giải bản đồ nghiệm",
    )
    return m


def latest_run(results_dir):
    runs = [p for p in Path(results_dir).iterdir()
            if all((p / f).exists() for f in REQUIRED_FILES)]
    if not runs:
        raise SystemExit(f"không có lần chạy nào trong {results_dir}, hãy chạy src.run trước")
    return max(runs, key=lambda p: p.stat().st_mtime)


def build_map(run_dir, boundary_path):
    """Dựng bản đồ HTML độc lập (chế độ sáng) cho thư mục kết quả chạy CLI hoặc tệp tải từ app."""
    tokens = MAP_LIGHT
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = summary.get("config") or {}
    try:
        stations = pd.read_csv(run_dir / "selected_stations.csv")
    except pd.errors.EmptyDataError:
        stations = pd.DataFrame()
    demand = pd.read_csv(run_dir / "assigned_demand.csv")

    radius = float(summary.get("radius_m", cfg.get("radius", 500)))
    alpha = summary.get("alpha", cfg.get("alpha"))
    objective = summary.get("objective_mode", cfg.get("objective_mode", "min_stations_then_cost"))
    demand_scale = summary.get("demand_scale", cfg.get("demand_scale", 1.0))
    cap_scale = summary.get("cap_scale", cfg.get("capacity_scale", 1.0))

    diamonds = []
    if not stations.empty and radius > 0:
        for r in stations.itertuples():
            diamonds.append((r, l1_diamond(float(r.lat), float(r.lon), radius)))

    bounds = _boundary_bounds(boundary_path)
    if bounds is None:
        lat_parts = [demand["lat"].to_numpy(float)]
        lon_parts = [demand["lon"].to_numpy(float)]
        if not stations.empty:
            lat_parts.append(stations["lat"].to_numpy(float))
            lon_parts.append(stations["lon"].to_numpy(float))
        for _, verts in diamonds:
            lat_parts.append(np.array([v[0] for v in verts], dtype=float))
            lon_parts.append(np.array([v[1] for v in verts], dtype=float))
        bounds = _bounds_from_coords(np.concatenate(lat_parts), np.concatenate(lon_parts))

    m = _base_map(
        center=(float(demand["lat"].mean()), float(demand["lon"].mean())),
        bounds=bounds,
    )
    _add_boundary(m, boundary_path, tokens)

    served_color = str(tokens["served"])
    unserved_color = str(tokens["unserved"])
    diamond_stroke = str(tokens["diamond_stroke"])
    diamond_fill = str(tokens["diamond_fill"])
    marker_border = str(tokens["marker_border"])
    fallback_color = str(tokens["station_fallback"])

    g_served = folium.FeatureGroup(name="Dân được phục vụ", show=True)
    g_missed = folium.FeatureGroup(name="Dân chưa phục vụ", show=True)
    for r in demand.itertuples():
        served = bool(r.served)
        if served:
            folium.CircleMarker(
                [r.lat, r.lon], radius=2.2, weight=0, fill=True, fill_opacity=0.72,
                fill_color=served_color,
                tooltip=f"Nhu cầu {fmt_vi_decimal(r.daily_demand, 1)}/ngày, đi {fmt_vi_int(r.distance_m)} m",
                class_name="demand-served",
            ).add_to(g_served)
        else:
            folium.CircleMarker(
                [r.lat, r.lon], radius=1.8, weight=1.0, color=unserved_color,
                opacity=0.65, fill=False,
                tooltip=f"Nhu cầu {fmt_vi_decimal(r.daily_demand, 1)}/ngày (chưa phục vụ)",
                class_name="demand-unserved",
            ).add_to(g_missed)
    g_served.add_to(m)
    g_missed.add_to(m)

    g_diamond = folium.FeatureGroup(name=f"Vùng L1 (R = {fmt_vi_int(radius)} m)", show=True)
    for _, verts in diamonds:
        folium.Polygon(
            locations=verts,
            color=diamond_stroke, weight=1.6, opacity=0.82,
            fill=True, fill_color=diamond_fill, fill_opacity=0.12,
            class_name="service-diamond",
        ).add_to(g_diamond)
    g_diamond.add_to(m)

    g_st = folium.FeatureGroup(name="Trạm được chọn", show=True)
    for r in stations.itertuples():
        color = COLORS.get(r.type, fallback_color)
        popup = (
            f"<b>{station_type_label(r.type)}</b> (ID {r.candidate_id})<br>"
            f"Tải {fmt_vi_int(r.load)} / sức chứa {fmt_vi_int(r.capacity)} "
            f"({fmt_vi_decimal(r.utilization * 100, 1)}%)<br>"
            f"Chi phí cố định: {fmt_vi_int(r.fixed_cost)} đồng"
        )
        folium.CircleMarker(
            [r.lat, r.lon], radius=5.2, color=marker_border, weight=2,
            fill=True, fill_color=color, fill_opacity=0.98,
            popup=folium.Popup(popup, max_width=270),
            class_name="station-marker",
        ).add_to(g_st)
    g_st.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    total = demand["daily_demand"].sum()
    served_total = demand.loc[demand["served"].astype(bool), "daily_demand"].sum()
    cost = stations["fixed_cost"].sum() if not stations.empty else 0
    counts = stations["type"].value_counts().to_dict() if not stations.empty else {}
    legend_rows = "".join(
        f'<div class="plan-legend-row"><span class="swatch-dot" style="background:{c}"></span>'
        f"<span>{station_type_label(t)} ({int(counts[t])})</span></div>"
        for t, c in COLORS.items()
        if counts.get(t, 0) > 0
    )
    alpha_txt = fmt_vi_decimal(float(alpha) * 100, 0) + "%" if alpha is not None else "—"
    objective_txt = OBJECTIVE_LABELS.get(str(objective), str(objective))
    body_html = f"""
      <div>Mục tiêu: {objective_txt}</div>
      <div>R = {fmt_vi_int(radius)} m · α = {alpha_txt} · sức chứa ×{fmt_vi_decimal(cap_scale, 1)} · nhu cầu ×{fmt_vi_decimal(demand_scale, 1)}</div>
      <div>Số trạm: {len(stations)} · Chi phí: {fmt_vi_decimal(cost / 1e9, 2)} tỷ đồng · Phủ: {fmt_vi_decimal(served_total / total * 100, 1)}%</div>
      <hr class="plan-legend-sep">
      <div class="plan-legend-grid">{legend_rows}</div>
    """
    _inject_map_chrome(
        m,
        body_html,
        legend_title=f"Lần chạy: {summary.get('run_id', run_dir.name)}",
        tokens=tokens,
        legend_position="bottom-left",
        aria_label="Thông tin lần chạy và chú giải bản đồ",
        collapsible_mobile=False,
    )
    return m


def boundary_path_for_run(run_dir: str | Path) -> Path:
    """Lấy đường dẫn ranh giới từ khóa `data_dir` trong `summary.json` (mặc định `data/boundary.geojson`)."""
    summary = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
    return Path(summary.get("data_dir") or "data") / "boundary.geojson"


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    run_dir = args.run or latest_run(args.results_dir)
    boundary_path = boundary_path_for_run(run_dir)
    m = build_map(run_dir, boundary_path)
    out = Path(run_dir) / "map.html"
    m.save(str(out))
    print(f"đã lưu {out}, mở bằng trình duyệt")


if __name__ == "__main__":
    main()
