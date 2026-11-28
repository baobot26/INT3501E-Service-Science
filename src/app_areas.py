"""Thành phần giao diện chọn khu vực và tạo vùng mới bằng danh sách điểm (bản đồ + bảng tọa độ)."""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_baselines import MAP_HEIGHT, render_app_header
from src.design_tokens import ui_tokens
from src.make_map import fmt_vi_decimal
from src.preprocessing import InputError

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"
AREAS_DIR = ROOT / "data" / "areas"
STORE_DIR = ROOT / "store" / "vn"

DEFAULT_LABEL = "Cầu Giấy (mặc định)"
DRAW_NEW_LABEL = "➕ Tạo vùng mới"

DEFAULT_MAP_CENTER: tuple[float, float] = (16.0, 106.0)
DEFAULT_MAP_ZOOM: int = 6

_NAME_RE = re.compile(r"^[a-z0-9_-]+$")
_COORD_DELIM_RE = re.compile(r"\s*[,;\t]\s*|\s+")

_ACTIVE_AREA_KEY = "_active_area"
_PENDING_AREA_KEY = "_pending_area"
_RESET_NAME_KEY = "_reset_draw_name"
_DRAW_NAME_KEY = "draw_area_name"

_POINTS_KEY = "area_points"
_LAST_CLICK_KEY = "_last_click_key"
_MAP_CENTER_KEY = "_pick_map_center"
_MAP_ZOOM_KEY = "_pick_map_zoom"
_MAP_INIT_CENTER_KEY = "_pick_map_init_center"
_MAP_INIT_ZOOM_KEY = "_pick_map_init_zoom"
_MAP_MOUNTED_KEY = "_pick_map_mounted"
_MAP_NONCE_KEY = "_pick_map_nonce"
_EDITOR_VER_KEY = "_points_editor_ver"
_EDITOR_BASE_KEY = "_points_editor_base"
_POINTS_SYNCED_KEY = "_points_last_synced"
_PASTE_TEXT_KEY = "paste_coords_text"
_RESET_PASTE_KEY = "_reset_paste_coords"
_PASTE_WARN_KEY = "_paste_coords_warn"


def _list_prepared_areas(areas_root: Path) -> list[str]:
    """Liệt kê tên khu vực đã chuẩn bị (không nạp geopandas/osmnx khi khởi động app)."""
    if not areas_root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in areas_root.iterdir()
        if entry.is_dir()
        and _NAME_RE.fullmatch(entry.name)
        and (entry / "area.json").is_file()
    )


def area_options(
    default_data_dir: Path | None = None,
    areas_dir: Path | None = None,
) -> dict[str, Path]:
    """Trả về ánh xạ tên khu vực → đường dẫn thư mục dữ liệu (mặc định Cầu Giấy đứng đầu)."""
    base_data = DEFAULT_DATA_DIR if default_data_dir is None else Path(default_data_dir)
    areas_root = AREAS_DIR if areas_dir is None else Path(areas_dir)
    opts: dict[str, Path] = {DEFAULT_LABEL: base_data}
    for name in _list_prepared_areas(areas_root):
        opts[name] = areas_root / name
    return opts


def relative_data_dir(data_dir: str | Path) -> str:
    """Đường dẫn POSIX của thư mục dữ liệu tương đối theo ROOT (ví dụ `data`, `data/areas/hcm_q1`)."""
    resolved = Path(data_dir).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(data_dir).as_posix()


def _clear_area_run_state() -> None:
    """Xoá kết quả các lần chạy (mô hình chính và baseline) khi chuyển sang khu vực khác."""
    st.session_state["runs"] = {}
    st.session_state.pop("last_run", None)
    st.session_state.pop("mc_p", None)
    st.session_state.pop("pm_p", None)


def select_area_sidebar(
    *,
    default_data_dir: Path | None = None,
    areas_dir: Path | None = None,
) -> Path | None:
    """Vẽ bộ chọn khu vực ở đầu thanh bên; trả về `Path` vùng chọn hoặc `None` nếu tạo vùng mới."""
    opts = area_options(default_data_dir=default_data_dir, areas_dir=areas_dir)
    choices = [*opts.keys(), DRAW_NEW_LABEL]

    pending = st.session_state.pop(_PENDING_AREA_KEY, None)
    if pending is not None and pending in choices:
        st.session_state["area"] = pending
    elif "area" not in st.session_state or st.session_state["area"] not in choices:
        st.session_state["area"] = DEFAULT_LABEL

    st.sidebar.html('<div class="sidebar-group-heading first">Khu vực</div>')
    selected = st.sidebar.selectbox(
        "Khu vực",
        options=choices,
        key="area",
        label_visibility="collapsed",
    )

    if selected == DRAW_NEW_LABEL:
        return None

    st.session_state.pop(_MAP_MOUNTED_KEY, None)
    prev = st.session_state.get(_ACTIVE_AREA_KEY)
    if prev is not None and prev != selected:
        _clear_area_run_state()
    st.session_state[_ACTIVE_AREA_KEY] = selected
    return opts[selected]


def apply_click(
    points: list[dict[str, float | None]],
    last_clicked: dict | None,
    last_handled: tuple[float, float] | None,
) -> tuple[list[dict[str, float | None]], tuple[float, float] | None]:
    """Hàm thuần xử lý click bản đồ: thêm 1 đỉnh mới khi khác click đã xử lý, bỏ qua `None`/lặp."""
    if not last_clicked or "lat" not in last_clicked or "lng" not in last_clicked:
        return list(points), last_handled
    try:
        lat = round(float(last_clicked["lat"]), 6)
        lon = round(float(last_clicked["lng"]), 6)
    except (TypeError, ValueError):
        return list(points), last_handled
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return list(points), last_handled
    click_key = (lat, lon)
    if click_key == last_handled:
        return list(points), last_handled
    return [*points, {"lat": lat, "lon": lon}], click_key


def parse_pasted_coords(text: str) -> tuple[list[tuple[float, float]], list[int]]:
    """Tách chuỗi nhiều dòng thành `(điểm (lat, lon) hợp lệ, danh sách số dòng lỗi 1-based)`.

    Chấp nhận dấu phẩy, dấu chấm phẩy, Tab hoặc khoảng trắng giữa 2 số; bỏ qua dòng trống.
    Kiểm tra phạm vi vĩ độ/kinh độ được chuyển cho `boundary_from_points`.
    """
    valid: list[tuple[float, float]] = []
    bad_lines: list[int] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = _COORD_DELIM_RE.split(stripped)
        if len(parts) != 2:
            bad_lines.append(line_no)
            continue
        try:
            lat = float(parts[0])
            lon = float(parts[1])
        except ValueError:
            bad_lines.append(line_no)
            continue
        if not (math.isfinite(lat) and math.isfinite(lon)):
            bad_lines.append(line_no)
            continue
        valid.append((round(lat, 6), round(lon, 6)))
    return valid, bad_lines


def create_area_from_points(
    points: Sequence[tuple[float, float]],
    name: str,
    *,
    store_dir: str | Path | None = None,
    areas_dir: str | Path | None = None,
) -> dict:
    """Hàm thuần dựng khu vực mới từ tập đỉnh `(lat, lon)` (không phụ thuộc thứ tự)."""
    from src.prepare_area import boundary_from_points, build_area

    boundary_gdf = boundary_from_points(points)
    return build_area(
        boundary_gdf,
        name,
        store_dir=STORE_DIR if store_dir is None else store_dir,
        areas_dir=AREAS_DIR if areas_dir is None else areas_dir,
        mode="B",
    )


def _normalize_points_rows(raw_rows: Sequence[dict] | None) -> list[dict[str, float | None]]:
    """Chuẩn hoá các dòng điểm về `{"lat": float | None, "lon": float | None}` (không còn NaN)."""
    out: list[dict[str, float | None]] = []
    for row in raw_rows or []:
        lat_v = row.get("lat") if isinstance(row, dict) else None
        lon_v = row.get("lon") if isinstance(row, dict) else None
        lat_f: float | None = None
        lon_f: float | None = None
        if lat_v is not None and not pd.isna(lat_v):
            try:
                val = float(lat_v)
                if math.isfinite(val):
                    lat_f = val
            except (TypeError, ValueError):
                lat_f = None
        if lon_v is not None and not pd.isna(lon_v):
            try:
                val = float(lon_v)
                if math.isfinite(val):
                    lon_f = val
            except (TypeError, ValueError):
                lon_f = None
        out.append({"lat": lat_f, "lon": lon_f})
    return out


def _points_to_tuples(rows: Sequence[dict[str, float | None]]) -> list[tuple[float | None, float | None]]:
    return [(r.get("lat"), r.get("lon")) for r in rows]


def _valid_coords(rows: Sequence[dict[str, float | None]]) -> list[tuple[int, float, float]]:
    """Trả danh sách `(số_thứ_tự_dòng_1_based, lat, lon)` của các điểm có đủ tọa độ hợp lệ."""
    valid: list[tuple[int, float, float]] = []
    for idx, r in enumerate(rows, start=1):
        lat, lon = r.get("lat"), r.get("lon")
        if (
            lat is not None
            and lon is not None
            and -90.0 <= lat <= 90.0
            and -180.0 <= lon <= 180.0
        ):
            valid.append((idx, float(lat), float(lon)))
    return valid


def _points_to_df(rows: Sequence[dict[str, float | None]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            {
                "#": pd.Series(dtype="object"),
                "lat": pd.Series(dtype="float64"),
                "lon": pd.Series(dtype="float64"),
            }
        )
    return pd.DataFrame(
        {
            "#": [str(i) for i in range(1, len(rows) + 1)],
            "lat": pd.Series([r.get("lat") for r in rows], dtype="float64"),
            "lon": pd.Series([r.get("lon") for r in rows], dtype="float64"),
        }
    )


def _df_to_points(df: pd.DataFrame) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    for lat_v, lon_v in zip(df["lat"].tolist(), df["lon"].tolist()):
        lat_f = None if (lat_v is None or pd.isna(lat_v)) else round(float(lat_v), 6)
        lon_f = None if (lon_v is None or pd.isna(lon_v)) else round(float(lon_v), 6)
        rows.append({"lat": lat_f, "lon": lon_f})
    return rows


def _fit_initial_view(coords: Sequence[tuple[int, float, float]]) -> tuple[list[float], int]:
    """Tính góc nhìn ban đầu: mặc định Việt Nam, hoặc vừa khít nếu đã có sẵn điểm."""
    if not coords:
        return [DEFAULT_MAP_CENTER[0], DEFAULT_MAP_CENTER[1]], DEFAULT_MAP_ZOOM
    lats = [c[1] for c in coords]
    lons = [c[2] for c in coords]
    center = [(min(lats) + max(lats)) / 2.0, (min(lons) + max(lons)) / 2.0]
    span = max(max(lats) - min(lats), max(lons) - min(lons))
    if span <= 0.008:
        zoom = 15
    elif span <= 0.025:
        zoom = 14
    elif span <= 0.06:
        zoom = 13
    elif span <= 0.15:
        zoom = 12
    elif span <= 0.4:
        zoom = 11
    elif span <= 1.0:
        zoom = 9
    else:
        zoom = DEFAULT_MAP_ZOOM
    return center, zoom


def _evaluate_points(rows: Sequence[dict[str, float | None]]) -> dict:
    """Kiểm tra trực tiếp danh sách điểm bằng chính `boundary_from_points` và `clean_boundary`."""
    from src.fetch_data import METRIC_CRS
    from src.prepare_area import boundary_from_points, clean_boundary

    valid_pts = _valid_coords(rows)
    n_valid = len(valid_pts)
    tuples = _points_to_tuples(rows)

    try:
        gdf = boundary_from_points(tuples)  # type: ignore[arg-type]
    except InputError as exc:
        msg = str(exc)
        return {
            "valid": False,
            "n_points": n_valid,
            "area_km2": None,
            "status_label": msg,
            "detail": msg,
            "polygon_latlons": None,
        }

    poly = gdf.geometry.iloc[0]
    polygon_latlons = [(float(y), float(x)) for x, y in poly.exterior.coords[:-1]]
    area_km2 = float(gdf.to_crs(METRIC_CRS).area.iloc[0]) / 1e6

    try:
        clean_boundary(gdf)
    except InputError as exc:
        msg = str(exc)
        if "vượt giới hạn" in msg or "exceeds limit" in msg:
            label = "Vượt 50 km²"
        elif "Việt Nam" in msg or "Vietnam" in msg:
            label = "Ngoài phạm vi Việt Nam"
        else:
            label = msg
        return {
            "valid": False,
            "n_points": n_valid,
            "area_km2": area_km2,
            "status_label": label,
            "detail": msg,
            "polygon_latlons": polygon_latlons,
        }

    return {
        "valid": True,
        "n_points": n_valid,
        "area_km2": area_km2,
        "status_label": "Hợp lệ",
        "detail": None,
        "polygon_latlons": polygon_latlons,
    }


def _set_points_external(new_rows: list[dict[str, float | None]]) -> None:
    """Cập nhật `area_points` từ nguồn ngoài bảng (click/dán/nút phụ) và tăng phiên bản `data_editor`."""
    normalized = _normalize_points_rows(new_rows)
    st.session_state[_POINTS_KEY] = normalized
    st.session_state[_POINTS_SYNCED_KEY] = list(normalized)
    st.session_state[_EDITOR_BASE_KEY] = list(normalized)
    st.session_state[_EDITOR_VER_KEY] = st.session_state.get(_EDITOR_VER_KEY, 0) + 1


def _reset_create_area_state() -> None:
    """Xoá toàn bộ trạng thái tạm của màn tạo vùng sau khi tạo thành công."""
    for key in (
        _POINTS_KEY,
        _LAST_CLICK_KEY,
        _MAP_CENTER_KEY,
        _MAP_ZOOM_KEY,
        _MAP_INIT_CENTER_KEY,
        _MAP_INIT_ZOOM_KEY,
        _MAP_MOUNTED_KEY,
        _EDITOR_BASE_KEY,
        _POINTS_SYNCED_KEY,
        _PASTE_WARN_KEY,
    ):
        st.session_state.pop(key, None)
    st.session_state[_MAP_NONCE_KEY] = st.session_state.get(_MAP_NONCE_KEY, 0) + 1
    st.session_state[_EDITOR_VER_KEY] = st.session_state.get(_EDITOR_VER_KEY, 0) + 1
    st.session_state[_RESET_NAME_KEY] = True
    st.session_state[_RESET_PASTE_KEY] = True


def _build_preview_feature_group(
    coords: Sequence[tuple[int, float, float]],
    status: dict,
    tokens: dict[str, str],
):
    """Tạo `folium.FeatureGroup` gồm đa giác xem trước (đúng hình sẽ dựng) và đỉnh đánh số theo bảng."""
    import folium

    fg = folium.FeatureGroup(name="Vùng đang chọn")
    polygon_latlons = status.get("polygon_latlons")

    if status["valid"]:
        stroke = tokens["accent"]
        fill = tokens["accent"]
        dash_array = None
        fill_opacity = 0.16
    else:
        stroke = tokens["error"] if polygon_latlons else tokens["accent"]
        fill = stroke
        dash_array = "6 6" if polygon_latlons else None
        fill_opacity = 0.08

    if polygon_latlons:
        folium.Polygon(
            locations=polygon_latlons,
            color=stroke,
            weight=2.5,
            opacity=0.92,
            dash_array=dash_array,
            fill=True,
            fill_color=fill,
            fill_opacity=fill_opacity,
            interactive=False,
        ).add_to(fg)
    elif len(coords) == 2:
        folium.PolyLine(
            locations=[(lat, lon) for _, lat, lon in coords],
            color=stroke,
            weight=2.2,
            opacity=0.85,
            dash_array="4 4",
            interactive=False,
        ).add_to(fg)

    badge_bg = tokens["accent"]
    badge_fg = tokens["on_accent"]
    for idx, lat, lon in coords:
        badge_html = (
            f'<div style="width:22px;height:22px;border-radius:50%;'
            f"background:{badge_bg};color:{badge_fg};border:2px solid {badge_fg};"
            f'font:700 11px/18px sans-serif;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.35);">'
            f"{idx}</div>"
        )
        folium.Marker(
            location=[lat, lon],
            tooltip=f"Điểm {idx} ({lat:.5f}, {lon:.5f})",
            icon=folium.DivIcon(
                html=badge_html,
                icon_size=(22, 22),
                icon_anchor=(11, 11),
                class_name="area-vertex-icon",
            ),
        ).add_to(fg)

    return fg


def render_draw_area_view() -> None:
    """Hiển thị bản đồ chọn đỉnh + bảng tọa độ + ô dán nhanh và biểu mẫu tạo khu vực."""
    import folium
    from streamlit_folium import st_folium

    from src.prepare_area import _check_store_files

    render_app_header(
        "Tạo vùng mới bằng danh sách điểm · Click các điểm bao quanh vùng (không phụ thuộc thứ tự) hoặc dán tọa độ vào ô Dán nhanh."
    )

    try:
        _check_store_files(STORE_DIR)
    except InputError as exc:
        st.info(
            "Chưa có đủ kho dữ liệu Việt Nam (`store/vn/`) để dựng vùng mới. "
            "Bạn vẫn có thể chọn và chạy bình thường các khu vực sẵn có (như Cầu Giấy mặc định).\n\n"
            f"```\n{exc}\n```"
        )
        return

    theme_ctx = getattr(st.context, "theme", None)
    tokens = ui_tokens(getattr(theme_ctx, "type", None))

    # Nguồn sự thật duy nhất: st.session_state["area_points"]
    if _POINTS_KEY not in st.session_state:
        st.session_state[_POINTS_KEY] = []
    points = _normalize_points_rows(st.session_state[_POINTS_KEY])
    st.session_state[_POINTS_KEY] = points

    # Phát hiện thay đổi từ bên ngoài (vd. lần mở đầu hoặc test gán sẵn session_state["area_points"])
    if _EDITOR_BASE_KEY not in st.session_state or points != st.session_state.get(_POINTS_SYNCED_KEY):
        st.session_state[_EDITOR_BASE_KEY] = list(points)
        st.session_state[_POINTS_SYNCED_KEY] = list(points)
        st.session_state[_EDITOR_VER_KEY] = st.session_state.get(_EDITOR_VER_KEY, 0) + 1

    coords = _valid_coords(points)
    status = _evaluate_points(points)

    # Khởi tạo góc nhìn khi mở màn tạo vùng (fit theo điểm nếu đã có, hoặc khôi phục góc nhìn đã lưu).
    # Giữ cố định props center/zoom trong suốt lần mở màn để st_folium không gọi lại setView khi người dùng đang pan/click.
    if not st.session_state.get(_MAP_MOUNTED_KEY):
        if _MAP_CENTER_KEY not in st.session_state or _MAP_ZOOM_KEY not in st.session_state:
            init_center, init_zoom = _fit_initial_view(coords)
            st.session_state[_MAP_CENTER_KEY] = init_center
            st.session_state[_MAP_ZOOM_KEY] = init_zoom
        st.session_state[_MAP_INIT_CENTER_KEY] = st.session_state[_MAP_CENTER_KEY]
        st.session_state[_MAP_INIT_ZOOM_KEY] = st.session_state[_MAP_ZOOM_KEY]
        st.session_state[_MAP_MOUNTED_KEY] = True

    col_map, col_panel = st.columns([57, 43], gap="medium")

    with col_map:
        m = folium.Map(location=list(DEFAULT_MAP_CENTER), zoom_start=DEFAULT_MAP_ZOOM)
        fg = _build_preview_feature_group(coords, status, tokens)
        map_nonce = st.session_state.get(_MAP_NONCE_KEY, 0)
        map_state = st_folium(
            m,
            key=f"pick_area_map_{map_nonce}",
            height=MAP_HEIGHT,
            use_container_width=True,
            center=st.session_state[_MAP_INIT_CENTER_KEY],
            zoom=st.session_state[_MAP_INIT_ZOOM_KEY],
            feature_group_to_add=fg,
            returned_objects=["last_clicked", "center", "zoom"],
        )

        # Dòng trạng thái trực tiếp ngay dưới bản đồ
        parts = [f"**{status['n_points']} điểm**"]
        if status["n_points"] >= 3 and status["area_km2"] is not None:
            parts.append(f"**{fmt_vi_decimal(status['area_km2'], 2)} km²**")
        parts.append(status["status_label"])
        status_line = " · ".join(parts)

        if status["valid"]:
            st.success(status_line)
        elif status["n_points"] >= 3:
            st.error(status_line)
        else:
            st.info(status_line)

        st.caption(
            "Click các điểm bao quanh vùng, thứ tự không quan trọng; dán tọa độ dạng \"vĩ độ, kinh độ\" "
            "vào ô **Dán nhanh tọa độ**; tối đa 50 km², chỉ trong Việt Nam."
        )

    # Lưu center/zoom người dùng đang xem và xử lý click mới (nếu có)
    if isinstance(map_state, dict):
        ret_center = map_state.get("center")
        if isinstance(ret_center, dict) and "lat" in ret_center and "lng" in ret_center:
            st.session_state[_MAP_CENTER_KEY] = [float(ret_center["lat"]), float(ret_center["lng"])]
        ret_zoom = map_state.get("zoom")
        if isinstance(ret_zoom, (int, float)) and ret_zoom > 0:
            st.session_state[_MAP_ZOOM_KEY] = int(round(float(ret_zoom)))

        new_points, new_handled = apply_click(
            points,
            map_state.get("last_clicked"),
            st.session_state.get(_LAST_CLICK_KEY),
        )
        if new_handled != st.session_state.get(_LAST_CLICK_KEY):
            st.session_state[_LAST_CLICK_KEY] = new_handled
            _set_points_external(new_points)
            st.rerun()

    with col_panel:
        st.markdown("**Danh sách đỉnh**")
        editor_ver = st.session_state.get(_EDITOR_VER_KEY, 0)
        base_rows = st.session_state[_EDITOR_BASE_KEY]
        base_df = _points_to_df(base_rows)
        edited_df = st.data_editor(
            base_df,
            key=f"area_points_editor_{editor_ver}",
            num_rows="dynamic",
            hide_index=True,
            disabled=["#"],
            width="stretch",
            height=260,
            column_config={
                "#": st.column_config.TextColumn("#", width="small"),
                "lat": st.column_config.NumberColumn(
                    "Vĩ độ",
                    format="%.6f",
                    min_value=-90.0,
                    max_value=90.0,
                    step=0.000001,
                ),
                "lon": st.column_config.NumberColumn(
                    "Kinh độ",
                    format="%.6f",
                    min_value=-180.0,
                    max_value=180.0,
                    step=0.000001,
                ),
            },
        )
        edited_points = _df_to_points(edited_df)
        if edited_points != points:
            valid_changed = _valid_coords(edited_points) != coords
            row_count_changed = len(edited_points) != len(base_rows)
            if valid_changed and row_count_changed:
                # Tính lại cột "#" cho các dòng mới thêm hoặc sau khi xoá dòng để không hiện None
                _set_points_external(edited_points)
                st.rerun()
            else:
                st.session_state[_POINTS_KEY] = edited_points
                st.session_state[_POINTS_SYNCED_KEY] = list(edited_points)
                if valid_changed or len(edited_points) < len(points):
                    if len(edited_points) < len(points):
                        _set_points_external(edited_points)
                    st.rerun()

        btn_col1, btn_col2 = st.columns(2, gap="small")
        with btn_col1:
            if st.button(
                "Xoá điểm cuối",
                width="stretch",
                disabled=len(points) == 0,
            ):
                _set_points_external(points[:-1])
                st.rerun()
        with btn_col2:
            if st.button(
                "Xoá hết",
                width="stretch",
                disabled=len(points) == 0,
            ):
                _set_points_external([])
                st.rerun()

        paste_warn = st.session_state.pop(_PASTE_WARN_KEY, None)
        if st.session_state.pop(_RESET_PASTE_KEY, False):
            st.session_state[_PASTE_TEXT_KEY] = ""

        with st.expander("Dán nhanh tọa độ (vĩ độ, kinh độ)", expanded=bool(paste_warn)):
            pasted_text = st.text_area(
                "Mỗi dòng một điểm (vd. copy từ Google Maps: 16.065, 108.215)",
                key=_PASTE_TEXT_KEY,
                height=100,
                placeholder="16.076084, 108.213015\n16.075671, 108.228893\n16.060825, 108.212585",
            )
            if paste_warn:
                st.warning(paste_warn)
            if st.button("Thêm vào danh sách", width="stretch"):
                parsed_pts, bad_lines = parse_pasted_coords(pasted_text or "")
                warn_msg = (
                    f"Đã bỏ qua dòng không hợp lệ: {', '.join(str(n) for n in bad_lines)}."
                    if bad_lines
                    else None
                )
                if parsed_pts:
                    added_rows = [{"lat": lat, "lon": lon} for lat, lon in parsed_pts]
                    _set_points_external([*points, *added_rows])
                    st.session_state[_RESET_PASTE_KEY] = True
                    if warn_msg:
                        st.session_state[_PASTE_WARN_KEY] = warn_msg
                    st.rerun()
                elif warn_msg:
                    st.warning(warn_msg)

        if st.session_state.pop(_RESET_NAME_KEY, False) or _DRAW_NAME_KEY not in st.session_state:
            st.session_state[_DRAW_NAME_KEY] = datetime.now().strftime("vung_%Y%m%d_%H%M%S")

        area_name = st.text_input(
            "Tên vùng",
            key=_DRAW_NAME_KEY,
            help="Chỉ dùng chữ thường không dấu, chữ số, dấu gạch dưới (_) và gạch ngang (-): [a-z0-9_-].",
        )
        st.caption("Quy tắc tên vùng: chỉ gồm `[a-z0-9_-]` (ví dụ `danang_order_free`, `hcm_q1`).")

        create_clicked = st.button(
            "Tạo vùng",
            type="primary",
            width="stretch",
            disabled=not status["valid"],
        )

        if create_clicked:
            cleaned_name = area_name.strip()
            try:
                with st.spinner("Đang dựng vùng từ kho dữ liệu (khoảng 30 giây)…"):
                    create_area_from_points(
                        [(lat, lon) for _, lat, lon in coords],
                        cleaned_name,
                        store_dir=STORE_DIR,
                        areas_dir=AREAS_DIR,
                    )
            except InputError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Không thể dựng vùng '{cleaned_name}': {exc}")
                with st.expander("Chi tiết lỗi"):
                    st.exception(exc)
            else:
                _reset_create_area_state()
                st.session_state[_PENDING_AREA_KEY] = cleaned_name
                st.rerun()
