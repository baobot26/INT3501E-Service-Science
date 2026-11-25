"""Centralized light and dark color/surface tokens for the Streamlit app and Folium maps.

Keys are named by semantic role rather than hue so theme variants stay decoupled
from layout CSS and map builders.
"""
from __future__ import annotations

UI_LIGHT: dict[str, str] = {
    "ground": "#F6F7F4",
    "panel": "#ECEFEA",
    "ink": "#1C2421",
    "ink_muted": "#46524D",
    "line": "#C9CFC7",
    "accent": "#165D4A",
    "accent_hover": "#10493A",
    "link": "#165D4A",
    "on_accent": "#FFFFFF",
    "selection_bg": "#CBE3DA",
    "focus_ring": "rgba(22, 93, 74, 0.22)",
    "table_header_bg": "#E4E9E2",
    "table_header_ink": "#1C2421",
    "error": "#B4342B",
    "error_ink": "#8B1E18",
    "error_bg": "#FBF0EE",
    "warning": "#9A6700",
    "warning_ink": "#6F4800",
    "warning_bg": "#FAF4E4",
    "orange": "#A84B14",
    "orange_ink": "#7A340C",
    "orange_bg": "#FCF2EA",
    "info": "#24577A",
    "info_ink": "#1A4666",
    "info_bg": "#ECF3F7",
    "success": "#165D4A",
    "success_ink": "#114A3B",
    "success_bg": "#E9F3EF",
}

UI_DARK: dict[str, str] = {
    "ground": "#121816",
    "panel": "#1A2220",
    "ink": "#E4E9E5",
    "ink_muted": "#A9B4AE",
    "line": "#34403B",
    "accent": "#2E7D66",
    "accent_hover": "#246A56",
    "link": "#7CC7AE",
    "on_accent": "#FFFFFF",
    "selection_bg": "#244A3E",
    "focus_ring": "rgba(124, 199, 174, 0.30)",
    "table_header_bg": "#212B28",
    "table_header_ink": "#E4E9E5",
    "error": "#D96B62",
    "error_ink": "#F6B4AE",
    "error_bg": "#2E1A19",
    "warning": "#D6A84E",
    "warning_ink": "#F2D08B",
    "warning_bg": "#2C2415",
    "orange": "#DC8A4E",
    "orange_ink": "#F4BE94",
    "orange_bg": "#2D2016",
    "info": "#5FA0CF",
    "info_ink": "#A6D1EE",
    "info_bg": "#162531",
    "success": "#3C997D",
    "success_ink": "#8CD5BC",
    "success_bg": "#152B24",
}

MAP_LIGHT: dict[str, object] = {
    "station_types": {
        "amenity=parking": "#2B5B84",
        "shop=supermarket": "#A84B14",
        "amenity=post_office": "#7B3B6B",
        "shop=convenience": "#826514",
        "building=apartments": "#58636A",
    },
    "station_fallback": "#58636A",
    "canvas_bg": "#EFEFE9",
    "tile_filter": "grayscale(88%) brightness(1.06) contrast(0.84) saturate(0.2)",
    "boundary": "#2F3A36",
    "served": "#3B7468",
    "unserved": "#7A716B",
    "preview_demand": "#5C6862",
    "diamond_stroke": "#165D4A",
    "diamond_fill": "#165D4A",
    "diamond_swatch_bg": "rgba(22, 93, 74, 0.14)",
    "marker_border": "#FFFFFF",
    "swatch_ring_shadow": "rgba(28, 36, 33, 0.28)",
    "legend_bg": "rgba(246, 247, 244, 0.96)",
    "legend_ink": "#1C2421",
    "legend_muted": "#46524D",
    "legend_line": "#C9CFC7",
    "legend_sep": "#D5DBD3",
    "legend_focus": "#165D4A",
    "attribution_bg": "rgba(246, 247, 244, 0.90)",
    "popup_ink": "#1C2421",
}

MAP_DARK: dict[str, object] = {
    "station_types": {
        "amenity=parking": "#6BA6D6",
        "shop=supermarket": "#E28E55",
        "amenity=post_office": "#C985B4",
        "shop=convenience": "#CCB054",
        "building=apartments": "#9AA7AE",
    },
    "station_fallback": "#9AA7AE",
    "canvas_bg": "#1B211F",
    "tile_filter": "invert(1) hue-rotate(180deg) grayscale(82%) brightness(0.84) contrast(0.92)",
    "boundary": "#C8D3CD",
    "served": "#5FB89C",
    "unserved": "#9E9790",
    "preview_demand": "#8E9B95",
    "diamond_stroke": "#5FC4A4",
    "diamond_fill": "#5FC4A4",
    "diamond_swatch_bg": "rgba(95, 196, 164, 0.18)",
    "marker_border": "#121816",
    "swatch_ring_shadow": "rgba(228, 233, 229, 0.32)",
    "legend_bg": "rgba(22, 30, 27, 0.95)",
    "legend_ink": "#E4E9E5",
    "legend_muted": "#A9B4AE",
    "legend_line": "#34403B",
    "legend_sep": "#2A3531",
    "legend_focus": "#7CC7AE",
    "attribution_bg": "rgba(22, 30, 27, 0.90)",
    "popup_ink": "#E4E9E5",
}

# Aliases cho chế độ sáng mặc định
UI = UI_LIGHT
MAP = MAP_LIGHT


def normalize_mode(mode: str | None) -> str:
    """Chuẩn hóa tên biến thể theme về 'light' hoặc 'dark'."""
    return "dark" if mode == "dark" else "light"


def ui_tokens(mode: str | None = "light") -> dict[str, str]:
    """Trả bảng token giao diện cho biến thể sáng hoặc tối."""
    return UI_DARK if normalize_mode(mode) == "dark" else UI_LIGHT


def map_tokens(mode: str | None = "light") -> dict[str, object]:
    """Trả bảng token bản đồ cho biến thể sáng hoặc tối."""
    return MAP_DARK if normalize_mode(mode) == "dark" else MAP_LIGHT
