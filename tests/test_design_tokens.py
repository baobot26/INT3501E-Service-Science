"""Verify .streamlit/config.toml stays in sync with src/design_tokens.py (light & dark)."""
from __future__ import annotations

import tomllib
from pathlib import Path

from src.design_tokens import (
    MAP_DARK,
    MAP_LIGHT,
    UI_DARK,
    UI_LIGHT,
    map_tokens,
    normalize_mode,
    ui_tokens,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / ".streamlit" / "config.toml"


def _assert_theme_variant_matches(variant: dict, ui: dict[str, str]) -> None:
    sidebar = variant["sidebar"]
    assert variant["primaryColor"] == ui["accent"]
    assert variant["backgroundColor"] == ui["ground"]
    assert variant["secondaryBackgroundColor"] == ui["panel"]
    assert variant["textColor"] == ui["ink"]
    assert variant["borderColor"] == ui["line"]
    assert variant["linkColor"] == ui["link"]
    assert variant["dataframeBorderColor"] == ui["line"]
    assert variant["dataframeHeaderBackgroundColor"] == ui["table_header_bg"]
    assert variant["dataframeHeaderTextColor"] == ui["table_header_ink"]
    assert variant["redTextColor"] == ui["error_ink"]
    assert variant["redBackgroundColor"] == ui["error_bg"]
    assert variant["yellowTextColor"] == ui["warning_ink"]
    assert variant["yellowBackgroundColor"] == ui["warning_bg"]
    assert variant["blueTextColor"] == ui["info_ink"]
    assert variant["blueBackgroundColor"] == ui["info_bg"]
    assert variant["greenTextColor"] == ui["success_ink"]
    assert variant["greenBackgroundColor"] == ui["success_bg"]

    assert sidebar["backgroundColor"] == ui["panel"]
    assert sidebar["secondaryBackgroundColor"] == ui["ground"]
    assert sidebar["textColor"] == ui["ink"]
    assert sidebar["borderColor"] == ui["line"]


def test_streamlit_config_matches_light_and_dark_ui_tokens():
    cfg = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    theme = cfg["theme"]

    _assert_theme_variant_matches(theme["light"], UI_LIGHT)
    _assert_theme_variant_matches(theme["dark"], UI_DARK)
    assert cfg["client"]["toolbarMode"] == "viewer"


def test_token_lookup_helpers_fallback_safely_to_light():
    assert normalize_mode(None) == "light"
    assert normalize_mode("light") == "light"
    assert normalize_mode("invalid") == "light"
    assert normalize_mode("dark") == "dark"

    assert ui_tokens(None) == UI_LIGHT
    assert ui_tokens("unknown") == UI_LIGHT
    assert ui_tokens("light") == UI_LIGHT
    assert ui_tokens("dark") == UI_DARK

    assert map_tokens(None) == MAP_LIGHT
    assert map_tokens("unknown") == MAP_LIGHT
    assert map_tokens("dark") == MAP_DARK


def test_map_tokens_distinct_light_and_dark_and_no_alarm_red():
    assert str(MAP_LIGHT["unserved"]).upper() != "#BF3A20"
    assert MAP_LIGHT["diamond_stroke"] == UI_LIGHT["accent"]
    assert "invert" in str(MAP_DARK["tile_filter"])
    assert "invert" not in str(MAP_LIGHT["tile_filter"])
    assert MAP_DARK["canvas_bg"] != MAP_LIGHT["canvas_bg"]
