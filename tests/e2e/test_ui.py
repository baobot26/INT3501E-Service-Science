"""Playwright end-to-end tests for the redesigned Streamlit app."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src import models
from src.make_map import fmt_vi_decimal

if TYPE_CHECKING:
    from playwright.sync_api import FrameLocator, Locator, Page

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SHOTS_006_DIR = REPO_ROOT / ".handoff" / "006-shots"
SHOTS_007_DIR = REPO_ROOT / ".handoff" / "007-shots"


@lru_cache(maxsize=1)
def _l1_inputs():
    return models.load_inputs("l1", DATA_DIR)


def _shot_006(page: Page, filename: str) -> None:
    SHOTS_006_DIR.mkdir(parents=True, exist_ok=True)
    _wait_settled_for_shot(page)
    page.screenshot(path=str(SHOTS_006_DIR / filename), full_page=False)


def _shot_007(page: Page, filename: str) -> None:
    SHOTS_007_DIR.mkdir(parents=True, exist_ok=True)
    _wait_settled_for_shot(page)
    page.screenshot(path=str(SHOTS_007_DIR / filename), full_page=False)


def _select_model(page: Page, label_text: str) -> None:
    sidebar = page.locator('[data-testid="stSidebar"]')
    radio = sidebar.locator('[data-testid="stRadio"]')
    expect(radio).to_be_visible(timeout=15_000)
    radio.locator("label", has_text=label_text).first.click()
    if label_text.startswith("Mô hình chính"):
        expect(sidebar.get_by_role("slider", name="Tỷ lệ phủ tối thiểu α")).to_be_visible(timeout=20_000)
    elif label_text == "Set covering":
        expect(sidebar.get_by_role("slider", name="Tỷ lệ phủ tối thiểu α")).to_have_count(0, timeout=20_000)
        expect(sidebar.get_by_role("spinbutton", name="Số trạm p", exact=True)).to_have_count(0, timeout=20_000)
        expect(sidebar.get_by_role("slider", name="Bán kính phục vụ R (m)")).to_be_visible(timeout=20_000)
    elif label_text == "Max covering":
        expect(sidebar.get_by_role("slider", name="Tỷ lệ phủ tối thiểu α")).to_have_count(0, timeout=20_000)
        expect(sidebar.get_by_role("slider", name="Bán kính phục vụ R (m)")).to_be_visible(timeout=20_000)
        expect(sidebar.get_by_role("spinbutton", name="Số trạm p", exact=True)).to_be_visible(timeout=20_000)
    elif label_text == "p-median":
        expect(sidebar.get_by_role("slider", name="Tỷ lệ phủ tối thiểu α")).to_have_count(0, timeout=20_000)
        expect(sidebar.get_by_role("slider", name="Bán kính đánh giá R (m)")).to_be_visible(timeout=20_000)
        expect(sidebar.get_by_role("spinbutton", name="Số trạm p", exact=True)).to_be_visible(timeout=20_000)
    expect(page.locator('[data-testid="stStatusWidget"]')).to_have_count(0, timeout=20_000)


def _assert_title_block_not_truncated(page: Page) -> None:
    issues = page.evaluate(
        """() => {
            const root = document.querySelector('.st-key-title_block');
            if (!root) return ['Missing .st-key-title_block'];
            const bad = [];
            const els = root.querySelectorAll('[data-testid="stMetricLabel"], [data-testid="stMetricValue"]');
            if (els.length === 0) return ['No metric elements inside .st-key-title_block'];
            for (const el of els) {
                if (el.scrollWidth > el.clientWidth) {
                    bad.push(`${el.getAttribute('data-testid')} "${el.innerText.trim()}": scrollWidth=${el.scrollWidth} > clientWidth=${el.clientWidth}`);
                }
            }
            return bad;
        }"""
    )
    assert not issues, f"Truncated title block cells: {issues}"


def expect(target):
    from playwright.sync_api import expect as pw_expect

    return pw_expect(target)


def _run_button(page: Page) -> Locator:
    return page.locator('[data-testid="stSidebar"] button', has_text="Chạy kịch bản")


def _metric_value(page: Page, label: str) -> Locator:
    card = page.locator('.st-key-title_block [data-testid="stMetric"]').filter(
        has=page.locator('[data-testid="stMetricLabel"] p', has_text=re.compile(rf"^{re.escape(label)}$"))
    )
    return card.locator('[data-testid="stMetricValue"]')


def _map_frame(page: Page) -> FrameLocator:
    return page.frame_locator('iframe[title*="Bản đồ"]').first


def _open_app(page: Page, app_url: str, width: int = 1440, height: int = 900, wait_map: bool = True) -> None:
    page.set_viewport_size({"width": width, "height": height})
    page.goto(app_url, wait_until="domcontentloaded")
    expect(
        page.get_by_role("heading", name="Quy hoạch vị trí đặt tủ đồ thông minh")
    ).to_be_visible(timeout=30_000)
    if wait_map:
        expect(_map_frame(page).locator(".leaflet-container")).to_be_visible(timeout=30_000)


def _click_until(toggle: Locator, done: Locator, want_visible: bool) -> None:
    """Click a sidebar toggle until `done` reaches the wanted visibility.

    After a viewport change Streamlit re-lays out the sidebar: the toggle can detach
    mid-click, or the sidebar can collapse on its own. A plain is_visible() + click()
    then waits 30 s on a button that is gone, so try briefly twice and check the end state.
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    for _ in range(2):
        if done.is_visible() == want_visible:
            break
        try:
            toggle.click(timeout=5_000)
        except PlaywrightTimeoutError:
            pass
    if want_visible:
        expect(done).to_be_visible(timeout=10_000)
    else:
        expect(done).not_to_be_visible(timeout=10_000)


def _ensure_sidebar_open(page: Page) -> None:
    toggle = page.locator('[data-testid="stExpandSidebarButton"]').first
    _click_until(toggle, _run_button(page), want_visible=True)


def _collapse_sidebar_if_open(page: Page) -> None:
    toggle = page.locator('[data-testid="stSidebarCollapseButton"] button').first
    _click_until(toggle, page.locator('[data-testid="stSidebar"]'), want_visible=False)


def _set_slider(page: Page, name: str, target: float, tol: float = 1e-4) -> None:
    slider = page.locator('[data-testid="stSidebar"]').get_by_role("slider", name=name)
    expect(slider).to_be_visible(timeout=10_000)
    slider.focus()
    for _ in range(80):
        cur = float(slider.input_value())
        if abs(cur - target) <= tol:
            return
        slider.press("ArrowRight" if cur < target else "ArrowLeft")
    final_val = slider.input_value()
    raise AssertionError(f"Could not move slider {name!r} to {target}, stuck at {final_val}")


def _assert_no_horizontal_overflow(page: Page) -> None:
    overflow = page.evaluate(
        "() => ({scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth})"
    )
    assert overflow["scrollWidth"] <= overflow["innerWidth"], (
        f"Horizontal overflow detected: scrollWidth={overflow['scrollWidth']} > "
        f"innerWidth={overflow['innerWidth']}"
    )


def _assert_boundary_fills_desktop_map(page: Page, min_ratio: float = 0.8) -> None:
    frame = _map_frame(page)
    container = frame.locator(".leaflet-container")
    boundary = frame.locator("path.district-boundary").first
    expect(boundary).to_be_visible(timeout=30_000)
    c_box = container.bounding_box()
    b_box = boundary.bounding_box()
    assert c_box is not None and b_box is not None
    ratio = b_box["height"] / c_box["height"]
    assert ratio >= min_ratio, (
        f"District boundary height ratio {ratio:.3f} < {min_ratio} "
        f"(boundary={b_box['height']:.1f}px, container={c_box['height']:.1f}px)"
    )


def _assert_boundary_inside_map(page: Page, tol_px: float = 4.0) -> None:
    frame = _map_frame(page)
    container = frame.locator(".leaflet-container")
    boundary = frame.locator("path.district-boundary").first
    expect(boundary).to_be_visible(timeout=30_000)
    c_box = container.bounding_box()
    b_box = boundary.bounding_box()
    assert c_box is not None and b_box is not None
    assert b_box["x"] >= c_box["x"] - tol_px, (b_box, c_box)
    assert b_box["y"] >= c_box["y"] - tol_px, (b_box, c_box)
    assert b_box["x"] + b_box["width"] <= c_box["x"] + c_box["width"] + tol_px, (b_box, c_box)
    assert b_box["y"] + b_box["height"] <= c_box["y"] + c_box["height"] + tol_px, (b_box, c_box)


def _scroll_main_to_top(page: Page) -> None:
    page.evaluate(
        """() => {
            window.scrollTo(0, 0);
            const main = document.querySelector('[data-testid="stMain"]');
            if (main) main.scrollTop = 0;
        }"""
    )


def _wait_settled_for_shot(page: Page) -> None:
    """Best-effort wait for Streamlit rerun & optional map tiles before saving a review screenshot."""
    expect(page.locator('[data-testid="stStatusWidget"]')).to_have_count(0, timeout=15_000)
    try:
        page.wait_for_load_state("networkidle", timeout=4_000)
    except Exception:
        pass


def test_initial_page_load_preview_map_and_theme_font(page: Page, app_url: str) -> None:
    _open_app(page, app_url)
    expect(page.locator('[data-testid="stSidebar"] [data-testid="stForm"]')).to_be_visible()
    expect(_run_button(page)).to_be_visible()
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    expect(page.locator('[data-testid="stAlert"]')).to_have_count(0)

    # toolbarMode="viewer": không hiển thị nút Deploy trên thanh header
    expect(page.locator("header")).not_to_contain_text("Deploy")

    # Khung tên trạng thái ban đầu
    expect(_metric_value(page, "Số trạm")).to_have_text("—")
    expect(_metric_value(page, "Trạng thái")).to_have_text("Chưa chạy")

    # Bản đồ xem trước: Leaflet, ứng viên SVG và khung ranh giới quận chiếm >= 80% chiều cao
    frame = _map_frame(page)
    expect(frame.locator(".leaflet-container")).to_be_visible(timeout=30_000)
    expect(frame.locator("path.candidate-marker").first).to_be_visible(timeout=30_000)
    assert frame.locator("path.candidate-marker").count() > 0
    _assert_boundary_fills_desktop_map(page, min_ratio=0.8)

    # Font Be Vietnam Pro thực sự đã tải (loaded FontFace) và áp trên nhãn & tiêu đề
    font_info = page.evaluate(
        """async () => {
            await document.fonts.ready;
            const loadedFaces = [...document.fonts]
                .filter(f => f.status === 'loaded' && f.family.replace(/['"]/g, '').includes('Be Vietnam Pro'))
                .map(f => ({family: f.family, weight: f.weight}));
            const label = document.querySelector('[data-testid="stMetricLabel"]');
            const heading = document.querySelector('.app-header h1');
            return {
                labelFont: label ? getComputedStyle(label).fontFamily : '',
                headingFont: heading ? getComputedStyle(heading).fontFamily : '',
                loadedFaces,
            };
        }"""
    )
    assert "Be Vietnam Pro" in font_info["labelFont"], font_info
    assert "Be Vietnam Pro" in font_info["headingFont"], font_info
    assert len(font_info["loadedFaces"]) >= 1, font_info


def test_default_time_limit_produces_valid_result(page: Page, app_url: str) -> None:
    """Chạy đúng tham số mặc định (không chỉnh slider nào): luôn có nghiệm hợp lệ."""
    _open_app(page, app_url)
    _run_button(page).click()

    status_el = _metric_value(page, "Trạng thái")
    expect(status_el).to_have_text(
        re.compile(r"^(Tối ưu|Khả thi, chưa chứng minh tối ưu)$"),
        timeout=60_000,
    )
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    expect(
        page.locator('[data-testid="stDownloadButton"] button', has_text="summary.json")
    ).to_be_visible()
    if status_el.inner_text().strip() == "Tối ưu":
        expect(_metric_value(page, "Số trạm")).to_have_text("31")


def test_default_scenario_results_map_and_download_persists(page: Page, app_url: str) -> None:
    _open_app(page, app_url)
    # Đặt giới hạn 60 giây cho test khẳng định "Tối ưu" để tránh chập chờn trên runner chậm
    _set_slider(page, "Giới hạn thời gian (giây)", 60)
    _run_button(page).click()

    expect(_metric_value(page, "Trạng thái")).to_have_text("Tối ưu", timeout=90_000)
    expect(_metric_value(page, "Số trạm")).to_have_text("31")
    expect(_metric_value(page, "Tổng chi phí")).to_have_text("10,6 tỷ đồng")
    expect(_metric_value(page, "Độ phủ")).to_have_text("60,1%")

    # Bản đồ nghiệm: đủ 31 marker trạm, 31 hình thoi L1, ranh giới >= 80% chiều cao, không dùng đỏ cam #BF3A20
    frame = _map_frame(page)
    expect(frame.locator(".leaflet-container")).to_be_visible(timeout=30_000)
    expect(frame.locator("path.station-marker")).to_have_count(31, timeout=30_000)
    expect(frame.locator("path.service-diamond")).to_have_count(31, timeout=30_000)
    _assert_boundary_fills_desktop_map(page, min_ratio=0.8)

    unserved_strokes = frame.locator("path.demand-unserved").evaluate_all(
        "els => [...new Set(els.map(e => (e.getAttribute('stroke') || getComputedStyle(e).stroke).toLowerCase()))]"
    )
    assert len(unserved_strokes) >= 1
    assert all("#bf3a20" not in s and "rgb(191, 58, 32)" not in s for s in unserved_strokes), unserved_strokes

    # Chú giải nghiệm ẩn các loại có 0 trạm
    expect(frame.locator("#plan-legend")).not_to_contain_text("(0)")

    # Tải assigned_demand.csv -> 1545 dòng (1 header + 1544 điểm cầu), kết quả không bị xóa
    dl_btn = page.locator('[data-testid="stDownloadButton"] button', has_text="assigned_demand.csv")
    expect(dl_btn).to_be_visible()
    with page.expect_download(timeout=30_000) as dl_info:
        dl_btn.click()
    download = dl_info.value
    assert download.suggested_filename == "assigned_demand.csv"
    downloaded_path = download.path()
    assert downloaded_path is not None
    lines = Path(downloaded_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1545

    # Kết quả vẫn giữ nguyên sau khi tải nhờ on_click="ignore"
    expect(_metric_value(page, "Số trạm")).to_have_text("31")
    expect(_metric_value(page, "Trạng thái")).to_have_text("Tối ưu")


def test_fixed_p_default_solves_to_31_stations(page: Page, app_url: str) -> None:
    _open_app(page, app_url)
    sidebar = page.locator('[data-testid="stSidebar"]')
    combo = sidebar.get_by_role("combobox", name="Mục tiêu")
    expect(combo).to_be_visible(timeout=10_000)
    combo.click()
    option = page.get_by_role(
        "option",
        name="Cố định số trạm p, tối thiểu chi phí (thử nghiệm)",
    )
    expect(option).to_be_visible(timeout=10_000)
    option.click()
    _set_slider(page, "Giới hạn thời gian (giây)", 60)

    _run_button(page).click()
    expect(page.locator(".app-header p")).to_contain_text("Cố định p = 31", timeout=90_000)
    expect(_metric_value(page, "Số trạm")).to_have_text("31")
    expect(
        page.locator('[data-testid="stDownloadButton"] button', has_text="summary.json")
    ).to_be_visible()
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def test_precheck_infeasible_shows_real_bound_and_fix_hint(
    page: Page, app_url: str, shots_dir: Path
) -> None:
    _open_app(page, app_url)
    # Giữ R = 300 m, nâng alpha lên 0.90 -> độ phủ tối đa 85,86% (hiển thị 85,9%)
    _set_slider(page, "Tỷ lệ phủ tối thiểu α", 0.9)
    _run_button(page).click()

    alert = page.locator('[data-testid="stAlert"]').first
    expect(alert).to_be_visible(timeout=30_000)
    expect(alert).to_contain_text("85,9")
    expect(alert).to_contain_text("Tăng R")
    expect(alert).to_contain_text("giảm α")

    expect(_map_frame(page).locator("path.candidate-marker").first).to_be_visible(timeout=30_000)
    _wait_settled_for_shot(page)
    page.screenshot(path=str(shots_dir / "after-precheck-error.png"), full_page=False)


def test_viewports_overflow_mobile_legend_and_review_screenshots(
    page: Page, app_url: str, shots_dir: Path
) -> None:
    # 1. Trước khi chạy — Desktop 1440x900
    _open_app(page, app_url, width=1440, height=900)
    expect(_map_frame(page).locator("path.candidate-marker").first).to_be_visible(timeout=30_000)
    _assert_boundary_fills_desktop_map(page, min_ratio=0.8)
    _assert_no_horizontal_overflow(page)
    _wait_settled_for_shot(page)
    page.screenshot(path=str(shots_dir / "before-desktop.png"), full_page=False)

    # 2. Trước khi chạy — Mobile 390x844 (sidebar tự đóng, chú giải thu gọn <= 15% diện tích bản đồ)
    _open_app(page, app_url, width=390, height=844)
    m_frame = _map_frame(page)
    expect(m_frame.locator("path.candidate-marker").first).to_be_visible(timeout=30_000)
    _assert_boundary_inside_map(page)
    legend = m_frame.locator("#plan-legend")
    expect(legend).not_to_have_attribute("open", "")
    c_box = m_frame.locator(".leaflet-container").bounding_box()
    l_box = legend.bounding_box()
    assert c_box is not None and l_box is not None
    area_ratio = (l_box["width"] * l_box["height"]) / (c_box["width"] * c_box["height"])
    assert area_ratio <= 0.15, f"Collapsed mobile legend covers {area_ratio:.1%} > 15%"
    _assert_no_horizontal_overflow(page)
    _wait_settled_for_shot(page)
    page.screenshot(path=str(shots_dir / "before-mobile.png"), full_page=False)

    # 3. Sau khi chạy mặc định (time_limit=60 để ổn định) — Desktop 1440x900
    _open_app(page, app_url, width=1440, height=900)
    _ensure_sidebar_open(page)
    _set_slider(page, "Giới hạn thời gian (giây)", 60)
    _run_button(page).click()
    expect(_metric_value(page, "Trạng thái")).to_have_text("Tối ưu", timeout=90_000)
    expect(_map_frame(page).locator("path.station-marker")).to_have_count(31, timeout=30_000)
    _assert_boundary_fills_desktop_map(page, min_ratio=0.8)
    _assert_no_horizontal_overflow(page)
    _wait_settled_for_shot(page)
    page.screenshot(path=str(shots_dir / "after-desktop.png"), full_page=False)
    page.screenshot(path=str(shots_dir / "light-after-desktop.png"), full_page=False)

    # 4. Tab Phân bổ nhu cầu — Desktop 1440x900 (cuộn xuống để thấy bảng và cụm nút tải)
    alloc_tab = page.get_by_role("tab", name="Phân bổ nhu cầu")
    alloc_tab.scroll_into_view_if_needed()
    alloc_tab.click()
    expect(alloc_tab).to_have_attribute("aria-selected", "true")
    expect(page.locator('[data-testid="stDataFrame"]').last).to_be_visible(timeout=15_000)
    page.locator('[data-testid="stDownloadButton"]').first.scroll_into_view_if_needed()
    page.screenshot(path=str(shots_dir / "after-tab-allocation.png"), full_page=False)

    # 5. Sau khi chạy — Mobile 390x844 (chú giải thu gọn mặc định, bấm mở/đóng được)
    page.set_viewport_size({"width": 390, "height": 844})
    _collapse_sidebar_if_open(page)
    _scroll_main_to_top(page)
    expect(_metric_value(page, "Số trạm")).to_have_text("31")
    after_frame = _map_frame(page)
    expect(after_frame.locator(".leaflet-container")).to_be_visible(timeout=30_000)
    _assert_boundary_inside_map(page)
    after_legend = after_frame.locator("#plan-legend")
    expect(after_legend).not_to_have_attribute("open", "")
    c_box2 = after_frame.locator(".leaflet-container").bounding_box()
    l_box2 = after_legend.bounding_box()
    assert c_box2 is not None and l_box2 is not None
    area_ratio2 = (l_box2["width"] * l_box2["height"]) / (c_box2["width"] * c_box2["height"])
    assert area_ratio2 <= 0.15, f"Collapsed mobile solution legend covers {area_ratio2:.1%} > 15%"
    _assert_no_horizontal_overflow(page)
    _wait_settled_for_shot(page)
    page.screenshot(path=str(shots_dir / "after-mobile.png"), full_page=False)

    # Mở chú giải trên mobile, kiểm có loại trạm và không có (0), chụp ảnh rồi đóng lại
    after_legend.locator("summary").click()
    expect(after_legend).to_have_attribute("open", "")
    expect(after_legend).to_contain_text("Cửa hàng tiện lợi")
    expect(after_legend).not_to_contain_text("(0)")
    page.screenshot(path=str(shots_dir / "after-mobile-legend-open.png"), full_page=False)

    after_legend.locator("summary").click()
    expect(after_legend).not_to_have_attribute("open", "")


def test_keyboard_tab_focus_visible_on_run_button(page: Page, app_url: str) -> None:
    _open_app(page, app_url, wait_map=False)
    run_btn = _run_button(page)
    expect(run_btn).to_be_visible(timeout=15_000)

    unfocused = run_btn.evaluate(
        """el => {
            const cs = getComputedStyle(el);
            return {outlineStyle: cs.outlineStyle, outlineWidth: cs.outlineWidth, boxShadow: cs.boxShadow};
        }"""
    )

    mip_slider = page.locator('[data-testid="stSidebar"]').get_by_role("slider", name="MIP gap")
    expect(mip_slider).to_be_visible(timeout=10_000)
    mip_slider.focus()

    reached = False
    for _ in range(15):
        page.keyboard.press("Tab")
        is_run_btn = page.evaluate(
            """() => {
                const el = document.activeElement;
                return !!el && el.tagName === 'BUTTON' && el.textContent.includes('Chạy kịch bản');
            }"""
        )
        if is_run_btn:
            reached = True
            break
    assert reached, "Could not reach 'Chạy kịch bản' button via Tab key"

    focused = page.evaluate(
        """() => {
            const el = document.activeElement;
            const cs = getComputedStyle(el);
            return {
                focusVisible: el.matches(':focus-visible'),
                outlineStyle: cs.outlineStyle,
                outlineWidth: cs.outlineWidth,
                boxShadow: cs.boxShadow,
            };
        }"""
    )
    assert focused["focusVisible"], focused
    has_outline = focused["outlineStyle"] != "none" and focused["outlineWidth"] != "0px"
    has_new_shadow = focused["boxShadow"] != "none" and focused["boxShadow"] != unfocused["boxShadow"]
    assert has_outline or has_new_shadow, f"Focus indicator not visible: {focused} vs {unfocused}"


def _assert_dark_map_styling(page: Page) -> None:
    frame = _map_frame(page)
    expect(frame.locator(".leaflet-container")).to_be_visible(timeout=30_000)
    info = frame.locator(".leaflet-tile-pane").evaluate(
        """el => {
            const tileFilter = getComputedStyle(el).filter || '';
            const legend = document.getElementById('plan-legend');
            const legendBg = legend ? getComputedStyle(legend).backgroundColor : '';
            const m = legendBg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
            const lum = m ? (0.2126 * (+m[1]/255) + 0.7152 * (+m[2]/255) + 0.0722 * (+m[3]/255)) : 1;
            return {tileFilter, legendBg, lum};
        }"""
    )
    assert "invert" in info["tileFilter"], info
    assert info["lum"] < 0.25, info


_CONTRAST_JS_HELPERS = """
function parseColor(str) {
    let m = str.match(/rgba?\\((\\d+(?:\\.\\d+)?),\\s*(\\d+(?:\\.\\d+)?),\\s*(\\d+(?:\\.\\d+)?)(?:,\\s*(\\d+(?:\\.\\d+)?))?\\)/);
    if (m) return [+m[1]/255, +m[2]/255, +m[3]/255, m[4] !== undefined ? +m[4] : 1.0];
    m = str.match(/color\\(srgb\\s+(\\d+(?:\\.\\d+)?)\\s+(\\d+(?:\\.\\d+)?)\\s+(\\d+(?:\\.\\d+)?)(?:\\s*\\/\\s*(\\d+(?:\\.\\d+)?))?\\)/);
    if (m) return [+m[1], +m[2], +m[3], m[4] !== undefined ? +m[4] : 1.0];
    return null;
}
function composite(fg, bg) {
    const a = fg[3];
    return [fg[0]*a + bg[0]*(1-a), fg[1]*a + bg[1]*(1-a), fg[2]*a + bg[2]*(1-a)];
}
function relLum(rgb) {
    const lin = c => c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    return 0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2]);
}
function cr(fgRgba, bgRgba) {
    const eff = composite(fgRgba, bgRgba);
    const l1 = relLum(eff), l2 = relLum(bgRgba);
    return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
}
"""


def _assert_title_block_contrast(page: Page, min_ratio: float = 4.5) -> None:
    ratios = page.evaluate(
        f"""() => {{
            {_CONTRAST_JS_HELPERS}
            const appBgStr = getComputedStyle(document.querySelector('[data-testid="stApp"]')).backgroundColor;
            const appBg = parseColor(appBgStr);
            const label = document.querySelector('.st-key-title_block [data-testid="stMetricLabel"] p');
            const value = document.querySelector('.st-key-title_block [data-testid="stMetricValue"]');
            const labelStr = getComputedStyle(label).color;
            const valueStr = getComputedStyle(value).color;
            const labelCol = parseColor(labelStr);
            const valueCol = parseColor(valueStr);
            if (!appBg || !labelCol || !valueCol) {{
                throw new Error(`Unparsed color: bg=${{appBgStr}}, label=${{labelStr}}, value=${{valueStr}}`);
            }}
            return {{
                appBg: appBgStr,
                labelStr,
                valueStr,
                labelRatio: cr(labelCol, appBg),
                valueRatio: cr(valueCol, appBg),
            }};
        }}"""
    )
    assert ratios["labelRatio"] >= min_ratio, ratios
    assert ratios["valueRatio"] >= min_ratio, ratios


def test_dark_mode_system_theme_map_contrast_and_screenshots(
    browser, app_url: str, shots_dir: Path
) -> None:
    context = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark")
    try:
        page = context.new_page()
        # 1. Trước khi chạy — Dark Desktop 1440x900
        _open_app(page, app_url, width=1440, height=900)
        expect(_map_frame(page).locator("path.candidate-marker").first).to_be_visible(timeout=30_000)
        bg = page.evaluate("() => getComputedStyle(document.querySelector('[data-testid=\"stApp\"]')).backgroundColor")
        assert bg == "rgb(18, 24, 22)", bg
        _assert_title_block_contrast(page, min_ratio=4.5)
        _assert_dark_map_styling(page)
        _assert_no_horizontal_overflow(page)
        _wait_settled_for_shot(page)
        page.screenshot(path=str(shots_dir / "dark-before-desktop.png"), full_page=False)

        # 2. Trước khi chạy — Dark Mobile 390x844
        m_page = context.new_page()
        _open_app(m_page, app_url, width=390, height=844)
        expect(_map_frame(m_page).locator("path.candidate-marker").first).to_be_visible(timeout=30_000)
        _assert_dark_map_styling(m_page)
        _assert_boundary_inside_map(m_page)
        _assert_no_horizontal_overflow(m_page)
        _wait_settled_for_shot(m_page)
        m_page.screenshot(path=str(shots_dir / "dark-before-mobile.png"), full_page=False)
        m_page.close()

        # 3. Sau khi chạy mặc định (time_limit=60) — Dark Desktop 1440x900
        _ensure_sidebar_open(page)
        _set_slider(page, "Giới hạn thời gian (giây)", 60)
        _run_button(page).click()
        expect(_metric_value(page, "Trạng thái")).to_have_text("Tối ưu", timeout=90_000)
        expect(_metric_value(page, "Số trạm")).to_have_text("31")
        expect(_metric_value(page, "Tổng chi phí")).to_have_text("10,6 tỷ đồng")
        expect(_metric_value(page, "Độ phủ")).to_have_text("60,1%")
        expect(_map_frame(page).locator("path.station-marker")).to_have_count(31, timeout=30_000)
        expect(_map_frame(page).locator("path.service-diamond")).to_have_count(31, timeout=30_000)
        _assert_title_block_contrast(page, min_ratio=4.5)
        _assert_dark_map_styling(page)
        _assert_no_horizontal_overflow(page)
        _wait_settled_for_shot(page)
        page.screenshot(path=str(shots_dir / "dark-after-desktop.png"), full_page=False)

        # 4. Tab Phân bổ nhu cầu — Dark Desktop 1440x900
        alloc_tab = page.get_by_role("tab", name="Phân bổ nhu cầu")
        alloc_tab.scroll_into_view_if_needed()
        alloc_tab.click()
        expect(alloc_tab).to_have_attribute("aria-selected", "true")
        expect(page.locator('[data-testid="stDataFrame"]').last).to_be_visible(timeout=15_000)
        page.locator('[data-testid="stDownloadButton"]').first.scroll_into_view_if_needed()
        page.screenshot(path=str(shots_dir / "dark-after-tab-allocation.png"), full_page=False)

        # 5. Sau khi chạy — Dark Mobile 390x844
        page.set_viewport_size({"width": 390, "height": 844})
        _collapse_sidebar_if_open(page)
        _scroll_main_to_top(page)
        expect(_metric_value(page, "Số trạm")).to_have_text("31")
        _assert_dark_map_styling(page)
        _assert_boundary_inside_map(page)
        _assert_no_horizontal_overflow(page)
        _wait_settled_for_shot(page)
        page.screenshot(path=str(shots_dir / "dark-after-mobile.png"), full_page=False)

        # 6. Precheck lỗi (R=300, alpha=0.9) — Dark Desktop 1440x900
        page.set_viewport_size({"width": 1440, "height": 900})
        _ensure_sidebar_open(page)
        _set_slider(page, "Tỷ lệ phủ tối thiểu α", 0.9)
        _run_button(page).click()
        alert = page.locator('[data-testid="stAlert"]').first
        expect(alert).to_be_visible(timeout=30_000)
        expect(alert).to_contain_text("85,9")
        expect(_map_frame(page).locator("path.candidate-marker").first).to_be_visible(timeout=30_000)
        _assert_dark_map_styling(page)
        _wait_settled_for_shot(page)
        page.screenshot(path=str(shots_dir / "dark-after-precheck-error.png"), full_page=False)
    finally:
        context.close()


def test_theme_menu_switch_light_dark_light(page: Page, app_url: str, shots_dir: Path) -> None:
    _open_app(page, app_url, width=1440, height=900)
    app_el = page.locator('[data-testid="stApp"]')
    assert app_el.evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(246, 247, 244)"

    # Mở menu ⋮ (toolbarMode="viewer" vẫn giữ bộ chọn System / Light / Dark)
    page.locator('[data-testid="stMainMenu"] button').first.click()
    switcher = page.locator('[data-testid="stThemeSwitcher"]')
    expect(switcher).to_be_visible(timeout=10_000)
    expect(switcher.locator('[data-testid="stMainMenuItem-theme-System"]')).to_be_visible()
    expect(switcher.locator('[data-testid="stMainMenuItem-theme-Light"]')).to_be_visible()
    expect(switcher.locator('[data-testid="stMainMenuItem-theme-Dark"]')).to_be_visible()

    # Chuyển Light -> Dark và chụp menu đang mở kèm giao diện đã đổi sang tối
    switcher.locator('[data-testid="stMainMenuItem-theme-Dark"]').click()
    expect(app_el).to_have_css("background-color", "rgb(18, 24, 22)", timeout=10_000)
    page.screenshot(path=str(shots_dir / "menu-theme-switch.png"), full_page=False)

    # Chuyển Dark -> Light -> kiểm nền trở về sáng tức thì
    switcher.locator('[data-testid="stMainMenuItem-theme-Light"]').click()
    expect(app_el).to_have_css("background-color", "rgb(246, 247, 244)", timeout=10_000)

    # Chuyển lại sang Dark và tải lại trang -> Streamlit gửi theme.type="dark" ở phiên mới, bản đồ đổi sang tối
    switcher.locator('[data-testid="stMainMenuItem-theme-Dark"]').click()
    expect(app_el).to_have_css("background-color", "rgb(18, 24, 22)", timeout=10_000)
    page.reload(wait_until="domcontentloaded")
    expect(
        page.get_by_role("heading", name="Quy hoạch vị trí đặt tủ đồ thông minh")
    ).to_be_visible(timeout=30_000)
    expect(app_el).to_have_css("background-color", "rgb(18, 24, 22)", timeout=10_000)
    _assert_dark_map_styling(page)


def test_model_selector_switches_form_parameters(page: Page, app_url: str) -> None:
    _open_app(page, app_url, width=1440, height=900)
    sidebar = page.locator('[data-testid="stSidebar"]')
    radio = sidebar.locator('[data-testid="stRadio"]')
    form = sidebar.locator('[data-testid="stForm"]')
    expect(radio).to_be_visible(timeout=15_000)
    expect(form).to_be_visible(timeout=15_000)

    # Bộ chọn mô hình nằm phía trên form trong cả DOM và bố cục
    r_box = radio.bounding_box()
    f_box = form.bounding_box()
    assert r_box is not None and f_box is not None
    assert r_box["y"] + r_box["height"] <= f_box["y"] + 2, (r_box, f_box)
    dom_before = page.evaluate(
        """() => {
            const r = document.querySelector('[data-testid="stSidebar"] [data-testid="stRadio"]');
            const f = document.querySelector('[data-testid="stSidebar"] [data-testid="stForm"]');
            return !!(r && f && (r.compareDocumentPosition(f) & Node.DOCUMENT_POSITION_FOLLOWING));
        }"""
    )
    assert dom_before

    for opt in ("Mô hình chính · MILP hai pha", "Set covering", "Max covering", "p-median"):
        expect(radio.locator("label", has_text=opt)).to_be_visible()

    _assert_title_block_not_truncated(page)
    _shot_006(page, "selector-desktop.png")

    # Set covering: không có α, không có p, chỉ có R và loại trạm
    _select_model(page, "Set covering")
    expect(sidebar.get_by_role("slider", name="Tỷ lệ phủ tối thiểu α")).to_have_count(0)
    expect(sidebar.get_by_role("spinbutton", name="Số trạm p", exact=True)).to_have_count(0)
    expect(sidebar.get_by_role("slider", name="Bán kính phục vụ R (m)")).to_be_visible()

    # Max covering: có R và p, không có α
    _select_model(page, "Max covering")
    expect(sidebar.get_by_role("slider", name="Tỷ lệ phủ tối thiểu α")).to_have_count(0)
    expect(sidebar.get_by_role("slider", name="Bán kính phục vụ R (m)")).to_be_visible()
    expect(sidebar.get_by_role("spinbutton", name="Số trạm p", exact=True)).to_be_visible()

    # p-median: có p và bán kính đánh giá R, không có α
    _select_model(page, "p-median")
    expect(sidebar.get_by_role("slider", name="Tỷ lệ phủ tối thiểu α")).to_have_count(0)
    expect(sidebar.get_by_role("slider", name="Bán kính đánh giá R (m)")).to_be_visible()
    expect(sidebar.get_by_role("spinbutton", name="Số trạm p", exact=True)).to_be_visible()

    # Quay lại mô hình chính: có α
    _select_model(page, "Mô hình chính · MILP hai pha")
    expect(sidebar.get_by_role("slider", name="Tỷ lệ phủ tối thiểu α")).to_be_visible()


def test_set_covering_r500_metrics_map_and_summary_json(page: Page, app_url: str) -> None:
    D_l1, pop, _ids = _l1_inputs()
    expected_selected, _coverable, expected_status = models.set_covering(D_l1, 500)
    expected_res = models.run_baseline("set_covering", D_l1, pop, R=500)
    assert expected_status == 0
    expected_n = len(expected_selected)

    _open_app(page, app_url, width=1440, height=900)
    _select_model(page, "Set covering")
    _set_slider(page, "Bán kính phục vụ R (m)", 500)
    _run_button(page).click()

    expect(_metric_value(page, "Trạng thái")).to_have_text("Tối ưu", timeout=60_000)
    expect(_metric_value(page, "Số trạm")).to_have_text(str(expected_n))
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    cap_text = page.locator('[data-testid="stMain"] [data-testid="stCaptionContainer"]').first.inner_text()
    assert "Mô hình này không xét sức chứa" in cap_text and "×3" in cap_text, cap_text

    frame = _map_frame(page)
    expect(frame.locator(".leaflet-container")).to_be_visible(timeout=30_000)
    expect(frame.locator("path.station-marker")).to_have_count(expected_n, timeout=30_000)
    expect(frame.locator("path.service-diamond")).to_have_count(expected_n, timeout=30_000)

    dl_btn = page.locator('[data-testid="stDownloadButton"] button', has_text="summary.json")
    expect(dl_btn).to_be_visible()
    with page.expect_download(timeout=30_000) as dl_info:
        dl_btn.click()
    download = dl_info.value
    assert download.suggested_filename == "summary.json"
    dl_path = download.path()
    assert dl_path is not None
    summary = json.loads(Path(dl_path).read_text(encoding="utf-8"))
    assert summary["model"] == "set_covering"
    assert summary["status"] == expected_res["status_label"]
    assert summary["station_count"] == expected_n
    assert summary["coverage_pct"] == expected_res["metrics"]["coverage_pct"]
    assert summary["config"]["radius"] == 500

    _scroll_main_to_top(page)
    _shot_006(page, "set-covering-desktop.png")
    _shot_007(page, "set-covering-desktop.png")


def test_max_covering_default_coverage_matches_models(page: Page, app_url: str) -> None:
    D_l1, pop, _ids = _l1_inputs()
    expected = models.run_baseline("max_covering", D_l1, pop, R=300, p=31)
    expected_cov = f"{fmt_vi_decimal(expected['metrics']['coverage_pct'], 1)}%"
    expected_n = str(len(expected["selected"]))

    _open_app(page, app_url, width=1440, height=900)
    _select_model(page, "Max covering")
    _set_slider(page, "Bán kính phục vụ R (m)", 300)
    p_input = page.locator('[data-testid="stSidebar"]').get_by_role("spinbutton", name="Số trạm p", exact=True)
    expect(p_input).to_have_value("31")
    _run_button(page).click()

    expect(_metric_value(page, "Trạng thái")).to_have_text("Tối ưu", timeout=60_000)
    expect(_metric_value(page, "Số trạm")).to_have_text(expected_n)
    expect(_metric_value(page, "Độ phủ trong R")).to_have_text(expected_cov)
    expect(_map_frame(page).locator("path.station-marker")).to_have_count(int(expected_n), timeout=30_000)

    _scroll_main_to_top(page)
    _shot_006(page, "max-covering-desktop.png")


def test_p_median_default_avg_dist_and_title_block_and_mobile(page: Page, app_url: str) -> None:
    D_l1, pop, _ids = _l1_inputs()
    expected = models.run_baseline("p_median", D_l1, pop, R=300, p=31, seed=0)
    expected_avg = f"{fmt_vi_decimal(expected['metrics']['avg_dist_m'], 1)} m"

    _open_app(page, app_url, width=1440, height=900)
    _select_model(page, "p-median")
    p_input = page.locator('[data-testid="stSidebar"]').get_by_role("spinbutton", name="Số trạm p", exact=True)
    expect(p_input).to_have_value("31")
    _set_slider(page, "Bán kính đánh giá R (m)", 300)
    _run_button(page).click()

    expect(_metric_value(page, "Trạng thái")).to_have_text(
        "Heuristic, chưa chứng minh tối ưu", timeout=60_000
    )
    expect(page.locator(".app-header p")).to_contain_text(
        "p-median · p = 31 · R = 300 m chỉ dùng để tính độ phủ · không xét sức chứa."
    )
    expect(_metric_value(page, "Số trạm")).to_have_text("31")
    expect(_metric_value(page, "KC trung bình")).to_have_text(expected_avg)
    expect(_map_frame(page).locator("path.station-marker")).to_have_count(31, timeout=30_000)
    _assert_title_block_not_truncated(page)

    _scroll_main_to_top(page)
    _shot_006(page, "p-median-desktop.png")
    _shot_007(page, "p-median-desktop.png")

    # Kiểm tra định dạng số nguyên trên bảng Trạm được chọn và chụp stations-table.png
    stations_tab = page.get_by_role("tab", name="Trạm được chọn")
    stations_tab.scroll_into_view_if_needed()
    stations_df = page.locator('[data-testid="stDataFrame"]').first
    expect(stations_df).to_be_visible(timeout=15_000)
    stations_df.scroll_into_view_if_needed()
    expect(stations_df.locator("table th").first).to_be_attached(timeout=15_000)
    st_headers = stations_df.locator("table th").evaluate_all(
        "els => els.map(e => e.innerText.trim())"
    )
    assert "Chi phí (triệu đồng)" in st_headers, st_headers
    _shot_007(page, "stations-table.png")

    # 1024x768 có thanh bên mở: kiểm tra khung tên không cắt chữ
    page.set_viewport_size({"width": 1024, "height": 768})
    _ensure_sidebar_open(page)
    _scroll_main_to_top(page)
    _assert_title_block_not_truncated(page)
    _shot_006(page, "title-block-1024.png")

    # 390x844 mobile trên kết quả baseline: không cuộn ngang
    page.set_viewport_size({"width": 390, "height": 844})
    _collapse_sidebar_if_open(page)
    _scroll_main_to_top(page)
    expect(_map_frame(page).locator(".leaflet-container")).to_be_visible(timeout=30_000)
    _assert_boundary_inside_map(page)
    _assert_no_horizontal_overflow(page)
    _shot_006(page, "baseline-mobile.png")


def test_session_comparison_table_and_model_switch_persistence(page: Page, app_url: str) -> None:
    D_l1, _pop, _ids = _l1_inputs()
    expected_sc_300 = len(models.set_covering(D_l1, 300)[0])

    _open_app(page, app_url, width=1440, height=900)
    _set_slider(page, "Giới hạn thời gian (giây)", 60)
    _run_button(page).click()
    expect(_metric_value(page, "Trạng thái")).to_have_text("Tối ưu", timeout=90_000)
    main_count = _metric_value(page, "Số trạm").inner_text().strip()
    main_cost = _metric_value(page, "Tổng chi phí").inner_text().strip()
    main_cov = _metric_value(page, "Độ phủ").inner_text().strip()

    _select_model(page, "Set covering")
    _run_button(page).click()
    expect(_metric_value(page, "Trạng thái")).to_have_text("Tối ưu", timeout=60_000)
    expect(_metric_value(page, "Số trạm")).to_have_text(str(expected_sc_300))

    comp_heading = page.get_by_role("heading", name="So sánh các mô hình đã chạy")
    expect(comp_heading).to_be_visible(timeout=15_000)
    comp_heading.scroll_into_view_if_needed()

    comp_df = page.locator('[data-testid="stDataFrame"]').last
    expect(comp_df).to_be_visible(timeout=15_000)
    comp_df.scroll_into_view_if_needed()
    expect(comp_df.locator("table tbody tr").first).to_be_attached(timeout=15_000)
    comp_info = comp_df.evaluate(
        """df => {
            const headers = [...df.querySelectorAll('table th')].map(e => e.innerText.trim());
            const modelNames = [...df.querySelectorAll('table tbody tr')].map(
                tr => (tr.querySelector('td')?.innerText || '').trim()
            );
            const scrollers = [...df.querySelectorAll('.dvn-scroller, [class*="scroller"]')];
            const overflows = scrollers.map(s => ({
                cls: s.className,
                scrollWidth: s.scrollWidth,
                clientWidth: s.clientWidth,
            })).filter(x => x.clientWidth > 0);
            return {headers, modelNames, overflows};
        }"""
    )
    assert comp_info["modelNames"] == ["Mô hình chính", "Set covering"], comp_info
    assert "KC TB · lớn nhất (m)" in comp_info["headers"], comp_info["headers"]
    assert "Trạm vượt sức chứa (×3)" in comp_info["headers"], comp_info["headers"]
    assert comp_info["overflows"], comp_info
    for ov in comp_info["overflows"]:
        assert ov["scrollWidth"] <= ov["clientWidth"] + 1, (
            f"Comparison table overflows horizontally at 1440x900: {ov}"
        )

    _shot_006(page, "comparison-desktop.png")
    _shot_007(page, "comparison-desktop.png")

    # Quay lại mô hình chính: vẫn giữ kết quả đã giải trước đó
    _select_model(page, "Mô hình chính · MILP hai pha")
    expect(_metric_value(page, "Trạng thái")).to_have_text("Tối ưu", timeout=20_000)
    expect(_metric_value(page, "Số trạm")).to_have_text(main_count)
    expect(_metric_value(page, "Tổng chi phí")).to_have_text(main_cost)
    expect(_metric_value(page, "Độ phủ")).to_have_text(main_cov)


def test_dark_mode_selector_and_comparison_contrast(browser, app_url: str) -> None:
    D_l1, _pop, _ids = _l1_inputs()
    expected_sc_300 = len(models.set_covering(D_l1, 300)[0])

    context = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark")
    try:
        page = context.new_page()
        _open_app(page, app_url, width=1440, height=900)
        _ensure_sidebar_open(page)

        selector_ratios = page.evaluate(
            f"""() => {{
                {_CONTRAST_JS_HELPERS}
                const sidebar = document.querySelector('[data-testid="stSidebar"]');
                const sbBgStr = getComputedStyle(sidebar).backgroundColor;
                const sbBg = parseColor(sbBgStr);
                const radio = sidebar.querySelector('[data-testid="stRadio"]');
                const labels = [...radio.querySelectorAll('[role="radiogroup"] [data-testid="stMarkdownContainer"] p')];
                const captions = [...radio.querySelectorAll('[data-testid="stCaptionContainer"] p')];
                return {{
                    sbBgStr,
                    labels: labels.map(el => ({{
                        text: el.innerText.trim(),
                        color: getComputedStyle(el).color,
                        ratio: cr(parseColor(getComputedStyle(el).color), sbBg),
                    }})),
                    captions: captions.map(el => ({{
                        text: el.innerText.trim(),
                        color: getComputedStyle(el).color,
                        ratio: cr(parseColor(getComputedStyle(el).color), sbBg),
                    }})),
                }};
            }}"""
        )
        assert len(selector_ratios["labels"]) == 4, selector_ratios
        assert len(selector_ratios["captions"]) == 4, selector_ratios
        for item in selector_ratios["labels"] + selector_ratios["captions"]:
            assert item["ratio"] >= 4.5, f"Low contrast in dark selector: {item} on {selector_ratios['sbBgStr']}"

        _shot_006(page, "selector-dark.png")

        # Chạy Set covering rồi đo tương phản bảng so sánh trong dark mode
        _select_model(page, "Set covering")
        _run_button(page).click()
        expect(_metric_value(page, "Trạng thái")).to_have_text("Tối ưu", timeout=60_000)
        expect(_metric_value(page, "Số trạm")).to_have_text(str(expected_sc_300))

        comp_heading = page.get_by_role("heading", name="So sánh các mô hình đã chạy")
        comp_heading.scroll_into_view_if_needed()
        comp_df = page.locator('[data-testid="stDataFrame"]').last
        expect(comp_df).to_be_visible(timeout=15_000)
        comp_df.scroll_into_view_if_needed()
        expect(comp_df.locator("table th").first).to_be_attached(timeout=15_000)

        table_ratios = page.evaluate(
            f"""() => {{
                {_CONTRAST_JS_HELPERS}
                function effectiveBg(el, fallback) {{
                    let cur = el;
                    while (cur) {{
                        const c = parseColor(getComputedStyle(cur).backgroundColor);
                        if (c && c[3] > 0.01) return c[3] >= 0.99 ? c : [...composite(c, fallback), 1.0];
                        cur = cur.parentElement;
                    }}
                    return fallback;
                }}
                const appBgStr = getComputedStyle(document.querySelector('[data-testid="stApp"]')).backgroundColor;
                const appBg = parseColor(appBgStr);
                const dfs = [...document.querySelectorAll('[data-testid="stDataFrame"]')];
                const df = dfs[dfs.length - 1];
                const headers = [...df.querySelectorAll('table th')].map(el => ({{
                    text: el.innerText.trim(),
                    color: getComputedStyle(el).color,
                    ratioVsApp: cr(parseColor(getComputedStyle(el).color), appBg),
                    ratioVsBg: cr(parseColor(getComputedStyle(el).color), effectiveBg(el, appBg)),
                }}));
                const cells = [...df.querySelectorAll('table td')].map(el => ({{
                    text: el.innerText.trim(),
                    color: getComputedStyle(el).color,
                    ratioVsApp: cr(parseColor(getComputedStyle(el).color), appBg),
                    ratioVsBg: cr(parseColor(getComputedStyle(el).color), effectiveBg(el, appBg)),
                }}));
                const canvas = df.querySelector('canvas');
                let maxCanvasFgRatio = 0;
                if (canvas) {{
                    const ctx = canvas.getContext('2d');
                    if (ctx) {{
                        const data = ctx.getImageData(0, 0, Math.min(canvas.width, 400), Math.min(canvas.height, 120)).data;
                        for (let i = 0; i < data.length; i += 4) {{
                            if (data[i+3] < 220) continue;
                            const r = cr([data[i]/255, data[i+1]/255, data[i+2]/255, 1.0], appBg);
                            if (r > maxCanvasFgRatio) maxCanvasFgRatio = r;
                        }}
                    }}
                }}
                return {{appBgStr, headers, cells, maxCanvasFgRatio}};
            }}"""
        )
        assert len(table_ratios["headers"]) >= 9, table_ratios
        assert len(table_ratios["cells"]) >= 9, table_ratios
        for item in table_ratios["headers"] + table_ratios["cells"]:
            assert item["ratioVsApp"] >= 4.5 and item["ratioVsBg"] >= 4.5, (
                f"Low contrast in dark comparison table: {item} on {table_ratios['appBgStr']}"
            )
        assert table_ratios["maxCanvasFgRatio"] >= 4.5, table_ratios
    finally:
        context.close()


