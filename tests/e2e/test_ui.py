"""Playwright end-to-end tests for the redesigned Streamlit app."""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from playwright.sync_api import FrameLocator, Locator, Page

pytestmark = pytest.mark.e2e


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


def _ensure_sidebar_open(page: Page) -> None:
    btn = _run_button(page)
    if not btn.is_visible():
        toggle = page.locator('[data-testid="stExpandSidebarButton"]').first
        if toggle.is_visible():
            toggle.click()
        expect(btn).to_be_visible(timeout=10_000)


def _collapse_sidebar_if_open(page: Page) -> None:
    collapse = page.locator('[data-testid="stSidebarCollapseButton"] button').first
    if collapse.is_visible():
        collapse.click()
        expect(page.locator('[data-testid="stSidebar"]')).not_to_be_visible(timeout=10_000)


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


def _assert_title_block_contrast(page: Page, min_ratio: float = 4.5) -> None:
    ratios = page.evaluate(
        """() => {
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
            const appBgStr = getComputedStyle(document.querySelector('[data-testid="stApp"]')).backgroundColor;
            const appBg = parseColor(appBgStr);
            const label = document.querySelector('.st-key-title_block [data-testid="stMetricLabel"] p');
            const value = document.querySelector('.st-key-title_block [data-testid="stMetricValue"]');
            const labelStr = getComputedStyle(label).color;
            const valueStr = getComputedStyle(value).color;
            const labelCol = parseColor(labelStr);
            const valueCol = parseColor(valueStr);
            if (!appBg || !labelCol || !valueCol) {
                throw new Error(`Unparsed color: bg=${appBgStr}, label=${labelStr}, value=${valueStr}`);
            }
            return {
                appBg: appBgStr,
                labelStr,
                valueStr,
                labelRatio: cr(labelCol, appBg),
                valueRatio: cr(valueCol, appBg),
            };
        }"""
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

