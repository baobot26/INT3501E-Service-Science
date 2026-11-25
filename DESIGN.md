# Design System — Quy hoạch vị trí đặt tủ đồ thông minh

Tài liệu đặc tả hệ thống thiết kế thực tế đang triển khai cho ứng dụng Streamlit (`app.py`, `.streamlit/config.toml`, `src/design_tokens.py`, `src/make_map.py`).

## 1. Bối cảnh sử dụng, kiến trúc thị giác và cơ chế chuyển theme
- **Chế độ sáng (mặc định, cảnh dùng chính):** tối ưu cho phòng học sáng đèn và máy chiếu có độ tương phản thấp — nền giấy quy hoạch dịu, chữ mực đậm, tương phản cao.
- **Chế độ tối (cảnh dùng thứ hai):** phục vụ giảng viên hoặc người chấm mở liên kết buổi tối trong phòng tối trên laptop bật dark mode — vẫn cùng thế giới bản vẽ quy hoạch trên nền tối ánh xanh mực, không dùng đen tuyền, không neon và không hào quang phát sáng.
- **Nguồn sự thật cho token màu:** tập trung tại `src/design_tokens.py` (`UI_LIGHT`, `UI_DARK`, `MAP_LIGHT`, `MAP_DARK`, qua `ui_tokens(mode)` / `map_tokens(mode)`) và `.streamlit/config.toml` (`[theme.light]` / `[theme.dark]` cùng `[theme.light.sidebar]` / `[theme.dark.sidebar]`), được kiểm khớp tự động bởi `tests/test_design_tokens.py`. Không có mã hex cứng rải rác trong `app.py` hay `src/make_map.py`.
- **Cơ chế chọn theme trong Streamlit 1.66.1 (đã kiểm chứng thực nghiệm):**
  - `[client] toolbarMode = "viewer"` ẩn nút `Deploy` và menu dev nhưng vẫn giữ bộ chọn **System / Light / Dark** (`stThemeSwitcher`) trực tiếp trong menu `⋮`.
  - Khi người dùng chọn **Light / Dark** trong menu `⋮`, giao diện Streamlit chuyển tức thì mà **không tự chạy lại script Python** và không gắn biến `--st-*` trên DOM trang chính; `st.context.theme.type` đọc giá trị theme tại thời điểm phiên trình duyệt kết nối (hoặc sau khi tải lại trang theo lựa chọn đã lưu trong `localStorage`).
  - Vì vậy, `CUSTOM_CSS` trong `app.py` dùng màu kế thừa (`color: inherit`, `background: transparent`, `color-mix(in srgb, currentColor 78%, transparent)` cho nhãn phụ và `22%` cho viền mảnh) để toàn bộ chrome đổi ngay theo menu `⋮`; chỉ vùng nền đặc của nút dính đáy sidebar (`--plan-panel`) và bản đồ HTML trong `st.iframe` (dựng theo `st.context.theme.type`, khóa cache gồm `theme_mode`) cập nhật khi tải lại trang hoặc ở phiên mở mới.
- **Bố cục bản vẽ quy hoạch + khung tên:** thanh bên trái (`st.sidebar.form("scenario")`, `initial_sidebar_state="auto"`) gồm 4 nhóm (**Phạm vi và mục tiêu** · **Hệ số** · **Loại trạm** · **Solver**) và nút chính **Chạy kịch bản** ghim dính đáy thanh bên; vùng chính gồm tiêu đề + 1 câu tóm tắt → **khung tên** 5 ô `st.metric` ngăn viền `1px` → **bản đồ toàn chiều rộng 560 px** (`410 px` trên mobile) → 2 tab bảng chi tiết → 3 nút tải tệp máy (`selected_stations.csv`, `assigned_demand.csv`, `summary.json`).

## 2. Token màu và kiểm định tương phản (WCAG 2.1)

Sàn chất lượng bắt buộc: chữ thường ≥ 4,5:1, chữ lớn ≥ 3,0:1, đồ họa phi văn bản trên bản đồ ≥ 3,0:1 (WCAG 1.4.11).

### Bề mặt, chữ, viền và màu nhấn giao diện (`UI_LIGHT` / `UI_DARK`)
| Vai trò | Khóa token | Sáng (`UI_LIGHT`) | Tỉ lệ sáng | Tối (`UI_DARK`) | Tỉ lệ tối |
|---|---|---|---:|---|---:|
| Nền chính | `ground` | `#F6F7F4` | — | `#121816` | — |
| Nền thanh bên / panel | `panel` | `#ECEFEA` | — | `#1A2220` | — |
| Viền mảnh 1px | `line` | `#C9CFC7` | — | `#34403B` | — |
| Chữ chính (trên ground / panel) | `ink` | `#1C2421` | 14,75:1 / 13,67:1 | `#E4E9E5` | 14,63:1 / 13,21:1 |
| Chữ phụ (`ink_muted` / `color-mix 78%`) | `ink_muted` | `#46524D` (`~#4C524F`) | 7,59:1 (7,43:1) | `#A9B4AE` (`~#B6BBB7`) | 8,41:1 (9,23:1) |
| Màu nhấn nút chính (vs ground / panel) | `accent` | `#165D4A` | 7,24:1 / 6,71:1 | `#2E7D66` | 3,63:1 / 3,28:1 |
| Chữ trắng trên nút chính | `on_accent` / `accent` | `#FFFFFF` | 7,78:1 | `#FFFFFF` | 4,95:1 |
| Chữ trắng trên nút hover | `on_accent` / `accent_hover` | `#FFFFFF` trên `#10493A` | 10,31:1 | `#FFFFFF` trên `#246A56` | 6,42:1 |
| Màu liên kết (trên ground / panel) | `link` | `#165D4A` | 7,24:1 / 6,71:1 | `#7CC7AE` | 9,11:1 / 8,22:1 |
| Tiêu đề cột bảng | `table_header_ink` / `bg` | `#1C2421` / `#E4E9E2` | 12,88:1 | `#E4E9E5` / `#212B28` | 11,86:1 |
| Vùng bôi chọn chữ | `ink` / `selection_bg` | `#1C2421` / `#CBE3DA` | 11,74:1 | `#E4E9E5` / `#244A3E` | 8,04:1 |

### Màu trạng thái / thông báo (Sáng và Tối)
| Trạng thái | Sáng (chữ / nền) | Tỉ lệ sáng | Tối (chữ / nền) | Tỉ lệ tối |
|---|---|---:|---|---:|
| Lỗi (`st.error`) | `#8B1E18` / `#FBF0EE` | 8,19:1 | `#F6B4AE` / `#2E1A19` | 9,44:1 |
| Cảnh báo (`st.warning`) | `#6F4800` / `#FAF4E4` | 7,35:1 | `#F2D08B` / `#2C2415` | 10,34:1 |
| Nhấn cam (`orange*`) | `#7A340C` / `#FCF2EA` | 8,54:1 | `#F4BE94` / `#2D2016` | 9,51:1 |
| Thông tin (`st.info`) | `#1A4666` / `#ECF3F7` | 8,87:1 | `#A6D1EE` / `#162531` | 9,66:1 |
| Thành công (`st.success`) | `#114A3B` / `#E9F3EF` | 8,97:1 | `#8CD5BC` / `#152B24` | 8,81:1 |

### Bảng màu dữ liệu bản đồ (`MAP_LIGHT` trên `#F4F4F2` và `MAP_DARK` trên `#1B211F`)
| Lớp dữ liệu | Hình dạng | Sáng (`MAP_LIGHT`) | Tỉ lệ sáng | Tối (`MAP_DARK`) | Tỉ lệ tối |
|---|---|---|---:|---|---:|
| Bãi đỗ xe (`amenity=parking`) | Chấm tròn (`r=5.2`) | `#2B5B84` | 6,50:1 | `#6BA6D6` | 6,27:1 |
| Siêu thị (`shop=supermarket`) | Chấm tròn (`r=5.2`) | `#A84B14` | 5,18:1 | `#E28E55` | 6,41:1 |
| Bưu điện (`amenity=post_office`) | Chấm tròn (`r=5.2`) | `#7B3B6B` | 7,12:1 | `#C985B4` | 5,84:1 |
| Cửa hàng tiện lợi (`shop=convenience`) | Chấm tròn (`r=5.2`) | `#826514` | 4,99:1 | `#CCB054` | 7,72:1 |
| Tòa chung cư (`building=apartments`) | Vòng rỗng khi tắt | `#58636A` | 5,60:1 | `#9AA7AE` | 6,63:1 |
| Viền hình thoi L1 (`diamond_stroke`) | Hình thoi `1.6px` | `#165D4A` | 7,07:1 | `#5FC4A4` | 7,73:1 |
| Nhu cầu được phục vụ (`served`) | **Chấm đặc** (`r=2.2`) | `#3B7468` | 4,91:1 | `#5FB89C` | 6,87:1 |
| Nhu cầu chưa phục vụ (`unserved`) | **Vòng rỗng** (`r=1.8`) | `#7A716B` | 4,33:1 | `#9E9790` | 5,67:1 |
| Mật độ nhu cầu xem trước | Chấm nhỏ (`r=2.1`) | `#5C6862` | 5,25:1 | `#8E9B95` | 5,66:1 |
| Ranh giới 3 phường (`boundary`) | Đường mảnh `1.6px` | `#2F3A36` | 10,72:1 | `#C8D3CD` | 10,64:1 |
| Chữ chú giải (chính / phụ) | Trên nền chú giải | `#1C2421` / `#46524D` | 14,75:1 / 7,59:1 | `#E4E9E5` / `#A9B4AE` | 13,83:1 / 7,95:1 |

## 3. Kiểu chữ (Typography)
- **Một họ chữ duy nhất:** **Be Vietnam Pro** (SIL OFL 1.1), tự host qua `[server] enableStaticServing = true` và `[[theme.fontFaces]]` (`static/fonts/BeVietnamPro-{Regular,Medium,SemiBold,Bold}.ttf` + `OFL.txt`).
- Cỡ chữ gốc `baseFontSize = 16`, chỉ số khung tên `metricValueFontSize = "1.375rem"` / 600, chữ số dạng tabular (`font-variant-numeric: tabular-nums`).
- **Định dạng số:** giao diện hiển thị theo chuẩn Việt Nam (`10,6 tỷ đồng`, `60,1%`, `1.544`); tệp CSV / JSON tải về giữ định dạng máy chuẩn (dấu chấm thập phân).

## 4. Bo góc, viền và khoảng cách
- `baseRadius = "0.25rem"` (4 px), `buttonRadius = "0.25rem"` (4 px), `showWidgetBorder = true`, `showSidebarBorder = true`, không đổ bóng trang trí.
- Khoảng trên tiêu đề nhóm (`0.95rem`–`1.35rem`) luôn lớn hơn khoảng dưới (`0.4rem`–`0.5rem`).

## 5. Quy tắc thành phần (Components)
- **Nút hành động:** chỉ nút `Chạy kịch bản` dùng `type="primary"` (theo `primaryColor` của theme hiện hành, `:material/play_arrow:`), đặt cuối form và ghim đáy thanh bên; 3 nút tải dùng kiểu secondary (`on_click="ignore"`, `:material/download:`).
- **Khung tên (`.st-key-title_block`):** `st.container(key="title_block")` + `st.columns(5, gap=None)` với `st.metric`, tự chuyển thành lưới 2 cột + ô cuối trải toàn chiều rộng ở `≤ 640px`. Nguồn nghiệm nằm trong `help=` của ô **Trạng thái**.
- **Bản đồ (`src/make_map.py` + `st.iframe`):**
  - Nhúng một tầng qua `st.iframe(m.get_root().render(), height=560, width="stretch", alt=...)`.
  - Zoom lẻ `zoomSnap=0.25, zoomDelta=0.5`, fit theo bbox `data/boundary.geojson` kèm tự refit khi resize (quận Cầu Giấy chiếm ≥ 80% chiều cao khung bản đồ desktop và nằm trọn khung mobile).
  - Nền OSM chuẩn: sáng dùng `grayscale(88%) brightness(1.06) contrast(0.84) saturate(0.2)`; tối dùng `invert(1) hue-rotate(180deg) grayscale(82%) brightness(0.84) contrast(0.92)`; nút zoom, thước tỷ lệ, popup và chú giải tự đổi theo `map_tokens(mode)`.
  - Chú giải `<details id="plan-legend"><summary>Chú giải</summary>` mở sẵn trên desktop, thu gọn mặc định (≤ 15% diện tích bản đồ) trên mobile `≤ 640px`, ẩn loại có 0 trạm trong chú giải nghiệm. CLI `python -m src.make_map` giữ chế độ sáng.
- **Bảng dữ liệu:** 2 tab **Trạm được chọn** và **Phân bổ nhu cầu** nằm dưới bản đồ; dòng chưa phục vụ hiển thị `—` ở cột **Trạm phục vụ** và **Khoảng cách L1 (m)** (không hiển thị `None`, không phát sinh `ArrowTypeError`).
- **Truy cập bàn phím:** `:focus-visible` hiển thị viền `2px solid var(--plan-accent)` + vòng sáng `4px var(--plan-focus)` trên mọi điều khiển.
