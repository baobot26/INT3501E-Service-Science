# Tối ưu vị trí đặt trạm dịch vụ (smart locker) — quận Cầu Giấy cũ

## Mô tả bài toán

Chọn **tập vị trí đặt trạm dịch vụ nhỏ nhất** trên địa bàn quận Cầu Giấy cũ
(3 phường: Cầu Giấy, Nghĩa Đô, Yên Hòa; khoảng 12,46 km²), thỏa mãn:

- Chỉ đặt tại **điểm ứng viên hợp lệ** lấy từ OpenStreetMap (bãi đỗ xe, siêu thị,
  cửa hàng tiện lợi, bưu điện; tòa chung cư hiện đang bị vô hiệu hóa trong
  `data/station_parameters.csv`). Ứng viên nằm trong **khu dân cư (H)** hoặc
  **khu vực cấm (C**: quân sự, đường sắt, nghĩa trang, mặt nước**)** bị loại.
- Mỗi người dân phát sinh **2 đơn vị nhu cầu/ngày** — tổng ≈ 588.166 đơn vị/ngày
  từ 1.544 điểm cầu (dữ liệu dân số WorldPop, lưới ~100 m).
- Mỗi điểm cầu được gán **trọn vẹn** cho **trạm được chọn gần nhất**; điểm cầu
  được coi là phục vụ nếu khoảng cách Manhattan (trên hệ chiếu UTM EPSG:32648)
  đến trạm đó ≤ R.
- Tổng nhu cầu được phục vụ ≥ α·100% và không trạm nào vượt công suất ngày.
- Hàm mục tiêu **hai pha**: (1) tối thiểu số trạm, sau đó (2) với số trạm tối ưu,
  tối thiểu tổng chi phí đầu tư.

Ba mô hình đối chứng được cài đặt thêm: **set covering**, **maximal covering**
và **p-median** (heuristic hoán đổi Teitz-Bart).

Kết quả kiểm tra polygon (R = 500 m, α = 0,95):

| Phương án xác định H | Ứng viên hợp lệ | Cận trên độ phủ | Khả thi? |
|---|---|---|---|
| A: `landuse=residential` | 129 / 481 | 92,73% | ❌ (chỉ đạt khi R = 800 m) |
| B: khối nhà ở (building footprints) | 184 / 481 | 97,94% | ✅ |

→ Kịch bản chính của MVP: **phương án B, R = 500 m, α = 0,95**.
Chi tiết kế hoạch: [`docs/implementation-plan-updated.md`](docs/implementation-plan-updated.md).

## Tech stack

- **Python 3.12** — môi trường ảo `.venv` (tạo bằng `uv` hoặc `python3 -m venv`)
- **osmnx** — tải ranh giới, điểm ứng viên và polygon từ OpenStreetMap/Overpass
- **GeoPandas / Shapely / pyproj** — làm sạch polygon, chuyển hệ chiếu EPSG:4326 → EPSG:32648
- **pandas / numpy** — bảng dữ liệu và ma trận khoảng cách
- **scipy.optimize.milp** (solver HiGHS) — giải bài toán MILP
- **networkx** — khoảng cách đi bộ trên đồ thị đường (baseline)
- **folium** — bản đồ HTML để kiểm tra trực quan

Cài đặt:

```bash
uv venv .venv && uv pip install --python .venv/bin/python osmnx scipy folium
# hoặc: pip install -r requirements.txt
```

## Cấu trúc mã nguồn

```
src/
├── fetch_data.py    # Tải dữ liệu đầu vào: ranh giới, ứng viên OSM, demand WorldPop, mạng đi bộ
├── polygons.py      # Kiểm tra polygon khu dân cư (H) / khu cấm (C), lọc ứng viên,
│                    #   tính 2 cận trên (độ phủ, năng lực) và xuất bản đồ kiểm tra
├── distances.py     # Ma trận khoảng cách đi bộ demand→ứng viên (baseline, ghi dist.npy)
└── models.py        # 3 mô hình đối chứng: set covering, max covering, p-median

data/
├── boundary.geojson, candidates.csv, demand.csv   # dữ liệu đầu vào
├── station_parameters.csv       # công suất / chi phí / bật-tắt theo loại trạm
├── candidates_feasible.csv      # ứng viên sau lọc polygon (kèm cờ feasible)
├── residential_areas.geojson, forbidden_areas.geojson
└── polygon_report*.json         # báo cáo kiểm tra theo cấu hình

results/
├── results_walking_all.csv      # kết quả baseline (khoảng cách đi bộ) — không ghi đè
└── results.csv                  # kết quả 3 mô hình đối chứng

docs/
├── implementation-plan.md          # kế hoạch gốc (đề bài)
└── implementation-plan-updated.md  # kế hoạch cập nhật theo số liệu polygon thật
```

## Cách chạy

```bash
.venv/bin/python src/fetch_data.py                # 1. tải dữ liệu (cần mạng + file WorldPop tif)
.venv/bin/python src/polygons.py --mode B         # 2. lọc ứng viên theo polygon (A hoặc B)
.venv/bin/python src/distances.py                 # 3. ma trận khoảng cách đi bộ (baseline)
.venv/bin/python src/models.py                    # 4. chạy 3 mô hình đối chứng
```

Các bước tiếp theo của MVP (ma trận khoảng cách Manhattan, mô hình chính 2 pha,
bộ kiểm tra ràng buộc độc lập, xuất kết quả theo `run_id`) được mô tả trong
`docs/implementation-plan-updated.md`.