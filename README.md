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

## Tech stack

- **Python 3.12** — môi trường ảo `.venv` (tạo bằng `uv` hoặc `python3 -m venv`)
- **osmnx** — tải ranh giới, điểm ứng viên và polygon từ OpenStreetMap/Overpass
- **GeoPandas / Shapely / pyproj** — làm sạch polygon, chuyển hệ chiếu EPSG:4326 → EPSG:32648
- **pandas / numpy** — bảng dữ liệu và ma trận khoảng cách
- **scipy.optimize.milp** (solver HiGHS) — giải bài toán MILP
- **networkx** — khoảng cách đi bộ trên đồ thị đường (baseline)
- **folium** — bản đồ HTML để kiểm tra trực quan
- **pytest** — bộ test offline (không gọi mạng, dùng fixture trong bộ nhớ)
- **Docker** — image `python:3.12-slim` đóng gói toàn bộ môi trường chạy

Cài đặt:

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
# hoặc: pip install -r requirements.txt -r requirements-dev.txt
```

## Cấu trúc mã nguồn

```
src/
├── fetch_data.py       # Tải dữ liệu đầu vào: ranh giới, ứng viên OSM, demand WorldPop, mạng đi bộ
├── polygons.py         # Kiểm tra polygon khu dân cư (H) / khu cấm (C), lọc ứng viên,
│                       #   tính 2 cận trên (độ phủ, năng lực) và xuất bản đồ kiểm tra
├── distances.py        # Ma trận khoảng cách đi bộ demand→ứng viên (baseline, ghi dist.npy)
├── models.py           # 3 mô hình đối chứng: set covering, max covering, p-median
├── preprocessing.py    # Đọc/validate đầu vào, xây ProblemData thống nhất cho pipeline
├── distances_l1.py     # Ma trận khoảng cách Manhattan (EPSG:32648) + manifest
├── model_main.py       # Mô hình chính: MILP hai pha (min trạm → min chi phí)
├── evaluate.py         # Kiểm định độc lập: gán lại nearest, kiểm coverage/capacity/objective
├── export.py           # Ghi kết quả per-run (summary, stations, assignments, validation, config)
└── run.py              # Entry point: một lệnh chạy toàn bộ pipeline

tests/
├── conftest.py         # Fixture chung: tham số trạm, ứng viên nhỏ, polygon H/C
├── test_polygons.py    # Lọc ứng viên, điểm trên biên, manhattan_reachable
├── test_models.py      # Đối chiếu brute-force: set/max covering, p-median, evaluate
├── test_model_main.py  # Brute-force oracle trên 40 instances ngẫu nhiên + edge cases
└── test_pipeline.py    # End-to-end pipeline, precheck, overwrite guard, alpha validation

data/
├── boundary.geojson, candidates.csv, demand.csv   # dữ liệu đầu vào
├── station_parameters.csv       # công suất / chi phí / bật-tắt theo loại trạm
├── candidates_feasible.csv      # ứng viên sau lọc polygon (kèm cờ feasible)
├── residential_areas.geojson, forbidden_areas.geojson
└── polygon_report*.json         # báo cáo kiểm tra theo cấu hình

results/
├── results_walking_all.csv      # kết quả baseline (khoảng cách đi bộ) — không ghi đè
├── results.csv                  # kết quả 3 mô hình đối chứng
└── <run_id>/                    # kết quả mô hình chính, mỗi lần chạy một thư mục riêng
    ├── summary.json, config.json, precheck.json
    ├── selected_stations.csv, assigned_demand.csv
    └── validation.json

Dockerfile, .dockerignore        # image chạy + loại trừ data/cache khi build
pytest.ini                       # cấu hình pytest (testpaths, pythonpath)
requirements.txt                 # dependencies chạy (pin phiên bản)
requirements-dev.txt             # pytest, pytest-cov
.github/workflows/ci.yml         # CI: docker build + pytest trong container
```

## Cách chạy

```bash
.venv/bin/python src/fetch_data.py                # 1. tải dữ liệu (cần mạng + file WorldPop tif)
.venv/bin/python src/polygons.py --mode B         # 2. lọc ứng viên theo polygon (A hoặc B)
.venv/bin/python src/distances.py                 # 3. ma trận khoảng cách đi bộ (baseline)
.venv/bin/python src/models.py                    # 4. chạy 3 mô hình đối chứng
.venv/bin/python -m src.distances_l1              # 5. ma trận Manhattan (EPSG:32648) cho mô hình chính
.venv/bin/python -m src.run --radius 500 --alpha 0.95 --cap-scale 1.0   # 6. mô hình chính: MILP hai pha
```

Bước 6 ghi kết quả vào `results/<run_id>/` (mỗi lần chạy một thư mục, không ghi đè).
Exit code: 0 = giải được + kiểm định PASS; 1 = bất khả thi (precheck hoặc solver);
2 = lỗi đầu vào; 3 = lỗi khác. Flag chính: `--radius`, `--alpha`, `--cap-scale`,
`--time-limit`, `--data-dir`, `--results-dir`.

### Chạy test

```bash
.venv/bin/python -m pytest -q                     # offline, không cần mạng
```

### Chạy bằng Docker

```bash
docker build -t service-science .
docker run --rm service-science                   # mặc định: chạy pytest
# chạy pipeline với dữ liệu thật (bind-mount data/ và results/):
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
  service-science python src/polygons.py --mode B
```

Lưu ý: image không chứa `data/` và `results/` (bị loại qua `.dockerignore`);
khi chạy pipeline thật phải bind-mount hai thư mục này như trên.
</sleev-id-m0035>