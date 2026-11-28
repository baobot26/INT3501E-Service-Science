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

| Phương án xác định H | Ứng viên hợp lệ | Cận trên độ phủ | Qua precheck? |
|---|---|---|---|
| A: `landuse=residential` | 129 / 481 | 92,73% | ❌ (chỉ đạt khi R = 800 m) |
| B: khối nhà ở (building footprints) | 184 / 481 | 97,94% | ✅ |

Lưu ý: precheck chỉ kiểm cận trên tổng độ phủ và tổng sức chứa; với quy tắc gán
trọn mỗi điểm cầu cho trạm gần nhất, bài toán MILP ở R = 500 m, α = 0,95 vô nghiệm
tại `cap_scale = 1,0` (và `1,5`) do quá tải cục bộ tại các cụm dân cư dày, và bắt
đầu khả thi từ `cap_scale ≈ 2,0`.

→ Kịch bản chính của MVP: **phương án B, R = 500 m, α = 0,95, `cap_scale = 3,0`** (khớp mặc định sức chứa của app).

## Tech stack

- **Python 3.12** — môi trường ảo `.venv` (tạo bằng `uv` hoặc `python3 -m venv`)
- **osmnx** — tải ranh giới, điểm ứng viên và polygon từ OpenStreetMap/Overpass
- **GeoPandas / Shapely / pyproj** — làm sạch polygon, chuyển hệ chiếu EPSG:4326 → EPSG:32648
- **pandas / numpy** — bảng dữ liệu và ma trận khoảng cách
- **scipy.optimize.milp** (solver HiGHS) — giải bài toán MILP
- **networkx** — khoảng cách đi bộ trên đồ thị đường (baseline)
- **folium** — bản đồ HTML để kiểm tra trực quan
- **streamlit** — giao diện web điều chỉnh tham số kịch bản
- **pytest** — bộ test offline (không gọi mạng, dùng fixture trong bộ nhớ)
- **Docker** — image `python:3.12-slim` đóng gói toàn bộ môi trường chạy

Cài đặt:

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
# hoặc: pip install -r requirements.txt -r requirements-dev.txt
```

## Cấu trúc mã nguồn

```
app.py                # Giao diện Streamlit: điều chỉnh tham số kịch bản, xem kết quả
src/
├── fetch_data.py       # Tải dữ liệu đầu vào: ranh giới, ứng viên OSM, demand WorldPop, mạng đi bộ
├── polygons.py         # Kiểm tra polygon khu dân cư (H) / khu cấm (C), lọc ứng viên,
│                       #   tính 2 cận trên (độ phủ, năng lực) và xuất bản đồ kiểm tra
├── distances.py        # Ma trận khoảng cách đi bộ demand→ứng viên (baseline, ghi dist.npy)
├── models.py           # 3 mô hình đối chứng: set covering, max covering, p-median
├── preprocessing.py    # Đọc/validate đầu vào, xây ProblemData thống nhất cho pipeline
├── distances_l1.py     # Ma trận khoảng cách Manhattan (EPSG:32648) + manifest
├── config.py           # SolveConfig: tham số kịch bản dùng chung cho CLI và Streamlit
├── model_main.py       # Mô hình chính: MILP hai pha (min trạm → min chi phí) + chế độ cố định p
├── evaluate.py         # Kiểm định độc lập: gán lại nearest, kiểm coverage/capacity/objective
├── export.py           # Ghi kết quả per-run (stations, assignments, validation, precheck)
├── run.py              # Entry point: một lệnh chạy toàn bộ pipeline
└── sweep.py            # Quét độ nhạy theo lưới R × α, xuất CSV tổng hợp

tests/
├── conftest.py         # Fixture chung: tham số trạm, ứng viên nhỏ, polygon H/C
├── test_polygons.py    # Lọc ứng viên, điểm trên biên, manhattan_reachable
├── test_models.py      # Đối chiếu brute-force: set/max covering, p-median, evaluate
├── test_model_main.py  # Brute-force oracle trên 40 instances ngẫu nhiên + edge cases
├── test_pipeline.py    # End-to-end pipeline, precheck, overwrite guard, alpha validation
├── test_sweep.py       # Kiểm thử quét lưới độ nhạy R × α
├── test_config.py      # Validate tham số của SolveConfig
└── test_dynamic_scenarios.py  # Bật/tắt loại trạm, hệ số, cố định p, CLI ↔ app khớp nhau

data/
├── boundary.geojson, candidates.csv, demand.csv   # dữ liệu đầu vào
├── station_parameters.csv       # công suất / chi phí / bật-tắt theo loại trạm
├── candidates_feasible.csv      # ứng viên sau lọc polygon (kèm cờ feasible)
├── dist_l1.npy, dist_l1_manifest.json  # ma trận Manhattan + manifest (có trong git)
├── residential_areas.geojson, forbidden_areas.geojson
└── polygon_report*.json         # báo cáo kiểm tra theo cấu hình

results/
├── results_walking_all.csv      # kết quả baseline (khoảng cách đi bộ) — không ghi đè
├── results.csv                  # kết quả 3 mô hình đối chứng (khoảng cách đi bộ)
├── results_l1.csv               # kết quả 3 mô hình đối chứng trên ma trận L1 + ứng viên feasible
├── sweep_<ts>.csv               # bảng phân tích độ nhạy R × α
└── <run_id>/                    # kết quả mô hình chính, mỗi lần chạy một thư mục riêng
    ├── summary.json, config.json, precheck.json
    ├── selected_stations.csv, assigned_demand.csv
    └── validation.json

Dockerfile, .dockerignore        # image test/app, loại trừ cache và kết quả chạy
pytest.ini                       # cấu hình pytest (testpaths, pythonpath)
requirements.txt                 # dependencies chạy (pin phiên bản)
requirements-dev.txt             # pytest, pytest-cov và rasterio cho fetch_data.py
.github/workflows/ci.yml         # CI: docker build + pytest trong container
```

## Cách chạy

```bash
.venv/bin/python src/fetch_data.py                # 1. tải dữ liệu (cần mạng + file WorldPop tif)
.venv/bin/python src/polygons.py --mode B         # 2. lọc ứng viên theo polygon (A hoặc B)
.venv/bin/python src/distances.py                 # 3. ma trận khoảng cách đi bộ (baseline)
.venv/bin/python src/models.py                    # 4a. chạy 3 mô hình đối chứng (đi bộ → results.csv)
.venv/bin/python src/models.py --metric l1        # 4b. chạy 3 mô hình đối chứng (L1 + feasible → results_l1.csv)
.venv/bin/python -m src.distances_l1              # 5. ma trận Manhattan (EPSG:32648) cho mô hình chính
.venv/bin/python -m src.run --radius 500 --alpha 0.95 --cap-scale 3.0   # 6. mô hình chính: MILP hai pha
.venv/bin/python -m src.sweep --radii 300 400 500 600 700 800 --alphas 0.6 0.7 0.8 0.9 0.95 --cap-scale 3.0  # 7. phân tích độ nhạy R × α
```

Bước 6 ghi kết quả vào `results/<run_id>/` (mỗi lần chạy một thư mục, không ghi đè).
Exit code: 0 = giải được + kiểm định PASS; 1 = bất khả thi (precheck hoặc solver);
2 = lỗi đầu vào; 3 = lỗi khác.

Flag chính:

| Flag | Ý nghĩa |
|---|---|
| `--radius` | bán kính phục vụ R (m) |
| `--alpha` | tỷ lệ phủ tối thiểu, 0…1 |
| `--cap-scale` | hệ số nhân sức chứa |
| `--demand-scale` | hệ số nhân nhu cầu |
| `--objective` | `min_stations_then_cost` (mặc định) hoặc `fixed_station_count` |
| `--station-count` | số trạm p, bắt buộc khi `--objective=fixed_station_count` |
| `--type-params` | file JSON ghi đè tham số theo loại trạm |
| `--time-limit` | tổng ngân sách giây cho cả hai pha |
| `--mip-rel-gap` | khoảng cách tương đối MIP |
| `--data-dir`, `--results-dir`, `--mode` | đường dẫn, nhãn chế độ polygon |

File `--type-params` có dạng
`{"shop=convenience": {"enabled": 0, "capacity": 15000, "fixed_cost": 650000000}}` —
ghi đè trong bộ nhớ, **không** sửa file CSV gốc.

### Giao diện Streamlit

```bash
.venv/bin/python -m streamlit run app.py
```

Người dùng tự điều chỉnh trên giao diện: R, α, mục tiêu (ít trạm nhất / cố định p),
hệ số nhu cầu, hệ số sức chứa, bật-tắt và sức chứa/chi phí của từng loại trạm,
giới hạn thời gian và MIP gap. Nút **Chạy kịch bản** mới kích hoạt solver (widget
trong `st.form` không chạy lại script tới khi bấm). Giao diện dùng đúng hàm giải
và tham số như CLI (`src/run.py`), nên hai đường cho kết quả giống nhau.

Deploy lên Streamlit Community Cloud: repo đã có `app.py` ở gốc, `requirements.txt`
(kể cả `streamlit`) và `.streamlit/config.toml`. Trong Cloud, chọn repo, nhánh và
file entrypoint `app.py`; Cloud tự cài `requirements.txt` và chạy `streamlit run app.py`.
`rasterio` chỉ nằm trong `requirements-dev.txt` vì không cần cho app.

Đặc tả giao diện, bảng màu sáng/tối, độ tương phản và quy tắc thành phần xem tại [DESIGN.md](DESIGN.md).

Lưu ý khi trình chiếu: ứng dụng hỗ trợ **System / Light / Dark** trong menu `⋮` (góc trên bên phải); nếu laptop đang đặt dark mode của hệ điều hành, hãy chọn **Light** trong menu `⋮` rồi tải lại trang trước khi chiếu trên máy chiếu phòng sáng (bản đồ cập nhật theo theme khi tải lại phiên).

### Chạy test

```bash
.venv/bin/python -m pytest -q                     # offline, không cần mạng (bỏ qua e2e)
```

### Kiểm thử giao diện (e2e)

Bộ kiểm thử Playwright (`tests/e2e/test_ui.py`, marker `e2e`) tách khỏi `pytest` mặc định và chạy trong một job CI riêng. Để chạy trên máy bằng Docker (PowerShell trên Windows):

```powershell
docker run --rm -v "${PWD}:/work" -w /work mcr.microsoft.com/playwright/python:v1.65.2-noble sh -c 'pip install -r requirements.txt -r requirements-dev.txt -r requirements-e2e.txt && pytest -m e2e tests/e2e -o addopts='
```

### Chạy bằng Docker

```bash
docker build -t service-science .
docker run --rm service-science                   # mặc định: chạy pytest

# mở giao diện Streamlit (truy cập http://localhost:8501):
docker run --rm -p 8501:8501 service-science streamlit run app.py --server.address=0.0.0.0

# chạy pipeline với dữ liệu thật (bind-mount data/ và results/):
# Bash / Linux / macOS:
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
  service-science python src/polygons.py --mode B

# PowerShell (Windows):
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/results:/app/results" `
  service-science python src/polygons.py --mode B
```

Lưu ý: image chứa dữ liệu đầu vào tối thiểu để chạy Streamlit smoke test;
`results/` vẫn không được đóng gói và nên bind-mount khi chạy pipeline thật.

## Chạy trên vùng tuỳ ý

Thay vì truy vấn Overpass cho từng vùng, có thể tải trước kho dữ liệu cả nước
vào `store/vn/` một lần rồi cắt dữ liệu theo bất kỳ tệp ranh giới (GeoJSON,
GeoPackage, Shapefile) nào hoàn toàn offline.

### 1. Tải kho dữ liệu Việt Nam (một lần)

- **OpenStreetMap (Geofabrik):** tạo thư mục `store/vn/` và tải tệp PBF cả nước:

  ```powershell
  # PowerShell (Windows)
  New-Item -ItemType Directory -Force -Path store/vn | Out-Null
  curl.exe -L -o store/vn/vietnam-latest.osm.pbf https://download.geofabrik.de/asia/vietnam-latest.osm.pbf
  ```

  ```bash
  # Bash (Linux / macOS)
  mkdir -p store/vn
  curl -L -o store/vn/vietnam-latest.osm.pbf https://download.geofabrik.de/asia/vietnam-latest.osm.pbf
  ```

- **Dân số WorldPop:** tải raster dân số Việt Nam độ phân giải 100 m (cùng sản
  phẩm với tệp `vnm_pop_100m.tif` nhóm đã dùng) từ
  [WorldPop Hub](https://hub.worldpop.org/) và lưu tại `store/vn/population.tif`.

### 2. Cắt vùng, tối ưu, xuất bản đồ và mở giao diện (Docker)

Ví dụ với ranh giới `data/boundary.geojson` (lưu kết quả chuẩn bị vào `data/areas/caugiay/`):

```powershell
# PowerShell (Windows)
# 1) Cắt dữ liệu theo ranh giới (offline, không gọi mạng)
docker run --rm -v "${PWD}/store:/app/store:ro" -v "${PWD}/data:/app/data" `
  service-science python -m src.prepare_area --boundary data/boundary.geojson --name caugiay --mode B

# 2) Chạy mô hình tối ưu hai pha
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/results:/app/results" `
  service-science python -m src.run --data-dir data/areas/caugiay --radius 500 --alpha 0.95 --cap-scale 3.0

# 3) Xuất bản đồ HTML cho lần chạy mới nhất với đúng ranh giới vùng
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/results:/app/results" `
  service-science python -m src.make_map --boundary data/areas/caugiay/boundary.geojson

# 4) Mở giao diện Streamlit trỏ vào vùng đã chuẩn bị (http://localhost:8501)
docker run --rm -p 8501:8501 -v "${PWD}/data:/app/data" -e APP_DATA_DIR=/app/data/areas/caugiay `
  service-science streamlit run app.py --server.address=0.0.0.0
```

Trên Bash (Linux / macOS), thay `${PWD}` bằng `$PWD` và ký tự xuống dòng `` ` `` bằng `\`.

### Giới hạn hiện tại

- Chỉ hỗ trợ vùng thuộc đất liền Việt Nam (tâm ranh giới trong khoảng 102–110°E,
  8–24°N; tính khoảng cách Manhattan trên hệ chiếu UTM `EPSG:32648`).
- Diện tích ranh giới mặc định ≤ 50 km² (điều chỉnh qua `--max-area-km2`); ma trận
  khoảng cách L1 từ chối nếu tích số điểm cầu × ứng viên vượt 25.000.000 ô (~100 MB).
- Dữ liệu OSM phản ánh thời điểm của tệp `vietnam-latest.osm.pbf` trong `store/vn/`.
