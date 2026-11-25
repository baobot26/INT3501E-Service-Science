# Product Truth — Quy hoạch vị trí đặt tủ đồ thông minh (Quận Cầu Giấy)

## Sản phẩm là gì
Công cụ web Streamlit (`app.py`, triển khai trên Streamlit Community Cloud) hỗ trợ lựa chọn vị trí đặt trạm tủ đồ thông minh (smart locker) trên địa bàn quận Cầu Giấy cũ gồm 3 phường (Cầu Giấy, Nghĩa Đô, Yên Hòa; ~12,46 km²), thuộc môn **INT3501E — Khoa học dịch vụ**.

## Người dùng mục tiêu và bối cảnh sử dụng
- **Nhóm sinh viên trình bày trên máy chiếu trong lớp:** phòng học sáng đèn, máy chiếu làm giảm tương phản và nhạt màu — giao diện cần nền sáng, chữ mực đậm, tương phản cao, kết quả đọc nhanh từ xa.
- **Giảng viên / người chấm bài mở link triển khai trên laptop:** cần hiểu ngay phạm vi địa bàn và tập ứng viên ngay khi mở trang, có thể thay đổi tham số kịch bản, chạy tối ưu và đối chiếu trực tiếp với báo cáo / CLI.

## Luồng công việc chính
1. Mở ứng dụng → quan sát ngay ranh giới 3 phường quận Cầu Giấy cũ, phân bố điểm nhu cầu và toàn bộ điểm ứng viên khả thi theo loại.
2. Điều chỉnh tham số kịch bản ở thanh bên (`R`, `α`, mục tiêu, `p`, hệ số nhu cầu / sức chứa, bật-tắt và công suất / chi phí từng loại trạm, giới hạn thời gian và MIP gap).
3. Bấm **Chạy kịch bản** → giải mô hình MILP hai pha (mất vài giây đến vài chục giây; kịch bản mặc định hoàn tất trong ~3–4 giây).
4. Đọc ngay kết quả tổng hợp (số trạm, tổng chi phí, độ phủ, trạng thái tối ưu, thời gian giải) và quan sát nghiệm trên bản đồ (trạm được chọn, vùng phục vụ hình thoi theo khoảng cách Manhattan L1, điểm nhu cầu được phục vụ / chưa phục vụ).
5. Tra cứu bảng chi tiết (**Trạm được chọn**, **Phân bổ nhu cầu**) và tải các tệp dữ liệu máy (`selected_stations.csv`, `assigned_demand.csv`, `summary.json`).

## Nguyên tắc nội dung và trung thực dữ liệu
- Toàn bộ giao diện dùng **tiếng Việt**, định dạng số hiển thị theo quy ước Việt Nam (dấu phẩy thập phân, dấu chấm phân cách hàng nghìn); các tệp CSV / JSON tải về giữ định dạng máy tiêu chuẩn (dấu chấm thập phân).
- Không tự bổ sung tuyên bố, chỉ số hay giả định nằm ngoài kết quả thực tế mà bộ giải MILP và bước kiểm định độc lập (`src/evaluate.py`) tạo ra.
- Cùng một cấu hình kịch bản trên giao diện Streamlit và CLI (`python -m src.run`) dùng chung `SolveConfig`, hàm dựng bài toán và hàm giải nên luôn cho kết quả khớp nhau.
