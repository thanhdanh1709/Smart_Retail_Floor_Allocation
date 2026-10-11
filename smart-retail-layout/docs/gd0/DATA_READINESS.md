# GĐ0B — Kiểm kê dữ liệu và mức sẵn sàng

Ngày kiểm kê: 2026-10-11. Đây là kiểm kê read-only; không tái sinh hay sửa dữ liệu đã xử lý.

## Nguồn và lineage

| Nguồn/đầu ra | Trạng thái hiện tại | Ghi chú |
|---|---|---|
| `data/raw/instacart/` | **MISSING** | Không có `products.csv`, `aisles.csv`, `order_products__prior.csv`; không thể tái lập input gốc trong checkout hiện tại. |
| `data/processed/group/` | **PRESENT, lineage incomplete** | 40 đơn vị nhóm hàng; manifest ghi `n_orders=3,214,236`, tên/kích thước nguồn, hash output 16 ký tự. |
| `data/processed/aisle/` | **PRESENT, lineage incomplete** | 132 aisle; cùng giới hạn manifest như mức group. |
| `data/mappings/groups_vn.csv` | **PRESENT** | 40 nhóm và mapping aisle; trường `value_v`, `needs_cold`, `dwell_level` là thiết lập/giả định dự án, không phải giá hoặc quan sát vật lý. |
| `data/floorplans/` | **PRESENT** | 9 lưới mẫu, 9 CSV slots, `case_minimart.txt`; lưới dùng ô 1 m theo mô hình, chưa khai báo khu tập kết online. |
| `data/qaplib/` | **PRESENT** | Bộ instance QAP phục vụ benchmark thuật toán, không phải dữ liệu cửa hàng. |

Full SHA-256 của 46 file đang có trong `data/` nằm ở [`DATA_FILE_INVENTORY_SHA256.csv`](DATA_FILE_INVENTORY_SHA256.csv). Hash của 14 output trong hai processed manifest đều khớp với prefix 16 ký tự ghi trong manifest. Đây chỉ xác nhận các file output hiện tại khớp prefix; không khôi phục được hash/raw identity của nguồn gốc.

Manifest group ghi `groups_vn.csv` nguồn 1,822 bytes, còn file mapping đang có là 1,863 bytes. Raw Instacart cũng vắng mặt, do đó kích thước lịch sử trong manifest không chứng minh chính xác phiên bản dữ liệu đầu vào đã dùng. Manifest cũ được giữ nguyên.

## Data dictionary và mô hình hóa hiện tại

Chi tiết từng trường ở [`DATA_DICTIONARY.csv`](DATA_DICTIONARY.csv). Phân loại chính:

- **Quan sát/derived từ giỏ hàng:** `f`, `w`, `lift`, `reorder_rate`, basket matrix; giá trị này được suy ra từ các order-product records mà manifest ghi nhận, nhưng raw files không còn để xác minh lại.
- **Proxy/assumption:** `impulse_p` hiện tính từ `1 - reorder_rate`; đó không phải xác suất impulse được quan sát. `value_v` là giá trị giả định theo nhóm, không có đơn vị giá/doanh thu/biên lợi nhuận. `dwell_median_s` 6/10/18 giây và `needs_cold` là tham số/nhãn mô hình.
- **Cấu trúc mặt bằng:** `n_slots=1` là giả định mỗi category hiện chiếm một đơn vị gán, không phải sức chứa kệ vật lý. `seg_len=2` là độ dài đoạn kệ của mô hình lưới, không đo chiều rộng kệ tại cửa hàng.
- Tất cả 9 layout mẫu và `case_minimart` đều được liệt kê tại [`FLOORPLAN_PICKUP_POINTS.csv`](FLOORPLAN_PICKUP_POINTS.csv) với `pickup_status=UNSPECIFIED`; không có staging point nào tự chọn.
- **Chưa có:** giá bán SKU, giá vốn/biên lợi nhuận, kích thước SKU, bề rộng lối/kệ, capacity thực của kệ, observation trajectory khách, vị trí xác nhận của `d_0`.
- `orders.csv` (giờ/ngày đặt hàng) và `departments.csv` có trong bộ dữ liệu online Instacart nhưng không được checkout giữ. `src/params.py` hiện không cần chúng để tính toán v3; chúng cần thiết nếu v4 muốn mô hình thời gian/department.

## Quyền dữ liệu và quyết định sử dụng

- Trang Kaggle ghi dữ liệu Instacart Market Basket Analysis chịu **Competition Rules**; không coi dataset là dữ liệu mở hoặc mặc định cho phép phát tán raw data. Cần lưu các quy tắc áp dụng cho bản tải cụ thể trước khi đưa data vào repo.
- Dunnhumby chính thức liệt kê **The Complete Journey** để tải cho classroom/academic research; trang điều khoản website giới hạn việc dùng tài liệu cho research/personal/non-commercial và giữ lại các quyền không cấp rõ. Điều khoản/giấy phép riêng cho tệp dataset cần được kiểm tra trong user guide trước khi dùng hoặc phát tán.
- Dunnhumby **Breakfast at the Frat** ghi rõ Base Price và Shelf Price nhưng chỉ cho 5 sản phẩm/4 nhóm. Đây là ứng viên để kiểm tra workflow giá, không đủ thay dữ liệu giá theo SKU/group của kế hoạch.
- Không tải dataset trong GĐ0. Không ghi raw data, API key, hoặc credentials vào repository.

## Tiêu chí 0B

Data dictionary và full checksums đã được lập. Tuy nhiên 0B chưa thể coi là hoàn tất theo nghĩa có raw lineage xác minh được: Instacart gốc không có trong checkout; giá/margin/kích thước/capacity và trajectory thực tế còn thiếu. Các trường mô hình giả định đã được ghi rõ để không dùng như quan sát.
