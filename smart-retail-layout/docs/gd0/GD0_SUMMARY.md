# Kết quả thực hiện GĐ0

Ngày: 2026-10-11<br>
Baseline mã nguồn: `main` tại `23d1f2e002fdb38f3ad6c313fb9764a77a2281ca`<br>
Trạng thái: **GĐ0 đã thực hiện; readiness gaps được ghi rõ để chặn việc giả định dữ liệu/nguồn chưa có.**

## 0A — Baseline và backup

- Tạo và kiểm chứng Git bundle cùng ZIP tracked HEAD trong [`backups/GD0_2026-10-11_23d1f2e/`](../../backups/GD0_2026-10-11_23d1f2e/). Backup ở workspace vì quyền ghi hiện tại giới hạn ở repository.
- Baseline test trên ZIP giải nén của HEAD gốc: **18 passed**. Suite hiện tại sau thay đổi: **23 passed**. Kiểm tra riêng API online-staging: **5 passed**.
- Full SHA-256 của 134 file kết quả cũ dưới `results/` giống hệt trước và sau; full SHA-256 của 46 file dưới `data/` cũng không đổi. Snapshot hash nằm trong [`results/GD0/baseline_23d1f2e_20261011/`](../../results/GD0/baseline_23d1f2e_20261011/).
- E1–E7 và các artifact `_v1`/`_v2` là file có sẵn; lượt này không chạy lại experiment và không coi các file đó là kết quả mới.

## 0B — Dữ liệu và giả định

- Lập [`DATA_READINESS.md`](DATA_READINESS.md), [`DATA_DICTIONARY.csv`](DATA_DICTIONARY.csv), và [`DATA_FILE_INVENTORY_SHA256.csv`](DATA_FILE_INVENTORY_SHA256.csv).
- Kiểm tra lại delimiter, số cột và UTF-8 của các CSV; nhãn tiếng Việt trong trường `dwell_median_s` đã được xác minh sau review.
- Không có raw Instacart trong checkout; processed data còn lại có manifest 16-hex output prefixes nhưng không có raw identity hash. Mọi metric-derived value được phân biệt với heuristic/proxy.
- `value_v` không phải price, revenue hay margin; `impulse_p` là `1 - reorder_rate`, không phải nhãn impulse được quan sát; `dwell_median_s`, cold requirement, `n_slots=1`, capacity/kích thước SKU đều được ghi đúng trạng thái.
- Không có giá/biên lợi nhuận, SKU dimension, capacity kệ, aisle width hoặc trajectory khách thực tế. Vì vậy bằng chứng v4 dạng revenue/price hoặc customer trajectory chưa sẵn sàng.

## 0C — Nguồn, quyền sử dụng và `d_0`/`d_0k`

- Lập [`REFERENCES_LEDGER.csv`](REFERENCES_LEDGER.csv) cho references ở mục 11, HRN4Customer, Instacart và Dunnhumby.
- P11-LEE được xác nhận trong proceedings AAMAS 2026 (DOI `10.65109/AJIK9102`); giấy phép paper CC BY 4.0 không xác minh hay cấp quyền dataset HRN4Customer. Flamand trong kế hoạch thiếu title cụ thể nên giữ hai citation ứng viên; Botsali–Peters chưa tìm thấy proceedings record sơ cấp; HRN4Customer chưa xác minh được landing page/owner/license. Không dùng HRN4Customer để calibration khi chưa giải quyết.
- Tạo [`FLOORPLAN_PICKUP_POINTS.csv`](FLOORPLAN_PICKUP_POINTS.csv); cả 9 layout mẫu và `case_minimart` đều `UNSPECIFIED`. Không tự gán cửa vào/quầy thu ngân làm staging point.
- `FloorPlan` nhận `pickup_cell` tùy chọn. Nếu truyền điểm, `d_0k` được tính theo shortest path trên graph hiện hữu từ d₀ tới `slot.access`; điểm phải nằm trong grid, trên ô walkable A/E/C, và nối được tới mọi slot. Không truyền thì `d_0k=None`; gọi `require_pickup()` sẽ báo lỗi rõ.
- Contract cho GĐ1/GĐ2: `fp.d_0k` theo thứ tự toàn bộ slot của `FloorPlan`; để lấy vector theo slot của một `Instance`, dùng `inst.fp.require_pickup()[inst.slot_idx]`.
- Điểm tập kết không được thêm vào `self.points`, do đó chỉ số điểm của v3 và `D`, `d_in`, `d_out`, `e` giữ nguyên. Objective v3 trong `Instance.lin1` không đổi; objective picking v4 chưa được đưa vào phase này.

## File code thay đổi

- `src/floorplan.py`: optional staging cell, validation, shortest-path vector `d_0k`, helper `require_pickup()`.
- `src/instance.py`: truyền optional `pickup_cell` từ `make_instance()` sang `FloorPlan`.
- `tests/test_core.py`: kiểm tra vector khoảng cách, lỗi khi chưa cấu hình/điểm sai hoặc disconnected, tính không đổi của ma trận v3 và forwarding từ `make_instance()`.

## Lệnh và kết quả xác nhận

Chạy trong môi trường `%TEMP%\smart-retail-gd0-venv-20261011`, với working directory `smart-retail-layout`:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_core.py -k online_staging
# 5 passed, 18 deselected

python -m pytest -q -p no:cacheprovider
# 23 passed

python -m pip check
# No broken requirements found
```

`compileall` cho hai module và test file mới cũng thành công. Không xóa hoặc ghi đè results cũ, raw/processed data, `KE_HOACH_v4.md`, backlog `CONG_VIEC_v4.md`, hay các untracked đã có trước GĐ0.

## Readiness còn lại trước các phase downstream

1. Raw Instacart/đúng version và điều khoản tải cụ thể cần được đưa vào quy trình có lineage để có thể xác minh từ nguồn đến output.
2. Cần xác minh citation/proceedings của Botsali–Peters và chốt paper Flamand nào được định dùng; kiểm tra rights dataset HRN4Customer nếu có URL chính thức.
3. Cần dữ liệu được phép dùng cho giá/margin và đo đạc vật lý, cùng tọa độ d₀ thật của cửa hàng trước khi cấu hình `pickup_cell` cho một store.
## Baseline pipeline smoke bổ sung sau review

Đã chạy `src.pipeline.run_pipeline()` từ snapshot tracked HEAD `23d1f2e` đã giải nén riêng trong thư mục tạm. Instance `grid/small` có 20 nhóm, 25 slot; seed `20261011`; hiệu chỉnh/sàng lọc dùng 100 khách; objective `model`; không refinement; đánh giá cuối 2 lần. Lượt chạy hoàn tất trong 2.387 giây, có 43 phương án Pareto và mọi phương án trả ra đều feasible. Chi tiết cấu hình, môi trường và KPI nằm trong [`baseline_pipeline_smoke.json`](../../results/GD0/pipeline_smoke_23d1f2e_20261011/baseline_pipeline_smoke.json). Đây là smoke test, không phải benchmark và không phải kết quả cửa hàng thật.

Các `p_value` không xác định được ghi là JSON `null`; artifact đã được kiểm tra bằng parser strict JSON. Quy tắc cho schema GĐ1B: chuẩn hóa mọi NaN/Infinity thành `null` trước khi xuất JSON.
