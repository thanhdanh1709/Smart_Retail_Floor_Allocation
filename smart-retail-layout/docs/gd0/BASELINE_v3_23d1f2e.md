# GĐ0A — Baseline v3 tại HEAD `23d1f2e`

Ngày ghi nhận: 2026-10-11<br>
Repository: `D:\Smart_Retail_Floor_Allocation`<br>
Nhánh: `main`<br>
Commit baseline: `23d1f2e002fdb38f3ad6c313fb9764a77a2281ca` (2026-10-05)

## Git và phạm vi baseline

- Tracked tree ban đầu sạch; nhánh `main` khớp `origin/main`.
- Các mục untracked đã có trước GĐ0 và được giữ nguyên: `.playwright-mcp/`, `AGENTS.md`, `de-cuong/`, `smart-retail-layout/docs/CONG_VIEC_v4.md`.
- Git cần `-c safe.directory=D:/Smart_Retail_Floor_Allocation` cho lệnh trong sandbox. Không thay Git config toàn máy.
- Baseline này là snapshot mã nguồn tại HEAD v3 nêu trên; các kết quả E1–E7 bên dưới là artifact cũ được kiểm kê, không phải kết quả vừa chạy.

## Bản sao lưu

Do workspace chỉ cho phép ghi trong repository, backup được đặt trong thư mục riêng, mới tạo và không đè file nào:

| Artifact | SHA-256 | Kiểm tra |
|---|---|---|
| [`repo_23d1f2e.bundle`](../../backups/GD0_2026-10-11_23d1f2e/repo_23d1f2e.bundle) | `a505fc02be98fd3a540b41479124408efa7d5f289bbef33662ad2544cec79c86` | `git bundle verify` xác nhận đầy đủ history và ref `main` |
| [`tracked_HEAD_23d1f2e.zip`](../../backups/GD0_2026-10-11_23d1f2e/tracked_HEAD_23d1f2e.zip) | `a3bfa2c2a11e64b38dddb6700d8c38074560d5a9eb91d561084c492630b8e39a2` | ZIP có 251 thành viên; `ZipFile.testzip()` không phát hiện lỗi |

ZIP/bundle chỉ chứa nội dung tracked của HEAD. Các file untracked và dữ liệu cục bộ không nằm trong snapshot đó; chúng vẫn ở nguyên vị trí.

## Artifact kết quả cũ

- Trước khi thay đổi code, đã tạo SHA-256, kích thước và thời gian sửa của **134 file** bên dưới `results/`, loại trừ thư mục GĐ0 mới. Danh sách nằm ở [`preexisting_results_sha256.csv`](../../results/GD0/baseline_23d1f2e_20261011/preexisting_results_sha256.csv).
- Các thư mục/báo cáo E1–E7 và các bản `_v1`, `_v2` là artifact có sẵn. Không gọi `experiments.run_all`, không chạy lại E1–E7, không gọi `params.build_all()`. Một lần `src.pipeline.run_pipeline()` smoke trên instance nhỏ của snapshot v3 được lưu riêng dưới `results/GD0/`.
- Hash inventory sẽ được đối chiếu lần cuối sau các thay đổi GĐ0 để xác nhận artifact cũ không đổi.

## Môi trường và kiểm tra baseline

- Python tìm thấy trên PATH: `3.12.10`.
- Python global ban đầu có `numpy 2.5.3` nhưng thiếu các package cốt lõi của dự án. Để không thay đổi Python global, đã tạo môi trường cô lập tại `%TEMP%\smart-retail-gd0-venv-20261011` và cài `smart-retail-layout/requirements.txt`.
- Trong môi trường cô lập, `python -m pip check` trả `No broken requirements found`.
- Kiểm thử baseline chạy trên bản giải nén từ ZIP tracked HEAD, không phải working tree đã sửa:

  ```powershell
  python -m pytest -q -p no:cacheprovider
  ```

  Kết quả: **18 passed in 11.73s**.

- `compileall` cho `src/floorplan.py`, `src/instance.py`, `tests/test_core.py` chạy thành công sau thay đổi. Suite đầy đủ trên working tree GĐ0 đạt **23 passed in 7.84s**; 5 kiểm tra mới riêng đạt **5 passed in 1.40s**. Các lệnh/chi tiết GĐ0 ghi trong [`GD0_SUMMARY.md`](GD0_SUMMARY.md).
- Pytest được gọi với `-p no:cacheprovider` vì checkout không cho ghi vào `.pytest_cache` cũ; không cần xóa hoặc sửa cache đó.

## Trạng thái nghiệm thu 0A

**Hoàn thành.** Bundle/archive đã kiểm chứng, baseline test và pipeline smoke chạy trên snapshot HEAD gốc, và full SHA-256 của 134 artifact trong `results/` khớp trước/sau GĐ0.

### Pipeline smoke trên snapshot gốc

- Artifact JSON: [`baseline_pipeline_smoke.json`](../../results/GD0/pipeline_smoke_23d1f2e_20261011/baseline_pipeline_smoke.json); SHA-256 `84463a7ae960a67d52250ec1c2b87efee09909b7d6741119f127ed10a37e9e69`.
- Nguồn chạy là ZIP tracked HEAD `23d1f2e`, giải nén vào thư mục tạm; không chạy từ working tree đã sửa và không đọc/ghi payoff cache của checkout hiện tại.
- Cấu hình smoke: `grid/small`, 20 nhóm, 25 slot, group level; objective `model`; seed `20261011`; 100 khách khi hiệu chỉnh/sàng lọc, một lần sàng lọc, 2 lần đánh giá cuối; không refinement; một hồ sơ `Cân bằng`.
- Kết quả: hoàn tất trong `2.387 s`; `lambda=0.3430819511`; Pareto có 43 phương án; cả ba phương án xuất ra đều feasible. KPI mô phỏng sơ đồ hiện trạng: quãng đường `28.06 m`, thời gian `2.7601 phút`, impulse revenue proxy `0.643`, `1.49` impulse items/khách, basket value `2.7535`, congestion cells `7`.
- Đây là smoke kiểm tra đường chạy pipeline v3 với dữ liệu/proxy sẵn có; không phải benchmark hiệu năng hay kết luận thực nghiệm cho cửa hàng thật.
