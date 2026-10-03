# Smart Retail Floor Allocation

Mã nguồn của đề án **"Tối ưu hóa không gian trưng bày và luồng di chuyển của khách hàng trong
cửa hàng bán lẻ số"**. Mã nguồn làm theo Phần B của đề cương: mô hình hóa mặt bằng, ước lượng
tham số từ dữ liệu giỏ hàng, mô hình quy hoạch nguyên song mục tiêu, GA/NSGA-II, mô phỏng tác
tử, các thí nghiệm E1–E7 và dashboard hỗ trợ ra quyết định.

Pipeline ra quyết định (`src/pipeline.py`, dashboard và E5–E7 dùng chung):

```
Data          giỏ hàng (Instacart / POS) → w, f, p, v ;  mặt bằng (lưới) → D, d_in, d_out
  ↓
Model         Z1 (quãng đường) + Z2 (giá trị) – HIỆU CHỈNH BẰNG MÔ PHỎNG trên sơ đồ hiện trạng:
              λ; e_k = tỷ lệ khách đi qua vùng tiếp xúc slot k; q_i = p_i(1 − f_i)(1 − e^(−λ t_i))
              → kiểm định: Spearman giữa Z và KPI mô phỏng
  ↓
Optimization  NSGA-II lai + GA hai đầu → tập Pareto (ILP / GA đã kiểm chứng ở E1–E4)
  ↓
Simulation    mô phỏng tác tử sàng lọc MỌI ứng viên (số ngẫu nhiên chung) → chọn theo hồ sơ
              (Tiện lợi / Cân bằng / Giá trị) → tinh chỉnh 2-swap đánh giá bằng mô phỏng
  ↓
Decision tool đánh giá cuối trên hạt giống độc lập (KTC 95%, Wilcoxon) → phương án khuyến nghị,
              danh sách kệ cần dời, xuất tệp (dashboard 5 thẻ ① … ⑤)
```

Lý do có bước hiệu chỉnh: với e_k hình học và p_i gốc, Z2 gần như không tương quan với doanh thu
ngẫu hứng trong mô phỏng (Spearman ≈ −0,1…0,3), nên phương án "max Z2" có thể bán kém hơn hiện trạng.
Z2 gốc bỏ qua việc khách đã có món đó trong danh sách (hệ số 1 − f_i), thời gian dừng (1 − e^(−λ t_i))
và việc lộ trình thực phụ thuộc sơ đồ. Kết quả trước khi hiệu chỉnh được lưu ở `results/_v1_model_goc/`.

## Cài đặt

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Dữ liệu

Bộ Instacart 2017 (khoảng 700 MB, không đưa lên Git) được tải bằng `kagglehub` (không cần
tài khoản) rồi chép vào `data/raw/instacart/`:

```bash
python -c "import kagglehub, shutil; p = kagglehub.dataset_download('psparks/instacart-market-basket-analysis'); shutil.copytree(p, 'data/raw/instacart', dirs_exist_ok=True)"
python -m experiments.build_data          # -> data/processed/{group,aisle}/, data/floorplans/
```

| Tệp | Nội dung |
|---|---|
| `data/mappings/groups_vn.csv` | Ánh xạ 132 aisle Instacart → 40 nhóm hàng kiểu cửa hàng Việt Nam; cột `needs_cold`, `value_v`, `dwell_level` là **giả định** (Instacart không có giá hay thời gian dừng) |
| `data/processed/<mức>/` | `w.npy, f.npy, lift.npy, p.npy, v.npy, categories.csv` (mẫu PL1), `baskets.npz` (200.000 đơn mẫu cho mô phỏng), `manifest.json` (mã băm để tái lập) |
| `data/floorplans/` | 9 mặt bằng chuẩn (`grid/racetrack/free` × `small/medium/large`) + `case_minimart.txt` (mặt bằng mẫu mục B2.1 dùng làm nghiên cứu tình huống) và `slots_*.csv` |
| `data/qaplib/` | 10 bộ QAPLIB có lời giải tối ưu đã công bố (E1) |

Định dạng lưới: `X` tường, `A` lối đi, `S` kệ, `R` kệ lạnh, `E` cửa vào, `C` thu ngân, 1 ô = 1 m.

## Chạy

```bash
pytest                                       # kiểm thử bắt buộc B8.3 (~15 s)
python -m experiments.run_all --profile quick    # chạy thử toàn bộ E1–E7 (~5 phút)
python -m experiments.run_all                    # đầy đủ theo đề cương: 30 lần chạy/cấu hình (vài giờ)
python -m experiments.e3_heuristics              # hoặc từng thí nghiệm riêng
streamlit run app/streamlit_app.py               # dashboard
```

Kết quả ghi vào `results/E1 … E7/` (CSV thô và bảng tổng hợp, `run_config.yaml` chứa cấu hình,
hạt giống, phiên bản thư viện) và `results/figures/`. Bảng payoff của mỗi bộ dữ liệu được tính
một lần và lưu ở `results/payoff_cache.json`, nên mọi lần chạy dùng cùng một phép chuẩn hóa.
Tham số thí nghiệm nằm trong `experiments/config.yaml`.

| Mã | Thí nghiệm | Script |
|---|---|---|
| E1 | Kiểm chứng GA trên QAPLIB | `e1_qaplib.py` |
| E2 | GA so với ILP (n = 8…20), dạng tuyến tính hóa (12) so với RLT | `e2_ga_vs_ilp.py` |
| E3 | GA vs SA vs tabu vs tham lam vs ngẫu nhiên vs hiện trạng; Wilcoxon, Friedman + Nemenyi | `e3_heuristics.py` |
| E4 | Tập Pareto: NSGA-II (thuần/lai) vs ε-ràng buộc vs quét α; hypervolume | `e4_pareto.py` |
| E5 | Toàn pipeline trên 9 + 1 mặt bằng: kiểm định mô hình (gốc vs hiệu chỉnh); 3 hồ sơ × {theo mô hình, chọn bằng mô phỏng, tinh chỉnh} so với hiện trạng, 30 lần lặp độc lập | `e5_simulation.py` |
| E6 | Độ nhạy của phương án khuyến nghị (λ, p, chiến lược đi, tốc độ); đường cong cải thiện theo R kèm KPI mô phỏng | `e6_sensitivity.py` |
| E7 | Bóc tách khâu Model (e_k đều / hình học / mô phỏng, p hay q, lặp e_k) và tìm kiếm cục bộ của GA | `e7_ablation.py` |

## Cấu trúc mã nguồn

| Mô-đun | Chức năng (mục đề cương) |
|---|---|
| `src/floorplan.py` | Lưới → slot → đồ thị lối đi → `D, d_in, d_out`, mức tiếp xúc hình học `e_k` (B2) |
| `src/layouts.py` | Sinh mặt bằng grid / racetrack / free, gán khu lạnh (B7.1) |
| `src/params.py` | Instacart → `w, f, lift, p, v`; nạp dữ liệu cửa hàng tải lên (B3) |
| `src/instance.py` | Bộ dữ liệu bài toán, hàm mục tiêu (2) (3) (11), ràng buộc, sơ đồ hiện trạng (B4) |
| `src/fastops.py` | Nhân numba: Δ(r, s) O(m) theo (13), tìm kiếm cục bộ 2-swap, hình phạt (B5.3) |
| `src/model_ilp.py` | ILP với PuLP + HiGHS/CBC; tuyến tính hóa (12) và RLT; payoff; ε-ràng buộc (B4.3–B4.5) |
| `src/ga.py` | GA: OX/PMX, đột biến, sửa lỗi, tinh hoa, tìm kiếm cục bộ, nhập cư chống hội tụ sớm (B5) |
| `src/moo.py` | NSGA-II (pymoo), biến thể lai, hypervolume, điểm gối, quét α (B5.5) |
| `src/baselines.py` | Ngẫu nhiên, tham lam, SA, tabu (Taillard) (B5.6) |
| `src/simulate.py` | Mô phỏng tác tử: lộ trình NN + 2-opt hoặc chữ S, dwell log-chuẩn, mua ngẫu hứng (14), KPI, bản đồ nhiệt, ùn tắc (B6) |
| `src/pipeline.py` | Pipeline Data → Model → Optimization → Simulation → Decision: hiệu chỉnh mô hình, kiểm định, tập ứng viên, sàng lọc/chọn/tinh chỉnh bằng mô phỏng, đánh giá cuối |
| `src/relocation.py` | Danh sách kệ cần dời theo thứ tự ưu tiên, kèm % cải thiện cộng dồn (B8.4) |
| `src/qaplib.py`, `src/viz.py`, `src/solvers.py` | QAPLIB, vẽ hình, bảng payoff |

## Giả định và giới hạn cần nêu khi bảo vệ

- **Đồng mua** lấy từ đơn online Instacart, phản ánh nhu cầu chứ không phản ánh đường đi
  trong cửa hàng (PL2 – câu 2). Quy trình vẫn dùng nguyên vẹn khi có dữ liệu POS: tải
  `categories.csv` + `baskets.csv` lên dashboard.
- **Giá trị `v_i`** và **mức dwell** là giả định có cơ sở (bảng `groups_vn.csv`), **`p_i` = 1 − tỷ lệ mua lại**
  (proxy (a) mục B3.4). Ảnh hưởng của các giả định này được kiểm tra trong E6.
- **λ** được hiệu chỉnh để trung bình mỗi lượt khách mua 1,5 món ngoài kế hoạch trên sơ đồ hiện trạng
  (giá trị giả định). Kết quả mô phỏng chỉ dùng để **so sánh tương đối** giữa các sơ đồ.
- **Sơ đồ hiện trạng** của các mặt bằng mẫu được dựng theo kinh nghiệm: nhóm hàng xếp theo
  thứ tự ngành hàng dọc đường rắn từ cửa vào, hàng lạnh ở khu lạnh phía sau.
- **Mức tiếp xúc `e_k`** phụ thuộc chính sơ đồ. Pipeline đo e_k trên luồng khách của sơ đồ hiện
  trạng; khi sơ đồ đổi nhiều, e_k lệch – vì vậy khâu Simulation đánh giá lại mọi ứng viên thay vì
  tin Z. E7 thử thêm lặp "tối ưu → mô phỏng → cập nhật e_k" với giảm chấn.
- **Z1** là tổng khoảng cách theo cặp đồng mua, không phải độ dài lộ trình nhiều điểm dừng. Ở cửa
  hàng nhỏ, quãng đường mô phỏng hầu như do hình dạng mặt bằng quyết định (các sơ đồ chênh ~3%)
  nên Z1 phân biệt kém; ở cửa hàng vừa/lớn Z1 xếp hạng tốt. Dạng Z1 theo cạnh lộ trình
  (trọng số 2/|giỏ|) đã thử và không cải thiện.
- **Chọn và tinh chỉnh** dùng hạt giống sàng lọc cố định; mọi con số báo cáo lấy từ đánh giá cuối
  trên hạt giống độc lập để tránh thiên lệch chọn lọc.
- **Tuyến tính hóa RLT** (sum_l y_ijkl = x_ik) có cùng tập nghiệm nguyên với (12) nhưng nới lỏng
  LP chặt hơn nhiều; E2 so sánh trực tiếp hai dạng.
