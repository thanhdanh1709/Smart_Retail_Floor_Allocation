# Smart Retail Floor Allocation

Mã nguồn của đề án **"Tối ưu hóa không gian trưng bày và luồng di chuyển của khách hàng trong
cửa hàng bán lẻ số"**. Mã nguồn làm theo Phần B của đề cương: mô hình hóa mặt bằng, ước lượng
tham số từ dữ liệu giỏ hàng, mô hình quy hoạch nguyên song mục tiêu, GA/NSGA-II, mô phỏng tác
tử, các thí nghiệm E1–E7 và dashboard hỗ trợ ra quyết định.

```
dữ liệu giỏ hàng ─┐
(Instacart)       ├─► tham số w, f, p, v ─┐
mặt bằng (lưới) ──┴─► D, d_in, d_out, e ──┴─► ILP / GA / NSGA-II ─► mô phỏng tác tử ─► dashboard
```

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
| E5 | Mô phỏng: hiện trạng vs 3 điểm Pareto trên 9 + 1 mặt bằng | `e5_simulation.py` |
| E6 | Độ nhạy (λ, p, chiến lược đi, tốc độ) và đường cong cải thiện theo R | `e6_sensitivity.py` |
| E7 | Bóc tách: bỏ e_k, bỏ tìm kiếm cục bộ, lặp cập nhật e_k | `e7_ablation.py` |

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
- **Mức tiếp xúc `e_k`** phụ thuộc chính sơ đồ. E7 thử lặp "tối ưu → mô phỏng → cập nhật e_k"
  với giảm chấn; nêu rõ đây là một hạn chế.
- **Tuyến tính hóa RLT** (sum_l y_ijkl = x_ik) có cùng tập nghiệm nguyên với (12) nhưng nới lỏng
  LP chặt hơn nhiều; E2 so sánh trực tiếp hai dạng.
