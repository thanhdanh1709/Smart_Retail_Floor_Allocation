# Thuật ngữ, ký hiệu và quy ước đặt tên

Tài liệu tra cứu dành cho người mới đọc mã nguồn `smart-retail-layout`. Mọi định nghĩa ở đây lấy từ docstring và
chú thích trong mã. Nếu có chỗ không khớp, **mã nguồn là chuẩn**; khi sửa mã, hãy cập nhật lại tài liệu này.

Tài liệu liên quan: `README.md` (cài đặt, cách chạy, mô hình v1), `docs/KE_HOACH_v4.md` (kế hoạch v4, công thức
đầy đủ), `docs/KIEM_CHUNG_TRICH_DAN.md` (nguồn trích dẫn).

---

## 1. Bức tranh chung trong 30 giây

Bài toán: **xếp các ngành/nhóm hàng vào các vị trí kệ (slot)** của một cửa hàng bán lẻ, sao cho cùng lúc:

- **luồng nhặt đơn online** (người nhặt – *picker*) đi ít nhất → `Z_P` nhỏ;
- **khách tại chỗ** (*walk-in*) đi ngang nhiều hàng để mua ngẫu hứng → `Z_W` lớn;
- hai luồng ít **chạm mặt** nhau → `C` nhỏ;
- bền vững trước việc ta **không biết chắc khách đi thế nào** (nhiều mô hình hành vi `m`) → cực tiểu *regret* lớn nhất.

Hệ thống chia làm **bốn tầng** (`src/system/`):

| Tầng | Việc | Đầu ra (hợp đồng dữ liệu) |
|---|---|---|
| T1 | Sinh bố trí kệ (lưới mặt bằng) | `ShelfLayout` |
| T4 (tính trước) | Tính luồng khách / người nhặt cho mặt bằng | `FlowBank` |
| T2 | Gán ngành hàng → slot (xương sống) | `CategoryPlan` |
| T3 | Xếp từng món trong kệ (planogram) | `Planogram`, phản hồi `SpaceDemand`, `CategoryValue` |
| T4 (đánh giá) | Đánh giá cuối: KPI, mô phỏng | `FlowReport` |

Luồng dữ liệu: `StoreSpec ─T1→ ShelfLayout ─T4→ FlowBank ─T2→ CategoryPlan ─T3→ Planogram`, sau đó `T4.evaluate → FlowReport`.
Bộ điều phối gói tất cả lại thành `StoreDesign`.

---

## 2. Từ viết tắt

### 2.1 Thuật ngữ bài toán và thuật toán

| Viết tắt | Đầy đủ | Nghĩa trong dự án |
|---|---|---|
| QAP | Quadratic Assignment Problem | Bài toán gán bậc hai; dạng gốc của `Z1` (luồng × khoảng cách) |
| QAPLIB | QAP Library (Burkard et al., 1997) | Bộ dữ liệu chuẩn để kiểm chứng GA (thí nghiệm E1) |
| ILP / MIP | Integer / Mixed-Integer Linear Programming | Quy hoạch nguyên, giải chính xác (`model_ilp.py`), chỉ dùng khi n nhỏ |
| LP | Linear Programming | Quy hoạch tuyến tính (vd. lịch nhặt theo giờ là bài vận tải LP) |
| RLT | Reformulation-Linearization Technique (Adams–Johnson) | Cách tuyến tính hóa chặt hơn cho ILP (`linearization="rlt"`) |
| GA | Genetic Algorithm | Thuật toán di truyền (`ga.py`) |
| NSGA-II | Non-dominated Sorting GA II | GA đa mục tiêu, cho tập Pareto (`moo.py`, `twoflow.py`) |
| OX / PMX | Order Crossover / Partially Mapped Crossover | Hai phép lai cho mã hóa hoán vị |
| SA | Simulated Annealing | Mô phỏng luyện kim (phương pháp so sánh, `baselines.py`) |
| TS / tabu | Tabu Search (Taillard, 1991) | Tìm kiếm tabu (phương pháp so sánh) |
| LS | Local Search | Tìm kiếm cục bộ (2-swap, 2-opt, Or-opt) |
| 2-swap | – | Đổi chỗ nhóm hàng ở hai slot `r ↔ s` |
| 2-opt / Or-opt | – | Cải thiện thứ tự điểm dừng của một lộ trình |
| TSP | Travelling Salesman Problem | Bài toán người bán hàng; ở đây là lộ trình qua các slot trong giỏ |
| MSA | Method of Successive Averages | Trung bình hóa dần để hội tụ (SUE, vòng ghép T2↔T3) |
| SUE | Stochastic User Equilibrium | Cân bằng người dùng ngẫu nhiên (mô hình hành vi có ùn tắc) |
| SSAP | Shelf Space Allocation Problem | Bài toán phân bổ không gian kệ cho từng món (tầng 3) |
| MOO | Multi-Objective Optimization | Tối ưu đa mục tiêu |
| KPI | Key Performance Indicator | Chỉ số đánh giá; **chỉ được tính trong `src/system/metrics.py`** |
| CSR | Compressed Sparse Row | Định dạng ma trận thưa (giỏ hàng, chặng đường, đường đi tác tử) |
| CRN | Common Random Numbers | Số ngẫu nhiên chung khi so sánh các phương án bằng mô phỏng |
| CLI | Command-Line Interface | `python -m src.system design ...` |

### 2.2 Mô hình hành vi khách (`behavior.py`, tập `M`)

Mỗi mô hình = (luật thứ tự điểm dừng, luật chọn đường mỗi chặng).

| Mã | Tên | Thứ tự điểm dừng | Chọn đường |
|---|---|---|---|
| `SP` | Shortest Path | gần nhất + 2-opt (≈ TSP) | đường ngắn nhất |
| `NN` | Nearest Neighbour | gần nhất kế tiếp | đường ngắn nhất |
| `SNK` | Snake | theo đường rắn (chữ S) | đường ngắn nhất |
| `RL` | Recursive Logit | gần nhất kế tiếp | recursive logit (Fosgerau et al., 2013) |
| `RL-A` | RL + Attraction | gần nhất kế tiếp | RL + sức hút `η` của ô trước mặt kệ |
| `PER` | Perimeter | theo vòng chu vi (góc) | RL + sức hút `ζ` của ô sát tường |
| `SUE` | Stochastic User Equilibrium | gần nhất kế tiếp | RL, chi phí ô tăng theo mật độ (MSA) |
| `LIN` | Linear | – | Không phải mô hình hành vi. Đây là hàng "bài thay thế tuyến tính/QAP" trong ma trận mất mát `L` |

### 2.3 Định tuyến người nhặt (`picking.py`)

| Mã | Nghĩa |
|---|---|
| `exact` | Held-Karp (quy hoạch động, chính xác) khi ≤ `MAX_EXACT = 15` điểm dừng; đơn lớn hơn dùng NN + 2-opt + Or-opt |
| `nn2opt` | Gần nhất + 2-opt |
| `sshape` | S-shape: đi rắn qua các dải lối có món |
| `largest_gap` | Largest gap: mỗi dải tách tại khoảng trống lớn nhất |

### 2.4 Mã giai đoạn, mức trung thực, thí nghiệm

| Mã | Nghĩa |
|---|---|
| `GĐ0` … `GĐ9` | Giai đoạn triển khai trong `KE_HOACH_v4.md` mục 6 (vd. GĐ4 = tầng 2 bền vững, `robust.py`) |
| `v1`, `v2`, `v4` | Phiên bản mô hình (thư mục `results/_v1_model_goc`, `_v2_hieu_chinh`, nhánh git `v4`) |
| `L0` / `L1` / `L2` | Mức trung thực (*fidelity*): L0 sàng lọc nhanh, chỉ giải tích; L1 trung bình, có mô phỏng 5 lần; L2 phương án cuối, 30 lần mô phỏng. Tham số cụ thể nằm ở `FIDELITY` trong `src/system/layers.py` |
| `E1` … `E20`, `E6'` | Mã thí nghiệm (`experiments/eN_*.py`, `results/EN/`); danh sách ở `KE_HOACH_v4.md` mục 7 |
| `B2`, `B4.3`, `PL1`, công thức `(11)` | Tham chiếu tới mục, phụ lục, công thức trong **đề cương** (bản v1). Thấy trong docstring cũ |
| `T1` … `T4` | Tầng của hệ thống (xem mục 1) |

### 2.5 Bộ dữ liệu

| Tên | Dùng cho |
|---|---|
| Instacart 2017 | Giỏ hàng thật → `w`, `f`, `p`, mẫu đơn, hồ sơ giờ đơn online `λ_P(h)`, dữ liệu cấp món |
| Open e-commerce 1.0 (Amazon) | Chỉ số giá **tương đối** giữa 40 nhóm |
| Tesco Grocery 1.0 | Kiểm tra độ khái quát của `f_i` |
| Lyon Dense Crowd | Hình dạng giản đồ tốc độ–mật độ `v(ρ)` (Weidmann) |
| QAPLIB | Kiểm chứng GA (E1) |

---

## 3. Ký hiệu toán học và tên biến tương ứng trong mã

### 3.1 Chỉ số và kích thước

| Ký hiệu | Tên trong mã | Nghĩa |
|---|---|---|
| `i`, `j` | `i`, `j`, `a`, `b` | Chỉ số nhóm/ngành hàng (*category*) |
| `k`, `l` | `k`, `l` | Chỉ số slot (vị trí trưng bày) |
| `r`, `s` | `r`, `s` | Hai slot được đổi chỗ trong một bước 2-swap |
| `n` | `n` | Số nhóm hàng thật |
| `m` | `m`, `inst.m` | Số slot. Nếu `m > n` thì đệm thêm `m − n` **nhóm rỗng** (chỉ số `≥ n`, mọi tham số = 0) |
| `m` (trong `Z_W^m`) | `m`, `model` | Một mô hình hành vi thuộc tập `M`. **Chú ý: cùng chữ `m` với số slot, phân biệt theo ngữ cảnh** |
| `h` | `h`, `HOURS` | Giờ trong ngày (0–23) |
| `π`, `perm` | `perm` | Lời giải: hoán vị độ dài `m`, `perm[k]` = nhóm đặt ở slot `k` |

### 3.2 Tham số nhóm hàng (`params.py`, `instance.py`)

| Ký hiệu | Mã | Nghĩa |
|---|---|---|
| `N` | – | Tổng số đơn trong dữ liệu |
| `w_ij` | `W` (ma trận `m×m`, đã đệm) | Luồng = `n_ij / N`, tỷ lệ đơn chứa cả `i` và `j` |
| `f_i` | `f` | Độ phổ biến = `n_i / N` |
| `lift_ij` | `lift` | `w_ij / (f_i f_j)` |
| `p_i` | `p` | Hệ số mua ngẫu hứng (proxy: `1 − tỷ lệ mua lại`) |
| `v_i` | `v`, `value_v` | Giá trị nhóm hàng (giả định có cơ sở; tầng 3 có thể cập nhật) |
| `q_i` | `q` | `p_i (1 − f_i)(1 − e^{−λ t_i})`: xác suất một lần đi ngang dẫn tới mua ngẫu hứng |
| `t_i` | `dwell_median_s`, `dwell_level` | Thời gian dừng trung vị (giây). Mức 1/2/3 tương ứng 6/10/18 s (`DWELL_MEDIAN_S`) |
| `σ` | `sigma` | Độ lệch log của thời gian dừng, `T ~ LogNormal(ln t, σ)`, mặc định 0.6 |
| `λ` | `lam` | Cường độ mua ngẫu hứng trong `1 − e^{−λT}`; `None` = hiệu chỉnh bằng mô phỏng |
| `coef_i` | `coef`, `unit_coef` | `unit_coef = clip(p)·E[1 − e^{−λT}]`; nhân với `v` ra `coef` |
| `s_i` | `SpaceDemand.s` | Số slot tối thiểu ngành `i` cần (phản hồi từ tầng 3) |

### 3.3 Mặt bằng (`floorplan.py`)

| Ký hiệu | Mã | Nghĩa |
|---|---|---|
| `d(k, l)` | `D` | Khoảng cách đi bộ giữa điểm tiếp cận slot `k` và `l` (m) |
| `d_in(k)` | `d_in` | Cửa vào → slot `k` |
| `d_out(k)` | `d_out` | Slot `k` → quầy thu ngân gần nhất |
| `d_0(k)` | – | Khu tập kết đơn online `P` → slot `k` |
| `e_k` | `e` | Mức tiếp xúc của slot `k` (tỷ lệ khách đi qua vùng tiếp xúc), trong [0, 1] |
| `e_W^m(k)`, `e_P(k)` | `exposure(...)`, `pick_exposure(...)` | Tiếp xúc theo khách (mô hình `m`) / theo người nhặt |
| `o_W^m(k)`, `o_P(k)` | `occ_*` | Chiếm chỗ (giây) tại vùng slot `k`, mỗi khách / mỗi đơn |
| – | `Slot.access` | Ô lối đi trước tâm slot (điểm tiếp cận) |
| – | `Slot.zone` | Các ô lối đi trước mặt slot (vùng tiếp xúc) |
| – | `Slot.face` | Hướng mặt kệ ra lối: `N`/`S`/`W`/`E` |
| – | `seg_len`, `min_len` | Độ dài một slot (số ô kệ) khi cắt dãy kệ |

**Ký tự ô lưới** (ô 1 m × 1 m, tệp `data/floorplans/*.txt`):

| Ký tự | Nghĩa |
|---|---|
| `A` | Lối đi (*aisle*) |
| `S` | Kệ thường (*shelf*) |
| `R` | Kệ khu lạnh (*refrigerated*) |
| `X` | Tường / vật cản |
| `E` | Cửa vào (*entrance*) |
| `C` | Quầy thu ngân (*checkout*) |
| `P` | Khu tập kết đơn online (*pick-up/staging*, từ v4). Đi lại được |

Đi lại được: `WALKABLE = "AECP"`; kệ: `SHELF = "SR"`.

### 3.4 Hàm mục tiêu

**Mô hình QAP gốc (v1, `instance.py`)**

| Ký hiệu | Mã | Nghĩa |
|---|---|---|
| `Z1` | `inst.z1(perm)` | `Σ_{i<j} w_ij d(k_i,k_j) + Σ_i f_i (d_in + d_out)`: quãng đường, cực tiểu |
| `Z2` | `inst.z2(perm)` | `Σ_i v_i p_i e(k_i)`: giá trị ngẫu hứng, cực đại |
| `Z` | `inst.z(perm, alpha)` | Tổng có trọng số đã chuẩn hóa theo bảng payoff, công thức (11) |
| `α` | `alpha` | Trọng số giữa `Z1` và `Z2` |
| `Z1min…Z2max` | `payoff["z1min"]`… | Bảng payoff (`solvers.compute_payoff`) |
| `F1`, `F2` | – | Hai mục tiêu chuẩn hóa về [0, 1] của NSGA-II (`moo.py`) |
| `lin1`, `lin2` | `inst.lin1`, `inst.lin2` | Phần tuyến tính `L[i,k]` của `Z1` và `Z2` |
| `L, cq, c0` | `inst.weighted(alpha)` | Dạng `Z = cq·quad + Σ L + c0` dùng trong nhân numba |
| `Δ(r,s)` | `delta_*` | Độ thay đổi `Z` khi đổi chỗ slot `r ↔ s`, O(m) (công thức 13) |

**Mô hình hai luồng theo định tuyến (v4, hàm mục tiêu chính)**

| Ký hiệu | Mã | Nghĩa |
|---|---|---|
| `Z_P` | `bank.z_p(perm)`, KPI `"Z_P"` | Quãng đường nhặt kỳ vọng mỗi đơn (m): khu tập kết → món → khu tập kết. Cực tiểu |
| `Z_W^m` | `bank.z_w(perm, m)`, KPI `"Z_W[m]"` | Doanh thu ngẫu hứng kỳ vọng mỗi khách theo mô hình `m`. Cực đại |
| `walk[m]` | `bank.walk(perm, m)` | Quãng đường kỳ vọng mỗi khách theo mô hình `m` |
| `Z_W^m*` | `z_w_star` | Giá trị tốt nhất **đã biết** (best-of-k), nên regret tính ra là **cận dưới** |
| `regret[m]` | `regret` | `(Z_W^m* − Z_W^m) / Z_W^m*` |
| `max_regret` | `max_regret` | `max_m regret[m]`, mục tiêu minimax của tầng 2 |
| `C^m` | KPI `"C[m]"`, `conflict` | Chỉ số chạm mặt giữa hai luồng (công thức trong docstring `metrics.py`) |
| `ε` | `eps` | Ràng buộc `Z_P ≤ ε · Z_P(hiện trạng)` (mặc định 1.10) |
| `ε_C` | `eps_c` | Ràng buộc `C^m ≤ ε_C · C^m(hiện trạng)` với mọi `m` (mặc định 1.2) |
| `eps_ratio`, `C_ratio` | cùng tên | Tỷ số so với hiện trạng, dùng để kiểm ràng buộc |
| `Z_P^lin`, `Z_W^lin`, `C^lin` | `twoflow.surrogate` | **Bài thay thế** tuyến tính/QAP (đóng băng `e`), chỉ dùng cho ILP, khởi tạo và đo giá của xấp xỉ |
| `L[A,B]` | `loss_matrix`, `RobustResult.loss` | Regret theo mô hình B của phương án tối ưu theo mô hình A |

### 3.5 Ràng buộc nghiệp vụ

| Ký hiệu | Mã | Nghĩa |
|---|---|---|
| tương thích | `allowed` (`m×m` bool), `needs_cold`, `is_cold` | Nhóm `i` được đặt ở slot `k` không (vd. hàng lạnh phải ở kệ `R`). Giữ **cứng** |
| cố định | `fixed` `{nhóm: slot}` | Nhóm bị khóa tại một slot |
| tách xa | `sep_i`, `sep_j`, `sep_d` (`δ`) | Cặp nhóm phải cách nhau ≥ δ m (`DEFAULT_SEPARATE`). Ràng buộc **mềm** |
| dời kệ | `R`, `relocation_R`, `k0` | Số nhóm tối đa được dời so với vị trí hiện trạng `k0[i]` (`−1` = không giới hạn). Ràng buộc mềm |
| phạt | `rho`, `PENALTY` | Hệ số phạt cho mỗi vi phạm ràng buộc mềm |
| `cold_slack` | cùng tên | Tỷ lệ slot lạnh dư so với số nhóm cần lạnh |

### 3.6 Hành vi, nhặt đơn, mô phỏng

| Ký hiệu | Mã | Nghĩa |
|---|---|---|
| `μ` | `mu` | Độ "lý trí" của RL (1/m). `None` = hiệu chỉnh sao cho đi vòng ≈ `target_detour = 1.28` |
| `η` | `eta` | RL-A: sức hút của ô lối đi trước mặt kệ |
| `ζ` | `zeta` | PER: sức hút của ô sát tường; `perimeter_band` = bề rộng dải chu vi |
| `κ` | `kappa` | Chi phí ô = `1 + κ·mật độ` (SUE và người nhặt tránh khách) |
| `M`, `z`, `Q`, `G` | – | Ma trận recursive logit (docstring `behavior.py`) |
| `λ_P(h)` | `hours` | Tỷ lệ đơn online theo giờ (Instacart, dữ liệu thật) |
| `λ_W(h)` | `hours` | Tỷ lệ khách tại chỗ theo giờ (**giả định** hai đỉnh, `schedule.py`) |
| `s` (trong `C`) | `online_share` | Tỷ lệ đơn online trên tổng lượt mua |
| `B` | `batch_size` | Số đơn tối đa trên một xe nhặt |
| `K` | `picker_capacity` | Số đơn nhặt được mỗi giờ |
| `Δ` (lịch nhặt) | `delivery_window_h`, `window_h` | Đơn đặt giờ `h` phải nhặt trong `[h, h + Δ]` |
| `v`, `v_free` | `speed`, `v_free`, `V_FREE` | Tốc độ đi bộ (m/s) |
| `ρ`, `ρ_max`, `γ` | `rho_max`, `gamma`, `WEIDMANN_GAMMA` | Mật độ (người/m²) và tham số giản đồ Weidmann. **Chú ý: `rho` trong GA là hệ số phạt, khác `ρ` mật độ** |
| `dt` | `dt` | Bước thời gian mô phỏng hai luồng (s) |
| `ψ_c`, `β`, `φ` | – | Tham số tầng 3 / vòng ghép (xem `KE_HOACH_v4.md` mục 1.3, 4) |
| `θ` | `ShelfLayout.theta` | Tham số sinh bố trí kệ ở tầng 1 |

---

## 4. Quy ước đặt tên trong mã

### 4.1 Chung (Python, PEP 8)

| Loại | Quy ước | Ví dụ |
|---|---|---|
| Module | `snake_case`, ngắn, theo chức năng | `floorplan.py`, `twoflow_sim.py` |
| Lớp | `PascalCase` | `FlowBank`, `RouteModel`, `GAConfig` |
| Lớp cấu hình | Hậu tố `Config`, là `@dataclass` | `GAConfig`, `SimConfig`, `RobustConfig`, `TwoFlowSimConfig` |
| Lớp kết quả | Hậu tố `Result` | `GAResult`, `RobustResult` |
| Hợp đồng giữa các tầng | `@dataclass` trong `src/system/contracts.py`, tên là danh từ nghiệp vụ | `StoreSpec`, `CategoryPlan`, `Planogram` |
| Hàm, biến | `snake_case` | `make_instance`, `relocation_sequence` |
| Hằng số module | `UPPER_SNAKE_CASE` | `MAX_EXACT`, `WALKABLE`, `FIDELITY` |
| Hàm/thuộc tính nội bộ | Tiền tố `_` | `_route_one`, `_digest`, `fp._dist` |
| Nhân numba | Trong `fastops.py` hoặc hàm `_...` có `@njit(cache=True)` | `quad_value`, `_evaluate` |

### 4.2 Biến mang tên ký hiệu toán học

Biến bám sát ký hiệu trong đề cương và kế hoạch, để đọc công thức và đọc mã như nhau:

- **Một chữ cái theo công thức:** `W`, `D`, `f`, `p`, `v`, `q`, `e`, `L`, `R`.
  Chữ **hoa** dùng cho ma trận (`W`, `D`, `L`, `M`, `Q`, `G`), chữ **thường** cho vectơ hoặc số vô hướng (`f`, `e`, `lam`).
- **Chữ Hy Lạp viết bằng tên tiếng Anh:** `alpha`, `lam` (vì `lambda` là từ khóa Python), `mu`, `eta`, `zeta`,
  `kappa`, `sigma`, `rho`, `gamma`, `theta`, `eps`.
- **Chỉ số dưới thành hậu tố `_`:** `Z_P` → `z_p`, `Z_W` → `z_w`, `Z_W^m*` → `z_w_star`, `d_in`, `d_out`, `sep_d`.
- **Tên KPI** trong dict trả về giữ đúng ký hiệu và đặt mô hình trong ngoặc vuông: `"Z_P"`, `"Z_W[RL]"`, `"regret[SP]"`, `"C[NN]"`, `"walk[PER]"`.

### 4.3 Hậu tố và tiền tố thường gặp

| Mẫu | Nghĩa | Ví dụ |
|---|---|---|
| `*_s`, `*_m`, `*_h`, `*_min` | Đơn vị: giây, mét, giờ, phút | `pick_time_s`, `shelf_len_m`, `window_h`, `trip_time_min` |
| `n_*` | Số lượng | `n_customers`, `n_orders`, `n_baskets`, `n_categories` |
| `*_idx`, `idx` | Mảng chỉ số | `slot_idx` |
| `*_ptr` + `*_idx`/`*_node` | Cặp mảng CSR | `b_ptr`/`b_idx`, `p_ptr`/`p_node`, `leg_ptr`/`leg_idx` |
| `*_frac`, `*_share`, `*_rate`, `*_ratio` | Tỷ lệ trong [0, 1] hoặc tỷ số | `ls_frac`, `online_share`, `feasible_rate`, `eps_ratio` |
| `*_star` | Tốt nhất đã biết | `z_w_star` |
| `*_lin` | Bài thay thế tuyến tính | `Z_P^lin` |
| `ga_*`, `nsga_*`, `ls_*`, `seq_*` | Tham số của GA, NSGA-II, tìm kiếm cục bộ, tuyến tính hóa tuần tự | `ga_pop`, `nsga_gens`, `ls_iter`, `seq_iters` |
| `delta_*` | Đánh giá chênh lệch khi đổi chỗ | `delta_swap` |
| `make_*`, `build`, `load` | Tạo đối tượng / dựng dữ liệu từ thô / đọc dữ liệu đã xử lý | `make_instance`, `items.build`, `items.load` |
| `digest` | Mã băm SHA-256 (16 ký tự) của nội dung, dùng cho cache và tái lập | `spec.digest()` |
| `current` | Sơ đồ hiện trạng | `inst.current` |

### 4.4 Viết tắt trong tên biến

| Viết tắt | Nghĩa |
|---|---|
| `fp` | `FloorPlan`: mặt bằng |
| `inst` | `Instance`: bộ dữ liệu bài toán (mặt bằng + nhóm hàng + ràng buộc) |
| `bank` | `FlowBank`: luồng đã tính sẵn |
| `rm` | `RouteModel`: mô hình định tuyến |
| `spec` | `StoreSpec`: đặc tả cửa hàng |
| `pg` | `Planogram` |
| `fid` | Mức trung thực (`Fidelity`) |
| `cfg` | Đối tượng cấu hình |
| `rng` | `numpy.random.Generator` |
| `pop` | Quần thể GA |
| `pc`, `pm` | Xác suất lai, xác suất đột biến |
| `cat`, `cat_at`, `slot_of` | Ngành hàng; ngành tại slot; slot của ngành (ánh xạ ngược của `perm`) |
| `meta` | Bảng thông tin nhóm (`category_id`, `name`, `needs_cold`, …) |
| `PL`, `PD`, `PC` | Ma trận quãng đường kỳ vọng của chặng / tới cửa / tới thu ngân trong `routing.py` |
| `sf` | `slots_frame`: bảng thông tin slot |

### 4.5 Mã định danh dữ liệu

| Mã | Định dạng | Nguồn |
|---|---|---|
| `slot_id` | `K001`, `K002`, … (đánh số từ 1) | `FloorPlan.slots_frame()` |
| `group_id` / `category_id` | `G01` … `G40` (mức `group`, 40 nhóm kiểu cửa hàng Việt Nam) | `data/mappings/groups_vn.csv` |
| `aisle_id` | Số nguyên Instacart (mức `aisle`, 132 aisle) | Instacart |
| `product_id` | Số nguyên Instacart | Instacart |
| `level` | `"group"` hoặc `"aisle"`: mức gom nhóm hàng | `StoreSpec.level` |

### 4.6 Tên tệp và thư mục

| Mẫu | Nghĩa |
|---|---|
| `{kind}_{scale}` | Tên mặt bằng. `kind ∈ {grid, racetrack, free}` (lưới song song / vòng / tự do), `scale ∈ {small, medium, large}`. Riêng `case_minimart` là cửa hàng tình huống |
| `specs/{name}.yaml` | Đặc tả `StoreSpec` để chạy hệ thống |
| `data/floorplans/{name}.txt`, `slots_{name}.csv` | Lưới mặt bằng và bảng slot |
| `data/raw/` | Dữ liệu gốc tải về (không sửa) |
| `data/processed/{group,aisle,items,external}/` | Dữ liệu đã xử lý (`w.npy`, `f.npy`, `manifest.json`, …) |
| `experiments/eN_*.py` → `results/EN/` | Script thí nghiệm và kết quả |
| `tests/test_{module}.py` | Kiểm thử cho từng module |

---

## 5. Quy ước khác cần biết khi đọc mã

- **Ngôn ngữ:** docstring và chú thích viết tiếng Việt; tên định danh viết tiếng Anh hoặc theo ký hiệu toán.
- **Đơn vị:** mét (m), giây (s), đơn/giờ. Lưới mặt bằng có ô 1 m × 1 m.
- **Một nguồn KPI duy nhất:** mọi chỉ số lấy từ `src/system/metrics.py`; không tầng nào được tự tính KPI riêng.
- **Hướng tối ưu:** `Z_P`, `Z1`, `C`, `regret` thì **cực tiểu**; `Z_W`, `Z2` thì **cực đại**. Trong NSGA-II mọi mục tiêu đều đưa về cực tiểu, nên `Z_W` được đổi dấu thành `−Z_W`.
- **"Định tuyến" và "tuyến tính":** từ v4, hàm mục tiêu chính tính bằng định tuyến (`FlowBank`, `routing.py`). Dạng tuyến tính/QAP chỉ là **bài thay thế** cho ILP (`KE_HOACH_v4.md` mục 0).
- **Cận dưới:** `Z_W^m*` là tốt nhất *đã biết* chứ không phải tối ưu thật, nên regret báo cáo là cận dưới.
- **Giả định và dữ liệu thật:** docstring ghi rõ `GIẢ ĐỊNH` (vd. `λ_W(h)`, `v_i`) để phân biệt với tham số lấy từ dữ liệu.
- **Tái lập:** mọi thứ có `seed`; `StoreDesign.manifest` cùng `digest` cho phép chạy lại đúng thiết kế. Pipeline v1 (`pipeline.py`) tách hạt giống sàng lọc (`SCREEN_SEED = 0`) khỏi hạt giống đánh giá cuối (`FINAL_SEED = 50_000`).
- **Ký hiệu trùng chữ:** `m` (số slot / mô hình hành vi), `R` (giới hạn dời kệ / ô kệ lạnh / ma trận), `L` (ma trận tuyến tính / ma trận mất mát / danh sách mua trong `simulate.py`), `rho` (hệ số phạt / mật độ), `e` (tiếp xúc / hằng số Euler). Phân biệt theo module và ngữ cảnh.
