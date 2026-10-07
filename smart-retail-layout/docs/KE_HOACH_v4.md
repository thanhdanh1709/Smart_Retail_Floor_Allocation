# Kế hoạch v4 – Sơ đồ bền vững cho hai luồng (khách tại chỗ + nhặt đơn online)

> Ngày lập: 2026-10-05 (bản 3: tầng 1, 3, 4 làm sâu; bốn tầng tích hợp thành một hệ thống). Kế thừa v3 (Cách A định tuyến, E1–E7 đã chạy full).
> Bốn tầng đều là phần nghiên cứu, mỗi tầng có: mô hình toán, thuật toán, kiểm chứng, thí nghiệm và đóng góp riêng.
> Tầng 2 vẫn là xương sống (hướng 1 + 3). Tầng 1, 3, 4 **ghép ngược** vào tầng 2 chứ không chỉ chạy nối tiếp.

---

## 0. Hàm mục tiêu: định tuyến là chính, tuyến tính chỉ là bài thay thế cho ILP

### 0.1 Vì sao không dùng dạng tuyến tính làm hàm mục tiêu chính
Kết quả v3 đã trả lời câu này:

| Bằng chứng v3 | Ý nghĩa |
|---|---|
| Spearman với mô phỏng: định tuyến Z1 0,996 / Z2 0,994; QAP hiệu chỉnh chỉ 0,31–0,40 | Dạng QAP/tuyến tính xếp hạng sơ đồ **sai** |
| E7(d): e_k phụ thuộc sơ đồ, CV 0,65; Spearman(e hình học, lưu lượng thật) 0,0–0,3 | Cố định e_k là bỏ đi phần lớn thông tin |
| E5: route "theo mô hình" +34% DT ngẫu hứng, 6/10 bộ trội; QAP −9,8% | Tối ưu theo dạng tuyến tính cho sơ đồ kém hơn |

Dạng Z_W = Σ v_i·q_i·e_k·x_ik sai ở **hai chỗ**, không chỉ một:
1. **e_k phụ thuộc x:** khách đi tới chỗ đặt món họ cần, nên sơ đồ đổi thì lưu lượng đổi. Lặp điểm bất động (MSA) chỉ sửa được chỗ này.
2. **Tương quan giữa giỏ và đường đi:** doanh thu ngẫu hứng chỉ tính cho ngành **không có trong giỏ** mà khách **đi ngang qua**. Mà khách đi ngang qua đâu lại phụ thuộc chính giỏ đó. Tổng Σ e_k·coef dùng lưu lượng trung bình nên bỏ mất tương quan này, kể cả khi e_k đúng. `RouteModel` tính đúng vì nó định tuyến **từng giỏ** và loại ngành đã có trong giỏ.

Dạng tuyến tính có trong bản trước chỉ vì nó cho phép ILP giải **chính xác**. Nhưng chính xác cho một hàm sai thì không có giá trị. Vì vậy v4 đổi lại như sau.

### 0.2 Phân vai mới

| | Hàm mục tiêu chính (định tuyến) | Bài thay thế tuyến tính / QAP |
|---|---|---|
| Z_W^m | `RouteModel` mở rộng: định tuyến từng giỏ theo mô hình m, doanh thu ngẫu hứng của ngành ∉ giỏ trên đường đi | Σ v·q·e_k^m(x_t)·x_ik, e đóng băng tại x_t |
| Z_P | Quãng đường nhặt kỳ vọng: định tuyến từng đơn từ **khu tập kết** (Held-Karp/2-opt) | Σ f_ij·d_kl·x_ik·x_jl + Σ g_i·d_0k·x_ik (QAP, RLT) |
| C^m | Σ_h Σ_k (lượt người nhặt qua k)·(mật độ khách tại k, giờ h), **cả hai lấy từ định tuyến** | Σ g_i·t_k^m·x_ik |
| Bộ giải | **GA / NSGA-II** (lời giải chính) | ILP |
| Vai trò | Thiết kế thật | Khởi tạo, kiểm tra GA, đo sai số của xấp xỉ |

### 0.3 ILP còn làm gì
- **Tuyến tính hóa tuần tự:** đóng băng e tại x_t (lấy từ định tuyến) → giải ILP → **chấm lại bằng định tuyến** → MSA → lặp. Giữ nghiệm tốt nhất **theo định tuyến**. Đây là một bộ giải heuristic dựa trên ILP, dùng để so với GA và làm hạt giống cho GA.
- **Kiểm tra GA cài đúng:** trên chính bài thay thế (n ≤ 15), GA phải trúng tối ưu ILP, như E2 đã làm.
- **Đo cái giá của xấp xỉ:** sơ đồ tối ưu theo bài tuyến tính, chấm bằng định tuyến, mất bao nhiêu. Thực chất đây là **một hàng của ma trận L[A,B]** (A = "mô hình tuyến tính"), nên trở thành một phát hiện của hướng 1 chứ không chỉ là chi tiết kỹ thuật.
- Phải ghi trong luận văn: **không còn khẳng định "ILP cho Pareto chính xác"** với hàm thật. ILP chỉ chính xác cho bài thay thế.

### 0.4 Hệ quả cho minimax regret
- Z_W^m* theo định tuyến không giải chính xác được → dùng **giá trị tốt nhất đã biết**: max của nhiều lần GA, ILP tuyến tính hóa tuần tự, và các sơ đồ tối ưu cho từng mô hình.
- Nếu Z_W^m* tốt nhất đã biết thấp hơn tối ưu thật thì regret tính ra **nhỏ hơn** regret thật. Phải báo cáo rõ điều này. Kiểm tra độ ổn định của Z_W^m* qua số lần chạy GA (đường cong "best-of-k").

### 0.5 Mở rộng định tuyến cho mọi mô hình hành vi
`RouteModel` hiện chỉ có một luật: NN + 2-opt, đường ngắn nhất, tập slot đi qua cố định cho mỗi chặng (`leg_idx`). Cần tổng quát thành:
- **Thứ tự dừng** theo mô hình m: TSP, NN, rắn, ưu tiên chu vi.
- **Mỗi chặng** a→b: thay tập slot đi qua bằng **xác suất đi qua** P^m[a,b,k] (đường ngắn nhất: 0/1; recursive logit: giải tích, tính trước một lần cho mỗi mặt bằng).
- Doanh thu ngẫu hứng tính theo **xác suất đi qua slot k ít nhất một lần trong chuyến**: 1 − Π_chặng (1 − P^m[a,b,k]), giả định các chặng độc lập. Mô phỏng tác tử dùng để kiểm chứng giả định này.

### 0.6 Tốc độ (điều kiện để GA chạy được)
Hiện tại ~1 ms/lần đánh giá (1.500 giỏ, một mô hình). Với |M| = 7 mô hình, cộng thêm định tuyến nhặt đơn, sẽ lên khoảng 10 ms. Cách giữ GA đủ nhanh:
- **Đánh giá chênh lệch khi đổi chỗ r↔s:** chỉ định tuyến lại các giỏ chứa ngành ở r hoặc s. Các giỏ còn lại chỉ đổi hệ số ngẫu hứng tại hai slot: dùng số lượt đi qua của từng slot đã lưu, tách theo "giỏ có chứa ngành j hay không".
- **Nhiều mức trung thực:** 300 giỏ khi sàng lọc, 1.500 ở mức L1, mẫu độc lập lớn ở mức L2. Tương ứng mức L0–L2 ở mục 1.4.
- Song song hóa theo mô hình m, và cache như hiện có.
- Tiêu chí: ≤ 3 ms/lần đánh giá chênh lệch với |M| = 7 (đo ở GĐ3).

---

## 1. Hệ thống hoàn chỉnh: bốn tầng liên kết

Mục tiêu: **một hệ thống duy nhất**. Đầu vào là mô tả cửa hàng, đầu ra là bộ thiết kế hoàn chỉnh. Các tầng trao đổi dữ liệu qua hợp đồng cố định, ghép ngược có kiểm soát hội tụ, và chạy lại được từng phần.
Mỗi tầng là một **mô-đun cắm được** sau một giao diện chung. Trong lúc triển khai, mọi tầng luôn có một bản chạy được (bản tạm → bản sâu), nên toàn hệ thống chạy được từ tuần đầu.

### 1.1 Sơ đồ hệ thống

```
                       ┌──────────────── StoreSpec (đầu vào duy nhất, store.yaml) ───────────────┐
                       │ mặt bằng W×H, cửa vào, thu ngân, khu tập kết, tường lạnh, w_min, w_cart, │
                       │ ngành hàng, dữ liệu đơn, tỉ lệ online, hồ sơ giờ, tập M, ε, hồ sơ ưu tiên │
                       └───────────────────────────────────┬──────────────────────────────────────┘
                                                           ▼
 ┌──────────── VÒNG NGOÀI (tầng 1 điều khiển) ─────────────────────────────────────────────────────┐
 │  θ ──► T1.build(θ) ──► ShelfLayout ──► kiểm tra khả thi (sức chứa ≥ Σ s_i, lối, liên thông)      │
 │                              │                                                                   │
 │                              ▼                                                                   │
 │                     T4.precompute(ShelfLayout, M) ──► FlowBank (E^μ mỗi mô hình; cache theo hash) │
 │                              │                                                                   │
 │   ┌──────── VÒNG TRONG (điểm bất động tầng 2 ↔ 3 ↔ 4) ───────────────────────────────────────┐   │
 │   │  T2.solve(layout, FlowBank, v, s, ε) ──► CategoryPlan (x, đoạn kệ, Z_P, Z_W^m, regret, C)│   │
 │   │  T3.solve(CategoryPlan, FlowBank)   ──► Planogram (món, tầng, cột, số mặt)              │   │
 │   │  T3.feedback(Planogram)             ──► SpaceDemand s_i, CategoryValue v_i ─────┐       │   │
 │   │  hội tụ? (Δv < tol và s không đổi) ── chưa ──► cập nhật MSA v, s ───────────────┘       │   │
 │   │  s_i vượt sức chứa ──► báo T1: θ không khả thi / tăng chiều dài kệ                     │   │
 │   └────────────────────────────────────────────────────────────────────────────────────────┘   │
 │                              ▼                                                                   │
 │   T4.evaluate(layout, plan, planogram, M, mức trung thực) ──► FlowReport                         │
 │   điểm(θ) = regret minimax tại ε (+ chạm mặt, lợi nhuận kệ) ──► successive halving / Bayes     │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                           ▼
                StoreDesign (bản vẽ kệ, bản đồ ngành, planogram, bản đồ nhiệt theo giờ,
                             đường + lịch nhặt, bảng chỉ số, manifest tái lập)
```

### 1.2 Hợp đồng dữ liệu giữa các tầng (`src/system/contracts.py`)

| Đối tượng | Sinh bởi | Dùng bởi | Nội dung chính |
|---|---|---|---|
| `StoreSpec` | người dùng (`store.yaml` / app) | tất cả | kích thước, điểm cố định, tham số vận hành, danh mục, nguồn dữ liệu, M, ε, hồ sơ ưu tiên, ngân sách tính toán |
| `ShelfLayout` | T1 | T2, T3, T4 | θ, lưới ô, `FloorPlan` (slot, đồ thị lối đi, D, d_in, d_0, lạnh, sức chứa ô), chiều dài lối |
| `FlowBank` | T4.precompute | T2, T3, T4 | với mỗi m ∈ M: ma trận đi qua E^m; hàm `exposure(x, m)`, `density(x, m, h)`, `aisle_direction(x, m)` |
| `CategoryPlan` | T2 | T3, T4, T1 | x / đoạn kệ mỗi ngành, Z_P, Z_W^m ∀m, regret, C^m, vi phạm ràng buộc |
| `Planogram` | T3 | T4, T2, T1 | mỗi ngành: (món, tầng, cột, số mặt), lợi nhuận kệ dự kiến |
| `SpaceDemand`, `CategoryValue` | T3.feedback | T2, T1 | s_i (số ô tối thiểu), v_i (giá trị ngành tính lại từ SSAP) |
| `FlowReport` | T4.evaluate | T1, đầu ra | bản đồ nhiệt (m, h), đường nhặt, lịch nhặt, chạm mặt (giải tích + mô phỏng), KPI |
| `StoreDesign` | bộ điều phối | app, xuất file | gói tất cả + `manifest` (config, git hash, sha dữ liệu, hạt giống) |

Quy ước bắt buộc (kiểm tra bằng test):
- Một hệ mã slot / ngành / món dùng xuyên suốt.
- Đơn vị thống nhất: m, s, đơn/giờ.
- **Một định nghĩa chỉ số duy nhất** ở `src/system/metrics.py`. Không tầng nào tự tính KPI riêng, tránh lệch định nghĩa giữa các thí nghiệm.

### 1.3 Ba vòng ghép và điều kiện hội tụ

| Vòng | Truyền gì | Cách lặp | Dừng khi |
|---|---|---|---|
| T4 → T2 (lưu lượng phụ thuộc sơ đồ) | định tuyến theo m (P^m đi qua, mật độ theo giờ) | GA: định tuyến ngay trong fitness. ILP (bài thay thế): tuyến tính hóa tuần tự + MSA, chấm lại bằng định tuyến | ‖Δe‖/‖e‖ < 2% hoặc 10 vòng, giữ nghiệm tốt nhất |
| T3 → T2 (giá trị và sức chứa ngành) | v_i, s_i | v ← v + (v' − v)/t (MSA); s_i chỉ tăng (đơn điệu) | ‖Δv‖/‖v‖ < 1% và s không đổi, tối đa 5 vòng |
| T4 → T3 (hướng dòng trong lối) | ψ_c | tính lại sau mỗi lần T2 đổi | theo vòng T3 → T2 |
| T3 → T1 (khả thi sức chứa) | Σ s_i so với sức chứa | θ không khả thi → loại hoặc sửa (kéo dài dãy) | — |

Báo cáo trong luận văn: số vòng thực tế, đường hội tụ, và **kiểm tra không phụ thuộc điểm xuất phát** (khởi tạo v từ giả định và từ ngẫu nhiên phải hội tụ về cùng sơ đồ hoặc chênh < 1%).

### 1.4 Mức trung thực (dùng chung cho mọi tầng)

| Mức | T2 | T3 | T4 | Dùng ở |
|---|---|---|---|---|
| L0 | GA 2 s | tham lam | chỉ giải tích | sàng lọc θ |
| L1 | GA 20 s + vòng ghép | heuristic + tìm kiếm cục bộ | giải tích + mô phỏng 5 lần | vòng giữa successive halving |
| L2 | GA 30 hạt giống (định tuyến, mẫu giỏ độc lập lớn) + ILP tuyến tính hóa tuần tự làm hạt giống khi n ≤ 15 | MIP | mô phỏng 30 lần, hạt giống độc lập | phương án cuối, thí nghiệm |

### 1.5 Bộ điều phối, bộ nhớ đệm, chạy lại từng phần
- `src/system/orchestrator.py`: `design(spec) -> StoreDesign` chạy toàn bộ; `rerun(design, changed=...)` chỉ chạy lại các tầng bị ảnh hưởng:

| Thay đổi | Chạy lại |
|---|---|
| mặt bằng, w_min | T1 → T4 (tất cả) |
| cố định kệ, đổi v_i / giá | T2 → T3 → T4 |
| đổi tập M / hiệu chỉnh hành vi | FlowBank → T2 (regret) → T3 → T4 |
| đổi tỉ lệ online / hồ sơ giờ | T2 (C^m) → T4 (lịch, chạm mặt) |
| chỉ chỉnh planogram một ngành | T3 (ngành đó) → T4 |

- `src/system/cache.py`: cache theo băm nội dung (FlowBank theo hash `ShelfLayout`, dữ liệu cấp món theo sha dữ liệu) → vòng ngoài không tính lại E^μ.
- **Chế độ cải tạo** (cửa hàng đang chạy): khóa T1 = mặt bằng hiện tại, T2 có ngân sách dời R (đã có), T3 có ngân sách đổi món → hệ thống dùng được cả cho thiết kế mới lẫn cải tạo.

### 1.6 Giao diện người dùng
- **CLI:** `python -m src.system design --spec store.yaml --fidelity L2 --out results/design_x/`.
- **App Streamlit** theo luồng: ① Đầu vào → ② Kệ → ③ Ngành hàng → ④ Món → ⑤ Luồng → ⑥ So sánh & quyết định.
  - Nút "Chạy toàn bộ" + khóa từng tầng để thử "nếu… thì…".
  - Mỗi thẻ hiển thị đầu vào nó nhận từ tầng trước.
- **Xuất:** JSON (`StoreDesign`), PNG/PDF bản vẽ, CSV planogram và lịch nhặt.

### 1.7 Kiểm chứng ở mức hệ thống
- **Test end-to-end** trên `case_minimart` và 1 mặt bằng nhỏ ở mức L0: phải chạy < 2 phút và luôn xanh sau mọi giai đoạn (CI cục bộ: `pytest -m e2e`).
- **Test hợp đồng:** mỗi tầng nhận/trả đúng kiểu, đúng mã, không vi phạm ràng buộc.
- **Tái lập:** cùng `store.yaml` + hạt giống → cùng `StoreDesign` (so băm).
- **Giá trị của tích hợp** (E19, E20): hệ thống tích hợp vs chạy tuần tự một lượt không ghép ngược.

---

## 2. Tầng 2 – Xếp ngành hàng hai luồng theo định tuyến, minimax regret (xương sống)

**Biến:** hoán vị π (≡ x_ik), giữ ràng buộc (4)(5)(8)(9). Mọi mục tiêu tính bằng **định tuyến** (mục 0.2). Dạng tuyến tính / QAP chỉ dùng cho ILP.

- **Nhặt đơn (min):** Z_P(π) = quãng đường nhặt kỳ vọng mỗi đơn, định tuyến từ khu tập kết qua các ngành trong đơn rồi về. Bài thay thế cho ILP: Σ f_ij·d_kl·x_ik·x_jl + Σ g_i·d_0k·x_ik.
- **Khách tại chỗ (max), mô hình m:** Z_W^m(π) = E_giỏ[Σ_k P^m(đi qua k | giỏ, π)·coef_π(k)·1{π(k) ∉ giỏ}].
- **Xung đột (min):** C^m(π) = Σ_h Σ_k λ_P(h)·pass_P(k|π)·λ_W(h)·pass_W^m(k|π)·dwell_k. Cả hai lượt đi qua lấy từ định tuyến; λ_P(h) từ `order_hour_of_day`.
- **Hướng 3:** NSGA-II trên (Z_P, −Z_W, C). ε-ràng buộc bằng GA có phạt; ILP ε-ràng buộc trên bài thay thế làm hạt giống và mốc so sánh.
- **Hướng 1:** min_π max_m (Z_W^m* − Z_W^m(π))/Z_W^m* s.t. Z_P(π) ≤ ε. Z_W^m* = giá trị tốt nhất đã biết (mục 0.4).
- **Mở rộng (từ tầng 3):** ngành i chiếm s_i ô liền nhau trên cùng một dãy kệ. GA: mã hóa hoán vị + giải mã tuần tự theo dãy (kiểu "space-filling decoding"). ILP: biến vị trí bắt đầu của đoạn kệ, chỉ dùng cho n nhỏ.

Giả định: giỏ khách tại chỗ cùng phân phối với đơn Instacart (+ độ nhạy kích thước giỏ); v_i giả định (+ độ nhạy).

---

## 3. Tầng 1 – Sinh bố trí kệ (làm sâu)

### 3.1 Bài toán
Cho: mặt bằng chữ nhật W×H, vị trí cửa vào / thu ngân / khu tập kết đơn online / tường lạnh. Tìm: phương án kệ θ (vị trí, hướng, độ dài, độ rộng lối đi). Mục tiêu: tốt nhất theo kết quả tầng 2 + tầng 4.

### 3.2 Không gian thiết kế (mẫu có tham số)
| Mẫu | Tham số θ | Ghi chú |
|---|---|---|
| Lưới (grid) | số dãy, độ dài dãy, độ rộng lối chính / lối phụ, số và vị trí lối cắt ngang, hướng dãy (dọc/ngang) | Mở rộng `grid_layout` |
| Đường vòng (racetrack) | số đảo, độ rộng vòng, độ dài đảo, khe giữa đảo | Mở rộng `racetrack_layout` |
| Đảo tự do (free) | số kệ, kích thước kệ, sinh vị trí bằng lấy mẫu Poisson-disk + sửa chữa | Thay seed ngẫu nhiên bằng tham số có kiểm soát |
| Lối chéo / xương cá (**mới**) | góc lối chéo, điểm giao | Ý từ kho hàng (Gue & Meller 2009), chưa thấy kiểm chứng cho cửa hàng có hai luồng |
| Lai | lưới phía trong + vòng chu vi | Phổ biến ở siêu thị |

### 3.3 Ràng buộc khả thi
- **Sức chứa:** tổng chiều dài kệ ≥ Σ_i s_i (từ tầng 3), ô lạnh ≥ nhu cầu ngành lạnh.
- **Lối đi:** độ rộng ≥ w_min. Lối có xe nhặt hàng ≥ w_cart (giả định 1,5–1,8 m, phân tích độ nhạy). Mọi ô kệ đến được từ cửa vào (kiểm tra liên thông đồ thị).
- **Thoát hiểm / tầm nhìn:** khoảng cách tối đa tới lối chính ≤ d_max (tham số).
- Kiểm tra hồi quy: bộ tham số mặc định tái tạo đúng 9 mặt bằng hiện có trong `data/floorplans/`.

### 3.4 Đánh đổi riêng của tầng 1 (câu hỏi nghiên cứu)
**Lối rộng ↔ chiều dài kệ:** lối rộng giảm chạm mặt và ùn tắc (tầng 4) nhưng bớt chiều dài kệ (giảm số mặt trưng bày ở tầng 3, giảm doanh thu). Đây là chỗ cả hai luồng cùng phụ thuộc vào hình học, chưa thấy nghiên cứu nào lượng hóa trên mặt bằng bán lẻ có nhặt đơn.

### 3.5 Thuật toán – tối ưu hai cấp, nhiều độ trung thực
- **Cấp ngoài (θ):** biến hỗn hợp rời rạc + liên tục.
  - Bước 1: lưới thô Latin Hypercube ~50 θ / mẫu.
  - Bước 2: **successive halving** – đánh giá tất cả bằng tầng 2 ngân sách thấp (GA 2 s, e giải tích), giữ 1/3, tăng ngân sách ×3, lặp.
  - Bước 3: tối ưu Bayes (Gaussian process, `scikit-optimize`) quanh vùng tốt, so với tìm ngẫu nhiên cùng ngân sách.
- **Cấp trong:** tầng 2 (GA minimax) + chỉ số nhanh tầng 4 (C giải tích, quãng đường nhặt ước lượng).
- **Hàm điểm cấp ngoài:** regret minimax tại ε cố định; báo cáo kèm mặt Pareto (Z_P, regret, chạm mặt).
- **Độ nhiễu:** dùng cùng hạt giống (CRN) giữa các θ; đo phương sai GA để biết khác biệt θ nào là thật.

### 3.6 Kiểm chứng
- Successive halving chọn trùng (top-3) với đánh giá đầy đủ trên tập nhỏ.
- Đường cong ngân sách – chất lượng: Bayes vs ngẫu nhiên vs lưới.

---

## 4. Tầng 3 – Xếp từng món trong kệ (làm sâu)

### 4.1 Bài toán: Shelf Space Allocation (SSAP) có vị trí
Mỗi ngành i đã có đoạn kệ (tầng 2): C_i cột × L tầng kệ. Mỗi món j trong ngành chọn: **số mặt trưng bày** n_j, **tầng kệ** l, **vị trí ngang** c (khối liền nhau).

**Cầu theo không gian (Corstjens & Doyle 1981; Drèze et al. 1994):**
D_j = a_j · n_j^β · φ_l · ψ_c
- a_j: cầu gốc = tần suất món j (Instacart `order_products__prior`).
- β: độ co giãn theo không gian (văn liệu ~0,1–0,2; phân tích độ nhạy).
- φ_l: hệ số tầng (ngang mắt > ngang tay > sát sàn; giả định từ văn liệu, độ nhạy).
- ψ_c: hệ số vị trí ngang, **lấy từ tầng 4**: lưu lượng qua đầu kệ theo hướng dòng khách.

**Mục tiêu (hai luồng ở cấp món):**
max Σ_j m_j·D_j  (lợi nhuận khách tại chỗ)  + γ·Σ_(j,j' kề) lift_jj'  (mua cùng)  − δ·Σ_j o_j·h_l  (thời gian với tay của người nhặt).
- o_j: tần suất món j trong đơn online. h_l: phạt tầng khó lấy (cúi/với).
- Xung đột ở cấp món: món bốc đồng muốn ngang mắt; món nhặt nhiều muốn ngang tay. Hai luồng tranh nhau "vùng vàng".

**Ràng buộc:**
- Σ_j độ rộng_j·n_j ≤ chiều dài tầng (độ rộng món giả định theo ngành, độ nhạy).
- n_min ≤ n_j ≤ n_max; n_j ≥ cầu ngày × số ngày tồn / sức chứa một mặt (đủ hàng, Hübner & Kuhn 2012).
- **Khối nhóm con:** món cùng aisle_id (nhóm con trong `groups_vn.csv`) nằm thành khối chữ nhật liền nhau (planogram blocking).
- Món lạnh chỉ vào kệ lạnh (đã kế thừa từ tầng 2).

### 4.2 Thuật toán
- **Hai cấp trong mỗi ngành:** (a) xếp khối nhóm con (aisle) trong đoạn kệ – bài gán nhỏ, vét cạn / ILP; (b) xếp món trong từng khối.
- **MIP chính xác** (PuLP + HiGHS) cho khối ≤ ~30 món: tuyến tính hóa n_j^β bằng biến nhị phân chọn mức n_j ∈ {1..n_max}; kề nhau bằng biến thứ tự cột.
- **Heuristic** cho khối lớn: tham lam theo lợi nhuận biên/cm → Hungarian gán (món, tầng) khi n_j đã cố định → tìm kiếm cục bộ (đổi chỗ, ±1 mặt).
- Các ngành độc lập → chạy song song.
- **Lọc dữ liệu:** 49.688 món; giữ món có ≥ 100 lượt mua, phần đuôi gộp thành "món khác" (báo % doanh số giữ lại).

### 4.3 Dữ liệu và giả định
| Cần | Nguồn | Xử lý |
|---|---|---|
| Tần suất món a_j, tần suất online o_j | `order_products__prior` | trực tiếp |
| Mua cùng cấp món lift_jj' | `order_products__prior` | ma trận thưa, chỉ cặp ≥ ngưỡng hỗ trợ |
| Nhóm con | `products.aisle_id` + `groups_vn.csv` | trực tiếp |
| Giá, lãi m_j | không có | giả định phân phối theo department; tùy chọn hiệu chỉnh từ dữ liệu có giá (vd. Dunnhumby – cần kiểm tra quyền dùng) |
| Độ rộng món, β, φ_l | không có | giả định từ văn liệu + phân tích độ nhạy |

### 4.4 Kiểm chứng
- MIP vs heuristic: khoảng cách tối ưu trên các khối nhỏ.
- So với baseline: xếp theo tần suất, theo bảng chữ cái, ngẫu nhiên, đều mặt.
- Độ nhạy β, φ_l, m_j: thứ hạng phương án có giữ không.

### 4.5 Ghép ngược
- s_i = ⌈Σ_j độ rộng_j·n_j^min / chiều dài ô⌉ → tầng 1 (sức chứa) và tầng 2 (ngành nhiều ô).
- Giá trị ngành v_i có thể **tính lại** = lợi nhuận tối ưu của SSAP trong ngành (thay giá trị giả định) → vòng lặp tầng 2 ↔ tầng 3 (1–2 vòng).

---

## 5. Tầng 4 – Luồng di chuyển (làm sâu)

Nói rõ trong bài: **luồng khách không ra lệnh được, chỉ dự đoán được.** Sơ đồ tạo ra luồng chứ không điều khiển luồng. **Luồng nhặt hàng thì tối ưu được thật** (đường đi + lịch).

### 5.1 Mô hình hành vi khách (dự đoán)
Mỗi mô hình = (luật thứ tự điểm dừng, luật chọn đường mỗi chặng, phản ứng với đông đúc):

| Mã | Thứ tự dừng | Đường mỗi chặng | Ghi chú |
|---|---|---|---|
| SP | TSP 2-opt | ngắn nhất | có sẵn |
| NN | gần nhất kế tiếp | ngắn nhất | có sẵn |
| SNK | rắn | rắn | có sẵn |
| RL-μ | NN | recursive logit (Fosgerau et al. 2013) | **mới** |
| RL-A | NN | RL + **sức hút** của kệ bốc đồng trên đường (lợi ích cạnh = −chi phí + η·hấp dẫn) | **mới**, nối với Z_W |
| PER | ưu tiên chu vi | RL | tái hiện "racetrack" (Larson et al. 2005) |
| SUE | — | RL + chi phí tăng theo mật độ, cân bằng ngẫu nhiên giải bằng MSA | **mới**, khách né chỗ đông |

### 5.2 Hiệu chỉnh mô hình bằng "sự thật cách điệu"
Không có quỹ đạo thật cho cửa hàng này → hiệu chỉnh tham số (μ, η, ưu tiên chu vi) bằng **khớp mô-men** với các chỉ tiêu công bố:
1. Đường đi dài hơn đường ngắn nhất trung bình ~28% (Lee et al.).
2. Tỉ lệ lối đi được ghé, ưu tiên chu vi (Larson, Bradlow & Fader 2005; Hui et al. 2009).
3. Phân phối thời gian chuyến.

Báo cáo độ khớp từng chỉ tiêu. Nếu tiếp cận được HRN4Customer hoặc bộ quỹ đạo công khai khác thì ước lượng hợp lý cực đại cho RL. **Việc đầu tiên: kiểm tra dữ liệu nào tải được thật.** Các mức μ chưa hiệu chỉnh được thì đưa vào tập M như một phần của sự bất định – đúng tinh thần hướng 1.

### 5.3 Tính luồng giải tích (nhanh, không nhiễu)
- Lưu lượng RL của chặng a→b chỉ phụ thuộc đồ thị lối đi → giải m+1 hệ tuyến tính **một lần cho mỗi mặt bằng**, lưu ma trận đi qua E^μ[a,b,k] (thưa).
- Với sơ đồ x: e_k^m(x) = Σ_giỏ Σ_chặng E^μ[a,b,k] → cắm vào `routing.RouteModel`.
- SUE: lặp MSA trên chi phí cạnh theo mật độ; kiểm tra hội tụ (khoảng cách tương đối).
- **Theo giờ:** cường độ khách tại chỗ λ_W(h) (giả định hồ sơ ngày; độ nhạy) và cường độ đơn online λ_P(h) từ `orders.order_hour_of_day`, `order_dow` → t_k^m(h) = λ_W(h)·e_k^m·thời gian dừng tại k.
- Kiểm thử: μ → ∞ trùng SP; điều kiện tồn tại (bán kính phổ < 1).

### 5.4 Nhặt đơn online (tối ưu thật)
- **Định tuyến từng đơn:**
  - Chính xác: Held-Karp (DP) cho đơn ≤ 12–15 điểm dừng, OR-Tools/LKH cho đơn lớn hơn.
  - Heuristic kho hàng: S-shape, largest gap, combined, **Ratliff–Rosenthal** (chính xác cho kho một khối – áp được cho mẫu lưới đơn khối).
  - So sánh: chênh lệch heuristic vs chính xác theo mẫu kệ.
- **Gộp đơn (batching):** xe nhặt chở B đơn (2–6). Thuật toán seed + tiết kiệm (Clarke–Wright) → tìm kiếm cục bộ; ràng buộc sức chứa xe.
- **Định tuyến tránh khách:** chi phí cạnh = độ dài + κ·mật độ khách t_k(h) → chuỗi đường cong (quãng đường, chạm mặt) theo κ. Đây là bài "hide and seek" đặt trên sơ đồ do mình sinh ra.
- **Lịch nhặt theo giờ (mới):** chia đơn vào các đợt (wave) theo giờ, thỏa khung giao hàng (giả định: đơn đặt giờ h phải xong trước h+Δ). Chọn giờ nhặt để min chạm mặt kỳ vọng Σ_h (số người nhặt trong giờ h)·mật độ khách(h). Đây là bài gán / luồng chi phí nhỏ nhất, giải chính xác.
- **Kiểm chứng xấp xỉ của tầng 2:** tương quan giữa Z_P (tổng theo cặp) và quãng đường nhặt thật (TSP + batching) trên nhiều sơ đồ → chứng minh Z_P là đại diện hợp lệ (như cách v3 đã kiểm chứng Z1).

### 5.5 Chạm mặt và ùn tắc
- **Giải tích:** chạm mặt kỳ vọng = Σ_h Σ_k ρ_W(k,h)·ρ_P(k,h)·thời gian chồng lấn (giả định Poisson).
- **Mô phỏng tác tử:** mở rộng `simulate.py` – khách (theo mô hình m) + người nhặt (theo đường đã tối ưu) cùng chạy theo thời gian; ô có sức chứa → chờ khi đông (hàng đợi đơn giản). Đếm chạm mặt, thời gian chờ, ùn tắc.
- So khớp giải tích vs mô phỏng (Spearman, sai số tương đối).

### 5.6 Đầu ra
Bản đồ nhiệt khách theo giờ (giải tích + mô phỏng, từng m), đường nhặt từng đợt, lịch nhặt, bản đồ điểm nóng chạm mặt.

---

## 6. Các giai đoạn triển khai

Nguyên tắc: **dựng khung hệ thống chạy được trước** (GĐ1), mỗi tầng ban đầu là bản tạm dùng code đã có. Các giai đoạn sau **thay từng bản tạm bằng bản sâu** phía sau cùng giao diện. Test end-to-end phải luôn xanh, nên lúc nào cũng có một hệ thống hoàn chỉnh để demo và lấy số liệu.

| GĐ | Nội dung | Tầng tạm → sâu | Ngày công |
|---|---|---|---|
| 0 | Sao lưu v3; kiểm chứng trích dẫn (mục 11); kiểm tra dữ liệu quỹ đạo (HRN4Customer…) và dữ liệu giá; thêm khu tập kết d_0k | — | 1,5 |
| 1 | **Khung hệ thống:** `contracts`, `metrics`, `orchestrator`, `cache`, `store.yaml`, CLI, test e2e + test hợp đồng. Bản tạm: T1 = `layouts.PRESETS`, T2 = GA hiện có, T3 = xếp món theo tần suất, T4 = `routing` + `simulate` hiện có | khung | 4 |
| 2 | **Tầng 2 hai luồng** (mode `twoflow_eps`, NSGA-II 3 mục tiêu, test vét cạn) | T2 | 3 |
| 3 | **Tầng 4a – hành vi:** RL giải tích, E^μ, RL-A, PER, SUE; hiệu chỉnh khớp mô-men; `FlowBank` + cache | T4 (dự đoán) | 5 |
| 4 | **Tầng 2 hướng 1:** RouteModel tổng quát cho M (P^m, đánh giá chênh lệch ≤ 3 ms), Z_W^m* tốt nhất đã biết, ma trận L (có hàng "mô hình tuyến tính"), GA minimax, ILP tuyến tính hóa tuần tự | T2 | 3 |
| 5 | **Tầng 4b – nhặt đơn:** Held-Karp/OR-Tools, S-shape/largest gap/Ratliff–Rosenthal, batching, tránh khách, lịch theo giờ | T4 (nhặt) | 5 |
| 6 | **Tầng 4c – mô phỏng hai luồng** theo thời gian, chạm mặt giải tích vs mô phỏng | T4 (đánh giá) | 3 |
| 7 | **Tầng 3:** dữ liệu cấp món, MIP SSAP + heuristic, khối nhóm con, planogram, `feedback` s_i, v_i | T3 | 5 |
| 8 | **Bật vòng ghép:** ngành nhiều ô (giải mã GA), vòng T2 ↔ T3 ↔ T4, kiểm tra hội tụ + không phụ thuộc điểm xuất phát | hệ thống | 3 |
| 9 | **Tầng 1:** không gian tham số 5 mẫu, khả thi, successive halving qua L0/L1/L2 + Bayes, hồi quy 9 mặt bằng | T1 | 5 |
| 10 | `rerun` từng phần, chế độ cải tạo, app 6 thẻ theo luồng, xuất file | hệ thống | 3 |
| 11 | Thí nghiệm E8–E20 (chạy nền), README, test (mục tiêu ≥ 45) | — | 5 |
| 12 | Viết Chương 3 (phương pháp + kiến trúc hệ thống) và Chương 4 (kết quả) | — | — |

**Tổng ~45 ngày công.** Phụ thuộc: GĐ1 trước tất cả. GĐ2 → 4 (lõi tầng 2). GĐ3 → 4, 5, 6. GĐ7 → 8. GĐ9 cần GĐ8 (vòng ngoài gọi cả vòng trong).

Mỗi giai đoạn chỉ coi là xong khi:
1. test e2e + test hợp đồng xanh;
2. có chế độ `quick` trong `config.yaml`;
3. đạt tiêu chí kiểm chứng của tầng đó (mục 3.6, 4.4, 5.2–5.5, 1.7).

---

## 7. Thí nghiệm

| Mã | Tầng | Câu hỏi | Kết quả chính |
|---|---|---|---|
| E1, E2 (giữ) | 2 | GA cài đúng? | QAPLIB, GA vs ILP |
| **E8** | 2+4 | Tối ưu theo A mất bao nhiêu khi chấm bằng B? | Ma trận L[A,B], 7 mô hình hành vi + hàng "mô hình tuyến tính/QAP" × 3 mặt bằng; mọi ô chấm bằng định tuyến (E5 cũ là một ô) |
| **E9** | 2 | Đánh đổi nhặt đơn ↔ doanh thu ngẫu hứng | Pareto NSGA-II theo định tuyến (n = 15, 40); so với Pareto ILP của bài thay thế sau khi chấm lại bằng định tuyến |
| **E10** | 2 | Giá của tính bền vững | Minimax vs tối ưu riêng vs kỳ vọng |
| **E11** | 4 | Hiệu chỉnh mô hình hành vi | Độ khớp sự thật cách điệu; độ nhạy μ, η |
| **E12** | 4 | Định tuyến nhặt hàng | Chính xác vs S-shape/largest gap/R–R theo mẫu kệ; lợi ích batching; tương quan Z_P ↔ quãng đường thật |
| **E13** | 4 | Xung đột hai luồng | Chạm mặt theo κ (tránh khách), theo lịch nhặt, tỉ lệ đơn online 10/30/50%; giải tích vs mô phỏng |
| **E14** | 3 | Xếp món | MIP vs heuristic vs baseline; độ nhạy β, φ, m; tranh chấp "vùng vàng" giữa hai luồng |
| **E15** | 2↔3 | Vòng ghép | Thay v_i giả định bằng v_i từ SSAP, ngành nhiều ô: sơ đồ thay đổi bao nhiêu |
| **E16** | 1 | Mẫu kệ nào bền nhất? | Regret tốt nhất theo mẫu (5 mẫu) |
| **E17** | 1 | Lối rộng ↔ chiều dài kệ | Mặt đánh đổi chạm mặt – lợi nhuận kệ theo độ rộng lối |
| **E18** | tất cả | Bền ngoài mẫu | Bỏ-một-mô-hình + mô hình ngoài M; chống "tự chấm điểm" |
| **E19** | hệ thống | Giá trị của tích hợp | Tích hợp (có vòng ghép) vs tuần tự một lượt vs tối ưu từng tầng riêng: regret, Z_P, chạm mặt, lợi nhuận kệ |
| **E20** | hệ thống | Nghiên cứu tình huống end-to-end | `case_minimart` + 1 mặt bằng thiết kế mới: chạy toàn bộ L2, chế độ cải tạo với R; hội tụ vòng ghép, thời gian chạy từng tầng |
| E6' | tất cả | Độ nhạy | v, β, φ, kích thước giỏ, hồ sơ giờ, w_cart |

Thống kê: 30 hạt giống; Wilcoxon + Holm, A12, Friedman (như `extra_stats.py`).

---

## 8. Rủi ro & cách xử lý

| Rủi ro | Xử lý |
|---|---|
| Không có quỹ đạo thật để hiệu chỉnh tầng 4 | Khớp mô-men với sự thật cách điệu; phần chưa chắc đưa vào M (hướng 1 sinh ra để xử lý đúng điều này) |
| Không có giá, độ rộng món (tầng 3) | Giả định + độ nhạy; báo cáo kết luận nào **không đổi** theo giả định |
| Tầng 1 tốn tính toán (mỗi θ chạy tầng 2) | Successive halving + e giải tích; ước lượng: 5 mẫu × 50 θ × 2 s ≈ 10 phút cho vòng đầu |
| SUE / vòng lặp e_k không hội tụ | MSA + giữ tốt nhất; báo cáo khoảng hội tụ |
| Định tuyến |M| mô hình làm GA chậm | Đánh giá chênh lệch, mẫu giỏ theo mức trung thực, song song theo m; ILP chỉ dùng ở n ≤ 15 |
| Regret bị một mô hình chi phối | Báo cáo có/không có mô hình đó + regret kỳ vọng |
| Bị phản biện "tối ưu theo chính mô phỏng" | E18; nêu rõ số % là tương đối |
| Khối lượng lớn (~45 ngày) | Lõi GĐ1–3 xong trước; mỗi tầng có bản `quick` chạy được độc lập để không chặn viết bài |

---

## 9. Đóng góp dự kiến

1. **Tầng 2:** ma trận thiệt hại khi mô hình hành vi sai + bố trí minimax regret **theo định tuyến** (GA, ILP tuyến tính hóa tuần tự làm hạt giống), có định lượng cái giá của xấp xỉ tuyến tính; xếp ngành hai luồng nhặt đơn – khách tại chỗ – xung đột.
2. **Tầng 4:** recursive logit giải tích (có sức hút, có né đông) đặt trong vòng tối ưu, không cần huấn luyện lại cho sơ đồ mới; hiệu chỉnh bằng sự thật cách điệu; lịch nhặt theo giờ giảm chạm mặt.
3. **Tầng 3:** SSAP hai luồng ở cấp món (tranh chấp vùng vàng giữa món bốc đồng và món hay nhặt), ghép ngược sức chứa và giá trị ngành lên tầng 1–2.
4. **Tầng 1:** so sánh độ bền của mẫu kệ trước bất định hành vi; đánh đổi độ rộng lối – chiều dài kệ dưới hai luồng; tối ưu hai cấp nhiều độ trung thực.
5. **Hệ thống tích hợp:** bốn tầng nối bằng hợp đồng dữ liệu, có vòng ghép ngược hội tụ (MSA), mức trung thực dùng chung, chạy lại từng phần. Dùng cho cả thiết kế mới và cải tạo. E19 chứng minh tích hợp tốt hơn chạy tuần tự.

---

## 10. Danh sách file

| File | Tầng | Việc |
|---|---|---|
| `src/system/contracts.py` (mới) | hệ thống | `StoreSpec`, `ShelfLayout`, `FlowBank`, `CategoryPlan`, `Planogram`, `SpaceDemand`, `CategoryValue`, `FlowReport`, `StoreDesign` |
| `src/system/metrics.py` (mới) | hệ thống | định nghĩa KPI duy nhất |
| `src/system/orchestrator.py` (mới) | hệ thống | `design`, `rerun`, vòng ngoài/vòng trong, hội tụ, mức trung thực |
| `src/system/cache.py` (mới) | hệ thống | cache theo băm nội dung |
| `src/system/export.py`, `src/system/__main__.py` (mới) | hệ thống | xuất JSON/PNG/CSV, CLI |
| `specs/*.yaml` (mới) | hệ thống | `store.yaml` mẫu cho 9 mặt bằng + minimart |
| `tests/test_e2e.py`, `tests/test_contracts.py` (mới) | hệ thống | e2e L0, hợp đồng, tái lập |
| `src/pipeline.py` | — | giữ cho v3; hệ thống mới gọi lại các hàm hiệu chỉnh/mô phỏng |
| `src/floorplan.py` | 1 | khu tập kết d_0k, sức chứa ô, kiểm tra liên thông |
| `src/layouts.py` | 1 | không gian tham số 5 mẫu (thêm lối chéo, lai), kiểm tra khả thi |
| `src/outer.py` (mới) | 1 | LHS, successive halving, Bayes |
| `src/model_ilp.py` | 2 | mode `twoflow_eps`, `minimax` (τ) |
| `src/twoflow.py` (mới) | 2 | dựng instance hai luồng, ngành nhiều ô |
| `src/robust.py` (mới) | 2 | Z_W^m* tốt nhất đã biết, ma trận L, GA minimax, ILP tuyến tính hóa tuần tự |
| `src/ga.py`, `src/moo.py` | 2 | fitness minimax, 3 mục tiêu, giải mã nhiều ô |
| `src/items.py` (mới) | 3 | dữ liệu cấp món: tần suất, lift thưa, lọc, giả định giá/độ rộng |
| `src/slotting.py` (mới) | 3 | MIP SSAP, heuristic, khối nhóm con, planogram |
| `src/behavior.py` (mới) | 4 | RL, RL-A, PER, SUE, E^μ, hiệu chỉnh khớp mô-men |
| `src/routing.py` | 2, 4 | RouteModel tổng quát: thứ tự dừng theo m, xác suất đi qua P^m, định tuyến nhặt đơn từ khu tập kết, đánh giá chênh lệch |
| `src/picking.py` (mới) | 4 | Held-Karp/OR-Tools, S-shape, largest gap, R–R, batching, tránh khách |
| `src/schedule.py` (mới) | 4 | hồ sơ giờ, lịch nhặt theo đợt |
| `src/simulate.py` | 4 | tác tử nhặt hàng, theo thời gian, sức chứa ô, chạm mặt |
| `src/viz.py` | 1–4 | planogram, bản đồ nhiệt theo giờ, đường nhặt, mặt đánh đổi |
| `experiments/e8…e18*.py`, `run_all.py`, `config.yaml` | — | thí nghiệm |
| `app/streamlit_app.py` | — | thẻ cho từng tầng |
| `tests/test_core.py` (+ file test theo tầng) | — | RL μ→∞ = SP, ILP vs vét cạn, Held-Karp vs vét cạn, SSAP MIP vs vét cạn, hồi quy 9 mặt bằng |

Thư viện thêm: `ortools`, `scikit-optimize` (hoặc `optuna`), `scipy.sparse`.

---

## 11. Tài liệu cần kiểm chứng trước khi trích dẫn

| Tài liệu | Dùng cho |
|---|---|
| Lee et al. (AAMAS 2026) – lệch 28%, RL theo sơ đồ | Tầng 2, 4 |
| Flamand et al. – bố trí cửa hàng + mô hình lưu lượng | Tầng 2 |
| "Playing hide and seek…" – nhặt hàng tránh khách | Tầng 4 |
| Fosgerau, Frejinger & Karlström (2013) – recursive logit | Tầng 4 |
| Larson, Bradlow & Fader (2005); Hui, Fader & Bradlow (2009) – quỹ đạo khách siêu thị | Tầng 4 (hiệu chỉnh) |
| Corstjens & Doyle (1981); Drèze, Hoch & Purk (1994) – co giãn không gian, hiệu ứng vị trí | Tầng 3 |
| Hübner & Kuhn (2012); Bianchi-Aguiar et al. (review SSAP) | Tầng 3 |
| Ratliff & Rosenthal (1983); Roodbergen & de Koster (2001) – định tuyến nhặt hàng | Tầng 4 |
| Gue & Meller (2009) – lối chéo / xương cá | Tầng 1 |
| Ozgormus & Smith; Botsali & Peters (2005) – bố trí cửa hàng | Tầng 1 |
| HRN4Customer – kiểm tra tồn tại và giấy phép | Tầng 4 |
