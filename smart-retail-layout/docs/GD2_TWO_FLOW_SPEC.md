# GĐ2 — Đặc tả mô hình hai luồng

## 1. Phạm vi

GĐ2 bổ sung một evaluator tách biệt cho luồng nhặt đơn trực tuyến và giá trị mua ngẫu hứng trong cửa hàng. Mã QAP/NSGA-II hai mục tiêu v3 và tuyến chạy GĐ1 tiếp tục giữ nguyên. GĐ2 chỉ được gọi qua API T2 opt-in.

Mô hình hiện xét một kịch bản danh nghĩa. Bất định nhiều kịch bản/minimax thuộc GĐ4. NSGA-II tạo xấp xỉ Pareto; kết quả của nó không phải chứng minh tối ưu toàn cục.

## 2. Tập chỉ số và ánh xạ

- `i, j`: ngành hàng, theo thứ tự `category_ids`.
- `k, l`: slot trưng bày, theo thứ tự `slot_ids`.
- `h`: khoảng thời gian, theo thứ tự `period_ids`.
- Một nghiệm là hoán vị `perm` dài `m`, theo quy ước v3: `perm[k] = i` là ngành hàng đặt tại slot `k`. Chỉ số từ `n` đến `m-1` là ngành rỗng. Mỗi chỉ số `0..m-1` phải xuất hiện đúng một lần.
- Các vector/hai ma trận phải căn theo đúng các ID trên; không tự ghép bằng thứ tự từ một nguồn khác.

## 3. Mục tiêu thô

### 3.1 Quãng đường nhặt đơn trực tuyến — tối thiểu hóa

`Z_P = Σ_i Σ_j f_ij d_{k_i,k_j} + Σ_i g_i d_{0,k_i}`

Trong đó `k_i` là slot được gán cho ngành `i`; `f_ij` là lưu lượng có hướng từ ngành `i` sang `j`; `d_{kl}` là khoảng cách đi bộ giữa hai slot; `g_i = Σ_h g_i(h)` là lưu lượng nhặt ngành `i`; `d_{0k}` là khoảng cách từ điểm tập kết đơn trực tuyến được cấu hình tường minh tới slot `k`.

- `pair_flow`: ma trận `n × n`, đơn vị demand/interval (đơn vị demand phải nhất quán với `period_category_flow`).
- `distance_m`: ma trận `m × m`, đơn vị mét.
- `period_category_flow`: ma trận `H × n`; tổng theo `h` tạo `g_i`.
- `pickup_distance_m`: vector `m`, đơn vị mét. Chỉ lấy từ pickup point cấu hình; không thay bằng cửa vào.
- `Z_P` có đơn vị demand·m theo đơn vị lưu lượng nguồn.

`period_category_flow[h,i]` là **số lượt nhặt trong toàn khoảng thời gian** `h`, không phải tốc độ. Nếu nguồn cung cấp lượt/giờ, phải nhân với độ dài khoảng thời gian trước khi tạo input. `pair_flow` phải dùng cùng quy ước số lượt theo khoảng thời gian để các hạng trong `Z_P` có cùng đơn vị.

### 3.2 Giá trị mua ngẫu hứng — tối đa hóa

`Z_W = Σ_i v_i q_i e_{k_i}`

- `category_value[i] = v_i`: giá trị quy đổi của ngành hàng. Mặc định unit là `model_value_unit` để không khẳng định đây là tiền thật; chỉ gắn unit tiền tệ khi nguồn dữ liệu và provenance hỗ trợ điều đó.
- `impulse_probability[i] = q_i`: xác suất/hệ số mua ngẫu hứng trong `[0, 1]`.
- `slot_exposure[k] = e_k`: mức tiếp xúc theo một mô hình được ghi provenance, không mặc định là số đã hiệu chỉnh thực tế.
- `Z_W` có đơn vị `value_unit` trên đơn vị demand; chỉ mang nghĩa tiền tệ khi unit và provenance của nguồn xác nhận.

### 3.3 Xung đột giữa picker — tối thiểu hóa

`C = Σ_h Σ_i g_i(h) t_{k_i}(h)`

- `slot_conflict[h,k] = t_k(h)`: chỉ số xung đột của slot `k` tại khoảng `h`, trên một đơn vị picker demand.
- `C` có đơn vị theo chỉ số và quy ước của bộ đo xung đột. Đơn vị/định nghĩa cụ thể phải được ghi trong cấu hình và provenance; không so sánh C giữa hai nguồn có thang đo khác nhau.

## 4. Chiều mục tiêu và chuẩn hóa cho solver

Evaluator luôn trả về giá trị thô `(Z_P, Z_W, C)` và đơn vị. NSGA-II giảm thiểu vector:

`F = (Z_P / s_P, -Z_W / s_W, C / s_C)`

`s_P`, `s_W`, `s_C` là các scale hữu hạn, dương, cố định theo cấu hình/hệ số. Không tính scale từ min/max của quần thể hiện tại. Epsilon luôn so sánh với giá trị thô, không so sánh F đã chuẩn hóa.

## 5. Ràng buộc

Một nghiệm khả thi phải thỏa đồng thời:

1. Hoán vị hợp lệ, không lặp/thiếu index.
2. Tương thích ngành hàng–slot (`allowed`) nếu hợp đồng cung cấp ma trận này.
3. Các vị trí cố định.
4. Khoảng cách tối thiểu của từng cặp cần tách.
5. Ngân sách di dời, nếu cấu hình hữu hạn.
6. Trong `twoflow_eps`: `Z_P ≤ ε_P` và, nếu cấu hình, `C ≤ ε_C`.

Solver dùng ràng buộc feasibility-first. Epsilon là ràng buộc solver thật, không chỉ là bộ lọc sau khi lấy mặt Pareto. Nếu thuật toán không tìm thấy nghiệm khả thi, trạng thái trả về là `NO_FEASIBLE_CANDIDATE_FOUND`; riêng trường hợp dữ liệu bắt buộc thiếu là `INPUT_NOT_READY`. Không tuyên bố bài toán vô nghiệm chỉ từ một lượt tìm kiếm heuristic.

## 6. Dữ liệu và provenance

Mỗi ma trận/vector bắt buộc phải đi kèm provenance gồm nguồn, lớp dữ liệu và SHA-256 nếu có. Input hash bao gồm ID, hệ số, đơn vị, ràng buộc và provenance để có thể tái lập/kiểm toán.

`data_class` phải thuộc danh mục: `observed`, `measured`, `processed`, `processed_raw_unverified`, `derived`, `assumption`, `model_proxy`, `layout_proxy`, `simulated_proxy`, `synthetic_fixture`, hoặc `synthetic_test`. Hai lớp synthetic phải được ghi nhãn rõ ràng; fixture GĐ2 chỉ dùng trong kiểm thử.

Các đầu vào bắt buộc cho đánh giá ba mục tiêu gồm `pair_flow`, `period_category_flow`, `distance_m`, `pickup_distance_m`, `category_value`, `impulse_probability`, `slot_exposure` và `slot_conflict`. Nếu thiếu dữ liệu hoặc provenance, API báo mã `INPUT_NOT_READY` kèm trường cần bổ sung. Không dùng 0, khoảng cách cửa vào, synthetic fixture hay proxy không gắn nhãn để lấp dữ liệu thiếu. Synthetic chỉ hợp lệ trong kiểm thử ghi rõ `synthetic_test`.

GĐ0/GĐ1 đã hỗ trợ tính `d_0k` khi layout khai báo `pickup_cell`; nếu không khai báo, GĐ2 vẫn chưa sẵn sàng. GĐ1 chưa tạo luồng xung đột theo giờ và chưa có mô hình giá trị/tiếp xúc đã được xác thực; vì vậy hiện chưa có bộ dữ liệu sản xuất đầy đủ để chạy T2 có ý nghĩa.

## 7. Chế độ tối ưu

### 7.1 `nsga2_3obj` (NSGA-II ba mục tiêu)

Tìm mặt Pareto xấp xỉ của `(Z_P, -Z_W, C)` dưới các ràng buộc nghiệp vụ. Lưu seed, cấu hình, điều kiện dừng, số lần đánh giá, input hash, giá trị thô, vector chuẩn hóa, feasibility, vi phạm và ánh xạ category–slot. Sắp kết quả xác định; khi trùng objective giữ hoán vị từ điển nhỏ nhất.

Artifact ghi `objective_scales` để kiểm tra lại vector chuẩn hóa, cùng `n_feasible`, `n_infeasible`, `n_pareto`. Ba bộ đếm mô tả quần thể cuối cùng của heuristic; riêng oracle vét cạn đếm trên toàn bộ hoán vị. `n_pareto` là 0 ở chế độ epsilon vì chế độ này chọn một phương án, không tạo Pareto front.

Ánh xạ `floorplan_slot_indices` gắn từng slot đầu vào với vị trí trong danh sách slot đầy đủ của floorplan. Nếu không có nguồn gốc layout để xác định chỉ số này, artifact để `null`; không suy diễn ánh xạ đồng nhất. Validator đối chiếu từng assignment với ánh xạ gốc.

### 7.2 `twoflow_eps`

Tối đa hóa `Z_W` dưới `Z_P ≤ ε_P`, tùy chọn `C ≤ ε_C`, và ràng buộc nghiệp vụ. Đây là bài toán một mục tiêu có ràng buộc thật. Nếu nhiều nghiệm cùng `Z_W`, chọn theo `Z_P` thấp nhất, rồi `C` thấp nhất, rồi hoán vị từ điển nhỏ nhất.

## 8. Đối chiếu v3

Chỉ so sánh một hạng mục khi ID, thứ tự index, đơn vị và hệ số tương ứng khớp. Hạng khoảng cách cặp có thể đối chiếu với hạng quadratic `z1`; hạng mua ngẫu hứng có thể đối chiếu `z2` khi `v_i q_i` và `e_k` giống nhau. Hạng pickup `g_i d_0k` không được gọi tương đương với lượt đi từ cửa vào/ra quầy của v3 nếu các vector và ý nghĩa vật lý khác nhau. Mục tiêu C không có đối ứng trong v3.

## 9. Ví dụ kiểm tra tính tay (synthetic test-only)

Hai ngành `A`, `B`; hai slot `K1`, `K2`; một khoảng thời gian. Dùng `f_AB=1`, `f_BA=2`, `d_12=d_21=1`; `g_A=1`, `g_B=2`; `d_0K1=4`, `d_0K2=1`; `v=(8,20)`, `q=(0.4,0.4)`, `e=(1,0.25)`; `t=(1,3)`.

| Gán | `Z_P` (min) | `Z_W` (max) | `C` (min) |
|---|---:|---:|---:|
| A→K1, B→K2 | 9 | 5.2 | 7 |
| A→K2, B→K1 | 12 | 8.8 | 5 |

Hai nghiệm không trội lẫn nhau. `ε_P=10` chỉ cho nghiệm thứ nhất. `ε_P=12, ε_C=6` chỉ cho nghiệm thứ hai. `ε_P=10, ε_C=6` không có nghiệm khả thi trong tập hai hoán vị. Đây là fixture tổng hợp để kiểm tra công thức, không phải kết quả cửa hàng thật.

## 10. Giới hạn

- Hoán vị giữ mô hình một ngành cho một slot; ngành nhiều ô/slotting thuộc công việc tiếp theo.
- Mô hình chỉ có một scenario; perturbation/minimax thuộc GĐ4.
- Pareto NSGA-II là gần đúng và phụ thuộc seed/cấu hình.
- Kết quả không đại diện cho hiệu quả thực tế của picker cho tới khi quy tắc lưu lượng, xung đột, xác suất/giá trị và provenance được xác thực.
