# Danh sách công việc chi tiết — Smart Retail Floor Allocation v4

Tài liệu này chuyển các giai đoạn trong [kế hoạch v4](KE_HOACH_v4.md) thành backlog có thể triển khai và nghiệm thu. Checkbox chưa đánh dấu nghĩa là chưa xác nhận hoàn thành.

## 1. Cách sử dụng

- Làm theo phụ thuộc ở mục 4; chỉ chạy song song các nhánh đã nêu.
- Chỉ đánh dấu hoàn tất khi đạt tiêu chí riêng của gói việc và Definition of Done ở mục 6.
- Giữ lại các lớp v3 đang hoạt động; chỉ thay thế khi có yêu cầu và kiểm chứng tương đương.
- Với dữ liệu hoặc quyết định nghiệp vụ còn thiếu, ghi giả định, nguồn và ảnh hưởng. Không xem dữ liệu giả lập là dữ liệu thực.

## 2. Hiện trạng mã nguồn dùng để lập backlog

Các khối nền hiện có thể tái sử dụng:

- src/floorplan.py, src/layouts.py: mô hình mặt bằng, đồ thị và kiểu bố trí.
- src/instance.py, src/params.py: cấu hình bài toán, ràng buộc, tạo instance.
- src/fastops.py, src/ga.py, src/model_ilp.py, src/moo.py, src/solvers.py: phép tính nhanh, GA, ILP, tối ưu đa mục tiêu và bộ giải.
- src/routing.py, src/simulate.py, src/pipeline.py, src/relocation.py, src/viz.py: định tuyến, mô phỏng, pipeline, di dời và trực quan hóa.
- app/streamlit_app.py: giao diện Streamlit có 5 tab.
- tests/test_core.py: bộ kiểm thử lõi hiện có.
- Thí nghiệm hiện có đến E1–E7.

Trong cây mã nguồn đã rà soát chưa thấy các mô-đun v4 src/system/, outer.py, twoflow.py, robust.py, items.py, slotting.py, behavior.py, picking.py, schedule.py; cũng chưa thấy test_contracts.py, test_e2e.py hoặc kịch bản E8–E20. Các mục này được xem là phần cần xây dựng, không ghi nhận là đã có.

src/params.py trỏ tới data/raw/instacart, nhưng thư mục dữ liệu thô này chưa có trong cây đã kiểm tra. Dữ liệu hiện thấy gồm mặt bằng, mapping, các mảng/metadata/basket đã xử lý và bộ QAPLIB; chưa thấy bộ dữ liệu cấp SKU đủ cho tầng SSAP. Đây là điều kiện chặn cần giải quyết trước khi nghiệm thu kết quả item-level trên dữ liệu thực.

## 3. Backlog theo giai đoạn

### GĐ0 — Đóng băng baseline v3 và xác nhận dữ liệu

**Mục tiêu:** Có mốc so sánh an toàn trước khi thêm tầng v4; thống nhất nguồn dữ liệu và giả định.

#### 0A. Lưu baseline và kiểm kê khả năng tái sử dụng

- [ ] Ghi lại commit, trạng thái Git, lệnh chạy, cấu hình và kết quả baseline của pipeline hiện hành.
- [ ] Lưu đầu ra baseline vào vị trí kế hoạch v4 quy định; không ghi đè dữ liệu/nghiệm cũ.
- [ ] Lập bảng đối chiếu module v3 → trách nhiệm v4, bắt đầu từ các module ở mục 2.
- [ ] Ghi thời gian chạy, seed, instance, objective và môi trường của từng phép đo.

**Nghiệm thu:** Baseline chạy lại được theo hướng dẫn; đầu ra có cấu hình, seed, mã instance; có danh sách thành phần giữ nguyên/tích hợp/mở rộng.

#### 0B. Chốt sổ dữ liệu và giả định nghiệp vụ

- [ ] Kiểm kê nguồn và trường dữ liệu cho basket, lượt khách, SKU/sản phẩm, giá trị hoặc biên lợi nhuận, kích thước mặt hàng, năng lực kệ và mặt bằng.
- [ ] Xác minh nguồn dự kiến cho Instacart và điều kiện sử dụng; chỉ bổ sung dữ liệu thô khi có nguồn hợp lệ.
- [ ] Lập data dictionary: tên trường, đơn vị, kiểu, khóa, nguồn, quy tắc thiếu/trùng và bước xử lý.
- [ ] Ghi nhận trường còn thiếu như giá/biên lợi nhuận, dung lượng và chiều rộng SKU; phân biệt giả định phục vụ thử thuật toán với dữ liệu thực.
- [ ] Xác định cách truy vết đầu ra về instance, dữ liệu, cấu hình và phiên bản mã.

**Nghiệm thu:** Có data dictionary và bảng giả định; dữ liệu giả lập được gắn nhãn; thiếu dữ liệu chặn được hiển thị trước khi chạy tầng liên quan.

#### 0C. Kiểm chứng nguồn nghiên cứu và dữ liệu mặt bằng

- [ ] Kiểm tra khả năng truy cập, trích dẫn và quyền sử dụng các nguồn dự kiến cho luận văn, gồm HRN4Customer và dữ liệu giá; ghi lại kết quả kiểm tra, không mặc định nguồn đã tải được.
- [ ] Bổ sung khu tập kết online d_0 vào readiness của floorplan và vector khoảng cách d_0k từ khu tập kết tới từng slot k; kiểm thử đây là đầu vào của objective picker.
- [ ] Ghi rõ nguồn nào đã xác minh, nguồn nào chỉ mới là dự kiến và ảnh hưởng nếu nguồn không truy cập được.

**Nghiệm thu:** Nguồn dữ liệu/trích dẫn có trạng thái và quyền sử dụng được ghi lại; floorplan nhỏ có d_0k hợp lệ hoặc trả lỗi cấu hình rõ ràng.

### GĐ1 — Hợp đồng hệ thống và vertical slice

**Mục tiêu:** Thiết lập giao diện ổn định giữa các tầng và chứng minh một luồng chạy đầu-cuối nhỏ nhất.

#### 1A. Định nghĩa hợp đồng đầu vào/đầu ra

- [x] Mô tả schema cho case, layout, demand/basket, sản phẩm, cấu hình solver, metric và kết quả.
- [x] Định nghĩa ID ổn định, đơn vị đo, phiên bản schema và biểu diễn lỗi/thiếu dữ liệu.
- [x] Quy định hợp đồng giữa tầng tạo layout, T2, T3, T4 và bộ điều phối.
- [x] Viết contract tests cho dữ liệu hợp lệ, thiếu trường bắt buộc, sai đơn vị/kiểu và trường hợp biên.

#### 1B. Chuẩn hóa metric và kết quả

- [x] Định nghĩa StoreDesign/result chung gồm case ID, trạng thái, seed, cấu hình, metric, runtime và lineage.
- [x] Chuẩn hóa objective, tính khả thi, nghiệm không khả thi và lý do thất bại.
- [x] Kiểm tra metric hữu hạn/đúng đơn vị và ghi rõ chiều tối ưu hóa.
- [x] Đưa manifest tối thiểu config, git hash, SHA dữ liệu và seed vào kết quả để tái lập/cache có cùng cách hiểu.
- [x] Định nghĩa content_hash canonical chỉ trên dữ liệu/kết quả quyết định xác định; loại runtime, timestamp và metadata thực thi không tất định khỏi payload đem so hash.

#### 1C. Tạo luồng chạy dọc tối thiểu

- [x] Chọn một case nhỏ hiện có, chẳng hạn case_minimart nếu tên đó thực sự tồn tại, cùng một mặt bằng nhỏ.
- [x] Đưa instance qua hợp đồng mới, gọi evaluator/solver hiện có và trả kết quả theo schema chuẩn.
- [x] Thêm E2E test kiểm tra kết quả có thể đọc, truy vết và tái lập bằng seed cố định.
- [x] Giữ vertical slice nhỏ; chưa thay GA, UI hoặc thêm SSAP vào bước này.

#### 1D. Bộ điều phối, cache, cấu hình và CLI

- [x] Tạo src/system/ chứa schema, validation, điều phối và adapter cần thiết theo kiến trúc được chốt.
- [x] Bổ sung store.yaml hoặc cấu hình tương đương; validate cấu hình trước khi chạy.
- [x] Thêm cache với khóa đủ phân biệt input/config/version để tránh trả nhầm kết quả cũ.
- [x] Bổ sung chế độ quick và lệnh CLI cho vertical slice; lỗi chỉ rõ case/trường/cấu hình liên quan.
- [x] Đo runtime case nhỏ; mục tiêu quick mode theo kế hoạch là dưới 2 phút trên môi trường được ghi nhận.

**Nghiệm thu 1A–1D:** Contract tests và vertical slice nhỏ chạy đạt; kết quả tuân schema; quick mode tái lập; cache phân biệt đúng đầu vào; có hướng dẫn chạy.

#### 1E. Tích hợp tạm đủ bốn tầng

- [x] Sau khi vertical slice nhỏ ổn định, nối T1 PRESETS → T2 GA hiện có → T3 xếp món tạm theo tần suất → T4 routing/simulate hiện có → StoreDesign.
- [x] Dùng fixture/synthetic item data có nhãn rõ cho T3 tạm vì repository chưa có nguồn SKU thực đủ dùng; không xem fixture này là nghiệm thu GĐ7 hay bằng chứng kết luận item-level thực tế.
- [x] Dùng adapter tạm theo contract; ghi nhãn rõ các tầng tạm và không xem kết quả này là nghiệm thu thuật toán v4 sâu.
- [x] Chạy E2E với case_minimart và một layout L0 nhỏ trong dưới 2 phút; kiểm tra contract về ID, đơn vị và ràng buộc.
- [x] Xác nhận cùng store.yaml và seed tạo cùng canonical content_hash của StoreDesign; runtime/timestamp không làm thay đổi hash; đưa lệnh pytest -m e2e vào hướng dẫn chạy.

**Nghiệm thu 1E:** Luồng bốn tầng tạm chạy đầu-cuối, tái lập và đạt tiêu chí mục 1.7 của kế hoạch; lỗi ở một tầng được quy về đúng trạng thái/nguồn.

**Nghiệm thu GĐ1:** Đạt 1A–1E; hệ thống tạm đủ bốn tầng chạy E2E theo mục 1.7 của kế hoạch.

**Trạng thái: DONE — 2026-10-11.** Được nghiệm thu sau review cuối của ChatGPT Project. T4 đang dùng `v3_customer_simulation` và chưa có picker flow; T3 dùng SKU tổng hợp có nhãn. Đây là bridge kỹ thuật L0, chưa phải thuật toán v4 đầy đủ hay bằng chứng thực nghiệm trên dữ liệu SKU thật.

### GĐ2 — T2: hai luồng và tối ưu đa mục tiêu

**Mục tiêu:** Mở rộng bài toán QAP thành đánh giá hai luồng và tìm tập nghiệm Pareto theo mục tiêu v4.

#### 2A. Chốt mô hình hai luồng và hàm mục tiêu

- [x] Đặc tả hai luồng, dữ liệu luồng, quy tắc ghép/mapping và ràng buộc khả thi (`docs/GD2_TWO_FLOW_SPEC.md`).
- [x] Chốt công thức twoflow_eps, ý nghĩa từng objective, đơn vị và chiều tối ưu hóa; thiếu trường bắt buộc/provenance trả `INPUT_NOT_READY`.
- [x] Tạo ví dụ nhỏ có thể tính tay để kiểm tra evaluator; dữ liệu được gắn `synthetic_test` và chỉ dùng trong kiểm thử.
- [x] Tạo adapter so sánh với evaluator/QAP hiện hữu ở các hạng thực sự tương đương; không ghép mục tiêu pickup hoặc conflict vào z1/z2 nếu khác ý nghĩa.

#### 2B. Tích hợp NSGA-II ba mục tiêu

- [x] Tạo `src/twoflow.py` để tách evaluator hai luồng khỏi bộ điều phối.
- [x] Tích hợp NSGA-II ba mục tiêu và `twoflow_eps` có epsilon constraints thật qua interface T2; dùng artifact riêng để không đổi hợp đồng GĐ1.
- [x] Ghi thống kê Pareto, feasibility/violations, seed, config, điều kiện dừng, số đánh giá, mapping, provenance và input hash.
- [x] Viết kiểm thử đối chiếu oracle vét cạn trên instance rất nhỏ, cùng kiểm thử dominance/Pareto và ràng buộc.
- [x] 2B.6 Bổ sung hồi quy dữ liệu/mapping/provenance, kiểm tra artifact và tương thích GĐ1/v3; hoàn tất vòng kiểm chứng.

**Nghiệm thu:** Ví dụ tính tay đúng; NSGA-II trả tập nghiệm hợp lệ; kết quả case nhỏ khớp nghiệm vét cạn theo tiêu chí đã định; không hồi quy case QAP cũ.

**Trạng thái triển khai (2026-10-11): DONE về nghiệm thu kỹ thuật.** Đã hoàn tất 2A và 2B.1–2B.6, bao gồm các sửa lỗi review cho dummy slot, mapping slot gốc, relocation budget, provenance, scale/objective lineage, bộ đếm kết quả và kiểm tra tính nhất quán artifact. ChatGPT Project xác nhận đạt nghiệm thu kỹ thuật theo bằng chứng được báo cáo. Sau khi dọn khai báo test trùng, xác minh cuối: 48 test GĐ2 passed; 8 E2E passed; toàn bộ 96 test passed; `compileall`, `pip check`, `git diff --check` và rà whitespace/newline đều passed. ChatGPT Project chưa độc lập kiểm tra diff cuối hoặc tự chạy lệnh. Chưa có dữ liệu picker conflict theo thời gian và bộ exposure/value GĐ2 đã xác thực trong luồng sản xuất GĐ1; do đó hiện chỉ kiểm chứng bằng fixture tổng hợp có nhãn, còn chạy T2 với dữ liệu thiếu phải trả `INPUT_NOT_READY`. NSGA-II là heuristic; chỉ oracle vét cạn cho instance nhỏ mới trả kết luận exact.

### GĐ3 — T4: hành vi khách hàng và calibration

**Mục tiêu:** Mô hình hóa lựa chọn/bỏ qua mặt hàng theo mức tiện lợi và hiệu chỉnh từ basket có thể truy vết.

#### 3A. Định nghĩa mô hình hành vi

- [ ] Đặc tả μ, E^μ, nhóm khách hàng, xác suất lựa chọn và sự kiện không mua/bỏ qua.
- [ ] Định nghĩa RL-A, PER, SUE, đầu vào/đầu ra và cách tác động tới đánh giá layout.
- [ ] Xác định biểu diễn FlowBank và trường cần thiết để tránh trộn case/config.
- [ ] Tách tham số đã hiệu chỉnh khỏi giả định cấu hình để truy vết nguồn.

#### 3B. Hiệu chỉnh, FlowBank và cache

- [ ] Tạo src/behavior.py theo hợp đồng GĐ1.
- [ ] Xây pipeline calibration từ basket/dữ liệu có sẵn; ghi rõ nếu dữ liệu chỉ đủ kiểm tra kỹ thuật.
- [ ] Thêm kiểm thử xác suất, dữ liệu biên, ổn định seed và round-trip FlowBank/cache.
- [ ] Kiểm tra giới hạn μ → ∞ tiến về shortest-path và điều kiện tồn tại của RL, gồm bán kính phổ nhỏ hơn 1 theo đặc tả.
- [ ] Báo độ khớp từng moment đã chọn trong calibration; SUE/MSA phải lưu residual và trạng thái hội tụ.
- [ ] Báo cáo độ phù hợp và giới hạn; không tuyên bố hiệu lực thực tế khi thiếu ground truth phù hợp.

**Nghiệm thu:** Mô hình có đặc tả; calibration tái lập và báo từng moment; giới hạn μ, điều kiện RL, residual/hội tụ SUE được kiểm tra; FlowBank/cache truy vết được; giới hạn dữ liệu được nêu.

### GĐ4 — T2: robust minimax

**Mục tiêu:** Đánh giá layout trong bất định demand/flow bằng mô hình robust có đặc tả rõ.

#### 4A. Đặc tả bất định và nghiệm robust

- [ ] Chốt tập bất định Z_W*, biến L, phạm vi perturbation và nguồn/cách chọn ngưỡng.
- [ ] Viết công thức minimax ILP/GA, điều kiện dừng và tiêu chí so sánh với nghiệm danh định.
- [ ] Tính Z_W^m* trên cùng miền khả thi với bài toán đang xét, gồm ràng buộc lạnh, cố định, tách xa và R; không dùng Hungarian vi phạm ràng buộc làm optimum tham chiếu. Nếu chỉ có cận trên, gắn nhãn bound và không diễn giải là optimum chính xác.
- [ ] Tạo case nhỏ tính tay gồm kịch bản danh định và kịch bản biên.
- [ ] Tạo src/robust.py và adapter cho solver hiện có nếu phù hợp với kiến trúc GĐ1.
- [ ] Tích hợp MSA theo đúng mục đích của kế hoạch; ghi log hội tụ và seed.

**Phụ thuộc:** GĐ2 và GĐ3.

**Nghiệm thu:** Kết quả minimax được kiểm chứng trên case nhỏ; perturbation thuộc tập bất định; báo cáo có giá trị danh định và robust; MSA có tiêu chí dừng.

### GĐ5 — T4: định tuyến và tối ưu picker

**Mục tiêu:** Tính thời gian di chuyển/nhặt hàng và so sánh chiến lược định tuyến, batching, tránh khách và lập lịch.

#### 5A. Bộ giải lộ trình

- [ ] Đặc tả đồ thị đi lại, điểm xuất phát/kết thúc, vị trí kệ, thời gian phục vụ và ràng buộc lối đi.
- [ ] Bổ sung Held–Karp cho case nhỏ và adapter OR-Tools nếu dependency/phiên bản được chấp nhận.
- [ ] Bổ sung các heuristic S-shape, largest gap, combined và Ratliff–Rosenthal; chuẩn hóa chất lượng nghiệm, runtime, giới hạn và trạng thái tối ưu.
- [ ] Chốt backend định tuyến đơn lớn (OR-Tools hoặc LKH) trước khi thêm dependency; giữ adapter tách khỏi contract.
- [ ] Kiểm tra đường đi hợp lệ trên đồ thị floorplan hiện có.

#### 5B. Batching, tránh khách và scheduling

- [ ] Tạo src/picking.py và src/schedule.py cho picking/batching và lịch theo thời gian.
- [ ] Định nghĩa ràng buộc batch, năng lực picker, cửa sổ thời gian và tương tác với luồng khách.
- [ ] Cài batching seed + Clarke–Wright savings + local search; cấu hình B trong khoảng 2–6 đơn và ràng buộc sức chứa xe.
- [ ] Định nghĩa lịch theo giờ thành bài gán/min-cost-flow cho các wave, deadline đơn đặt giờ h phải xong trước h+Δ và mục tiêu giảm chạm mặt kỳ vọng.
- [ ] Mô hình hóa chính sách tránh đông/khách và chi phí theo mật độ khách bằng đầu vào cấu hình; phần này phụ thuộc GĐ3 và phải đánh dấu kịch bản giả lập nếu chưa có dữ liệu.
- [ ] Bổ sung benchmark so sánh exact với heuristic trên case nhỏ và cỡ lớn hơn.
- [ ] Thêm toy case để nghiệm scheduling exact được kiểm tra bằng tay hoặc vét cạn; kiểm tra batch không vượt sức chứa xe.

**Phụ thuộc:** 5A có thể bắt đầu sau GĐ1; batching cần data/config GĐ0. Mật độ khách và routing tránh khách trong 5B phụ thuộc GĐ3.

**Nghiệm thu:** Route hợp lệ; exact khớp vét cạn trên case nhỏ; heuristic báo giới hạn so với exact; batching tuân B/capacity; lịch exact khớp toy case và deadline.

### GĐ6 — T4: mô phỏng hai luồng theo thời gian

**Mục tiêu:** Nối hành vi khách với picker và so sánh đánh giá phân tích với mô phỏng theo thời gian.

#### 6A. Mô hình tương tác động

- [ ] Xác định trạng thái mô phỏng, bước thời gian, luồng khách, picker, sự kiện xung đột và quy tắc cập nhật.
- [ ] Kết nối FlowBank/mô hình hành vi với lịch picker; ghi seed và cấu hình trong kết quả.
- [ ] Tạo scenario tái lập từ tải nhẹ đến cao; phân biệt kiểm thử kỹ thuật và scenario dựa trên dữ liệu thực.
- [ ] Đối chiếu metric phân tích với mô phỏng; định nghĩa sai số chấp nhận và cách báo bất định.
- [ ] Tính encounter kỳ vọng giải tích từ mật độ khách ρ_W, picker ρ_P và thời gian chồng lấn theo giả định Poisson; so sánh với mô phỏng bằng Spearman và sai số tương đối.
- [ ] Mô phỏng sức chứa cell, hàng đợi/chờ đơn giản và xuất metric encounter, wait time, congestion cùng trạng thái seed/config.
- [ ] Thêm kiểm thử bất biến trạng thái, kết thúc mô phỏng, seed và trường hợp không có khách/picker.

**Phụ thuộc:** GĐ3 và GĐ5.

**Nghiệm thu:** Mô phỏng tái lập, không vi phạm invariant; metric giải tích/mô phỏng gồm encounter và wait/congestion; Spearman và sai số tương đối được báo; sai số vượt ngưỡng được đánh dấu.

### GĐ7 — T3: dữ liệu SKU và item-level SSAP

**Mục tiêu:** Bổ sung quyết định đặt sản phẩm cấp SKU theo nhu cầu, giá trị, dung lượng và kích thước.

#### 7A. Sẵn sàng dữ liệu sản phẩm

- [ ] Chốt nguồn dữ liệu sản phẩm được phép dùng và cách nối SKU với basket/demand.
- [ ] Bổ sung/kiểm tra trường demand, giá trị hoặc biên lợi nhuận, capacity, width và category theo data dictionary GĐ0.
- [ ] Nếu thiếu trường vật lý/giá trị, thiết lập fixture tổng hợp có nhãn rõ, chỉ dùng kiểm tra thuật toán.
- [ ] Đưa ngưỡng lọc support (mốc kế hoạch ban đầu là tối thiểu 100 lượt mua) thành tham số có phiên bản; gộp đuôi thành nhóm “món khác” khi được cấu hình.
- [ ] Báo số SKU được giữ và tỷ lệ doanh số/demand giữ lại sau preprocessing; không hard-code ngưỡng nghiệp vụ.
- [ ] Validate SKU thiếu mapping, demand sai, capacity/width không dương và đơn vị không đồng nhất.

#### 7B. SSAP, blocking và planogram

- [ ] Tạo src/items.py và src/slotting.py theo hợp đồng hệ thống.
- [ ] Định nghĩa SSAP/MIP: biến quyết định, objective, ràng buộc kệ và ý nghĩa nghiệm khả thi.
- [ ] Thêm heuristic đối chứng và blocking theo định nghĩa nghiệp vụ; giải thích SKU bị chặn/không được xếp.
- [ ] Nếu giữ mô hình blocking của kế hoạch, kiểm tra các món cùng aisle_id tạo thành một khối chữ nhật liên tục; nếu đổi hình thức blocking, ghi quyết định mô hình và lý do trước khi nghiệm thu.
- [ ] Tạo output planogram đọc lại được và liên kết với layout, SKU, cấu hình, metric.
- [ ] Đưa feedback s_i/v_i vào pipeline theo công thức v4; kiểm tra chiều, miền và dữ liệu thiếu.
- [ ] Kiểm thử toy cho công thức cầu theo không gian D_j = a_j · n_j^β · φ_l · ψ_c và từng thành phần của objective SSAP: lợi nhuận, adjacency/lift và picker penalty.
- [ ] Kiểm tra round-trip planogram → s_i, v_i → T2 để xác nhận feedback giữ đúng mapping và đơn vị.

**Phụ thuộc:** GĐ0 data readiness và GĐ1 contract.

**Nghiệm thu:** MIP khớp vét cạn ở case cực nhỏ; công thức D_j/objective được kiểm tra bằng toy case; assignment thỏa capacity/width/ràng buộc; round-trip feedback đúng; fixture tổng hợp có nhãn; thiếu nguồn SKU thì chưa nghiệm thu trên dữ liệu thực.

### GĐ8 — Coupling ngược giữa các tầng

**Mục tiêu:** Lặp cập nhật T2/T3/T4 đến khi đạt quy tắc hội tụ hoặc giới hạn dừng đã định.

#### 8A. Thiết kế vòng lặp ghép tầng

- [ ] Định nghĩa biến truyền giữa item placement, flow/behavior, picker và objective T2.
- [ ] Xác định điểm khởi tạo, thứ tự cập nhật, điều kiện dừng, ngưỡng hội tụ và giới hạn số vòng.
- [ ] Đặt mặc định T4→T2 theo kế hoạch: relative change của exposure dưới 2% hoặc tối đa 10 vòng, giữ nghiệm tốt nhất; các ngưỡng phải cấu hình được.
- [ ] Đặt mặc định T3→T2 theo kế hoạch: cập nhật v bằng MSA, s_i chỉ tăng, dừng khi relative change v dưới 1% và s không đổi hoặc tối đa 5 vòng.
- [ ] Ghi các giá trị trên là mặc định/provisional của kế hoạch; nếu thay đổi phải lưu rationale, cấu hình và phiên bản.
- [ ] Phân biệt hội tụ, dừng do giới hạn và thất bại; ghi lịch sử metric từng vòng.
- [ ] Xử lý nghiệm bất khả thi tại một tầng mà không làm mất nguyên nhân.

#### 8B. Multi-slot category GA và kiểm tra độ bền

- [ ] Bổ sung GA cho category nhiều vị trí/slot với decoder hợp lệ và cơ chế sửa nghiệm xác định.
- [ ] Kiểm tra hội tụ từ nhiều seed/điểm khởi đầu và báo độ nhạy.
- [ ] So sánh coupling với pipeline không ghép tầng trên cùng case/config.

**Phụ thuộc:** Prototype/L0 cần GĐ2, GĐ3, GĐ5 và GĐ7 nhưng có thể dùng T4 analytic; coupling L2/full-fidelity chỉ nghiệm thu sau GĐ6 có mô phỏng hai luồng.

**Nghiệm thu:** Quy tắc dừng xác định; không gọi là hội tụ nếu chạm max-iteration; lịch sử truy vết được; decoder/repair chỉ chấp nhận nghiệm khả thi và báo thất bại rõ nếu không có nghiệm khả thi. Độ nhạy điểm khởi tạo (cùng layout hoặc chênh dưới 1%) được ghi là tiêu chí nghiên cứu cần đo, không ép kết quả.

### GĐ9 — T1: tối ưu layout outer loop

**Mục tiêu:** Tìm cấu hình mặt bằng bằng layout tham số hóa, lọc tính khả thi và đánh giá đa mức.

#### 9A. Không gian layout và tính khả thi

- [ ] Chuyển 5 kiểu layout trong kế hoạch thành tham số sinh layout; đối chiếu với src/layouts.py để tránh nhân đôi logic.
- [ ] Định nghĩa miền tham số, ràng buộc hình học/lối đi/kệ và điều kiện layout khả thi.
- [ ] Tạo kiểm thử hình học cho biên, kích thước nhỏ, giao nhau và layout không sinh được.

#### 9B. Multi-fidelity và chọn ứng viên

- [ ] Định nghĩa L0/L1/L2: metric, độ chính xác, runtime và quy tắc chuyển mức.
- [ ] Dùng profile fidelity mặc định/provisional của v4: L0 = GA 2 giây + T3 tham lam + T4 giải tích; L1 = GA 20 giây + coupling + T3 heuristic/local search + T4 giải tích và 5 mô phỏng; L2 = GA 30 seed hoặc ILP khi n ≤ 15 + T3 MIP + T4 30 mô phỏng với seed độc lập.
- [ ] Đặt budget fidelity trong config; nếu đổi mặc định 2s/20s, 30 seed, 5/30 mô phỏng hoặc điều kiện ILP thì lưu rationale và phiên bản.
- [ ] Tích hợp successive halving/Bayesian search theo thiết kế v4; khóa seed, ghi candidate bị loại và lý do.
- [ ] Định nghĩa regression set 9 layouts theo kế hoạch và lưu baseline.
- [ ] Bổ sung src/outer.py điều phối tìm kiếm qua coupling GĐ8.
- [ ] Đo runtime/số lần đánh giá; cache phải phân biệt layout/config.

**Phụ thuộc:** GĐ8.

**Nghiệm thu:** 5 kiểu layout có kiểm thử feasibility; L0/L1/L2 có tiêu chí; regression 9 layouts tái lập; outer loop lưu lịch sử search và kết quả.

### GĐ10 — Chạy lại từng phần, cải tạo và giao diện

**Mục tiêu:** Hỗ trợ đánh giá layout mới/cải tạo và xuất kết quả có thể kiểm tra.

#### 10A. Rerun một phần và cải tạo

- [ ] Định nghĩa input thay đổi nào làm mất hiệu lực cache/tầng nào cần chạy lại.
- [ ] Mã hóa và kiểm thử dependency invalidation tối thiểu: đổi mặt bằng/w_min → T1 đến T4; cố định kệ đổi v_i/giá → T2→T3→T4; đổi tập hành vi M → FlowBank→T2→T3→T4; đổi online ratio/hour profile → T2(C^m) và T4; đổi một planogram ngành → T3 ngành đó→T4.
- [ ] Hỗ trợ scenario cải tạo giữ nguyên phần layout không đổi và tính lại phần phụ thuộc.
- [ ] Hiển thị lineage từ kết quả mới về baseline, thay đổi đầu vào và tầng đã chạy/bỏ qua.
- [ ] So sánh partial rerun với full rerun ở các case có thể đối chiếu.

#### 10B. Luồng ứng dụng 6 bước

- [ ] Cập nhật app/streamlit_app.py theo luồng 6 bước v4, tái sử dụng UI hiện có khi phù hợp.
- [ ] Theo đúng semantic 6 bước: Đầu vào → Kệ → Ngành hàng → Món → Luồng → So sánh & quyết định.
- [ ] Cho phép khóa từng tầng để chạy what-if; mỗi bước hiển thị dữ liệu nhận từ tầng trước.
- [ ] Hiển thị trạng thái, lỗi dữ liệu, tiến độ và giới hạn quick/full mode rõ ràng.
- [ ] Giao diện gọi hợp đồng hệ thống, không cài đặt lại thuật toán.

#### 10C. Export và truy vết

- [ ] Xuất kết quả/planogram/metric cùng metadata, cấu hình, seed và phiên bản schema.
- [ ] Hỗ trợ các artifact theo kế hoạch: StoreDesign JSON, bản vẽ PNG/PDF, planogram CSV và picking schedule CSV.
- [ ] Kiểm tra đọc lại file export; không đưa secret nội bộ vào output.
- [ ] Cập nhật hướng dẫn chạy ứng dụng và định dạng dữ liệu.

**Phụ thuộc:** GĐ1, GĐ8 và GĐ9; các tầng UI trình bày cần có kết quả ổn định.

**Nghiệm thu:** 6 bước chạy với case mẫu; rerun/cached invalidation khớp các tình huống đã liệt kê; partial rerun có lineage chính xác; export đọc lại được và metric khớp giao diện.

### GĐ11 — E8–E20, kiểm thử tổng thể và tài liệu

**Mục tiêu:** Hoàn tất bằng chứng thực nghiệm, kiểm thử tích hợp và tài liệu theo kế hoạch.

#### 11A. Thí nghiệm E8–E20

- [ ] Lập bảng E8–E20: giả thuyết, dữ liệu/case, baseline, biến số, số lần chạy/seed, metric và tiêu chí kết luận.
- [ ] Đánh dấu thí nghiệm cần SKU thật. Không dùng fixture/synthetic thay dữ liệu thật để đưa ra kết luận thực nghiệm; nếu thiếu dữ liệu bắt buộc, chỉ báo kiểm thử kỹ thuật/độ nhạy và đánh dấu thí nghiệm chưa đủ bằng chứng.
- [ ] Giữ E1/E2 làm benchmark và bổ sung E6' độ nhạy theo biến trong kế hoạch.
- [ ] Bổ sung script/cấu hình chạy, lưu raw result và summary có phiên bản.
- [ ] Tạo bảng/biểu đồ tái lập; ghi sai số, runtime và giới hạn.
- [ ] Khóa protocol thống kê trước full campaign: 30 seed, Wilcoxon + Holm, A12 và Friedman khi áp dụng.

#### 11B. Bộ kiểm thử và chất lượng

- [ ] Hoàn thiện unit/contract/e2e/regression tests cho các tầng đã triển khai.
- [ ] Bổ sung test_contracts.py, test_e2e.py hoặc vị trí tương đương theo cấu trúc thực tế.
- [ ] Đạt tối thiểu 45 tests theo kế hoạch, tập trung vào hành vi quan trọng.
- [ ] Chạy lint/static analysis/build phù hợp cấu hình repository và lưu kết quả.

#### 11C. README và hướng dẫn vận hành

- [ ] Cập nhật README: cài đặt, cấu hình, dữ liệu, CLI, quick, Streamlit, tests và giới hạn.
- [ ] Viết hướng dẫn tái lập E8–E20 và giải thích output.
- [ ] Đối chiếu từng hướng dẫn với lệnh/file thực tế.

**Nghiệm thu:** E8–E20 có protocol và kết quả tái lập; tối thiểu 45 test hữu ích chạy đạt; README làm theo được trên môi trường được hỗ trợ.

### GĐ12 — Viết chương 3 và chương 4 luận văn

**Mục tiêu:** Trình bày phương pháp và kết quả dựa trên implementation/experiment đã xác nhận.

#### 12A. Hoàn thiện nội dung luận văn

- [ ] Chương 3: mô hình, dữ liệu, hợp đồng tầng, thuật toán, coupling, giả định và cách đánh giá.
- [ ] Chương 4: thiết lập E8–E20, baseline, kết quả, độ nhạy, runtime và giới hạn.
- [ ] Tạo bảng/biểu đồ từ artifact thực nghiệm; liên kết số liệu với script và phiên bản dữ liệu/mã.
- [ ] Kiểm tra thuật ngữ, đơn vị, công thức và kết luận với mã nguồn/đầu ra.

**Phụ thuộc:** GĐ11 và kết quả đã xác nhận; không trình bày số liệu minh họa như số liệu thực.

**Nghiệm thu:** Kết luận định lượng truy về artifact; phương pháp đủ để tái lập; giới hạn dữ liệu/mô hình được nêu.

## 4. Thứ tự thực hiện và nhánh song song

### Đường găng đề xuất

GĐ0 → GĐ1 → (GĐ2 + GĐ3 + GĐ5 + chuẩn bị dữ liệu GĐ7) → GĐ4/GĐ6/GĐ7 → GĐ8 → GĐ9 → GĐ10 → GĐ11 → GĐ12

### Quy tắc phụ thuộc

- GĐ1 là điều kiện chung trước khi tích hợp tầng mới.
- GĐ2 và GĐ3 có thể chạy song song sau khi hợp đồng tầng được khóa.
- GĐ5 có thể bắt đầu sau GĐ1; phần tương tác khách cần GĐ3.
- GĐ4 cần mô hình T2 từ GĐ2 và bất định/flow từ GĐ3.
- GĐ6 cần hành vi GĐ3 và picker/scheduling GĐ5.
- GĐ7 cần data readiness GĐ0 và contract GĐ1; fixture/solver có thể phát triển song song, nhưng nghiệm thu dữ liệu thực cần nguồn SKU hợp lệ.
- GĐ8 chỉ bắt đầu khi interface T2/T3/T4 cần ghép đã ổn định.
- GĐ9 cần GĐ8; GĐ10 tích hợp luồng ổn định; GĐ11 cung cấp bằng chứng cho GĐ12.

### Mốc ưu tiên

1. Đóng băng baseline, data dictionary, nguồn và floorplan readiness (0A–0C).
2. Hoàn thiện hợp đồng, vertical slice quick và E2E tạm bốn tầng (1A–1E).
3. Tạo contract cho T2/T4/T3 rồi triển khai nhánh có thể kiểm tra độc lập.
4. Tích hợp robust, mô phỏng, item-level và coupling theo phụ thuộc.
5. Thêm outer search, partial rerun, UI/export.
6. Chạy thí nghiệm, hoàn thiện tài liệu và luận văn.

## 5. Vertical slice đầu tiên

1. Chọn case/layout nhỏ đang tồn tại và xác minh tên/file thực tế.
2. Tạo input theo schema GĐ1 và validate.
3. Gọi evaluator/solver v3 hiện có qua adapter.
4. Trả metric theo result schema chuẩn, có seed/config/lineage.
5. Kiểm tra bằng contract test và E2E test.
6. Chạy chế độ quick và đo runtime theo tiêu chí v4.

Vertical slice chứng minh luồng điều phối/hợp đồng, chưa chứng minh thuật toán v4 mới.

## 6. Definition of Done chung

Một giai đoạn chỉ hoàn tất khi:

- [ ] Đạt tiêu chí nghiệm thu riêng.
- [ ] Contract và E2E liên quan chạy đạt; có regression test cho hành vi v3 bị ảnh hưởng.
- [ ] Chế độ quick chạy được theo phạm vi định và runtime được đo trên môi trường ghi nhận.
- [ ] Output có seed, cấu hình, phiên bản/schema và lineage cần thiết.
- [ ] Lỗi, nghiệm không khả thi, thiếu dữ liệu và trạng thái dừng được biểu diễn rõ.
- [ ] Tài liệu lệnh chạy và giới hạn được cập nhật cùng implementation.
- [ ] Build/test/lint được báo theo đúng lệnh đã chạy; không suy diễn thành công nếu chưa chạy.

## 7. Quyết định và rủi ro cần giải quyết

| Chủ đề | Cần quyết định/xác minh | Tác động nếu chưa chốt |
|---|---|---|
| Dữ liệu sản phẩm | Nguồn SKU hợp lệ; demand, giá trị/biên lợi nhuận, capacity, width, category | Không nghiệm thu SSAP/planogram trên dữ liệu thực hoặc kết luận E8–E20 liên quan SKU |
| Đơn vị và chi phí | Đơn vị khoảng cách, thời gian picker, giá trị sản phẩm và chi phí layout | Metric giữa tầng không so sánh đáng tin cậy |
| Hai luồng/minimax | Công thức objective, tập Z_W*, biến L, perturbation | Kết quả robust/Pareto thiếu chuẩn đối chiếu |
| Layout outer loop | 5 kiểu layout là template hay topology; miền tham số/feasibility | Search có thể đánh giá sai không gian nghiệm |
| Coupling | Biến truyền, thứ tự cập nhật, ngưỡng hội tụ, giới hạn vòng | Không phân biệt hội tụ với dừng cưỡng bức |
| E8–E20 | Protocol, baseline, số seed, metric và tiêu chí kết luận | Không đủ bằng chứng tái lập cho báo cáo/luận văn |
| Cache | Thành phần input/config/version trong cache key | Có thể trả kết quả cũ cho case/layout khác |
| Backend routing | Chọn OR-Tools hay LKH cho đơn lớn; phiên bản và license/dependency | Khó tái lập runtime và nghiệm giữa môi trường |
| Backend Bayes | Chọn scikit-optimize hay Optuna trước GĐ9 | Search và cấu hình thí nghiệm có thể không nhất quán |

Giải quyết các mục này trong đặc tả hoặc cấu hình có phiên bản trước khi dùng để kết luận thực nghiệm.

## 8. Nhật ký cập nhật

- Backlog được lập từ kế hoạch v4 và cây mã nguồn tại commit 23d1f2e trong lần rà soát này.
- Tài liệu không tự xác nhận giai đoạn nào đã xong; chỉ cập nhật checkbox sau khi có bằng chứng từ mã nguồn và lệnh chạy.
