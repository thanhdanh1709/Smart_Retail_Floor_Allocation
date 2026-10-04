# Smart Retail Floor Allocation

Mã nguồn của đề án **“Tối ưu hóa không gian trưng bày và luồng di chuyển của khách hàng trong cửa hàng bán lẻ số”**. Dự án triển khai mô hình hóa mặt bằng, ước lượng tham số từ giỏ hàng, quy hoạch nguyên song mục tiêu, GA/NSGA-II, mô phỏng khách hàng, bộ thí nghiệm E1–E7 và dashboard hỗ trợ ra quyết định.

Bài toán chính là gán các nhóm hàng vào vị trí trưng bày để cân đối **tiện lợi khi mua sắm** và **giá trị mua ngẫu hứng theo mô hình**. Đây là công cụ nghiên cứu và so sánh phương án. Dữ liệu Instacart, hệ số giá trị và hành vi mô phỏng cần được hiệu chỉnh bằng dữ liệu cửa hàng trước khi dùng cho quyết định thực tế.

<!-- AUTO-GENERATED -->
<!-- Phần kỹ thuật được đối chiếu từ source/config/tests; không có script tự sinh README. -->
<!-- Baseline kiểm tra: f4ab327, ngày 2026-10-04. -->

## Mục lục

- [1. Tổng quan và luồng thực hiện](#tong-quan)
- [2. Cài đặt và chạy nhanh](#cai-dat)
- [3. Cấu trúc dự án](#cau-truc)
- [4. Dữ liệu và tiền xử lý](#du-lieu)
- [5. Mặt bằng, slot và đồ thị](#mat-bang)
- [6. Instance, biểu diễn nghiệm và ràng buộc](#instance)
- [7. Mô hình toán và chuẩn hóa](#mo-hinh)
- [8. Thuật toán tối ưu](#thuat-toan)
- [9. Mô phỏng khách hàng](#mo-phong)
- [10. Pipeline hiệu chỉnh, chọn và đánh giá](#pipeline)
- [11. Dashboard Streamlit](#dashboard)
- [12. Thí nghiệm E1–E7](#thi-nghiem)
- [13. Đọc kết quả và tái lập](#tai-lap)
- [14. Kiểm thử và xử lý lỗi chạy](#kiem-thu)
- [15. Giới hạn hiện tại](#gioi-han)
- [16. Quy trình phát triển với ECC và AGENTS.md](#quy-trinh-ai)

<a id="tong-quan"></a>

## 1. Tổng quan và luồng thực hiện

Pipeline dùng chung tại [src/pipeline.py](src/pipeline.py), dashboard và thí nghiệm E5:

~~~mermaid
flowchart TD
    A["Instacart / CSV cửa hàng"] --> B["params: w, f, p, v, metadata, giỏ CSR"]
    C["Lưới mặt bằng"] --> D["floorplan: slot, D, d_in, d_out, e hình học"]
    B --> E["instance: nhóm hàng + slot + ràng buộc + hiện trạng"]
    D --> E
    E --> F["Model: hiệu chỉnh lambda, e và q trên hiện trạng"]
    F --> G["Optimization: NSGA-II lai + GA hai đầu trên QAP"]
    G --> H{"objective"}
    H -->|model| I["Ứng viên QAP hiệu chỉnh"]
    H -->|route| J["NSGA-II với mục tiêu theo tuyến đi, khởi tạo từ QAP"]
    I --> K["Simulation: sàng lọc, chọn hồ sơ, tinh chỉnh 2-swap"]
    J --> K
    K --> L["Decision: đánh giá cuối bằng seed riêng, KPI, bảng gán, danh sách dời"]
~~~

| Tầng | Trách nhiệm | Sản phẩm |
|---|---|---|
| Data | Chuyển giỏ hàng và hình học thành tham số | Ma trận đồng mua, độ phổ biến, giỏ CSR, khoảng cách giữa slot |
| Model | Xây dựng instance; hiệu chỉnh hành vi và tiếp xúc trên hiện trạng | Mô hình gốc, mô hình hiệu chỉnh, bảng payoff |
| Optimization | Tìm các phương án đánh đổi giữa hai mục tiêu | Tập Pareto xấp xỉ và các nghiệm cực biên |
| Simulation | Đánh giá lại ứng viên theo chuyến mua sắm mô phỏng | KPI sàng lọc, phương án chọn theo hồ sơ, phương án tinh chỉnh |
| Decision | Đánh giá cuối và giải thích phương án | Trung bình/KTC, so sánh với hiện trạng, CSV và PNG |

**Ba khái niệm cần phân biệt:** Z1/Z2 của QAP là mục tiêu đại diện; Z1/Z2 của RouteModel được tính trên tuyến đi của mẫu giỏ cố định; KPI của Simulator được tính từ các chuyến mua sắm có hành vi ngẫu nhiên. Tối ưu được mục tiêu đại diện chưa đủ để kết luận KPI cuối sẽ tốt hơn.

<a id="cai-dat"></a>

## 2. Cài đặt và chạy nhanh

### 2.1. Môi trường

[requirements.txt](requirements.txt) ghi Python 3.10+ và khóa phiên bản các thư viện. Lần kiểm tra baseline của README dùng **Python 3.12.10 trên Windows**, với các phiên bản cài đặt khớp requirements. Đây là môi trường đã kiểm tra, không phải cam kết mọi phiên bản Python/nền tảng đều tương đương.

| Nhóm | Thư viện chính |
|---|---|
| Tính toán và dữ liệu | NumPy, SciPy, pandas, Numba |
| Tối ưu | PuLP, highspy/HiGHS, pymoo |
| Thống kê | SciPy, scikit-posthocs |
| Giao diện và hình ảnh | Streamlit, Plotly, Matplotlib |
| Công cụ | PyYAML, pytest, kagglehub |

Mọi lệnh module ở các phần sau phải chạy từ thư mục **smart-retail-layout/**. Repo có sẵn dữ liệu processed; có thể mở dashboard và chạy tests trước khi tải raw Instacart.

### 2.2. Windows PowerShell

Nếu đang ở thư mục gốc repository:

~~~powershell
Set-Location .\smart-retail-layout
py -3.12 -m venv .venv
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
~~~

Gọi trực tiếp Python của venv giúp chạy được mà không cần thay ExecutionPolicy để kích hoạt Activate.ps1. Nếu đã có venv thì bỏ bước tạo lại. Khi thấy lệnh dùng python ở phần sau, thay bằng .\.venv\Scripts\python.exe nếu terminal chưa kích hoạt venv.

### 2.3. Linux/macOS

~~~bash
cd smart-retail-layout
python3 -m venv .venv
source .venv/bin/activate
export PYTHONIOENCODING=utf-8
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
python -m streamlit run app/streamlit_app.py
~~~

Mở địa chỉ Streamlit in trong terminal. Dừng server bằng Ctrl+C. Lần chạy đầu có thể mất thêm thời gian để Numba biên dịch và tính payoff; thời gian phụ thuộc máy và cấu hình.

### 2.4. Lệnh thường dùng

~~~bash
python -m experiments.run_all --help
python -m experiments.e5_simulation --help
python -m experiments.run_all --profile quick --workers 2 --only e1 e3
python -m experiments.e3_heuristics --profile quick --workers 2
~~~

Chạy thí nghiệm sẽ ghi kết quả vào results/ và có thể ghi đè kết quả cùng tên. Đọc [phần thí nghiệm](#thi-nghiem) trước khi chạy toàn bộ.

<a id="cau-truc"></a>

## 3. Cấu trúc dự án

Từ gốc repository, đề cương nằm ở [docx/De_cuong_Smart_Retail_Floor_Allocation.docx](../docx/De_cuong_Smart_Retail_Floor_Allocation.docx); mã thực thi nằm trong smart-retail-layout/:

~~~text
smart-retail-layout/
├── app/                    Dashboard Streamlit
├── src/                    Dữ liệu, mô hình, tối ưu, mô phỏng, quyết định
├── experiments/            Script E1–E7, cấu hình và thống kê bổ sung
├── tests/                  Kiểm thử lõi
├── data/
│   ├── mappings/           Ánh xạ aisle sang nhóm hàng
│   ├── processed/          group/ và aisle/
│   ├── floorplans/         Lưới mẫu và bảng slot
│   └── qaplib/             Bộ kiểm chứng QAP
├── results/                Kết quả thí nghiệm, hình và cache
├── requirements.txt
├── pytest.ini
└── README.md
~~~

| Tệp | Trách nhiệm chính |
|---|---|
| [app/streamlit_app.py](app/streamlit_app.py) | Giao diện năm tab, upload dữ liệu, lưu kết quả các khâu trong session |
| [src/params.py](src/params.py) | Nạp raw Instacart/CSV upload; tính w, f, lift, p, v và giỏ CSR |
| [src/layouts.py](src/layouts.py), [src/floorplan.py](src/floorplan.py) | Sinh lưới, gán vùng lạnh, tách slot, đồ thị và đường đi |
| [src/instance.py](src/instance.py) | Instance, ma trận tương thích, hiện trạng, mục tiêu, kiểm tra vi phạm |
| [src/fastops.py](src/fastops.py) | Nhân Numba: objective, delta swap, penalty, local search |
| [src/model_ilp.py](src/model_ilp.py), [src/solvers.py](src/solvers.py) | ILP, epsilon-front, payoff và chuẩn hóa |
| [src/ga.py](src/ga.py) | GA với toán tử hoán vị, repair, local search và nhập cư |
| [src/moo.py](src/moo.py) | NSGA-II, lọc Pareto, hypervolume, điểm gối, quét alpha |
| [src/baselines.py](src/baselines.py) | Random, Greedy, Simulated Annealing và Tabu Search |
| [src/simulate.py](src/simulate.py) | Simulator, hiệu chỉnh lambda, tiếp xúc, lặp mô phỏng và KTC |
| [src/routing.py](src/routing.py) | RouteModel và RouteObjective: khoảng cách/giá trị theo tuyến đi |
| [src/pipeline.py](src/pipeline.py) | Hiệu chỉnh → sinh ứng viên → sàng lọc → chọn/tinh chỉnh → đánh giá cuối |
| [src/relocation.py](src/relocation.py) | Chuỗi đổi chỗ ưu tiên để đưa nhóm hàng về slot đích |
| [src/qaplib.py](src/qaplib.py), [src/viz.py](src/viz.py) | Đọc QAPLIB, vẽ mặt bằng/Pareto/heatmap |
| [experiments/common.py](experiments/common.py) | Cấu hình, dựng instance, payoff cache, workers và metadata lần chạy |
| [tests/test_core.py](tests/test_core.py) | Kiểm tra công thức, thuật toán, mô phỏng và pipeline |

<a id="du-lieu"></a>

## 4. Dữ liệu và tiền xử lý

Nguồn triển khai: [src/params.py](src/params.py), [experiments/build_data.py](experiments/build_data.py).

### 4.1. Dữ liệu processed có sẵn

Hai mức biểu diễn dùng cùng nguồn đơn hàng:

| Mức | Số loại hàng | Ma trận w/lift | Giỏ mẫu hiện có |
|---|---:|---|---|
| group | 40 nhóm theo mapping Việt Nam | 40 × 40 | CSR 200.000 × 40 |
| aisle | 132 aisle sau khi bỏ other/missing | 132 × 132 | CSR 200.000 × 132 |

[Manifest group](data/processed/group/manifest.json) và [manifest aisle](data/processed/aisle/manifest.json) ghi **3.214.236 đơn** dùng để ước lượng tham số. 200.000 giỏ là mẫu cho mô phỏng, không phải toàn bộ dữ liệu dùng tính w và f. Hàm compute mặc định chọn tối đa 200.000 giỏ không hoàn lại với seed=0.

| Artifact trong `data/processed/<level>/` | Nội dung và định dạng |
|---|---|
| categories.csv | Metadata; thứ tự dòng tương ứng với chỉ số trong các mảng |
| w.npy | Support đồng mua, ma trận đối xứng, đường chéo bằng 0 |
| f.npy | Độ phổ biến của từng loại hàng, vector dài n |
| lift.npy | Đồng mua so với giả định độc lập, ma trận n × n |
| p.npy | Proxy hành vi từ tỷ lệ không mua lại |
| v.npy | Hệ số giá trị từ mapping |
| baskets.npz | Các mảng indptr, indices, shape của CSR; giá trị nhị phân được phục dựng khi nạp |
| manifest.json | Ngày tạo, n_orders, n_units, kích thước nguồn và mã băm đầu ra |

Manifest hiện lưu **sources: số bytes**, **outputs: SHA-256 rút gọn 16 ký tự hex**. Nó chưa lưu hash nội dung raw input hoặc toàn bộ tham số tiền xử lý. Muốn tái lập chặt chẽ cần ghi bổ sung phiên bản nguồn, hash raw/mapping và seed sử dụng.

### 4.2. Raw Instacart và cách xây lại dữ liệu

Raw Instacart không nằm trong Git. Đặt trực tiếp các tệp sau vào data/raw/instacart/:

| Tệp | Cột dùng |
|---|---|
| products.csv | product_id, aisle_id |
| order_products__prior.csv | order_id, product_id, reordered |
| aisles.csv | aisle_id, aisle |

Mapping tại [data/mappings/groups_vn.csv](data/mappings/groups_vn.csv) gồm group_id, name, aisle_ids phân cách bằng dấu chấm phẩy, needs_cold, value_v và dwell_level. Hai aisle 6 và 100 bị loại; aisle còn lại phải có mapping.

Có thể tải dataset bằng công cụ kagglehub đã có trong requirements:

~~~bash
python -c "import kagglehub; print(kagglehub.dataset_download('psparks/instacart-market-basket-analysis'))"
~~~

Từ thư mục được in ra, giải nén nếu cần rồi đặt ba CSV vào đúng đường dẫn trên. Quyền truy cập tải dữ liệu phụ thuộc dịch vụ và phiên làm việc. Sau đó chạy:

~~~bash
python -m experiments.build_data
~~~

Script tính lại cả group/aisle và ghi lại chín mặt bằng chuẩn cùng slots_*.csv. Nó có thể thay đổi dữ liệu processed và floorplan đã có; giữ lại phiên bản dùng cho báo cáo trước khi xây lại.

### 4.3. Cách tính tham số

Cho N là số đơn còn lại sau lọc, X_oi=1 nếu đơn o chứa loại i. Một loại xuất hiện nhiều lần trong cùng đơn vẫn chỉ tính một lần. CSR được nhị phân hóa trước khi tính XᵀX.

$$
f_i=\frac{1}{N}\sum_o X_{oi},\qquad
w_{ij}=\frac{1}{N}\sum_o X_{oi}X_{oj}\ (i\ne j),\qquad w_{ii}=0.
$$

$$
\operatorname{lift}_{ij}=\frac{w_{ij}}{f_if_j},\qquad
p_i=1-\operatorname{mean}(\mathrm{reordered}\mid i).
$$

Khi f_if_j=0, loader đặt lift bằng 0. w là dữ liệu đầu vào mục tiêu QAP; lift phục vụ phân tích liên kết, không thay w trong mục tiêu mặc định.

p_i được tính từ các dòng đơn–sản phẩm thuộc nhóm i, còn w và f tính trên đơn nhị phân. p_i là proxy từ tỷ lệ không reorder; dữ liệu này không quan sát trực tiếp việc mua ngoài kế hoạch. v_i là hệ số giá trị giả định, không phải giá bán. Dwell trung vị lấy từ mapping:

| dwell_level | dwell_median_s |
|---:|---:|
| 1 | 6 giây |
| 2 | 10 giây |
| 3 | 18 giây |

### 4.4. CSV của cửa hàng tải lên

categories.csv cần một dòng cho mỗi loại hàng, với category_id duy nhất:

| Cột | Bắt buộc | Giá trị khi thiếu | Ý nghĩa |
|---|---|---|---|
| category_id | Có | — | ID dùng để nối với giỏ hàng |
| name | Có | — | Tên hiển thị |
| group_id | Không | category_id | Nhóm ngành hàng của loại đó |
| needs_cold | Không | 0 | 0: thường; 1: cần lạnh |
| value_v | Không | 0.5 | Hệ số giá trị |
| impulse_p | Không | 0.3 | Hệ số mua ngẫu hứng đầu vào |
| dwell_median_s | Không | 10.0 | Trung vị thời gian dừng, giây |
| n_slots | Không | 1 | Metadata; hiện chưa triển khai nhiều slot cho một loại |

Ví dụ cấu trúc CSV, các hệ số chỉ để minh họa:

~~~csv
category_id,name,group_id,needs_cold,value_v,impulse_p,dwell_median_s,n_slots
G01,Rau củ,Tươi,0,0.6,0.25,10,1
G02,Sữa,Lạnh,1,0.8,0.20,6,1
G03,Bánh kẹo,Khô,0,0.5,0.40,18,1
~~~

baskets.csv dùng dạng dài, mỗi dòng là một loại có trong một đơn:

~~~csv
order_id,category_id
1001,G01
1001,G02
1002,G02
1002,G03
~~~

Loader from_uploaded tính lại w/f/lift; p/v lấy từ metadata. ID không có trong categories bị lọc. Dòng trùng order_id/category_id được gộp nhị phân; không mô hình hóa số lượng sản phẩm trong giỏ.

Trước khi upload, kiểm tra ID không trùng, liên kết category đầy đủ, giỏ sau lọc không rỗng, các số hữu hạn, impulse_p trong [0,1], needs_cold là 0/1 và dwell_median_s>0. Loader hiện chưa kiểm tra đầy đủ các điều kiện này. Hai tệp phải dùng cùng kiểu ID và CSV UTF-8.

<a id="mat-bang"></a>

## 5. Mặt bằng, slot và đồ thị

Nguồn triển khai: [src/floorplan.py](src/floorplan.py), [src/layouts.py](src/layouts.py).

### 5.1. Lưới mặt bằng

| Ký hiệu | Ý nghĩa | Có thể đi qua |
|---|---|---|
| X | Tường/vật cản | Không |
| A | Lối đi | Có |
| S | Kệ thường | Không |
| R | Kệ lạnh | Không |
| E | Cửa vào | Có |
| C | Thu ngân | Có |

Mỗi ô có cạnh 1 m. Các dòng phải cùng chiều dài; cần có E, C và các đoạn kệ tiếp cận được. Đồ thị nối ô đi được theo bốn hướng, mỗi cạnh dài 1 m. Khoảng cách Dijkstra được tính trên đồ thị này, không phải khoảng cách Euclid giữa các kệ.

Chín lưới chuẩn nằm tại [data/floorplans](data/floorplans), gồm grid/racetrack/free × small/medium/large; [case_minimart.txt](data/floorplans/case_minimart.txt) là tình huống siêu thị mini. Số nhóm và slot khi dựng instance mặc định:

| Kiểu | small: n / m | medium: n / m | large: n / m |
|---|---|---|---|
| grid | 20 / 25 | 40 / 55 | 132 / 161 |
| racetrack | 20 / 32 | 40 / 58 | 132 / 145 |
| free | 20 / 35 | 40 / 59 | 132 / 157 |

Các số trên phụ thuộc bộ sinh và cấu hình hiện tại; thay n, restrict_slots hoặc lưới sẽ thay instance.

### 5.2. Tách slot và đo khoảng cách

Slot là một đoạn **mặt kệ** hướng ra lối đi. Mặc định seg_len=2, min_len=2; phần dư quá ngắn nối vào đoạn trước. Hai mặt của kệ đảo có thể sinh hai slot khác nhau.

Mỗi slot có vùng tiếp xúc zone và điểm tiếp cận access. Khoảng cách đến kệ được đo từ access:

| Cột slots_*.csv | Ý nghĩa |
|---|---|
| slot_id | K001, K002…; nhãn hiển thị |
| cells | Ô kệ, tọa độ (row,col) bắt đầu từ 0, phân cách bằng dấu chấm phẩy |
| face | N/S/W/E: hướng mặt kệ |
| access_r, access_c | Ô đi được trước tâm slot |
| is_cold | Slot lạnh 0/1 |
| d_in | Cửa vào → access |
| d_out | Access → thu ngân gần nhất |
| e_geom | Chỉ số tiếp xúc hình học |

FloorPlan dùng **E đầu tiên** theo thứ tự duyệt lưới. Với nhiều C, mỗi chặng cuối chọn C gần nhất từ điểm xuất phát chặng đó. D là khoảng cách giữa các access; lối đi được cache khi mô phỏng.

e_geom đếm tiếp xúc của các tuyến cửa vào → từng slot → thu ngân, rồi chuẩn hóa min–max. Đây là chỉ số hình học, không phải tỷ lệ khách quan sát thực tế.

### 5.3. Vùng lạnh và lưới upload

make_instance gọi assign_cold để bố trí đủ slot lạnh, mặc định có cold_slack=0.2. Luồng này cũng áp dụng khi truyền grid tùy chỉnh, nên ký hiệu S/R trong tệp upload có thể bị thay đổi khi dựng instance. Cần đối chiếu slot lạnh sau dựng với bố trí vật lý của cửa hàng.

<a id="instance"></a>

## 6. Instance, biểu diễn nghiệm và ràng buộc

Nguồn triển khai: [src/instance.py](src/instance.py), [src/fastops.py](src/fastops.py).

Instance chọn n loại phổ biến nhất theo f, giữ thứ tự metadata của các loại đã chọn, lấy các cột tương ứng của giỏ CSR và bỏ giỏ trở thành rỗng. Nếu m>n, W/f/p/v được đệm bằng 0 cho m−n loại rỗng.

### 6.1. Hoán vị

~~~text
perm[k] = chỉ số nội bộ của loại hàng tại slot k
~~~

perm phải là hoán vị đầy đủ của 0…m−1. Ví dụ n=3, m=4, perm=[2,0,3,1]: slot 0 chứa loại nội bộ 2; slot 2 chứa dummy 3 và được xem là trống. Chỉ số loại lấy từ inst.meta; nó không phải category_id trong CSV.

slot_k là chỉ số 0-based trong instance. inst.slot_idx[slot_k] ánh xạ sang slot của FloorPlan; slot_id là nhãn Kxxx của FloorPlan. Chúng có thể khác thứ tự khi restrict_slots=True.

### 6.2. Ràng buộc

Với k_i là slot chứa loại i và k_i⁰ là vị trí hiện trạng:

| Ràng buộc | Quy tắc |
|---|---|
| Gán duy nhất | Mỗi loại thật một slot; mỗi slot tối đa một loại thật |
| Tương thích | Loại lạnh vào slot lạnh; loại thường vào slot thường |
| Cố định | fixed={category_id: slot_k}; slot đó không dành cho loại khác |
| Tách xa | D(k_i,k_j) ≥ delta cho cặp được khai báo |
| Ngân sách dời | Số loại thật có k_i khác k_i⁰ ≤ R |
| Không giới hạn dời | R=-1 |

R đếm **số loại hàng đổi slot**, không phải số thao tác swap, số kệ vật lý hay chi phí vận chuyển. Dummy không tính vào R. separate=None dùng các cặp mặc định; separate=[] bỏ các cặp đó khi bài toán thực sự không có yêu cầu tách xa.

Sơ đồ hiện trạng mẫu xếp theo ngành hàng và thứ tự snake trong từng lớp lạnh/thường. Nó không được dựng bằng cách giải toàn bộ ràng buộc separation/fixed. Vì vậy, phải kiểm tra cả hiện trạng:

~~~python
from src.instance import make_instance

inst = make_instance("grid", "medium")
print(inst.violations(inst.current))
print(inst.feasible(inst.current))
~~~

violations trả compat, separate, relocation_excess; feasible kiểm tra tổng bằng 0. API này giả định đầu vào đã là hoán vị đúng: nó không thay thế kiểm tra độ dài, miền chỉ số và tính duy nhất. Một số hiện trạng mẫu lớn có thể vi phạm cặp tách xa mặc định.

<a id="mo-hinh"></a>

## 7. Mô hình toán và chuẩn hóa

Nguồn triển khai: [src/instance.py](src/instance.py), [src/solvers.py](src/solvers.py).

### 7.1. Mục tiêu tiện lợi Z1 của QAP

$$
\min Z_1(\pi)=
\sum_{i<j} w_{ij}D(k_i,k_j)
+\sum_i f_i\left[d_{\mathrm{in}}(k_i)+d_{\mathrm{out}}(k_i)\right].
$$

Hạng đầu đưa các loại hay đồng mua gần nhau. Hạng sau đưa loại phổ biến đến vị trí thuận tiện với cửa vào và thu ngân. Chỉ cộng i<j để tránh đếm đôi.

Z1 là **đại lượng đại diện cho khoảng cách** dựa trên đồng mua. Nó không bằng độ dài một chuyến ghé nhiều điểm trong Simulator.

### 7.2. Mục tiêu giá trị Z2 của QAP

$$
\max Z_2(\pi)=\sum_i v_i r_i e(k_i),\qquad
r_i=
\begin{cases}
p_i & \text{khi Instance.q là None},\\
q_i & \text{khi đã có hệ số hiệu chỉnh}.
\end{cases}
$$

Mô hình gốc dùng p và e hình học. Mô hình hiệu chỉnh dùng q và e từ mô phỏng hiện trạng. e được giữ cố định trong từng instance QAP, dù luồng khách sẽ thay đổi khi đổi phương án.

### 7.3. Tổng có trọng số và payoff

Gọi pi₁ là nghiệm min Z1, pi₂ là nghiệm max Z2 dùng lập payoff:

| Mốc | Cách tính |
|---|---|
| Z1min | Z1(pi₁) |
| Z1max | Z1(pi₂) |
| Z2min | Z2(pi₁) |
| Z2max | Z2(pi₂) |

Z1max/Z2min là giá trị chéo tại hai nghiệm payoff, không phải cực đại/cực tiểu toàn cục riêng của từng mục tiêu.

$$
F_1=\frac{Z_1-Z_{1\min}}{Z_{1\max}-Z_{1\min}},\qquad
F_2=\frac{Z_{2\max}-Z_2}{Z_{2\max}-Z_{2\min}},\qquad
\min F_\alpha=\alpha F_1+(1-\alpha)F_2.
$$

alpha=1 ưu tiên tiện lợi; alpha=0 ưu tiên giá trị. compute_payoff điều chỉnh các mốc khi khoảng chuẩn hóa không dương; Instance.weighted còn chặn mẫu số tối thiểu 1e−9.

method="auto" chọn ILP khi n≤15 và m=n; ngoài trường hợp đó, min Z1 dùng GA và max Z2 dùng bài toán gán Hungarian. Nhánh heuristic trả exact=False. Nhánh ILP chỉ có exact=True khi cả hai lần giải chứng nhận tối ưu.

Hungarian tối ưu phần Z2 tuyến tính với ma trận allowed; nó chưa thực thi separation/R. Payoff heuristic có thể dùng nghiệm vi phạm để định thang. Cả payoff chéo và payoff xấp xỉ đều có thể làm F1/F2 vượt [0,1]; không so điểm chuẩn hóa từ hai instance/payoff khác nhau như cùng một thang đo.

<a id="thuat-toan"></a>

## 8. Thuật toán tối ưu

### 8.1. ILP: basic và RLT

[src/model_ilp.py](src/model_ilp.py) dùng PuLP với HiGHS mặc định, hoặc CBC. x_ik∈{0,1} biểu diễn loại i ở slot k; chỉ tạo x cho cặp tương thích. Biến y_ijkl∈[0,1] tuyến tính hóa tích x_ik x_jl cho i<j, k≠l và w_ij>0.

| Dạng | Ràng buộc liên kết chính |
|---|---|
| basic | y_ijkl ≥ x_ik+x_jl−1; hệ số khoảng cách không âm trong bài toán cực tiểu |
| rlt | Tổng theo l của y_ijkl = x_ik; tổng theo k = x_jl, trên các cặp biến được tạo |

RLT siết relaxation bằng các đẳng thức biên. Số biến y có thể tăng theo O(n²m²), nên ILP chủ yếu dùng để kiểm chứng quy mô nhỏ.

Các mode: z1, z2, weighted và eps. eps giải min Z1 với Z2≥epsilon. Kết quả có status, perm, số biến/ràng buộc, thời gian dựng/giải và optimal; HiGHS còn cung cấp bound/mip_gap. Giới hạn thời gian không đồng nghĩa đã tìm được tối ưu; kiểm tra perm không phải None, tính khả thi và chứng nhận/gap.

### 8.2. Genetic Algorithm

[src/ga.py](src/ga.py) thực hiện:

1. Khởi tạo khoảng 90% ngẫu nhiên, 10% tham lam.
2. Chọn bố mẹ bằng tournament.
3. Lai OX hoặc PMX, giữ cấu trúc hoán vị.
4. Đột biến swap/inversion/mixed.
5. Repair tương thích bằng swap; dự phòng bài toán gán Hungarian.
6. Giữ tinh hoa; local search 2-swap trên một phần cá thể tốt.
7. Chèn cá thể mới khi trì trệ; dừng theo số thế hệ, stall hoặc thời gian.

| GAConfig | Mặc định | Ý nghĩa |
|---|---:|---|
| pop_size / tournament_k | 100 / 3 | Quần thể / kích thước giải đấu |
| crossover / pc | ox / 0.9 | Lai ghép / xác suất |
| mutation / pm | swap / 0.2 | Đột biến / xác suất |
| elite / ls_frac | 2 / 0.1 | Tinh hoa / tỷ lệ tìm kiếm cục bộ |
| greedy_frac | 0.1 | Tỷ lệ khởi tạo tham lam |
| immigrant_every / immigrant_frac | 20 / 0.5 | Ngưỡng trì trệ / tỷ lệ thay |
| max_gens / stall_gens | 1000 / 100 | Giới hạn thế hệ / trì trệ |
| time_limit | 0.0 | 0: không giới hạn thời gian |
| rho / seed | 10.0 / 0 | Hệ số phạt / seed |

Fitness gồm mục tiêu và rho×(vi phạm tương thích + số cặp tách xa vi phạm + số loại dời vượt R). Repair chủ yếu bảo đảm compatibility khi bài toán gán tương thích có nghiệm; separation/R vẫn dùng penalty. **Không giả định mọi nghiệm GA đều khả thi.** Khi báo cáo mục tiêu thuần, tính lại bằng inst.z1/z2/z; fitness không phải cùng đại lượng đó.

Với a=perm[r], b=perm[s], W/D đối xứng, kernel [src/fastops.py](src/fastops.py) dùng:

$$
\Delta =
c_q\sum_{k\ne r,s}(W_{b,\pi_k}-W_{a,\pi_k})(D_{r,k}-D_{s,k})
+L_{b,r}+L_{a,s}-L_{a,r}-L_{b,s}.
$$

L và c_q do Instance.weighted tạo. Delta của phần mục tiêu tính O(m), thay cho tính lại O(m²); phần thay đổi penalty được xử lý riêng. Local search dùng first improvement. ls_max_moves=0 được triển khai thành mức tối đa 10m bước, không phải vòng lặp vô hạn.

### 8.3. NSGA-II và tập Pareto

[src/moo.py](src/moo.py) dùng pymoo NSGA2 với OrderCrossover, InversionMutation và CompatRepair. Hai mục tiêu cực tiểu là F1/F2; vi phạm được cộng penalty vào cả hai. ls_prob>0 bật biến thể lai, local search theo alpha ngẫu nhiên.

front_from_perms lọc khả thi, loại nghiệm bị trội, bỏ trùng giá trị mục tiêu và sắp theo Z1. Kết quả gồm perms, Z (Z1/Z2) và F (F1/F2). Một điểm trội điểm khác nếu không tệ hơn ở mọi mục tiêu và tốt hơn ít nhất một mục tiêu.

Điểm gối có khoảng cách lớn nhất tới đường nối hai cực biên trong không gian chuẩn hóa; với tập rất nhỏ/đường suy biến có fallback. Hypervolume mặc định dùng reference=(1.1,1.1); phải giữ cùng payoff/reference khi so các thuật toán. Quét alpha với GA chỉ xấp xỉ phần trade-off được weighted sum hỗ trợ, có thể bỏ sót phần Pareto không lồi.

Tập Pareto có thể rỗng khi không tìm được nghiệm khả thi. Các bước chọn min/gối trong pipeline chưa xử lý đầy đủ trường hợp đó.

### 8.4. Các phương pháp đối chứng

| Phương pháp tại [src/baselines.py](src/baselines.py) | Cách tìm nghiệm |
|---|---|
| Random | Lấy nhiều hoán vị tương thích, giữ nghiệm có fitness tốt nhất |
| Greedy | Gán loại phổ biến vào slot có e cao trong từng lớp lạnh/thường |
| SA | Đề xuất swap, chấp nhận cải thiện hoặc bước xấu theo nhiệt độ giảm |
| Tabu | Tìm swap tốt, dùng danh sách cấm và điều kiện aspiration |
| Hiện trạng | Sơ đồ ngành hàng/snake làm mốc so sánh |

Ngân sách E3 là thời gian tìm kiếm; biên dịch, chuẩn bị, bước polish hoặc hoàn tất một thế hệ có thể làm thời gian tổng khác ngân sách cấu hình.

<a id="mo-phong"></a>

## 9. Mô phỏng khách hàng

Nguồn triển khai: [src/simulate.py](src/simulate.py).

### 9.1. Một chuyến mua sắm

Simulator lấy mẫu có hoàn lại một giỏ từ CSR, xác định slot của các loại cần mua, sắp thứ tự ghé rồi nối các chặng bằng shortest path.

| strategy | Thứ tự ghé |
|---|---|
| tsp | Nearest neighbor từ E, rồi 2-opt cho đường mở kết thúc tại C |
| nn | Nearest neighbor, không 2-opt |
| snake | Theo thứ hạng snake cố định của slot |
| mixed | Trộn tsp/snake; mix_tsp mặc định 0.7 |

tsp là tên chiến lược trong code, không phải lời giải TSP tối ưu được chứng minh. snake_order được dựng bằng nearest neighbor trên các slot, nên không bảo đảm là đường chữ S vật lý chính xác trên mọi lưới.

Các slot có zone chạm đường đi được tính tiếp xúc tối đa một lần/chuyến. Với loại i tại slot có hàng:

$$
T_i=\exp(\log t_i+\sigma \xi),\qquad \xi\sim\mathcal N(0,1).
$$

t_i là dwell_median_s, sigma mặc định 0.6. Nếu i nằm trong danh sách mua, cộng pick_time_s=8 giây. Nếu chưa có trong danh sách, xác suất mua ngẫu hứng là:

$$
P_i=\operatorname{clip}(p_i\,p_{\mathrm{scale}},0,1)
\left(1-\exp(-\lambda T_i)\right).
$$

Một loại được tính một đơn vị giá trị v_i, không mô phỏng số lượng SKU. Tốc độ đi mặc định 1 m/s; thời gian chuyến gồm đi bộ và dừng.

### 9.2. KPI và bản đồ nhiệt

| KPI | Ý nghĩa/đơn vị | Diễn giải |
|---|---|---|
| distance_m | Khoảng cách trung bình, m/khách | Thấp hơn: ít đi bộ hơn |
| trip_time_min | Thời gian trung bình, phút/khách | Phụ thuộc đường đi, tốc độ, dwell và pick |
| exposed_slots | Số slot có hàng tiếp xúc trung bình | Độ phủ tiếp xúc theo mô hình |
| impulse_revenue | Tổng v của loại mua ngoài kế hoạch, TB/khách | Proxy giá trị, không mặc định là tiền |
| impulse_items | Số loại mua ngoài kế hoạch, TB/khách | Dùng hiệu chỉnh lambda |
| basket_value | Giá trị hàng kế hoạch + ngẫu hứng, TB/khách | Cùng đơn vị của v |
| congestion_cells | Số ô đạt ngưỡng mật độ trong lần chạy | Chỉ báo toàn lượt mô phỏng |
| planned_items | Số loại trong giỏ kế hoạch, TB/khách | Có trong kết quả run; không nằm trong KPI_COLS mặc định |

Kết quả run còn có traffic (lượt đi qua/khách), dwell (giây dừng/khách), peak, exposure_rate và tùy chọn customers.

Khách đến theo khoảng chờ mũ, mặc định 120 khách/giờ. Ùn tắc dùng bucket 30 giây, ngưỡng 4 khách cùng ô. Thời gian dừng được phân bổ xấp xỉ dọc tuyến khi tính thời điểm qua ô; khách không điều chỉnh đường/tốc độ để né nhau.

### 9.3. Lặp và khoảng tin cậy

replicate dùng seed_r=seed+1000r. summarize và final_eval trả mean cùng ci95, trong đó ci95 là **nửa độ rộng** khoảng tin cậy:

$$
\bar x\ \pm\ t_{0.975,r-1}\frac{s}{\sqrt r}.
$$

Đơn vị quan sát ở đây là KPI của mỗi lần lặp, không phải từng khách gộp lại. Với một lần lặp, code trả ci95=0; không diễn giải điều đó là không có bất định.

<a id="pipeline"></a>

## 10. Pipeline hiệu chỉnh, chọn và đánh giá

Nguồn triển khai: [src/pipeline.py](src/pipeline.py), [src/routing.py](src/routing.py).

### 10.1. Hiệu chỉnh Model

calibrate_model tìm lambda bằng mở rộng cận và chia đôi để số loại mua ngẫu hứng trên hiện trạng gần target_impulse_items, mặc định 1.5. Đây là mục tiêu giả định; thuật toán chưa xác nhận đầy đủ mục tiêu có đạt được trong dữ liệu hiện tại hay không.

Sau đó, pipeline lấy e từ exposure_rate và tính:

$$
q_i=p_i(1-f_i)\left(1-\exp(-\lambda t_i)\right).
$$

Hệ số 1−f_i đại diện khả năng loại i chưa nằm trong giỏ kế hoạch. Việc dùng t_i trung vị trong q là xấp xỉ; nó khác việc lấy kỳ vọng trên toàn phân phối dwell.

exposure_rate lấy tỷ lệ khách tiếp xúc từng slot; khi rate=0, code thay bằng traffic trung bình trong zone. Traffic tính lượt qua và có thể gồm lượt quay lại, nên **e fallback có thể lớn hơn 1**. Calibration không dùng min–max. Helper simulated_exposure mới chuẩn hóa min–max; hai hàm không có cùng ý nghĩa.

model_validity dùng Spearman giữa Z1 và distance/trip_time, giữa Z2 và impulse/basket. Đây là kiểm tra mức phù hợp về thứ hạng với Simulator, không phải kiểm định nhân quả hoặc chứng minh mô hình phản ánh cửa hàng thật. sample_layouts chỉ sửa compatibility, chưa bảo đảm separation/R.

### 10.2. Hai lựa chọn mục tiêu và RouteModel

| objective | Cách đánh giá khi tối ưu |
|---|---|
| model | QAP hiệu chỉnh: e/q cố định |
| route | Tuyến NN + open 2-opt của từng giỏ mẫu; tiếp xúc phụ thuộc phương án |

Nhánh model tạo NSGA-II lai và thêm GA ở alpha=0/1. Nhánh route tạo tập QAP trước, rồi dùng hiện trạng và một phần nghiệm QAP làm seeds cho NSGA-II theo tuyến đi.

RouteModel chọn mẫu cố định mặc định tối đa 1.500 giỏ không hoàn lại, gom giỏ trùng thành tập loại kèm trọng số. Khoảng cách và tiếp xúc của các chặng được tiền tính; đánh giá phương án tính lại thứ tự tuyến NN+2-opt bằng Numba và cache theo hoán vị.

Với L_b là tập loại kế hoạch, E_b(pi) là tập slot tuyến giỏ b tiếp xúc, omega_b là tỷ trọng giỏ:

$$
Z_1^{\mathrm{route}}(\pi)=\sum_b\omega_b\,\mathrm{length}(\mathrm{route}_b(\pi)),
$$

$$
Z_2^{\mathrm{route}}(\pi)=
\sum_b\omega_b
\sum_{\substack{i\notin L_b\\k_i\in E_b(\pi)}}
v_i\,\operatorname{clip}(p_i p_{\mathrm{scale}},0,1)\,
\mathbb E\left[1-\exp(-\lambda T_i)\right].
$$

Kỳ vọng dwell được xấp xỉ bằng 20.000 mẫu chuẩn chung, seed=0. Đánh giá tất định khi mẫu/cấu hình cố định; không đồng nghĩa chính xác với mọi hành vi khách. RouteModel dùng tuyến tsp, chưa mô phỏng trực tiếp nn/snake/mixed, ùn tắc hoặc thời gian chuyến.

RouteObjective định thang từ min/max của các seed layouts, không phải cực trị toàn cục. Hybrid local search nhận kernel của QAP bên dưới; **không phải mọi bước tìm kiếm ở nhánh route đều cải thiện trực tiếp mục tiêu route**. Các ứng viên cuối vẫn được đánh giá lại bằng RouteModel và lọc khả thi.

### 10.3. Sàng lọc, hồ sơ và tinh chỉnh

screen mô phỏng mọi ứng viên với seed chung, mặc định SCREEN_SEED=0. include_current=True thêm hiện trạng để so sánh/chọn; API có thể tắt tùy chọn này.

Với d/r là distance_m/impulse_revenue và d₀/r₀ là hiện trạng:

$$
S_\beta=100\left[
\beta\frac{r-r_0}{\max(r_0,10^{-12})}
-(1-\beta)\frac{d-d_0}{d_0}
\right].
$$

Điểm cao hơn tốt hơn; cần baseline có d₀>0.

| Hồ sơ | beta | Cách chọn |
|---|---:|---|
| Tiện lợi | 0 | Ưu tiên giảm khoảng cách |
| Cân bằng | 0.5 | Ưu tiên ứng viên không tệ hơn hiện trạng ở cả hai KPI nếu có; rồi tối đa điểm |
| Giá trị | 1 | Ưu tiên tăng giá trị ngẫu hứng |

**alpha là trọng số khoảng cách trong mục tiêu QAP; beta là trọng số giá trị trong quyết định bằng KPI.** Điểm gối theo mô hình và lựa chọn Cân bằng theo mô phỏng là hai cách chọn khác nhau.

refine thử swap tương thích, không nhận bước làm tăng tổng vi phạm. Nó nhận bước có score cao hơn **hoặc** có ít vi phạm hơn. Từ nghiệm khả thi, score sàng lọc không giảm; từ nghiệm vi phạm, score có thể giảm để giảm vi phạm. n_evals là số lần thử, không bảo đảm từng lần đều là swap hợp lệ được mô phỏng.

Với Cân bằng, refine đặt `need_dom=True` nếu điểm xuất phát không tệ hơn baseline ở cả hai KPI. Khi đó, mọi bước được nhận cũng phải thỏa điều kiện này trên mẫu sàng lọc. Nếu điểm xuất phát không đạt điều kiện, safeguard đó không được bật. Điều kiện này không bảo đảm hai KPI trên đánh giá cuối độc lập. route_search dùng nguyên tắc không tăng vi phạm và cải thiện điểm hoặc giảm vi phạm trên mục tiêu route với ngân sách R; muốn giữ khả thi phải bắt đầu từ nghiệm khả thi.

### 10.4. Đánh giá cuối và đầu ra

final_eval dùng seed=50.000+1000r, tách khỏi các seed dùng chọn/tinh chỉnh ở cấu hình mặc định. Các phương án dùng cùng seed theo lần lặp để so sánh cặp. Nếu tăng số lần sàng lọc đủ lớn, cần kiểm tra lại các miền seed không chồng lấn.

Kết quả gồm final_runs và final: mean, ci95, change_pct và p_value Wilcoxon theo KPI so với Hiện trạng. change_pct dương với distance nghĩa là đi xa hơn; dương với impulse nghĩa là giá trị tăng. Pipeline không tự hiệu chỉnh Holm cho tất cả kiểm định.

recommendation chọn bản tinh chỉnh nếu có, nếu không chọn bản đã sàng lọc. Nó không dùng bảng final để chọn lại; đánh giá cuối phục vụ báo cáo. Seed chung hỗ trợ so sánh, nhưng do đường đi khác nhau làm số lần rút ngẫu nhiên khác nhau, không phải mọi biến hành vi đều được ghép hoàn toàn giữa hai phương án.

| PipelineConfig | Mặc định |
|---|---|
| n_customers / target_impulse_items | 2.000 / 1.5 |
| nsga_time / seed | 20 giây / 0 |
| screen_rep / refine_evals | 2 / 150 |
| final_customers / final_rep | 3.000 / 30 |
| n_random_validity | 40 |
| profiles | Tiện lợi, Cân bằng, Giá trị |
| objective / route_baskets | model / 1.500 |
| include_current | True |

nsga_time là ngân sách từng lượt NSGA-II, không phải thời gian toàn pipeline. Payoff, GA cực biên, tiền tính route, hiệu chỉnh và mô phỏng tốn thời gian thêm.

### 10.5. Ví dụ gọi API

Chạy đoạn sau bằng Python từ smart-retail-layout/. Ví dụ giảm quy mô để kiểm tra luồng API; final_rep=3 chỉ phù hợp chạy thử:

~~~python
from src.instance import make_instance
from src.pipeline import PipelineConfig, recommendation, run_pipeline

inst = make_instance("grid", "small", n=8, restrict_slots=True)
if not inst.feasible(inst.current):
    raise ValueError(f"Hiện trạng vi phạm: {inst.violations(inst.current)}")

cfg = PipelineConfig(
    objective="model",
    n_customers=100,
    nsga_time=1,
    screen_rep=1,
    refine_evals=0,
    final_customers=100,
    final_rep=3,
    n_random_validity=2,
    profiles=("Cân bằng",),
    seed=0,
)
res = run_pipeline(inst, cfg)
name = recommendation(res, "Cân bằng")
perm = res.plans[name]
if not inst.feasible(perm):
    raise ValueError(f"Phương án vi phạm: {inst.violations(perm)}")
print(name)
print(inst.assignment_frame(perm))
print(res.final)
~~~

PipelineResult còn cung cấp base, cal, calib, validity, front, screen, ref, plans, plan_source, refine_hist và obj. Đổi objective="route" để thêm bước tối ưu theo tuyến; không đổi ý nghĩa Simulator đánh giá cuối.

<a id="dashboard"></a>

## 11. Dashboard Streamlit

Nguồn triển khai: [app/streamlit_app.py](app/streamlit_app.py). Khởi chạy bằng python -m streamlit run app/streamlit_app.py.

Sidebar chọn mặt bằng mẫu/upload, nguồn giỏ hàng, số loại phổ biến, ràng buộc tách xa/cố định, R và số khách mô phỏng. Mặt bằng large với dữ liệu group được chuyển sang mức aisle trong giao diện.

| Tab | Thao tác chính | Kết quả |
|---|---|---|
| ① Dữ liệu | Đọc liên kết đồng mua, metadata và mặt bằng | Heatmap/bảng cặp liên kết, sơ đồ hiện trạng |
| ② Mô hình | Hiệu chỉnh lambda/e/q; chạy kiểm tra thứ hạng | Hệ số hiệu chỉnh và bảng Spearman |
| ③ Tối ưu | Chọn mục tiêu, tạo tập Pareto | Biểu đồ Z1/Z2, điểm cực biên/gối |
| ④ Mô phỏng | Sàng lọc toàn bộ ứng viên, chọn hồ sơ và tinh chỉnh | KPI so hiện trạng, phương án theo mô hình/mô phỏng |
| ⑤ Quyết định | Chạy đánh giá cuối bằng seed riêng | KTC, kiểm định, sơ đồ và các tệp xuất |

Giao diện mặc định chọn mục tiêu **Định tuyến**; PipelineConfig của API mặc định **model**. Các ngân sách slider trên UI cũng khác mặc định API/config thí nghiệm.

Tab quyết định xuất PNG sơ đồ, CSV gán nhóm→slot và CSV danh sách cần dời. relocation_sequence ưu tiên swap theo weighted QAP hiệu chỉnh; cột cải thiện cộng dồn là tiến độ mục tiêu QAP từ hiện trạng tới đích, không phải tỷ lệ tăng doanh thu route/mô phỏng. Chuỗi trung gian chưa bảo đảm separation/R hoặc hoàn thành mọi phép dời.

Kết quả trong session được gắn với key cấu hình. Tuy nhiên key chưa bao gồm đầy đủ nội dung lưới, file upload và các tùy chọn; thay dữ liệu cần làm sạch cache/kết quả cũ và chạy lại các khâu phụ thuộc. Trước khi dùng bảng xuất, kiểm tra hoán vị, inst.feasible và các ràng buộc thực tế.

<a id="thi-nghiem"></a>

## 12. Thí nghiệm E1–E7

Nguồn triển khai: [experiments/config.yaml](experiments/config.yaml), [experiments/common.py](experiments/common.py), [experiments/run_all.py](experiments/run_all.py).

### 12.1. Mục tiêu từng thí nghiệm

| Mã / script | Câu hỏi | Đầu ra tiêu biểu |
|---|---|---|
| E1 — [e1_qaplib.py](experiments/e1_qaplib.py) | GA khớp lời giải chuẩn QAPLIB đến mức nào? | runs.csv, summary.csv |
| E2 — [e2_ga_vs_ilp.py](experiments/e2_ga_vs_ilp.py) | GA so với ILP; basic so với RLT ở quy mô nhỏ | ilp.csv, runs.csv, summary.csv |
| E3 — [e3_heuristics.py](experiments/e3_heuristics.py) | GA/SA/Tabu so với Greedy/Random/Hiện trạng | deterministic.csv, runs.csv, summary.csv, wilcoxon.csv, nemenyi.csv |
| E4 — [e4_pareto.py](experiments/e4_pareto.py) | NSGA-II thuần/lai, epsilon-constraint và quét alpha | runs.csv, summary.csv; hypervolume/Pareto |
| E5 — [e5_simulation.py](experiments/e5_simulation.py) | Toàn pipeline trên 9 mặt bằng + minimart; model so với route, 3 hồ sơ và các cách chọn | model_validation.csv, screen.csv, final_runs.csv, final_summary.csv, plans.npz |
| E6 — [e6_sensitivity.py](experiments/e6_sensitivity.py) | Độ nhạy lambda/p, nhiễu p, cách đi, tốc độ; hiệu quả theo R | runs.csv, summary.csv, rank_stability.csv, R_curve.csv, R_curve_route.csv |
| E7 — [e7_ablation.py](experiments/e7_ablation.py) | Bóc tách exposure, p/q, lặp cập nhật exposure, route và local search GA | ls_runs.csv, exposure_variants.csv, exposure_iterations.csv, exposure_dependence.csv |
| [extra_stats.py](experiments/extra_stats.py) | Cận dưới E2; so cặp E3 bằng Mann–Whitney, Holm và A12 | E2/lower_bound.csv, E3/pairwise_holm_a12.csv |

Các artifact là đầu ra theo code path tương ứng; không phải mọi cấu hình quick đều sinh đầy đủ mọi bảng phụ. E3 có Wilcoxon, Friedman/Nemenyi; extra_stats áp dụng Holm cho họ so cặp trong từng instance. A12>0.5 nghĩa là thuật toán a có xu hướng đạt Z thấp hơn b. Điều này không áp dụng tự động cho mọi p_value E5.

### 12.2. full và quick

full.n_runs=30, quick.n_runs=3 cho những script dùng trường n_runs. Số lần lặp mô phỏng E5/E6/E7 được cấu hình riêng; không gọi toàn bộ chiến dịch là “30 lần cho mọi khâu”.

| Khâu | full | quick |
|---|---|---|
| E1 | 10 bộ QAPLIB; 10 giây/lần | nug12, had12, nug20; 3 giây/lần |
| E2 | n=8,10,12,15,20; ILP 1.800 giây; basic tới n=12 | n=8,10; ILP 120 giây; basic tới n=8 |
| E3 | 3 kiểu × medium/large; ngân sách 10/30 giây | grid/medium; 2 giây |
| E4 | Epsilon 15 điểm; NSGA 20 giây, pop 100 | Epsilon 5 điểm; NSGA 3 giây, pop 60 |
| E5 | 2.000 khách sàng lọc, 2 lần lặp, 150 lần thử tinh chỉnh; cuối 3.000 khách × 30 lần | 600 khách, 1 lần, 10 lần thử; cuối 600 × 3 |
| E6 | 3.000 khách × 10; route_iters=4.000; R=0,2,4,6,8,10,15,20,30,-1 | 800 × 2; route_iters=300; R=0,4,-1 |
| E7 | Exposure tối đa 8 vòng; route_iters=20.000; 3.000 khách × 10 | 2 vòng; route_iters=500; 800 × 2 |

load_config dùng **ghi đè nông ở cấp đầu tiên**: quick.e5 thay toàn bộ full.e5, không trộn sâu từng field. Nếu bổ sung cấu hình, giữ đủ trường script truy cập hoặc kiểm tra default thực sự được sử dụng. E5 quick không ghi versions nhưng script mặc định vẫn chạy cả model và route.

### 12.3. Cách chạy và workers

~~~bash
# Một phần thí nghiệm để kiểm tra
python -m experiments.run_all --profile quick --workers 2 --only e1 e3

# Toàn bộ cấu hình quick hoặc full
python -m experiments.run_all --profile quick --workers 2
python -m experiments.run_all --profile full --workers 2

# Chạy riêng E5
python -m experiments.e5_simulation --profile quick --workers 2

# Sau khi đã có dữ liệu E2/E3, tính thống kê bổ sung
python -m experiments.extra_stats
~~~

--only **phân biệt hoa/thường** và nhận token e1…e7, extra. E1/E3 hoặc extra_stats không phải token lọc hợp lệ. Với extra_stats chạy trực tiếp, không thêm --profile/--workers vì module này không xử lý các cờ đó.

Thứ tự mặc định là E1 → E3 → E4 → E5 → E6 → E7 → E2 → extra_stats. Workers mặc định max(1, CPU−2); có thể đặt 1 để hạn chế RAM/CPU. Mỗi worker warmup Numba trước tìm kiếm; nhiều worker và luồng solver có thể cạnh tranh tài nguyên.

run_all ghi log mã trả về rc của từng subprocess nhưng chưa biến rc khác 0 thành mã lỗi tổng thể. **Phải kiểm tra từng rc và output**, không chỉ trạng thái process cha. Thời gian tổng còn gồm hiệu chỉnh, payoff và mô phỏng; quick không có thời gian hoàn thành cố định.

<a id="tai-lap"></a>

## 13. Đọc kết quả và tái lập

### 13.1. Vị trí kết quả

| Đường dẫn | Vai trò |
|---|---|
| [results/E1](results/E1)…[results/E7](results/E7) | Dữ liệu thô, bảng tổng hợp và run_config.yaml từng thí nghiệm |
| [results/figures](results/figures) | Hình Pareto, mặt bằng, heatmap và thống kê |
| results/run_all.log | Log nối thêm: tên module, rc, thời gian |
| `results/<module>.out.txt` | Stdout/stderr từng module; có thể bị ghi đè |
| [results/payoff_cache.json](results/payoff_cache.json) | Mốc chuẩn hóa dùng chung các lần chạy |
| [results/_v1_model_goc](results/_v1_model_goc), [results/_v2_hieu_chinh](results/_v2_hieu_chinh) | Kết quả lưu của các phiên bản mô hình trước |

Có đủ thư mục E1–E7 không chứng minh một chiến dịch hoàn chỉnh đã được chạy lại trên HEAD hiện tại. Đối chiếu số instance, seed, rep, profile và log trước khi dùng các bảng cho báo cáo.

E5 có recommendations.csv với cột source để truy nguồn chọn phương án; trong API, thông tin tương ứng nằm ở PipelineResult.plan_source. change_vs_current.csv và by_method.csv tổng hợp thay đổi. plans.npz dùng khóa version|instance|plan; phân biệt model/route khi đọc. Các số liệu cuối phải lấy từ final_runs/final_summary, không thay bằng KPI sàng lọc.

### 13.2. Thông tin cần lưu cho một lần chạy

Ghi commit Git, thay đổi chưa commit, Python/nền tảng, requirements thực tế, hash raw/mapping/processed, profile và cấu hình đầy đủ, command, workers/solver threads, seed, payoff và toàn bộ output.

run_config.yaml ghi thời điểm, Python/platform, một số phiên bản thư viện và phần config mà script truyền vào; không tự ghi mọi yếu tố trên, không tự ghi Git commit hay hash input. Manifest cung cấp hash đầu ra rút gọn để phát hiện lệch dữ liệu.

Các seed được dùng ở nhiều tầng: lấy mẫu giỏ, GA/NSGA-II, mẫu RouteModel, mô phỏng sàng lọc và đánh giá cuối. Cùng seed với thuật toán dừng theo thời gian vẫn có thể cho số thế hệ khác giữa các máy. Không cam kết kết quả bitwise giống nhau chỉ từ seed.

### 13.3. Cache và kết quả cũ

Payoff cache được khóa theo tên spec (kiểu/quy mô/n/restrict/R), chưa bao gồm toàn bộ nội dung dataset, level, grid và tham số mô hình. Dashboard upload key dựa tên/kích thước tệp, còn key của các bước chưa đầy đủ. Khi đổi dữ liệu/cấu hình, giữ bản kết quả cũ rồi vô hiệu hóa cache liên quan và chạy lại; không tái dùng payoff cũ chỉ vì tên spec giống nhau.

<a id="kiem-thu"></a>

## 14. Kiểm thử và xử lý lỗi chạy

### 14.1. Kiểm tra cơ bản

~~~bash
python -m pip check
python -m pytest -q
python -m experiments.run_all --help
~~~

[tests/test_core.py](tests/test_core.py) kiểm tra:

- Khoảng cách metric, tách slot và gán vùng lạnh.
- Delta swap so với tính lại mục tiêu; local search không làm xấu fitness.
- ILP basic/RLT so với vét cạn n=7; GA so với ILP ở bài toán nhỏ.
- Tính lặp lại mô phỏng theo seed và các cực trị hành vi.
- Hệ số q/e hiệu chỉnh, chọn hồ sơ, refinement từ nghiệm khả thi.
- RouteModel so với mô phỏng trong dung sai; route_search giữ R từ điểm xuất phát khả thi.
- Giá trị QAPLIB và tính hợp lệ của OX/PMX.

Baseline f4ab327 được kiểm tra ngày 2026-10-04: **18 tests passed**, Python 3.12.10/Windows, pip check sạch. Số test là ảnh chụp tại baseline; chạy lại để kiểm tra checkout đang dùng. Tests không chứng minh campaign E1–E7 đầy đủ hoặc mô hình đúng với cửa hàng thực.

### 14.2. Lỗi thường gặp

| Hiện tượng | Kiểm tra/cách xử lý |
|---|---|
| No module named experiments/src | Chạy lệnh từ smart-retail-layout/, dùng Python đúng venv |
| UnicodeEncodeError trên Windows | Đặt $env:PYTHONIOENCODING="utf-8" trước khi chạy |
| Không có `processed/<level>/w.npy` | Kiểm tra checkout dữ liệu; xây lại bằng build_data khi đủ raw CSV |
| Các dòng lưới không cùng chiều dài | Sửa file txt; dùng ký hiệu quy định, có E/C và lối tiếp cận |
| Không đủ slot lạnh/thường hoặc m<n | So số loại cần gán với số slot và tương thích sau dựng |
| GA trả nghiệm vi phạm | Xem violations, kiểm tra tính khả thi của ràng buộc và ngân sách; không dùng fitness tốt làm bằng chứng |
| Pareto rỗng/lỗi chọn cực biên | Kiểm tra đầu vào/ràng buộc; code chưa xử lý đầy đủ tập ứng viên rỗng |
| run_all kết thúc nhưng thiếu output | Đọc rc trong log và traceback tại `<module>.out.txt` |
| Đổi upload nhưng thấy kết quả cũ | Làm sạch cache/session liên quan và chạy lại pipeline |

<a id="gioi-han"></a>

## 15. Giới hạn hiện tại

- **Nguồn dữ liệu:** Instacart là đơn online; đồng mua không trực tiếp mô tả luồng khách tại cửa hàng. p từ reorder, v và dwell là giả định/đầu vào, cần đối chiếu dữ liệu thực.
- **Sức chứa:** Một loại thật dùng một slot; n_slots chưa điều khiển phân bổ nhiều slot. Chưa mô hình hóa SKU, số lượng, mặt trưng bày và sức chứa tồn kho chi tiết.
- **Hình học:** Lưới 1 m, bốn hướng, cửa vào đầu tiên và thu ngân gần nhất. Gán vùng lạnh tự động có thể thay lưới upload.
- **Khả thi:** GA dùng penalty; repair và random layouts chủ yếu xử lý compatibility. Hiện trạng/fixed/separation có thể xung đột. Pipeline có thể thêm hiện trạng vi phạm vào screening; dashboard chưa có chốt đầy đủ ngăn xuất mọi phương án vi phạm.
- **Mô hình đại diện:** Z1 QAP không phải độ dài chuyến. Exposure QAP cố định theo baseline; fallback traffic chưa luôn nằm [0,1]. Payoff heuristic chưa luôn khả thi và không phải cận tối ưu.
- **Theo tuyến đi:** Mẫu giỏ hữu hạn, NN+2-opt và kỳ vọng dwell xấp xỉ; local search lai vẫn dùng kernel QAP. RouteModel không thay toàn bộ Simulator.
- **Mô phỏng/thống kê:** Không có né va chạm/hàng chờ động đầy đủ; ùn tắc là xấp xỉ. Final dùng seed riêng theo mặc định, nhưng cải thiện trên screen không bảo đảm trên final. Holm không áp dụng tự động cho mọi so sánh.
- **Dời hàng:** R chỉ đếm nhóm đổi slot. Chuỗi dời tham lam chưa kiểm tra mọi ràng buộc ở trạng thái trung gian, chưa bảo đảm hoàn thành hoặc tối ưu chi phí vận hành.
- **Vận hành/tái lập:** Cache chưa nhận diện đầy đủ nội dung đầu vào; run_all chưa tổng hợp lỗi subprocess thành thất bại chung. Kết quả cũ có thể thuộc nhiều lần chạy/cấu hình.

Trước khi áp dụng cho cửa hàng, đo/hiệu chỉnh p, v, dwell và lambda; đối chiếu hình học/vùng lạnh; kiểm tra current và phương án bằng feasible; đánh giá độ nhạy và KPI trên dữ liệu độc lập. Các yêu cầu vận hành chưa mô hình hóa phải được người quản lý đánh giá bổ sung.

<a id="quy-trinh-ai"></a>

## 16. Quy trình phát triển với ECC và AGENTS.md

Quy tắc AI riêng của workspace được mô tả trong [AGENTS.md ở gốc repository](../AGENTS.md), nếu tệp này có trong checkout. ECC hỗ trợ cập nhật tài liệu; skill source-command-update-docs yêu cầu đối chiếu source/config/commands thay vì suy diễn từ tài liệu cũ.

Với công việc không đơn giản, AGENTS.md quy định ChatGPT Web trong Project đã cấu hình làm tác nhân phân tích/lập kế hoạch/review; Codex thu thập ngữ cảnh, sửa tệp, chạy kiểm tra và báo kết quả. ChatGPT Project được truy cập bằng MCP playwright và phải được xác minh đúng Project trước khi gửi ngữ cảnh.

~~~text
Yêu cầu → đọc AGENTS.md + trạng thái Git + source hiện tại
        → tham vấn ChatGPT Project
        → Codex sửa đúng phạm vi
        → build/test/kiểm tra liên quan
        → gửi kết quả cho ChatGPT Project review
        → áp dụng điều chỉnh cần thiết → kiểm tra cuối → hoàn thành
~~~

Khi thay cấu trúc/mục tiêu/loader/config, cập nhật các phần tương ứng của README, kiểm tra lệnh, link và công thức. Giữ thay đổi chưa commit của người dùng; không gửi secrets; không dùng hội thoại ngoài Project làm phương án thay thế khi Project không truy cập/xác minh được.

Các phần kỹ thuật của README bám baseline ghi ở đầu; khi source thay đổi, source và bằng chứng chạy hiện tại là căn cứ để cập nhật tài liệu.

<!-- END AUTO-GENERATED -->
