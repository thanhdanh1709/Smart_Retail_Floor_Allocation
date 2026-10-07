# Kiểm chứng trích dẫn và nguồn dữ liệu (GĐ0 kế hoạch v4)

> Kiểm tra ngày 2026-10-07 bằng tìm kiếm web. "Đã xác nhận" = tìm thấy trang nhà xuất bản / kho lưu trữ khớp tác giả, năm, nơi đăng.
> Trước khi nộp luận văn vẫn phải đọc bản gốc để trích đúng số liệu.

## Trích dẫn

| Tài liệu (kế hoạch mục 11) | Trạng thái | Thông tin chính xác | Ghi chú cho luận văn |
|---|---|---|---|
| Lee et al. (AAMAS 2026) | Đã xác nhận | Ken Ming Lee, Paul Barde, Maxime C. Cohen, Derek Nowrouzezahrai. *Modelling Customer Trajectories with Reinforcement Learning for Practical Retail Insights*. AAMAS 2026; arXiv 2605.18449 | Lệch trung bình **28%** so với đường ngắn nhất: đúng. Mô hình của họ là **RL entropy cực đại** (tương đương recursive logit trên đồ thị), không gọi là "recursive logit" – trích đúng tên |
| Flamand et al. | Đã xác nhận | Flamand, Ghoniem, Maddah (và cộng sự). *Store-Wide Shelf-Space Allocation with Ripple Effects Driving Traffic*. Operations Research, 2023 (doi 10.1287/opre.2023.2437) | Mô hình hồi quy lưu lượng phụ thuộc phân bổ ("ripple effect") + MINLP; +65% lợi nhuận ngẫu hứng (Beirut). Là công trình gần nhất với "e_k phụ thuộc x" – **phải trích và so sánh** |
| "Playing hide and seek…" | Đã xác nhận | *Playing hide and seek: tackling in-store picking operations while improving customer experience*, arXiv 2301.02142 | Bài toán diPRP: người nhặt tránh khách, giảm >50% lần chạm mặt. Họ **cho sẵn sơ đồ**; v4 tối ưu sơ đồ – khác biệt cần nêu |
| Fosgerau, Frejinger & Karlström (2013) | Đã xác nhận | *A link based network route choice model with unrestricted choice set*. Transportation Research Part B 56:70–80. doi 10.1016/j.trb.2013.07.012 | Nguồn của recursive logit |
| Larson, Bradlow & Fader (2005) | Đã xác nhận | *An exploratory look at supermarket shopping paths*. Int. J. Research in Marketing 22(4):395–414 | RFID trên xe đẩy; 14 kiểu đường đi; bác bỏ một phần "racetrack" – lưu ý khi gọi mô hình PER là "tái hiện racetrack" |
| Hui, Fader & Bradlow (2009) | Đã xác nhận (2 bài) | (a) *Path data in marketing: an integrative framework…* Marketing Science 28(2):320–335; (b) *The traveling salesman goes shopping: the systematic deviations of grocery paths from TSP optimality*. Marketing Science 28(3):566–572 | Kế hoạch ghi "Hui et al. 2009" – dùng bài (b) cho sự thật cách điệu |
| Corstjens & Doyle (1981) | Đã xác nhận | *A model for optimizing retail space allocations*. Management Science 27(7):822–833 | Co giãn không gian chính + chéo |
| Drèze, Hoch & Purk (1994) | Đã xác nhận | *Shelf management and space elasticity*. Journal of Retailing 70(4):301–326 | **Vị trí** ảnh hưởng mạnh, **số mặt** ít ảnh hưởng khi trên ngưỡng tối thiểu → ủng hộ β nhỏ và φ_l lớn trong tầng 3 |
| Bianchi-Aguiar et al. (review SSAP) | Đã xác nhận, **sửa tác giả** | Bianchi-Aguiar, **Hübner, Carravilla, Oliveira** (2021). *Retail shelf space planning problems: a comprehensive review and classification framework*. EJOR 289(1):1–16 | Kế hoạch ghép nhầm với Kuhn |
| Hübner & Kuhn (2012) | Chưa xác nhận trên web lần này | Dự kiến: *Retail category management: state-of-the-art review of quantitative research and software applications in assortment and shelf space management*, Omega 40(2) | Kiểm tra lại trước khi trích |
| Ratliff & Rosenthal (1983); Roodbergen & de Koster (2001) | Chưa tìm lần này (tài liệu kinh điển) | R&R: *Order-picking in a rectangular warehouse: a solvable case of the TSP*, Operations Research 31(3). R&dK: *Routing methods for warehouses with multiple cross aisles*, IJPR 39(9) | Kiểm tra số trang |
| Gue & Meller (2009) | Đã xác nhận | *Aisle configurations for unit-load warehouses*. IIE Transactions 41(3):171–182 | Lối chéo giảm >20% quãng đường **chu trình đơn (unit-load)** – không phải nhặt nhiều điểm; nêu rõ khi chuyển ý sang cửa hàng |
| Ozgormus & Smith | Đã xác nhận một phần | Ozgormus & Smith (2020) – bố trí khối cửa hàng tạp hóa, doanh thu + ưu tiên kề nhau, kiểm chứng ở Migros (+3–4% doanh thu); luận án Auburn *Optimization of Block Layout for Grocery Stores* | Cần tra tên tạp chí chính xác |
| Botsali & Peters (2005) | Đã xác nhận nội dung | Mô hình mạng cho bố trí **serpentine**, tối đa doanh thu ngẫu hứng với hệ số nhìn thấy theo số lần sản phẩm nằm cạnh đường đi | Tra tên tạp chí chính xác |

## Dữ liệu quỹ đạo khách (để hiệu chỉnh tầng 4)

| Nguồn | Tình trạng | Dùng được không |
|---|---|---|
| "HRN4Customer" | **Không tìm thấy** bộ dữ liệu nào tên này | Bỏ khỏi kế hoạch |
| Shopper Trajectories Dataset (VRAI, ĐH Politecnica delle Marche) – 3 cửa hàng (Indonesia, Đức) | Chỉ cấp cho nghiên cứu, phải xin | Có thể xin; nếu được, dùng ước lượng hợp lý cực đại μ của RL. Không phụ thuộc vào nó |
| Wharton PathTracker (Larson et al.) | Không công khai | Không |
| Dữ liệu TSURUHA (Nhật), cửa hàng tiện lợi (Lee et al.) | Riêng tư | Không |

**Kết luận GĐ0:** giữ phương án hiệu chỉnh bằng **khớp mô-men với sự thật cách điệu** (lệch 28% – Lee et al.; độ lệch TSP – Hui et al.; kiểu đường – Larson et al.). μ chưa hiệu chỉnh được → đưa vào tập M.

## Dữ liệu giá (tầng 3)
Chưa có nguồn giá công khai phù hợp với Instacart (Dunnhumby "The Complete Journey" cần đăng ký, điều khoản cần đọc). GĐ7 dùng giả định giá/lãi theo department + phân tích độ nhạy.

## Dữ liệu đã dựng ở GĐ0
- `data/processed/items/items.csv`: 47.874 món; giữ món ≥ 100 lượt mua → 19.870 món, **97,3%** lượt mua.
- `data/processed/hour_profile.csv`: λ_P(h) từ `orders.order_hour_of_day` (thật; đỉnh 10–15h); λ_W(h) giả định hai đỉnh 11h30, 18h (giờ mở 7–22h).
- `FloorPlan`: ô `P` = khu tập kết đơn online; `d_0` = khoảng cách P → slot (không có P thì lấy cửa vào).
