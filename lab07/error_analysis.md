# LAB 07 — Error analysis

Model: **Transformer T_h4** (embedding 128 + sinusoidal PE → 2 block, 4 head, d_ff 256 → mean pooling → linear), IMDb, max_length 256, test 9,916 review (split 60/20/20, seed 2026). Code và hình nằm trong `transformer_classification.ipynb`.

## Confusion matrix (xem `confusion_matrix.png`)

| Model | TN | FP | FN | TP | Accuracy | F1 |
|---|--:|--:|--:|--:|--:|--:|
| LSTM | 4385 | 554 | 688 | 4289 | 0.8747 | 0.8735 |
| Transformer | 4344 | 595 | 722 | 4255 | 0.8672 | 0.8660 |

Cả hai đều có FN > FP (bỏ sót review positive nhiều hơn).

## 3 mẫu đúng (chọn ngẫu nhiên seed 2026, review ≤ 150 token)

| True → Pred (p) | Nội dung (rút gọn) | Attention top-5 | Occlusion |
|---|---|---|---|
| positive → positive (0.99) | "MY MOST FAVOURITE MOVIE ... absolute masterpiece ... He's incredible!!!" | great, great, second, which, far | incredible (rất nhỏ) |
| negative → negative (0.00) | "Bad acting. Bad writing. ... poorly written film" | poorly, bored, bad, bad, bad | ≈ 0 với mọi token |
| negative → negative (0.00) | "plain confusing, slow, and uninteresting ... Avoid this one" | hoping, mst, ending, aliens, was | ≈ 0 với mọi token |

Ở mẫu đúng, model rất chắc chắn (p ≈ 0 hoặc 1) nên bỏ một token gần như không đổi gì (occlusion ≈ 0): nhiều từ cùng chiều sentiment bù cho nhau. Mẫu 2 có attention vào đúng các từ chê; mẫu 3 thì attention cao nhất vào "hoping", "mst", không phải các từ chê.

## 3 mẫu sai và phân loại lỗi (nguyên nhân là hypothesis)

| # | True → Pred (p) | Loại lỗi | Attention top-5 | Occlusion (mạnh nhất) | LSTM | Đối chiếu |
|---|---|---|---|---|---|---|
| 1 | positive → negative (0.25) | domain-specific term + ambiguous: bình luận đấu vật ("The Rock won the title", "crappy champ", "pleased with Storm and Christian") | jeff, van, best, next, booker | van, pleased, booker, am, best | sai (0.09) | Tên riêng (van, booker) ảnh hưởng mạnh dù không mang sentiment; nhãn positive không rõ từ nội dung. |
| 2 | negative → positive (0.56) | contrast + long sequence: "not all bad ... excellent camera work. BUT ... long and boring" | bad, nice, went, long, very | excellent (0.30), graduated, boring, entertaining, bad | **đúng** (0.02) | Lời khen kỹ thuật ở giữa thắng kết luận sau "BUT"; mean pooling cộng đều mọi token. LSTM đọc tuần tự nên phần kết có trọng lượng lớn. |
| 3 | positive → negative (0.16) | ambiguous sentiment / noise: khen nhân vật Carmen nhưng chê show ("Its way worse", "ugh! Awful") | awful, its, kyle, best, fit | awful (0.58), worse (0.41), awesome | sai (0.07) | Model dựa đúng vào từ chê có trong text; nhãn positive khó suy ra (nhãn mơ hồ). |

## Các token quan trọng có xuất hiện trong attention pattern không?
Attention ở đây là lượng attention mỗi token **nhận được** ở layer cuối (trung bình các head và query); token quan trọng đo bằng occlusion (bỏ token, đo thay đổi P(positive)).

| Chỉ số (200 review test ≤ 120 token) | Giá trị |
|---|--:|
| Token ảnh hưởng nhất nằm trong top-5 attention | 34.5% |
| Trùng top-5 attention và top-5 occlusion | 28.6% |
| Trùng top-5 chọn ngẫu nhiên và top-5 occlusion | 5.3% |

- Attention **có liên quan** đến token quan trọng (gấp ~5 lần ngẫu nhiên; mẫu 3: "awful" đứng đầu cả hai), nhưng phần lớn trường hợp không trùng (mẫu 2: token ảnh hưởng nhất "excellent" không nằm trong top attention).
- Application: câu negation "I did not enjoy this movie at all." đúng (0.045); occlusion chỉ ra "not" (Δp −0.50) và "did", nhưng attention nhận nhiều nhất là "at" (0.22).
- Kết luận: phân biệt **model behavior** (occlusion: đầu ra thay đổi thế nào) với **human interpretation** (heatmap attention). Attention ≠ explanation.

## Thống kê lỗi trên toàn test set

| Tỉ lệ lỗi | Transformer | LSTM |
|---|--:|--:|
| ≤ 256 token | 0.118 | 0.105 |
| > 256 token (bị truncate) | 0.169 | 0.177 |
| Có từ phủ định | 0.137 | 0.131 |
| Không có phủ định | 0.108 | 0.091 |
| Có từ tương phản (but, however, ...) | 0.143 | 0.136 |
| Không có tương phản | 0.095 | 0.088 |

- Truncation là nguồn lỗi lớn: review > 256 token lỗi nhiều hơn ~5 điểm phần trăm ở cả hai model.
- Negation và contrast làm tăng lỗi; contrast tăng nhiều nhất với Transformer (0.095 → 0.143), phù hợp với câu contrast bị sai ở phần Application (0.432; LSTM đúng 0.927).
- Hai model sai trên các mẫu khá khác nhau: cùng sai 684, chỉ Transformer sai 633, chỉ LSTM sai 558.
