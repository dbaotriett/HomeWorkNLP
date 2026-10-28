# LAB 06 — Error analysis

Model: **ATT** (embedding + positional embedding → self-attention một head → attention pooling → linear), IMDb, max_length 200, test 9,916 review (split 60/20/20, seed 2026). Code và hình nằm trong `self_attention.ipynb`.

## Confusion matrix (xem `confusion_matrix.png`)

| Model | TN | FP | FN | TP | Accuracy | F1 |
|---|--:|--:|--:|--:|--:|--:|
| LSTM | 4209 | 730 | 715 | 4262 | 0.8543 | 0.8551 |
| ATT | 4140 | 799 | 675 | 4302 | 0.8514 | 0.8537 |

ATT hơi nghiêng về positive (FP 799 > FN 675); LSTM cân bằng hơn.

## 3 mẫu đúng (chọn ngẫu nhiên, seed 2026, review ≤ 120 token)

| True → Pred (p) | Nội dung (rút gọn) | Attention top-5 | Occlusion top-5 |
|---|---|---|---|
| negative → negative (0.33) | "avoid watching this movie ... nonsense movies" | of, do, acting, rickshaw, paid | definitely, goers, movie, though, you |
| negative → negative (0.18) | "annoyingly unfunny jokes ... truly dreadful songs ... colossal bore" | dreadful, save, jokes, director, only | anyone, party, for, a, jimmy |
| positive → positive (0.91) | "Wile E. continues hilarious ... A real classic." | happens, finish, off, redoubling, much | hilarious, you've, you, as, to |

Ngay cả ở mẫu đúng, attention pooling gần như đều (mỗi token ≈ 0.01) và top-5 attention không trùng các từ sentiment rõ ràng (avoid, nonsense, classic).

## 3 mẫu sai và phân loại lỗi (nguyên nhân là hypothesis)

| # | True → Pred (p) | Loại lỗi | Token mình cho là quan trọng | Attention top-5 | Occlusion (mạnh nhất) | Đối chiếu |
|---|---|---|---|---|---|---|
| 1 | positive → negative (0.04) | ambiguous wording ("eerie and campy fun" nhưng "bad musical", "awful children's recital") | fun, campy, awful, bad | evangelist, macliamm, there's, halfway, recital | nothing, awful, bad | Model dựa vào từ chê có thật trong review; nhãn positive khó suy ra từ chữ. Attention không chỉ vào các từ này. |
| 2 | negative → positive (0.59) | sarcasm ("Scenes to make you squeal with delight", "Is it better than ...? Nah.") | delight, Nah, Nope | put, instead, kimono, climactic, do | instead, them, climactic | Model không nhận ra mỉa mai; "Nah", "Nope" không nằm trong top của cả hai cách đo. |
| 3 | negative → positive (0.79) | ambiguous wording + rare/lost token ("great names ... acting for food", "barely entertaining", "4/10" mất chữ số khi tokenize) | barely, 4/10, not at all | guess, names, whereas, luxury, barely | great (0.23), surprising (0.11) | "great" bị hiểu như lời khen dù đang nói về dàn diễn viên. |

## Attention có tập trung vào token quan trọng không? (kiểm chứng trên 300 review test ≤ 120 token)

| Chỉ số | Giá trị |
|---|--:|
| Token ảnh hưởng nhất (occlusion) nằm trong top-5 attention | 17.7% |
| Trùng top-5 attention và top-5 occlusion | 10.4% |
| Trùng top-5 chọn ngẫu nhiên và top-5 occlusion | 6.7% |

- **Không**: attention pooling gần như phân phối đều, chỉ hơn chọn ngẫu nhiên một chút. Thông tin từ quan trọng đã được self-attention trộn vào vector của mọi vị trí trước khi pooling, nên weight pooling không cho biết token nào quyết định.
- Vì vậy **không kết luận** "attention nhìn sai từ nên model sai": ở mẫu đúng attention cũng không nhìn vào từ quyết định. Occlusion (thay đổi đầu ra khi bỏ token) khớp với prediction hơn (mẫu 1: awful, bad; mẫu 3: great).

## Thống kê thêm
- Tỉ lệ lỗi theo độ dài: ≤ 200 token 13.0%, > 200 token 17.6%, phù hợp giả thuyết truncation (phần sau token 200 bị cắt).
- Review có từ phủ định: lỗi 15.3%, không có: 12.9%.
- Application: "I like this movie." bị dự đoán Negative (0.091) ở **mọi** model attention (ATT_nopos 0.020, POOL 0.008), chỉ LSTM đúng (0.578), nên lỗi không do positional embedding; giả thuyết: domain shift (câu rất ngắn, không có từ khen mạnh).
