# LAB 07 — Reflection

## Kết quả chính (test 9,916 review IMDb; Kaggle GPU; d_model/embedding 128, Adam 1e-3, batch 64, 8 epoch, chọn epoch theo validation loss)

| Model | Parameters | Training time | Accuracy | Precision | Recall | F1 |
|---|--:|--:|--:|--:|--:|--:|
| LSTM | 2,692,225 | 50.9 s | 0.8747 | 0.8856 | 0.8618 | 0.8735 |
| Transformer (2 block, 4 head, mean pooling) | 2,825,089 | 134.1 s | 0.8672 | 0.8773 | 0.8549 | 0.8660 |

| Biến thể Transformer | F1 | Ghi chú |
|---|--:|---|
| 1 / 2 / 4 / 8 head | 0.868 / 0.867 / 0.866 / 0.853 | Cùng 2,825,089 tham số; thời gian 83 / 102 / 134 / 201 s |
| max_length 32 / 64 / 128 / 256 | 0.714 / 0.792 / 0.835 / 0.866 | Peak GPU 114 / 189 / 414 / 1,166 MB |
| CLS pooling | 0.862 | Mean pooling 0.866 |
| Không positional encoding | 0.863 | Không phân biệt được "dog bites man" / "man bites dog" |

## 28. Prediction verification (dự đoán gốc giữ nguyên trong `prediction.md`)

| Prediction | Dự đoán | Kết quả | Đúng/Sai | Giải thích |
|---|---|---|---|---|
| 1 | Transformer xử lý token song song hơn RNN | Benchmark: T = 512 Transformer 12.1 ms so với LSTM 20.6 ms; nhưng train ở T = 256 chậm hơn (16.7 so với 6.2 s/epoch) | Đúng (về cơ chế) | Không có phụ thuộc giữa các timestep nên tính song song được; nhưng "song song hơn" không có nghĩa "luôn nhanh hơn": LSTM dùng cuDNN, Transformer tự viết với 2 layer × 4 head attention matrix. |
| 2 | Bỏ PE thì không biết thứ tự | T_nope: xáo trộn từ của 1,000 review → 0% prediction thay đổi | Đúng | Self-attention + FFN permutation equivariant, mean pooling bất biến. |
| 3 | Không PE thì không phân biệt "dog bites man" / "man bites dog" | Vector câu trùng (khoảng cách ≈ 10⁻⁶), cùng P = 0.999745 | Đúng | Như trên; có PE thì khoảng cách 0.150. |
| 4 | Hai head không nhất thiết học cùng pattern | Cosine giữa các head: layer 1 0.006–0.018, layer 2 0.06–0.16 | Đúng | Mỗi head có W^Q, W^K, W^V riêng, khởi tạo khác nhau. Không thấy head redundant. |
| 5 | Tăng head không chắc tăng accuracy | F1 0.868 (1 head) → 0.853 (8 head) | Đúng | d_model cố định nên d_k giảm (128 → 16); 8 head mỗi head chỉ 16 chiều, lại chậm hơn. |
| 6 | Transformer không luôn nhanh hơn RNN | Transformer / LSTM: 1.67 (T = 32), 0.59 (T = 512), 3.78 (T = 4096) | Đúng | Nhanh hơn ở độ dài vừa nhờ song song; chậm hơn ở T rất nhỏ (chi phí cố định) và T lớn (O(T²)). |

Không có prediction sai về hướng. Điểm mình không lường trước: ở cấu hình train thật (T = 256), Transformer chậm hơn LSTM, và LSTM có F1 cao hơn một chút.

## 27. So sánh toàn bộ LAB 04 → LAB 07

| Property | TF-IDF | Avg Word2Vec | LSTM | Attention | Transformer |
|---|---|---|---|---|---|
| Sparse | Có | Không | Không | Không | Không |
| Dense | Không | Có | Có | Có | Có |
| Word order | Limited (n-gram) | Không | Có | Có (cần vị trí) | Có* |
| Context-dependent | Không | Không | Có | Có | Có |
| Direct token interaction | Không | Không | Limited | Có | Có |
| Sequential computation | Không | Không | Có | Không | Không |
| Parallelizable | Có | Có | Limited | Có | Có |
| Long-range interaction | Limited | Không | Difficult | Improved | Improved |
| Attention | Không | Không | Không | Có | Có |
| Positional information | Không | Không | Implicit | Required | Required |
| Cost with sequence length | ~ linear | ~ linear | ~ linear | O(T²) | O(T²) |

\* Transformer cần positional information (permutation experiment: không PE thì thứ tự mất hoàn toàn).

**Từ LAB 01 tới LAB 07, mỗi bước giải quyết limitation nào của bước trước?**

- **TF-IDF.** Limitation: sparse, mỗi từ một chiều riêng nên "good" và "great" không liên quan; không có thứ tự từ.
- **Word2Vec.** Giải quyết: dense vector, từ gần nghĩa có vector gần nhau. Limitation: Average Word2Vec mất thứ tự và mất tương tác giữa từ ("not good" ≈ "good"); một vector cố định cho mỗi từ.
- **LSTM.** Giải quyết: đọc tuần tự, hidden state phụ thuộc ngữ cảnh và thứ tự. Limitation: tính tuần tự (không song song theo thời gian), thông tin xa phải đi qua nhiều bước (vanishing gradient), nén cả câu vào một hidden state.
- **Attention.** Giải quyết: mỗi vị trí nhìn trực tiếp mọi vị trí khác qua weight, song song được. Limitation: không biết thứ tự nếu không thêm vị trí; một head chỉ một kiểu quan hệ; chi phí O(T²).
- **Transformer.** Giải quyết: positional encoding (thứ tự), multi-head (nhiều kiểu quan hệ song song), FFN (biến đổi phi tuyến từng vị trí), residual + LayerNorm (chồng nhiều layer ổn định). Remaining limitation: chi phí O(T²) theo độ dài (max_length 256 đã cần 1.2 GB GPU khi train); học từ đầu cần nhiều dữ liệu (ở đây không thắng LSTM trên 30k review); attention weight không phải lời giải thích → động lực cho pretrained Transformer (BERT, LAB 08).

## Điều rút ra
- Transformer là **ghép các thành phần**: self-attention (tương tác token) + PE (thứ tự) + FFN (biến đổi từng vị trí) + residual/LayerNorm (ổn định); mỗi thành phần có vai trò riêng, kiểm chứng được bằng ablation.
- Kiến trúc mạnh hơn trên lý thuyết không tự động thắng: với dữ liệu vừa và train từ đầu, LSTM nhỉnh hơn.
- Tăng head hay tăng sequence length đều có chi phí; cần đo thay vì giả định.
- Attention pattern chỉ là phân phối weight (layer 1 bão hòa gần one-hot, layer 2 phân tán), không phải explanation.

## 31. Ôn tập Individual learning check
1. **Tại sao Transformer cần positional encoding?** Self-attention chỉ dựa vào nội dung token, hoán vị input chỉ hoán vị output; không PE thì "dog bites man" và "man bites dog" cho cùng representation (đã kiểm chứng: khoảng cách ≈ 10⁻⁶). PE cộng vào embedding một vector riêng cho mỗi vị trí.
2. **Multi-head attention khác một attention head ở đâu?** h head chạy song song, mỗi head có W^Q, W^K, W^V riêng (chiều d_k = d_model/h) nên có thể học kiểu quan hệ khác nhau; output được concat rồi nhân W^O để trộn thông tin giữa các head.
3. **d_model = 128, 8 heads thì d_k?** d_k = 128 / 8 = 16.
4. **Tại sao attention matrix có kích thước T × T?** Mỗi trong T token (query) tính score với mỗi trong T token (key): QKᵀ có shape (T × d_k)(d_k × T) = T × T.
5. **Residual connection có vai trò gì?** y = x + F(x): giữ đường truyền thẳng thông tin đầu vào, F chỉ học phần thay đổi; gradient có thành phần I đi thẳng về layer trước nên chồng nhiều layer vẫn train được.
6. **FFN làm gì trong Transformer block?** Biến đổi phi tuyến (Linear → ReLU → Linear, 128 → 256 → 128) áp dụng độc lập và giống nhau cho từng vị trí; attention trộn thông tin giữa token, FFN xử lý thông tin tại mỗi token.
7. **Tại sao Transformer xử lý token song song hơn RNN?** Toàn bộ score là một phép nhân ma trận QKᵀ, không có bước nào phải chờ bước trước; RNN cần hₜ₋₁ để tính hₜ.
8. **Sequence dài gấp đôi thì attention matrix thay đổi thế nào?** Số phần tử tăng 4 lần (128² = 16,384 → 256² = 65,536); bộ nhớ attention cũng ×4.
9. **Vì sao attention weight cao không phải "lý do" của prediction?** Weight chỉ cho biết token được trộn bao nhiêu vào representation; output còn qua W^V, W^O, FFN, nhiều layer, pooling. Trong lab: token ảnh hưởng nhất (occlusion) chỉ nằm trong top-5 attention 34.5% số lần; ở câu negation "at" nhận attention cao nhất dù "not" mới quyết định.
