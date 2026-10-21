# LAB 05 — Reflection

## Kết quả chính (test set 9,916 review, IMDb, train trên Kaggle GPU)

| Model | Representation | Accuracy | Precision | Recall | F1 | Tham số | Thời gian train |
|---|---|--:|--:|--:|--:|--:|--:|
| Logistic Regression | TF-IDF | 0.9063 | 0.9002 | 0.9148 | 0.9074 | — | 5.3 s |
| RNN | Embedding | 0.7301 | 0.6985 | 0.8133 | 0.7516 | 2,029,569 | 45.1 s |
| LSTM | Embedding | 0.8543 | 0.8538 | 0.8563 | 0.8551 | 2,117,889 | 48.2 s |
| GRU | Embedding | 0.8787 | 0.8788 | 0.8796 | 0.8792 | 2,088,449 | 46.6 s |

## So sánh với prediction (xem `prediction.md`)
- **P1** (RNN xử lý word order trực tiếp hơn TF-IDF): đúng về cơ chế, và Experiment 6 cho thấy dự đoán của RNN phụ thuộc vị trí từ quyết định (margin 0.00 ở đầu, 0.70 ở cuối), nhưng RNN vẫn kém TF-IDF về accuracy (0.730 so với 0.906).
- **P2** (LSTM xử lý dependency dài tốt hơn RNN): đúng. LSTM F1 0.855 so với 0.752; margin theo vị trí cân bằng hơn. GRU còn tốt hơn (0.879).
- **P3** (sequence length 50 → 200, RNN): thời gian train tăng 20.0 → 45.1 s; F1 0.702 → 0.697 → 0.752, không đơn điệu.
- **P4** (truncate 30 token): chưa thử 30; với 50 token LSTM F1 giảm xuống 0.786 (so với 0.855 ở 200) nên dự đoán mất thông tin là hợp lý.
- **P5** (LSTM không chắc luôn hơn RNN): trong thí nghiệm này LSTM hơn RNN, nhưng chỉ là một dataset và một seed nên không kết luận được cho mọi dataset.

## So sánh các representation (LAB 04 → LAB 05)

| Property | TF-IDF + LR | Avg Word2Vec + LR | RNN | LSTM | GRU |
|---|---|---|---|---|---|
| Sparse representation | Có | Không | Không | Không | Không |
| Dense representation | Không | Có | Có (embedding) | Có | Có |
| Word order | Không | Không | Có | Có | Có |
| Context-dependent state | Không | Không | Có (hₜ) | Có (hₜ, cₜ) | Có (hₜ) |
| Long dependency | Không áp dụng | Không | Hạn chế | Tốt hơn | Tốt hơn |
| One vector per word | Có (một chiều/từ) | Có | Có (embedding tĩnh) | Có | Có |
| Computational cost | Thấp | Thấp | Cao hơn (tuần tự) | Cao hơn nữa | Cao hơn (ít hơn LSTM) |

**RNN/LSTM/GRU giải quyết được gì mà Average Word2Vec không giải quyết tốt?** Thứ tự từ và trạng thái phụ thuộc ngữ cảnh: hₜ được cập nhật tuần tự nên "dog bites man" khác "man bites dog", và "not" có thể ảnh hưởng lên "like" ở bước sau, thay vì chỉ cộng vào trung bình. Trong thí nghiệm, đầu ra của model thay đổi theo vị trí của từ quyết định, điều mà vector trung bình không thể làm.

**RNN/LSTM/GRU còn vấn đề gì?**
- RNN thường vẫn khó giữ thông tin xa (kết quả Exp 6: margin ≈ 0 khi từ quyết định ở đầu/giữa); LSTM/GRU tốt hơn nhưng vẫn giảm khi từ quyết định ở xa.
- Cần `max_length` cố định: phần bị truncate mất hẳn. Kiểm chứng: cùng model GRU đạt accuracy 0.850 khi đọc 200 token đầu nhưng 0.909 khi đọc toàn bộ trên các review dài (> 200 token); 2 trong 4 lỗi phân tích do truncation.
- Mỗi từ vẫn một embedding tĩnh (từ nhiều nghĩa như "bank" vẫn dùng chung vector đầu vào).
- Tính toán tuần tự, không song song hoá được theo chiều thời gian; chi phí tăng theo độ dài.
- Dễ overfit (val loss tăng sau epoch 3–4 ở GRU/LSTM) và chưa xử lý tốt cấu trúc tương phản như "I expected the movie to be bad, but it was actually wonderful".
- Đây là động lực của attention và Transformer.

## Điều rút ra
- Sequence model không tự động tốt hơn bag-of-words: TF-IDF + LR (0.906) vẫn vượt GRU (0.879) trong thiết lập này, vì TF-IDF dùng trọng số từng từ trực tiếp còn model sequence học từ đầu trên dữ liệu vừa.
- Không chỉ nhìn accuracy: RNN có accuracy 0.730 nhưng lệch hẳn về positive (precision 0.699 so với recall 0.813).
- Không chỉ nhìn epoch cuối: LSTM/GRU overfit rõ sau epoch 4 và 3, nên chọn epoch theo validation loss.
- Phân biệt quan sát thực nghiệm (margin theo vị trí) với giải thích lý thuyết (vanishing gradient).

## Ôn tập Individual learning check
1. **RNN khác Average Word2Vec ở điểm nào?** Average Word2Vec cộng các vector từ rồi chia trung bình nên mất thứ tự và các từ không tương tác. RNN đọc tuần tự, hidden state hₜ = tanh(Wₓₕ xₜ + Wₕₕ hₜ₋₁ + b) mang thông tin các token trước và phụ thuộc thứ tự.
2. **Hidden state hₜ có ý nghĩa gì?** Là "bộ nhớ" tóm tắt chuỗi từ đầu đến bước t; vừa là đầu ra của bước t vừa là đầu vào của bước t+1; trạng thái cuối thường dùng để phân loại.
3. **Tại sao RNN có thể gặp vanishing gradient?** Gradient qua nhiều timestep là tích của nhiều Jacobian diag(1 − h²)·Wₕₕ; nếu các thừa số nhỏ hơn 1 thì tích giảm theo hàm mũ (ví dụ 0.5⁵⁰ ≈ 10⁻¹⁵) nên tín hiệu từ token xa không cập nhật được tham số.
4. **Forget gate của LSTM dùng để làm gì?** fₜ quyết định giữ bao nhiêu cell state cũ: gần 1 thì giữ (cₜ ≈ cₜ₋₁), gần 0 thì xóa. Với f = 0.9, i = 0.2, c̃ = 0.5, cₜ₋₁ = 0.8 thì cₜ = 0.82.
5. **Tại sao tăng sequence length có thể làm training chậm hơn?** Số timestep tăng nên phải tính tuần tự nhiều bước hơn (và BPTT qua nhiều bước hơn): trong thí nghiệm LSTM mất 21.9 → 30.7 → 48.2 s cho 50 → 100 → 200 token.
6. **GRU khác LSTM ở điểm nào?** GRU không có cell state riêng và dùng ít gate hơn (update/reset), nên ít tham số hơn (88,320 so với 117,760 ở phần recurrent), thường nhanh hơn và dễ tối ưu hơn; LSTM có cell state và 3 gate (forget, input, output).
7. **Tại sao training loss giảm nhưng validation loss tăng?** Model đang overfit: học thuộc dữ liệu train (GRU: train 0.020, val 0.761 ở epoch 8) nên tổng quát kém và quá tự tin vào dự đoán sai; cần early stopping theo validation loss.
8. **Nếu một review bị truncate ở 30 token, thông tin gì có thể bị mất?** Phần lớn nội dung: với trung vị 172 token, 30 token chỉ giữ khoảng 13% token; mất phần kết luận/đánh giá ở cuối, các từ tương phản ("but ..."), và nhiều từ sentiment.
