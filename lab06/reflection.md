# LAB 06 — Reflection

## Kết quả chính (test 9,916 review IMDb, train trên Kaggle GPU, 8 epoch, chọn epoch theo validation loss)

| Model | Accuracy | F1 | Training time | Parameters | Best epoch |
|---|--:|--:|--:|--:|--:|
| LSTM (hidden state cuối) | 0.8543 | 0.8551 | 49.6 s | 2,117,889 | 4 |
| ATT (self-attention + vị trí + attention pooling) | 0.8514 | 0.8537 | 21.7 s | 2,050,401 | 1 |
| ATT_nopos (bỏ positional embedding) | 0.8721 | 0.8717 | 21.6 s | 2,030,401 | 2 |
| POOL (chỉ attention pooling) | 0.8783 | 0.8811 | 11.9 s | 2,020,201 | 3 |

- ATT ngang LSTM về F1 nhưng train nhanh gấp ≈ 2.3 lần vì tính song song mọi vị trí.
- ATT overfit rất nhanh (best ở epoch 1, val loss 0.35 → 0.92); model đơn giản hơn (POOL) lại tốt nhất trên bài toán sentiment này.

## So sánh với prediction (xem `prediction.md`)
- **P1** (dot product lớn nhất → weight lớn nhất): đúng; Experiment 1 "good" score 8.66 → weight 0.964.
- **P2** (score bằng nhau → đều): đúng, [0.25, 0.25, 0.25, 0.25].
- **P3** ([1, 1, 10] → gần one-hot): đúng, 0.9998.
- **P4** (10 → 1000 token, chi phí ×10,000): đúng về số phần tử (100 → 1,000,000). Thời gian đo thật tăng theo T²: 1000 → 4000 token tăng ≈ 16–19 lần; ở T nhỏ tăng chậm hơn do chi phí cố định.
- **P5** (attention không luôn chọn đúng token con người cho là quan trọng): đúng. Top-5 attention chỉ trùng top-5 occlusion 10.4% (ngẫu nhiên 6.7%); heatmap có pattern hợp lý (adopted/sleeping → cat) lẫn khó giải thích (cat → yesterday 0.92).

## 28. So sánh LAB 05 và LAB 06

| Property | RNN/LSTM/GRU | Attention |
|---|---|---|
| Sequence order | Có (đọc tuần tự) | Có nếu thêm positional embedding (self-attention thuần không biết thứ tự) |
| Hidden state | Có | Không bắt buộc |
| Sequential computation | Có | Không bắt buộc |
| Direct token-to-token interaction | Hạn chế (qua nhiều bước) | Có (1 bước) |
| Long-range interaction | Khó | Dễ hơn |
| Parallel computation | Hạn chế | Có (ATT 21.7 s so với LSTM 49.6 s) |
| Cost theo sequence length | Gần tuyến tính theo timestep | O(T²) cho full self-attention |
| Context representation | Recurrent state | Weighted combination |

**Attention giải quyết vấn đề gì của recurrent processing?** RNN phải nén cả sequence vào một hidden state, thông tin xa đi qua nhiều bước nên dễ mất (vanishing gradient), và phải tính tuần tự. Attention cho mỗi vị trí nhìn trực tiếp mọi vị trí khác qua weight αᵢⱼ, không phụ thuộc khoảng cách, tính song song được; context vector là tổng có trọng số của mọi value thay vì chỉ trạng thái cuối.

**Attention tạo ra vấn đề mới nào?**
- Chi phí O(T²) về bộ nhớ và tính toán: T = 10,000 cần 10⁸ phần tử (≈ 400 MB float32) cho một attention matrix.
- Không biết thứ tự nếu không thêm positional information.
- Dễ overfit trên dữ liệu vừa (ATT best ở epoch 1).
- Attention weight dễ bị hiểu nhầm là lời giải thích: weight cao không có nghĩa là token đó quyết định prediction.

## 29. Attention không miễn phí
T = 1,000 → 1,000,000 phần tử; T = 10,000 → 100,000,000 phần tử. Attention giải quyết một số vấn đề của RNN nhưng chi phí tăng theo bình phương độ dài, nên "attention tốt hơn RNN nên luôn thay RNN" là sai. Trong lab này LSTM cũng không thua: F1 0.855 so với 0.854, và LSTM là model duy nhất đúng câu "I like this movie.".

## Điều rút ra
- Attention phụ thuộc query: cùng câu, query sentiment chọn "good" (0.964), query topic chọn "movie" (0.961).
- Scaling √d_k giữ softmax không bão hòa khi d_k lớn (max weight ≈ 0.31 ở mọi d_k, so với 0.97 khi không scale ở d_k = 1024).
- Attention pattern ≠ linguistic explanation ≠ reasoning; cần kiểm chứng (occlusion) trước khi dùng heatmap để giải thích.

## 31. Ôn tập Individual learning check
1. **Attention score được tính như thế nào?** sᵢ = qᵀkᵢ (dot product giữa query và key), scaled: sᵢ = qᵀkᵢ / √d_k.
2. **Tại sao phải dùng softmax?** Biến score bất kỳ thành weight dương, tổng bằng 1, để context vector là trung bình có trọng số; softmax đồng biến nên score lớn hơn → weight lớn hơn, và khả vi để huấn luyện.
3. **Query, Key và Value có vai trò gì?** Query: "đang cần thông tin gì"; Key: "mỗi vị trí có gì để so khớp" (so với query ra score); Value: "nội dung thật được lấy ra" theo weight. Trong self-attention cả ba tạo từ cùng X: Q = XW_Q, K = XW_K, V = XW_V.
4. **Context vector được tính như thế nào?** c = Σᵢ αᵢvᵢ với α = softmax(scores); dạng ma trận C = AV.
5. **Tại sao chia √d_k?** Với vector ngẫu nhiên, độ lệch chuẩn của qᵀk tăng theo √d_k; không chia thì softmax gần one-hot và gradient gần 0. Chia √d_k giữ score ở cùng thang đo.
6. **Self-attention khác attention thông thường ở đâu?** Attention thông thường: query từ nguồn khác (ví dụ decoder) so với key/value của sequence. Self-attention: Q, K, V cùng từ một sequence, mỗi token attend tới các token khác trong chính câu đó, ra ma trận T × T.
7. **Sequence dài gấp đôi thì attention matrix thay đổi thế nào?** Số phần tử tăng 4 lần ((2T)² = 4T²).
8. **Attention có chứng minh model đang reasoning không?** Không. Attention chỉ cho biết phân phối weight; trong lab này top-5 attention gần như không trùng token ảnh hưởng đến prediction (10.4% so với ngẫu nhiên 6.7%).
9. **Tại sao attention xử lý long-range interaction tốt hơn recurrent state?** Hai vị trí bất kỳ tương tác trực tiếp qua một weight αᵢⱼ (đường đi độ dài 1), không phải truyền qua T bước cập nhật hidden state nên thông tin và gradient không bị co lại theo khoảng cách.
