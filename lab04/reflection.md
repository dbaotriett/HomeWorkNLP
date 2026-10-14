# LAB 04 — Reflection

## Kết quả chính (test set, IMDb, chia 60/20/20 seed 2026: 29,746 / 9,916 / 9,916)

| Representation | Classifier | Accuracy | Precision | Recall | F1 |
|---|---|--:|--:|--:|--:|
| TF-IDF | Logistic Regression | 0.9063 | 0.9002 | 0.9148 | 0.9074 |
| Avg Word2Vec | Logistic Regression | 0.8712 | 0.8656 | 0.8800 | 0.8728 |
| TF-IDF | Linear SVM | 0.9060 | 0.8996 | 0.9148 | 0.9072 |
| TF-IDF (1-2 gram) | Logistic Regression | 0.9103 | 0.9030 | 0.9202 | 0.9115 |

## Tại sao hai representation cho kết quả khác nhau?
- **TF-IDF** là sparse, lexical, giữ thông tin từng từ (48,587 feature); sentiment trên IMDb dựa nhiều vào từ đặc trưng (worst, waste, excellent, great) nên Logistic Regression gán trọng số riêng cho từng từ rất hiệu quả.
- **Average Word2Vec** là dense, semantic nhưng: mất thứ tự, mỗi từ chỉ một vector, và trung bình hóa hàng trăm từ vào 100 chiều làm loãng các từ sentiment quan trọng. Word2Vec học theo ngữ cảnh nên `good` và `bad` gần nhau (cos 0.754): embedding đặt hai từ trái nghĩa gần nhau vì cùng context, không tách được cực sentiment.
- Kết quả: TF-IDF hơn Avg Word2Vec khoảng 3.5 điểm accuracy. Logistic Regression và Linear SVM trên cùng TF-IDF gần như bằng nhau (0.9063 và 0.9060) → ở đây **representation ảnh hưởng nhiều hơn classifier**.

## Word order: "good" vs "not good"
- cos(avg(S1), avg(S2)) = 0.977: hai câu gần như cùng một vector vì `not` chỉ được cộng vào trung bình.
- Avg Word2Vec + LR: P(positive) S1 = 0.928, S2 = 0.550 → chỉ giảm độ tự tin, **không đảo nhãn** (vẫn > 0.5).
- TF-IDF unigram + LR: S1 = 0.867, S2 = 0.395 → ở đây có đảo nhãn, vì `not` có trọng số âm lớn (−4.42) trong dữ liệu IMDb. Tuy vậy unigram không biểu diễn "phủ định của good": nó chỉ cộng `not` âm với `good` dương, nên thành công hay không phụ thuộc vào con số trọng số, không phải hiểu cấu trúc.
- Bigram `not good` có trọng số riêng −2.95 (còn `good` là +2.87, `very good` +3.09), nên biểu diễn được cụm phủ định một cách trực tiếp.

## n-gram
Uni+bigram tăng F1 từ 0.9074 lên 0.9115 (cải thiện nhỏ) nhưng vocabulary tăng từ 48,587 lên 491,451 (~10 lần) và thời gian train tăng từ 5.2s lên 19.2s. Cái giá là feature space rất lớn và phần lớn feature hiếm.

## Data leakage
Fit TF-IDF trên toàn bộ dataset làm vocabulary thêm 10,894 từ (chỉ xuất hiện nhờ val/test) và IDF thay đổi; test F1 chỉ khác 0.9034 và 0.9029 vì IMDb lớn. Chênh lệch nhỏ không có nghĩa là đúng quy trình: test set đã ảnh hưởng đến feature của train, và với dataset nhỏ ảnh hưởng này lớn hơn.

## Model interpretation
- Top positive: excellent (+10.59), great (+10.42), perfect, loved, amazing; top negative: worst (−13.39), waste (−10.24), awful, bad, boring → đều hợp lý với sentiment.
- Có feature **không có nghĩa** với task: `today` (+5.01) và `it` (+4.89) có trọng số dương lớn, nhiều khả năng là tương quan ngẫu nhiên/phong cách viết, không phải cảm xúc.
- Accuracy cao không đồng nghĩa model dựa vào lý do đúng.

## Giới hạn còn lại
Mỗi từ một vector, không có thứ tự, không hiểu thành ngữ, mỉa mai hay ý gián tiếp ("killed me", "awful amount of"). Đây là động lực chuyển sang mô hình có thứ tự/ngữ cảnh (RNN/LSTM rồi Transformer).

## Ôn tập Individual learning check
1. **Tại sao accuracy 95% chưa chắc tốt?** Nếu lớp đa số chiếm 95% thì model đoán toàn lớp đó đã đạt 95% nhưng recall lớp thiểu số = 0. Cần xem precision/recall/F1/confusion matrix.
2. **TF-IDF khác average Word2Vec ở điểm nào?** TF-IDF: sparse, lexical, giữ từng term. Average Word2Vec: dense, semantic, mất thứ tự, một từ một vector, trung bình làm loãng từ quan trọng.
3. **Tại sao "not good" có thể gây lỗi?** Averaged embedding chỉ cộng `not` và `good` độc lập, không biểu diễn vai trò phủ định; S1 và S2 gần như cùng vector (cos 0.977) và model chỉ giảm nhẹ độ tự tin.
4. **Data leakage xảy ra ở đâu nếu fit TF-IDF trước khi split?** Vocabulary và IDF được tính từ cả test set, nên thông tin test đi vào feature của train.
5. **Precision và recall khác nhau thế nào?** Precision = TP/(TP+FP): trong số dự đoán positive, bao nhiêu đúng. Recall = TP/(TP+FN): trong số positive thật, bắt được bao nhiêu.
6. **Tại sao thêm bigram có thể cải thiện sentiment classification?** Bigram như "not good", "very good" thành feature riêng nên bắt được phủ định và nhấn mạnh; đổi lại vocabulary tăng ~10 lần.
