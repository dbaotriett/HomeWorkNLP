# Error analysis — LAB 02

Chọn 2 prediction đúng và 2 prediction sai từ phần next-word prediction.
Nguyên nhân chọn trong: insufficient training data, unseen n-gram, vocabulary limitation, sparsity, context quá ngắn, domain mismatch, preprocessing, smoothing.

## Case 1 — prediction đúng (Trigram)
```
Context: with each
Model prediction: other
Expected: other
Probability: p(pred@1) = 0.0008, p(actual) = 0.00082
```

**Nguyên nhân:** context quen thuộc, collocation mạnh.

**Giải thích:** Cụm "with each other" xuất hiện nhiều lần trong C4. Trigram context "with each" → "other" là dự đoán chính xác. Xác suất tuy nhỏ (0.0008) nhưng vẫn là cao nhất trong phân phối vì các ứng viên khác còn hiếm hơn. Đây là trường hợp model nắm được collocation phổ biến.

## Case 2 — prediction đúng (Trigram)
```
Context: need to
Model prediction: be
Expected: be
Probability: p(pred@1) = 0.0044, p(actual) = 0.00436
```

**Nguyên nhân:** cấu trúc ngữ pháp phổ biến "need to be".

**Giải thích:** "need to" là collocation rất phổ biến trong tiếng Anh. Model trigram học được rằng sau "need to" thì "be" thường xuất hiện nhất. Đây là trường hợp model nắm được cả cú pháp lẫn ngữ nghĩa đơn giản. Xác suất 0.0044 cao hơn hẳn so với các case khác.

## Case 3 — prediction sai (Trigram)
```
Context: after you
Model prediction: have
Expected: leave
Probability: p(pred@1) = 0.0002, p(actual) = 0.00002
```

**Nguyên nhân:** context quá ngắn + sparsity.

**Giải thích:** "after you" có thể đi với rất nhiều từ: leave, have, get, arrive, finish,... Training corpus không đủ để model học hết. Model chọn "have" vì đây là từ phổ biến chung, nhưng trong test set "leave" mới là từ đúng. Đây là lỗi do context quá ngắn — bigram/trigram không đủ thông tin để phân biệt. Xác suất p(actual) = 0.00002 rất nhỏ, cho thấy model gần như không nắm được continuation này.

## Case 4 — prediction sai (Trigram)
```
Context: as rss
Model prediction: (không có — top-5 tùy ý)
Expected: straight
Probability: p(actual) = 0.00002, context seen = False
```

**Nguyên nhân:** unseen n-gram.

**Giải thích:** Trigram context "as rss" chưa từng xuất hiện trong training corpus, nên C("as rss") = 0.

- Với MLE: mọi P(w | "as rss") = 0 → không có dự đoán.
- Với Laplace: P(w | "as rss") = (0+1)/(0+V) = 1/V cho **mọi** w. Tất cả từ có xác suất bằng nhau → top-5 trở nên tùy ý, model không có thông tin để dự đoán.

Đây là biểu hiện của sparsity ở bậc n cao: context càng dài, càng dễ chưa từng thấy. Smoothing chỉ ngăn PPL = inf, **không thêm thông tin** để dự đoán.
