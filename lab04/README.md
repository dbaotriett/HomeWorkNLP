# LAB 04 — Text Classification (IMDb)

- Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng — HK I 2026
- Raw text → representation (TF-IDF / Average Word2Vec) → classifier → evaluation → error analysis.

| File | Nội dung |
|---|---|
| `calculations.pdf` | Tính tay (bản scan): mục 6, 7, 8, 9 |
| `prediction.pdf` | Dự đoán trước experiment (bản scan): mục 10 |
| `preprocessing.py` | Load IMDb, làm sạch, bỏ trùng, chia train/val/test (có test) |
| `classification.ipynb` | Experiment 1–5, evaluation, confusion matrix, error analysis, model weights, application (có biểu đồ) |
| `results.csv` | Số liệu các thí nghiệm |
| `reflection.md` | So sánh representation, interpretation, ôn tập |

## Chạy lại

Đặt `imdb.zip` (chứa `IMDB Dataset.csv`) vào `lab04/data/` (không commit), rồi:

```bash
pip install scikit-learn gensim pandas matplotlib nbformat ipykernel
python preprocessing.py
jupyter nbconvert --to notebook --execute --inplace classification.ipynb
```

Dữ liệu: 50,000 review, bỏ 418 review trùng còn 49,578; chia stratified **60/20/20, seed 2026** = 29,746 / 9,916 / 9,916. Vectorizer chỉ fit trên train; siêu tham số C chọn trên validation.

## Kết quả chính (test set)

| Representation | Classifier | Accuracy | F1 |
|---|---|--:|--:|
| TF-IDF | Logistic Regression | 0.9063 | 0.9074 |
| Avg Word2Vec | Logistic Regression | 0.8712 | 0.8728 |
| TF-IDF | Linear SVM | 0.9060 | 0.9072 |
| TF-IDF (1-2 gram) | Logistic Regression | 0.9103 | 0.9115 |

- Dataset cân bằng 50/50.
- Bigram tăng F1 nhẹ nhưng vocabulary tăng từ 48,587 lên 491,451.
- Fit TF-IDF trước khi split (leakage) làm vocabulary thêm 10,894 từ, nhưng F1 chỉ khác 0.0005 vì IMDb lớn.

## AI assistance statement
- Tool: Claude AI
- Purpose: Giải thích API, gợi ý code, kiểm tra implementation.
- Generated content: Giải thích API và gợi ý code co-occurrence/Word2Vec.
- Modified content: viết lại, sửa theo corpus.
- Verification: Chạy lại code.
