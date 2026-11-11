# LAB 08 — Pretrained Transformers and BERT Fine-Tuning

- Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng — HK I 2026
- Chủ đề: From training a Transformer to using a pretrained language model.

| File | Nội dung |
|---|---|
| `calculations.pdf` | Tính tay: attention mask (mục 7), MLM loss (mục 9), tokenization (mục 10) |
| `prediction.pdf` | 3 dự đoán trước experiment (mục 11) |
| `tokenization.ipynb` | Tokenizer, `input_ids`, `attention_mask`, special/subword tokens, words vs tokens trên IMDb, kiểm chứng attention mask, MLM với model thật |
| `pretrained_classification.py` | `PretrainedClassifier` (encoder pretrained + head [CLS] → Linear), train/predict, freeze, trích feature (có test) |
| `fine_tuning.ipynb` | Experiment 1–5, application, confusion matrix, confidence analysis, error analysis, reproducibility |
| `results.csv` | Số liệu các thí nghiệm |
| `data_efficiency.png` | F1 theo % training data: fine-tuned, frozen, from scratch |
| `confusion_matrix.png` | Confusion matrix của fine-tuned, frozen, from scratch |
| `confidence_analysis.png` | Reliability diagram và phân bố confidence |
| `error_analysis.md` | 3 FP, 3 FN, 2 lỗi confidence cao, kiểm chứng giả thuyết |
| `reflection.md` | Kiểm chứng prediction, câu hỏi bắt buộc, reflection, learning check |

## Cách chạy

- Huấn luyện chạy trên **Kaggle (GPU Tesla T4, có internet)**: https://www.kaggle.com/code/botritinh/lab08 (notebook private).
- Kết quả từng run, feature của chế độ frozen, dữ liệu domain shift và trọng số fine-tuned (fp16) lưu trong `cache/` (không commit, khoảng 570 MB). Chạy lại notebook ở local sẽ nạp cache; xóa `cache/` để train lại (cần GPU).
- Notebook dùng lại `../lab04/preprocessing.py` và `../lab04/data/imdb.zip`.

```bash
pip install torch transformers datasets scikit-learn pandas matplotlib nbformat ipykernel
python pretrained_classification.py
jupyter nbconvert --to notebook --execute --inplace tokenization.ipynb
jupyter nbconvert --to notebook --execute --inplace fine_tuning.ipynb
```

## Reproducibility

| | |
|---|---|
| Model | `distilbert-base-uncased` (66.4M tham số) + Dropout(0.1) → Linear(768, 2) trên [CLS] |
| Tokenizer | WordPiece uncased, vocabulary 30,522 |
| Dataset | IMDb 50k (Kaggle `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`), bỏ HTML, bỏ 418 review trùng → 49,578 |
| Train / Validation / Test split | 29,746 / 9,916 / 9,916 (60/20/20, stratified, seed 2026; giống LAB 04–07) |
| Max sequence length | 256 (Experiment 4: 64 / 128 / 256) |
| Batch size | 32 |
| Learning rate | 2e-5 fine-tuning, 1e-4 from scratch, 1e-3 frozen head; warmup 10% + linear decay |
| Epochs | 3, chọn epoch theo validation loss |
| Optimizer | AdamW (weight decay 0.01), gradient clipping 1.0, fp16 autocast |
| Random seed | 2026 |
| Hardware | Kaggle Tesla T4 |
| Training time | Fine-tuning 573 s (3 epoch, 191 s/epoch) |

## Kết quả chính (test set)

| Model | Accuracy | F1 | Training time | Trainable params |
|---|--:|--:|--:|--:|
| DistilBERT from scratch | 0.8719 | 0.8658 | 573 s | 66.4M |
| DistilBERT frozen + head | 0.8627 | 0.8628 | 66 s + 25 s | 1,538 |
| **DistilBERT fine-tuned** | **0.9142** | **0.9159** | 573 s | 66.4M |

- Fine-tuned với 10% dữ liệu (F1 0.883) vượt from scratch với 100% (0.866).
- max_length 64 / 128 / 256: F1 0.837 / 0.884 / 0.916, peak GPU 1.6 / 2.1 / 2.9 GB.
- Domain shift (accuracy): Amazon 0.931, Yelp 0.899, SST-2 0.864, Tweets 0.816.
- 107 lỗi có confidence ≥ 0.95 (ECE 0.0125): confidence ≠ correctness.

## AI assistance statement
- AI-assisted code: Claude Code
- What was generated:
- What was modified:
- What was verified: chạy test `pretrained_classification.py` (shape, attention mask, freeze), chạy notebook trên Kaggle rồi chạy lại ở local, kiểm tra tokenizer và các con số trong file .md với output notebook.
