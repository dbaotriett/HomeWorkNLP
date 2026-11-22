# LAB 09 — Named Entity Recognition and Information Extraction

- Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng — HK I 2026
- Chủ đề: From sequence classification to token classification (CoNLL-2003, BIO tagging).

| File | Nội dung |
|---|---|
| `calculations.pdf` | Tính tay: BIO (mục 6), entity boundary (mục 7), alignment + IGNORE loss (mục 8–11), shape (mục 15), P/R/F1 entity vs token (mục 19–20) |
| `prediction.pdf` | 3 dự đoán trước experiment (mục 12) |
| `ner_utils.py` | BIO → entity spans, entity-level / token-level metrics (khớp `seqeval`), label alignment Strategy A/B (có test) |
| `baseline_ner.py` | Baseline: token features (±1 word) + Logistic Regression |
| `tokenizer_alignment.ipynb` | `is_split_into_words`, `word_ids`, alignment; Experiment 1 (tokenization analysis) |
| `ner_finetuning.ipynb` | BERT fine-tuning, Experiment 2–6, confusion matrix, context analysis, strategy A vs B, mini IE |
| `information_extraction.py` | `TokenClassifier`, `NERExtractor` (text → JSON), 30 câu gold cho mini IE |
| `results.csv` | Số liệu các thí nghiệm |
| `entity_metrics.csv` | P/R/F1 theo loại entity (baseline và BERT) |
| `entity_length_analysis.png` | Entity F1 theo độ dài |
| `confusion_matrix.png` | Confusion matrix word-level của BERT |
| `error_analysis.md` | 5 FP, 5 FN, 5 boundary error, lỗi sai loại |
| `reflection.md` | Kiểm chứng prediction, reflection, learning check |

## Cách chạy

- Fine-tuning chạy trên **Kaggle (GPU Tesla T4, có internet)**: https://www.kaggle.com/code/botritinh/lab09 (notebook private).
- Kết quả và trọng số BERT fine-tuned (fp16) lưu trong `cache/` (không commit, khoảng 220 MB); chạy lại notebook ở local sẽ nạp cache. Baseline chạy ở local (`python baseline_ner.py` tạo `cache/baseline.json`).

```bash
pip install torch transformers datasets scikit-learn seqeval pandas matplotlib nbformat ipykernel
python ner_utils.py
python baseline_ner.py
jupyter nbconvert --to notebook --execute --inplace tokenizer_alignment.ipynb
jupyter nbconvert --to notebook --execute --inplace ner_finetuning.ipynb
```

Cấu hình: CoNLL-2003 (HF `eriktks/conll2003`, bản parquet) train 14,041 / validation 3,250 / test 3,453 câu; `bert-base-cased`, Strategy A (subword đầu nhận nhãn, còn lại −100), max_length 256 (không câu nào bị cắt), AdamW lr 3e-5, warmup 10%, batch 16, 3 epoch, fp16, seed 2026, chọn epoch theo validation entity F1.

## Kết quả chính (test, entity-level)

| Model | Precision | Recall | F1 | Token accuracy |
|---|--:|--:|--:|--:|
| Baseline (features + LR) | 0.722 | 0.778 | 0.749 | 0.959 |
| **BERT fine-tuned** | **0.908** | **0.923** | **0.915** | 0.983 |

- Theo loại (BERT): PER 0.962, LOC 0.935, ORG 0.902, MISC 0.796.
- Entity chưa gặp trong train: BERT 0.863 so với baseline 0.573.
- Entity 4+ word: BERT 0.886 so với baseline 0.289.
- Mini IE (30 câu tin tức tự viết, 71 entity): F1 0.944.

## AI assistance statement
- Tool: Claude Code
- Purpose: Hỗ trợ xây dựng model nhận dạng thực thể (NER) và trích xuất thông tin (IE) sử dụng mô hình BERT trên bộ dữ liệu CoNLL-2003.
- Generated: Tạo mã nguồn xử lý BIO tagging, chuyển đổi nhãn BIO thành entity spans và tính toán các chỉ số Precision, Recall, F1.
             Xây dựng baseline NER sử dụng Logistic Regression.
- Modified: Chỉnh sửa và hoàn thiện các hàm xử lý BIO tagging và entity-level metrics.
- Verified: chạy test `ner_utils.py` (khớp `seqeval`) và `information_extraction.py`, chạy notebook trên Kaggle rồi chạy lại ở local, đối chiếu số liệu giữa notebook và các file .md.
