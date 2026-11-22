# LAB 10 — Extractive Question Answering

- Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng, HK I 2026
- Chủ đề: from token classification to span prediction (SQuAD v1.1, start/end position).

| File | Nội dung |
|---|---|
| `calculations.pdf` | Tính tay: start/end (mục 6), start loss (mục 7), vì sao cần hai prediction (mục 8), tokenization QA (mục 10), character span (mục 12), sliding window (mục 13), EM/F1 (mục 18–19) |
| `prediction.pdf` | 4 dự đoán trước experiment (mục 9) |
| `tokenizer_alignment.ipynb` | Input QA, `token_type_ids` (BERT / DistilBERT / RoBERTa), `offset_mapping`, character → token span, các trường hợp khó, truncation và sliding window; thống kê mất đáp án và độ phủ trên 20,000 câu train |
| `qa_model.py` | EM/F1 chuẩn SQuAD, `char_to_token_span`, chuẩn bị feature train/eval, `QAModel` (backbone + Linear(d, 2)), loss, post-processing n-best qua các window, baseline keyword overlap (có test) |
| `qa_finetuning.ipynb` | Fine-tuning DistilBERT, Experiment 1–5, robustness (irrelevant / distractor / shuffle), error analysis, Document QA, reproducibility |
| `results.csv` | Số liệu các thí nghiệm |
| `learning_curves.png` | Training loss, validation loss, validation EM/F1 của 3 cấu hình max_length |
| `em_f1_analysis.png` | EM vs F1 (baseline vs fine-tuned), phân bố F1 của các câu EM = 0 |
| `answer_length_analysis.png` | EM/F1 theo độ dài đáp án |
| `error_analysis.md` | Thống kê wrong start / wrong end / wrong span, 5 ví dụ mỗi loại, giả thuyết nguyên nhân |
| `document_qa.py` | Mini Document QA: câu hỏi + tài liệu dài, sliding window, answer span; tài liệu FAQ tự viết + 15 câu gold |
| `reflection.md` | Kiểm chứng prediction, reflection LAB 08–10, learning check, QA vs RAG |

## Cách chạy

- Fine-tuning chạy trên **Kaggle (GPU Tesla T4, có internet)**: https://www.kaggle.com/code/botritinh/lab10 (notebook private). Ba cấu hình × 2 epoch mất khoảng 65 phút.
- Kết quả và trọng số fine-tuned (fp16, khoảng 130 MB) lưu trong `cache/`, không commit. Chạy lại notebook ở local sẽ nạp cache, không train lại.

```bash
pip install torch transformers datasets pandas matplotlib nbformat ipykernel
python qa_model.py
python document_qa.py
jupyter nbconvert --to notebook --execute --inplace tokenizer_alignment.ipynb
jupyter nbconvert --to notebook --execute --inplace qa_finetuning.ipynb
```

## Reproducibility

| | |
|---|---|
| Model | `distilbert-base-uncased` + Linear(768, 2) (start/end), 66.4M tham số |
| Tokenizer | WordPiece uncased, 30,522 token |
| Dataset | SQuAD v1.1 (HF `rajpurkar/squad`) |
| Train split | 83,590 câu hỏi (420 bài): train chính thức trừ 22 bài làm validation |
| Validation split | 4,009 câu hỏi (22 bài, tách theo title để không lộ context) |
| Test split | 10,570 câu hỏi (SQuAD dev chính thức, 48 bài, 3 gold mỗi câu) |
| Max sequence length | 384 (Experiment 1: 128 / 256 / 384) |
| Stride | 128 (85 / 42 cho 256 / 128) |
| Batch size | 32 |
| Learning rate | 3e-5, warmup 10% + linear decay |
| Epochs | 2 (chọn epoch theo validation F1) |
| Optimizer | AdamW (weight decay 0.01), grad clip 1.0, fp16 |
| Random seed | 2026 |
| Hardware | Tesla T4 (Kaggle) |
| Training time | 1,393 s (L = 384, 2 epoch); suy luận toàn bộ test 26 s |

Post-processing: n-best = 20, độ dài span ≤ 30 token, i ≤ j, span phải nằm trong context; score = start_logit + end_logit, chọn span tốt nhất trên mọi window của câu hỏi.

## Kết quả chính (test = SQuAD dev, 10,570 câu)

| Model | EM | F1 |
|---|--:|--:|
| Baseline (keyword overlap → chọn câu → span) | 10.55 | 22.51 |
| **DistilBERT fine-tuned (L = 384)** | **75.83** | **84.26** |

- **Experiment 1 (max_length):**
  - L = 128 / 256 / 384 cho EM/F1 gần như bằng nhau (75.88 / 84.20, 75.74 / 84.32, 75.83 / 84.26) nhờ sliding window.
  - L = 128 sinh gấp 2.1 lần số window nhưng chỉ cần 2.1 GB VRAM, so với 3.8 GB của L = 384.
- **Experiment 4 (lexical overlap):** F1 tăng từ 73.4 (overlap ≤ 25%) lên 90.1 (> 75%).
- **Experiment 5 (độ dài đáp án):** EM giảm từ 79.8 (1 word) xuống 54.1 (7+ word); F1 chỉ giảm từ 84.8 xuống 75.5.
- **Robustness (2,000 câu):**
  - Câu không liên quan: +0.2 F1.
  - Xáo thứ tự câu: −0.8 F1.
  - **Câu mồi (distractor): −51.5 F1**, model chọn đáp án giả 58%.
- **Error analysis:** trong 2,555 câu sai có 654 wrong start, 549 wrong end, 174 chồng lấp, 1,177 sai chỗ.
- **Document QA** (tài liệu 425 token, 2 window, 15 câu): EM 73.3 / F1 88.4.
- Mục 26 (SQuAD 2.0, no-answer) là phần mở rộng không bắt buộc, chưa làm.

## AI assistance statement
- Tool: Deepseek
- Purpose: Hỗ trợ xây dựng, triển khai và kiểm thử hệ thống hỏi đáp trích xuất (Extractive Question Answering) sử dụng mô hình DistilBERT trên bộ dữ liệu SQuAD v1.1.
- What was generated: Xây dựng các hàm tính toán Exact Match (EM) và F1 theo tiêu chuẩn SQuAD.
                      Tạo hàm chuyển đổi vị trí ký tự thành vị trí token (Character-to-Token Span).
                      Xây dựng quy trình chuẩn bị dữ liệu, căn chỉnh đáp án và tạo các cửa sổ dữ liệu (Sliding Window).
                      Triển khai kiến trúc mô hình QA gồm DistilBERT và lớp dự đoán vị trí bắt đầu, kết thúc của đáp án.
                      Xây dựng hàm tính loss.
                      Xây dựng baseline.
- What was modified: Hoàn thiện các hàm xử lý tokenization, offset mapping và căn chỉnh vị trí đáp án.
                     Cải thiện quy trình huấn luyện mô hình DistilBERT.
- What was verified: chạy test `qa_model.py` và `document_qa.py`; kiểm tra `char_to_token_span` trích lại đúng đáp án trên 20,000 câu train (99.3% window khôi phục đúng, phần còn lại là gold cắt giữa từ); chạy notebook trên Kaggle rồi chạy lại ở local; đối chiếu số liệu giữa notebook, `results.csv` và các file .md.
