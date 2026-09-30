# LAB 02 — Language Models

- Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng — HK I 2026

N-gram language models, smoothing và perplexity.

| File | Nội dung |
|---|---|
| `calculations.pdf` | Tính tay (bản scan): mục 7, 9, 11, 18 |
| `prediction.pdf` | Dự đoán trước experiment (bản scan): mục 12 |
| `ngram_lm.py` | Implementation: mục 14–15 |
| `experiments.ipynb` | Experiment 1–3, application, context length: mục 13, 16, 19, 20, 21, 23 |
| `results.csv` | Perplexity, context length, next-word prediction, sentence ranking |
| `error_analysis.md` | Phân tích lỗi: mục 22 |
| `reflection.md` | Câu hỏi tổng kết: mục 24 |

## Chạy lại

Đặt corpus vào `lab02/data/` (không commit), rồi:

```bash
pip install pandas matplotlib nbformat
python ngram_lm.py        # 12 unit tests
Corpus: c4-train.00000-of-01024-30K.json.gz, dùng 10,000 documents đầu, chia theo document 80/10/10 (seed 42),
vocabulary min_count = 2 (từ hiếm → <unk>). Mở experiments.ipynb và chạy từ trên xuống (~3 phút).

Kết quả chính
Model	Train PPL	Valid PPL	Test PPL
Unigram MLE	1,413.5	1,219.7	1,172.5
Unigram Laplace	1,415.4	1,224.1	1,177.2
Bigram MLE	125.0	inf (24.0% token p=0)	inf (23.4%)
Bigram Laplace	3,014.3	3,605.3	3,458.7
Trigram MLE	9.9	inf (63.9% token p=0)	inf (63.3%)
Trigram Laplace	11,583.0	19,314.2	19,018.4
Nhận xét nhanh:

MLE fit training rất tốt nhưng PPL = ∞ trên valid/test vì zero probability.

Laplace loại bỏ zero-prob nhưng làm training PPL tăng mạnh ở bậc cao.

Trên corpus nhỏ này, Unigram Laplace có valid/test PPL thấp nhất — tăng n không cải thiện chất lượng model.