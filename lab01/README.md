# LAB 01 — From Text Processing to Search

| File | Nội dung |
|---|---|
| `calculations.md` | Part B — bài tính tay |
| `prediction.md` | Part C — dự đoán trước khi chạy thí nghiệm |
| `implementation.py` | Part E — TF, IDF, TF-IDF, cosine tự cài đặt + unit tests + so sánh với sklearn |
| `experiments.ipynb` | Part D, F, G, H, I, J trên corpus 30K (C4) |
| `pipelines.py` | Hàm dùng chung cho notebook: 3 pipeline preprocessing, search |
| `eval_set.json` | 8 queries + relevance labels (gán thủ công bằng pooling top-10 của 3 pipeline) |
| `results.csv` | Top-5 của mỗi query × pipeline, kèm P@5, R@5, RR |
| `reflection.md` | Reflection |

## Chạy lại

```bash
pip install numpy scipy scikit-learn pandas tokenizers
python implementation.py        # unit tests + so sánh sklearn
```
Đặt `c4-train.00000-of-01024-30K.json.gz` vào `lab01/data/` (không commit vì dung lượng), rồi mở `experiments.ipynb`.

## Kết quả chính

| Metric | Pipeline A (minimal) | Pipeline B (normalized) | Pipeline C (subword) |
|---|---|---|---|
| Vocabulary size | 473,388 | 186,302 | 28,553 |
| Avg tokens/doc | 361.1 | 197.5 | 459.0 |
| Matrix sparsity | 0.999611 | 0.999329 | 0.993311 |
| OOV rate (held-out 3K) | 3.98% | 3.27% | 0.13% |
| P@5 / R@5 / MRR | 0.275 / 0.509 / 0.354 | 0.250 / 0.454 / 0.406 | 0.300 / 0.620 / 0.479 |

## AI contribution
- Claude (Anthropic) sinh code ban đầu cho `implementation.py`, `pipelines.py` và các cell code trong `experiments.ipynb`.
- Relevance labels trong `eval_set.json` được gán với sự hỗ trợ của AI; em đã kiểm tra lại: …
- Phần prediction, bài tính tay, giải thích kết quả, error analysis và reflection do em tự làm.
