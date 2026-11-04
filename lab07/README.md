# LAB 07 — Transformer Architecture

- Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng — HK I 2026
- Chủ đề: From attention to Transformer: Embedding + Positional Encoding → Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm → Pooling → Classifier.

| File | Nội dung |
|---|---|
| `calculations.pdf` | Tính tay: mục 6, 7, 10, 14, 15, 16, 18, 19, 22 |
| `prediction.pdf` | 6 dự đoán trước experiment (mục 8) |
| `transformer_block.py` | `MultiHeadSelfAttention`, `FeedForward`, `TransformerBlock`, `TransformerClassifier` tự ghép (test: shape, khớp `nn.TransformerEncoderLayer`, permutation equivariance) |
| `positional_encoding.py` | Sinusoidal positional encoding (test khớp tính tay mục 10) |
| `transformer_classification.ipynb` | Experiment 1–6, permutation experiment, learning curves, confusion matrix, application, error analysis |
| `results.csv` | Số liệu các thí nghiệm |
| `positional_encoding.png` | Heatmap PE và độ giống nhau giữa các vị trí |
| `attention_heads.png` | Attention matrix của 4 head × 2 layer |
| `learning_curves.png` | Train/validation loss theo epoch |
| `confusion_matrix.png` | Confusion matrix LSTM và Transformer |
| `error_analysis.md` | 3 mẫu đúng, 3 mẫu sai, attention vs occlusion |
| `reflection.md` | Prediction verification, so sánh LAB 04 → 07, câu hỏi ôn tập |

## Cách chạy

- Huấn luyện chạy trên **Kaggle (GPU)**: https://www.kaggle.com/code/botritinh/lab07 (notebook private, dataset `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`).
- Model và kết quả benchmark lưu trong `cache/` (không commit); chạy lại notebook ở local sẽ nạp cache (xóa `cache/` để train lại).
- Notebook dùng lại `../lab04/preprocessing.py` và `../lab04/data/imdb.zip`.

```bash
pip install numpy torch scikit-learn pandas matplotlib nbformat ipykernel
python transformer_block.py
python positional_encoding.py
jupyter nbconvert --to notebook --execute --inplace transformer_classification.ipynb
```

Dữ liệu: IMDb 49,578 review (sau khi bỏ trùng), split stratified 60/20/20 seed 2026 = 29,746 / 9,916 / 9,916 (giống LAB 04–06). Vocabulary 20,000 (chỉ từ train). Mọi model: embedding/d_model 128, Adam lr 1e-3, batch 64, 8 epoch, dropout 0.1, chọn epoch theo validation loss.

## Kết quả chính (test set)

| Model | Parameters | Training time | Accuracy | F1 |
|---|--:|--:|--:|--:|
| LSTM | 2,692,225 | 50.9 s | 0.8747 | 0.8735 |
| Transformer (2 block, 4 head) | 2,825,089 | 134.1 s | 0.8672 | 0.8660 |

- Không positional encoding: "dog bites man" và "man bites dog" cho cùng vector câu; xáo trộn từ của 1,000 review không đổi prediction nào.
- 1 / 2 / 4 / 8 head: cùng số tham số, F1 0.868 / 0.867 / 0.866 / 0.853, thời gian tăng theo số head.
- max_length 32 → 256: F1 0.714 → 0.866, peak GPU 114 → 1,166 MB.
- Mean pooling F1 0.866, CLS pooling 0.862.
- Benchmark một batch: Transformer nhanh hơn LSTM ở T = 128–512, chậm hơn 3.8 lần ở T = 4096.

## AI assistance statement
- Tool: Claude Code
- Purpose:
- Generated:
- Modified:
- Verified: chạy test `transformer_block.py` (khớp `nn.TransformerEncoderLayer`) và `positional_encoding.py` (khớp tính tay), chạy notebook trên Kaggle rồi chạy lại ở local, đối chiếu số liệu giữa notebook và các file .md.
