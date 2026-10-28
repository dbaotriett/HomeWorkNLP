# LAB 06 — Attention Mechanism

- Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng — HK I 2026
- Chủ đề: From recurrent state to context-aware representation (Query, Key, Value → attention weights → context vector).

| File | Nội dung |
|---|---|
| `calculations.pdf` | Tính tay: mục 6, 8–11 (Bài 1–4), 14, 15, 17, 21, 25/29 |
| `prediction.pdf` | Dự đoán trước experiment: mục 12 |
| `attention_numpy.py` | `scaled_dot_product_attention`, softmax ổn định số, `self_attention` bằng NumPy (có test: khớp tính tay và khớp PyTorch) |
| `self_attention.ipynb` | Core implementation, Experiment 1–5, temperature/scaling, attention heatmap, LSTM vs Attention, application, error analysis |
| `results.csv` | Số liệu các thí nghiệm |
| `reflection.md` | So sánh LAB 05 và LAB 06, câu hỏi ôn tập |

## Cách chạy

- Experiment 4 (huấn luyện LSTM và các model attention) chạy trên **Kaggle (GPU)**: https://www.kaggle.com/code/botritinh/lab06 (notebook private, dataset `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`).
- Kết quả huấn luyện và Word2Vec lưu trong `cache/` (không commit); chạy lại notebook ở local sẽ nạp cache, xóa `cache/` để train lại.
- Notebook dùng lại `../lab04/preprocessing.py` và `../lab04/data/imdb.zip`.

```bash
pip install numpy torch scikit-learn pandas matplotlib gensim nbformat ipykernel
python attention_numpy.py
jupyter nbconvert --to notebook --execute --inplace self_attention.ipynb
```

Dữ liệu: IMDb 49,578 review (sau khi bỏ trùng), chia stratified 60/20/20 seed 2026 = 29,746 / 9,916 / 9,916 (giống LAB 04–05). Vocabulary 20,000 từ (chỉ từ train), max_length 200, embedding 100, Adam lr 2e-3, batch 64, 8 epoch, chọn epoch theo validation loss.

## Kết quả chính (test set)

| Model | Accuracy | F1 | Training time | Parameters |
|---|--:|--:|--:|--:|
| LSTM | 0.8543 | 0.8551 | 49.6 s | 2,117,889 |
| Self-attention (ATT) | 0.8514 | 0.8537 | 21.7 s | 2,050,401 |
| ATT không positional embedding | 0.8721 | 0.8717 | 21.6 s | 2,030,401 |
| Chỉ attention pooling | 0.8783 | 0.8811 | 11.9 s | 2,020,201 |

- Cùng câu, đổi query thì attention đổi: query sentiment chọn "good" (0.964), query topic chọn "movie" (0.961).
- Temperature 0.5 / 1 / 2 trên score [2, 1, 0]: max weight 0.867 / 0.665 / 0.506, entropy 0.44 / 0.83 / 1.02.
- Không chia √d_k: max weight tăng 0.49 → 0.97 khi d_k = 4 → 1024; có chia: ổn định ≈ 0.31.
- Attention matrix T × T: T = 1,000 → 10⁶ phần tử, T = 10,000 → 10⁸ (≈ 400 MB float32).
- Top-5 attention chỉ trùng top-5 occlusion 10.4% (ngẫu nhiên 6.7%): attention weight không phải lời giải thích.

## AI assistance statement
- Tool: Claude Code
- Purpose:
- Generated content:
- Modified content:
- Verification: chạy test `attention_numpy.py` (khớp tính tay và `torch.nn.functional.scaled_dot_product_attention`), chạy notebook trên Kaggle rồi chạy lại ở local, kiểm tra số liệu giữa notebook và các file .md.
