# LAB 05 — Sequence Models: RNN / LSTM / GRU (IMDb)

- Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng — HK I 2026
- Raw text → tokenizer → token IDs → padding/truncation → embedding → RNN/LSTM/GRU → linear → sentiment.

| File | Nội dung |
|---|---|
| `calculations.md` | Tính tay: mục 6, 7, 8, 9, 17 |
| `prediction.md` | Dự đoán trước experiment: mục 10 |
| `rnn_cell.py` | `SimpleRNNCell` tự implement bằng NumPy (có test: khớp tính tay và `torch.nn.RNNCell`) |
| `sequence_classification.ipynb` | Experiment 1–6, evaluation, learning curves, error analysis, application |
| `results.csv` | Số liệu các thí nghiệm |
| `learning_curves.png` | Train/validation loss theo epoch (RNN, LSTM, GRU) |
| `confusion_matrix.png` | Confusion matrix của RNN, LSTM, GRU |
| `error_analysis.md` | 3 mẫu đúng, 4 mẫu sai, phân tích nguyên nhân |
| `reflection.md` | So sánh với LAB 04, câu hỏi ôn tập |

## Cách chạy

- Huấn luyện chạy trên **Kaggle (GPU)**: https://www.kaggle.com/code/botritinh/lab05 (notebook private, dùng dataset `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`).
- Kết quả từng cấu hình lưu trong `cache/` (không commit). Chạy lại notebook ở máy local sẽ nạp cache thay vì train lại; xóa `cache/` để train lại từ đầu (rất chậm trên CPU).
- Notebook dùng lại `../lab04/preprocessing.py` và `../lab04/data/imdb.zip`.

```bash
pip install torch scikit-learn pandas matplotlib nbformat ipykernel psutil
python rnn_cell.py
jupyter nbconvert --to notebook --execute --inplace sequence_classification.ipynb
```

Dữ liệu: IMDb 50,000 review, bỏ 418 review trùng còn 49,578; chia stratified 60/20/20 seed 2026 = 29,746 / 9,916 / 9,916 (giống LAB 04). Vocabulary 20,000 từ (chỉ từ train). Embedding 100, hidden 128, dropout 0.3, Adam lr 2e-3, batch 64, 8 epoch, chọn epoch theo validation loss, truncate giữ 200 token đầu (pad bên phải).

## Kết quả chính (test set)

| Model | Accuracy | F1 | Tham số | Epoch tốt nhất |
|---|--:|--:|--:|--:|
| TF-IDF + Logistic Regression | 0.9063 | 0.9074 | — | — |
| RNN | 0.7301 | 0.7516 | 2,029,569 | 6 |
| LSTM | 0.8543 | 0.8551 | 2,117,889 | 4 |
| GRU | 0.8787 | 0.8792 | 2,088,449 | 3 |

- LSTM và GRU vượt RNN rõ rệt; GRU tốt nhất trong ba model nhưng vẫn kém TF-IDF + LR.
- LSTM/GRU overfit sau epoch 3–4 (val loss tăng, train loss giảm).
- LSTM, max_length 50 / 100 / 200: F1 = 0.786 / 0.832 / 0.855; thời gian train 22 / 31 / 48 s.
- Experiment 6 (vị trí từ quyết định): RNN gần như chỉ dựa vào token cuối (margin 0.00 ở đầu, 0.70 ở cuối); GRU ổn định nhất (0.21 / 0.20 / 0.25).

## AI assistance statement
- Tool: Claude Code
- Purpose: Giải thích API, gợi ý code.
- Generated content: Giải thích API và gợi ý code.
- Modified content: viết lại.
- Verification: chạy test `rnn_cell.py`, chạy thử pipeline ở quy mô nhỏ, chạy notebook trên Kaggle rồi kiểm tra lại ở local
