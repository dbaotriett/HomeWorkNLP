# LAB 03 — Word Representations and Embeddings

- Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng — HK I 2026
- Co-occurrence → PPMI → Word2Vec → similarity, analogy, semantic search.

| File | Nội dung |
|---|---|
| `calculations.pdf` | Tính tay (bản scan): mục 6, 7, 8, 16, 23 |
| `prediction.pdf` | Dự đoán trước experiment (bản scan): mục 9 |
| `cooccurrence.py` | `build_vocabulary`, `build_cooccurrence_matrix`, `cosine_similarity`, `most_similar`, PPMI (có test) |
| `word_embedding.ipynb` | Experiment 1–4, evaluation, semantic search, polysemy |
| `results.csv` | Số liệu các thí nghiệm |
| `error_analysis.md` | 3 similarity đúng, 3 sai/bất ngờ, polysemy `bank` |
| `reflection.md` | Bảng Static → Contextual và câu hỏi `bank` |

## Chạy lại

```bash
pip install gensim scikit-learn scipy pandas nbformat ipykernel
python cooccurrence.py
jupyter nbconvert --to notebook --execute --inplace word_embedding.ipynb
```

Corpus: 8,000 document đầu của `../lab02/data/c4-train.00000-of-01024-30K.json.gz` (không commit).
Word2Vec: skip-gram + negative sampling (5), dim 100, window 5, min_count 3, epochs 10.

## Kết quả chính
- doctor → dentist, **physician**, ophthalmologist, surgeon, dermatologist.
- Similarity vs trực giác (20 cặp): Spearman ρ = 0.813; analogy hit@1 = 0.53 (15 truy vấn).
- Dim 50/100/300: ρ = 0.774 / 0.813 / 0.836; thời gian 42 / 38 / 87 s.
- Semantic search (P@20, "medical treatment"): TF-IDF 0.80, embedding trung bình 0.90.

## AI assistance statement
- Tool: Claude Code
- Purpose: Giải thích API, gợi ý code, kiểm tra implementation.
- Generated content: Giải thích API và gợi ý code co-occurrence/Word2Vec.
- Modified content: Tôi viết lại, sửa theo corpus.
- Verification: Chạy lại code, so sánh với tính tay.