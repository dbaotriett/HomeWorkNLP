# LAB 08 — Reflection

## Kết quả chính (test 9,916 review IMDb; Kaggle Tesla T4; max_length 256; 3 epoch; chọn epoch theo validation loss)

| Model | Accuracy | F1 | Training time | Trainable params |
|---|--:|--:|--:|--:|
| Transformer LAB 07 (from scratch, 2 block, tham khảo) | 0.8672 | 0.8660 | 134 s | 2.8M |
| DistilBERT from scratch (khởi tạo ngẫu nhiên) | 0.8719 | 0.8658 | 573 s | 66.4M |
| DistilBERT frozen + head ([CLS]) | 0.8627 | 0.8628 | 66 s + 25 s | 1,538 |
| **DistilBERT fine-tuned** | **0.9142** | **0.9159** | 573 s | 66.4M |

| Training data | 10% | 25% | 50% | 100% |
|---|--:|--:|--:|--:|
| Fine-tuned F1 | 0.883 | 0.906 | 0.909 | 0.916 |
| Frozen F1 | 0.843 | 0.854 | 0.856 | 0.863 |
| From scratch F1 | 0.827 | 0.840 | 0.867 | 0.866 |

Domain shift (accuracy, model fine-tuned trên IMDb): IMDb 0.914, Amazon 0.931, Yelp 0.899, SST-2 0.864, Tweets 0.816.

## Kiểm chứng prediction (dự đoán gốc giữ nguyên trong `prediction.md`)

| Prediction | Dự đoán | Kết quả | Đúng/Sai | Giải thích |
|---|---|---|---|---|
| 1 | Fine-tuned tốt hơn from scratch 3–5 điểm | 0.914 so với 0.872 (+4.2); so với LAB 07 +4.7 | Đúng | Representation pretrained đã có kiến thức ngôn ngữ. |
| 2 | C. Frozen kém hơn fine-tuning 5–8 điểm | 0.863 so với 0.914 (−5.1) | Đúng | [CLS] chưa được dạy tóm tắt sentiment; frozen vẫn được 86% nhờ representation có sẵn. |
| 3 | From scratch giảm mạnh hơn khi ít dữ liệu | 100% → 10%: scratch −3.9, fine-tuned −3.3 F1 | Đúng một phần | Đúng hướng nhưng chênh nhỏ hơn mình nghĩ. Bằng chứng mạnh hơn: fine-tuned với 10% (0.883) đã vượt scratch với 100% (0.866). |

## 28. Error analysis — câu hỏi bắt buộc
**Q1. Model nào hưởng lợi nhiều hơn khi training data ít?** Observation: ở 10% dữ liệu fine-tuned hơn from scratch 5.5 điểm F1 (0.883 so với 0.827), ở 100% hơn 5.0. Evidence: fine-tuned 10% vượt cả from scratch 100%. Interpretation: model pretrained hưởng lợi nhiều nhất khi dữ liệu ít, vì không phải học ngôn ngữ từ đầu.

**Q2. Pretrained tốt hơn from scratch chứng minh điều gì?** Chỉ chứng minh rằng với **cùng kiến trúc, cùng dữ liệu và cùng số epoch**, khởi tạo từ trọng số pretrained cho kết quả tốt hơn trên IMDb test. Nó không chứng minh model "hiểu" ngôn ngữ; lợi ích có thể đến từ kiến thức từ vựng/cú pháp, từ khởi tạo tốt giúp tối ưu dễ hơn, hoặc cả hai.

**Q3. Frozen gần bằng fine-tuned có ý nghĩa gì?** Nghĩa là representation pretrained đã chứa sẵn gần đủ thông tin cho task; khi đó có thể dùng frozen để tiết kiệm (train 1,538 tham số thay vì 66M). Ở đây frozen kém 5 điểm, nên chưa "gần bằng".

**Q4. Fine-tuning tốt hơn frozen nhiều gợi ý điều gì?** Representation pretrained chưa được sắp xếp theo đúng task: thông tin có trong encoder nhưng [CLS] chưa tóm tắt nó theo sentiment (mean pooling frozen 0.867 nhỉnh hơn [CLS] 0.863). Task cần điều chỉnh các layer bên trong (ví dụ xử lý phủ định, tương phản).

**Q5. Performance giảm mạnh trên domain khác, nguyên nhân?** Khác style và độ dài (tweet trung vị 16 từ so với 172), từ ngữ mới (slang "lit", emoji thành [UNK]), sarcasm phổ biến hơn, và cách gán nhãn khác giữa các dataset. Model fine-tuned chỉ thấy review phim dài.

**Q6. Sequence length tăng mà performance gần như không đổi trong khi training time tăng mạnh, có nên tăng tiếp không?** Không. Trong lab, 128 → 256 vẫn tăng 3.2 điểm F1 (thời gian ×1.8) nên đáng tăng; nhưng nếu tiếp tục mà F1 gần như không đổi thì chi phí attention O(T²) và bộ nhớ (2.9 GB ở 256) không đáng. Nên dùng cách khác, ví dụ đọc phần cuối review (đọc 256 token cuối sửa được 65% lỗi review dài).

## 29. Reflection (khoảng 480 từ)
Sau LAB 07 mình đã tự ghép được một Transformer, nhưng LAB 08 cho thấy trong thực tế người ta hiếm khi train Transformer từ đầu.

**Data.** Một Transformer học từ đầu phải học đồng thời nghĩa của từ, cú pháp và task chỉ từ dữ liệu có nhãn của bài toán. Với IMDb (29,746 review), DistilBERT from scratch chỉ đạt F1 0.866, ngang Transformer nhỏ của LAB 07, dù lớn gấp 23 lần. Dữ liệu có nhãn đắt và ít, trong khi văn bản không nhãn thì rất nhiều.

**Pretrained representation.** Pretraining bằng masked language modeling biến văn bản không nhãn thành tín hiệu học: che từ rồi đoán lại. Model học được rằng "eats" hợp với "The cat ... fish" (P = 0.35), "france" hợp với "Paris is the capital of ...". Kiến thức này nằm trong trọng số trước khi model thấy một nhãn sentiment nào. Ngay cả khi đóng băng toàn bộ encoder, chỉ một lớp Linear 1,538 tham số đã đạt 86%.

**Computational cost.** Pretraining rất tốn kém (hàng tỷ từ, nhiều GPU), nhưng chỉ cần làm một lần và chia sẻ. Fine-tuning trên một GPU T4 mất khoảng 3 phút mỗi epoch, và model tốt nhất đã đạt ở epoch 1. Bù lại, model 66M tham số chậm hơn và tốn bộ nhớ hơn nhiều so với LSTM hay Transformer nhỏ; max_length 256 cần 2.9 GB GPU, và chi phí attention tăng theo T².

**Downstream task.** Classification head là phần duy nhất dành riêng cho task. Cùng một encoder có thể gắn head khác cho NER, QA hay phân loại khác. Điều quan trọng nhất mình thấy là hiệu quả dữ liệu: fine-tune với 10% dữ liệu (2,975 review) đã vượt model from scratch dùng toàn bộ 100%.

**Fine-tuning.** Fine-tuning cho phép mọi layer điều chỉnh theo task và tốt hơn frozen 5 điểm. Các câu phủ định và tương phản mà model LAB 07 làm sai thì nay đúng. Fine-tuning cũng overfit nhanh (best ở epoch 1), nên cần learning rate nhỏ (2e-5) và chọn epoch theo validation.

**Limitation và domain shift.** Pretrained không có nghĩa là hiểu mọi domain. Model fine-tuned trên review phim giữ được 0.93 trên Amazon nhưng chỉ còn 0.82 trên tweet. Model vẫn sai với sarcasm, double negation và review dài bị truncate (lỗi gần gấp đôi), và có 107 lỗi với confidence ≥ 0.95. Accuracy 91% chỉ là observation trên một tập test, không phải bằng chứng model "hiểu ngôn ngữ tốt".

Vì vậy cách làm thực tế là: chọn một model pretrained phù hợp domain, fine-tune với dữ liệu task, đánh giá cẩn thận (confusion matrix, confidence, domain shift). Train từ đầu chỉ dành cho khi có rất nhiều dữ liệu và tài nguyên, hoặc domain quá khác (ngôn ngữ mới, mã gen...).

## 27. Individual learning check
1. **Pretraining khác fine-tuning ở đâu?** Pretraining: học representation ngôn ngữ chung trên corpus rất lớn không nhãn với mục tiêu tự giám sát (MLM). Fine-tuning: bắt đầu từ trọng số pretrained, tiếp tục train trên dữ liệu có nhãn của một task cụ thể với loss của task.
2. **MLM là gì?** Che ngẫu nhiên một số token bằng [MASK] (BERT: 15%), model đoán token gốc từ ngữ cảnh hai phía; loss = −Σ log P(token đúng) trên các vị trí bị mask.
3. **Tại sao cần tokenizer?** Model chỉ nhận số: tokenizer chuyển text thành token và token IDs theo đúng vocabulary model đã học lúc pretraining, thêm special tokens, padding/truncation và attention mask.
4. **Subword token giải quyết vấn đề gì?** Out-of-vocabulary: từ hiếm/tên riêng/từ sai chính tả được cắt thành mảnh có trong vocabulary (macliammoir → mac ##lia ##mm ##oir) thay vì [UNK], với vocabulary vừa phải (30,522).
5. **[CLS] thường được sử dụng như thế nào?** Token đặc biệt ở vị trí 0; nhờ self-attention, output của nó có thể tổng hợp cả câu; head phân loại lấy vector [CLS] cuối cùng → Linear → softmax.
6. **Attention mask dùng để làm gì?** Đánh dấu token thật (1) và padding (0) để attention bỏ qua [PAD]; trong lab, có mask đúng thì thêm [PAD] không đổi vector [CLS] (sai khác 4·10⁻⁶), mask toàn 1 thì đổi 3.24.
7. **Frozen encoder nghĩa là gì?** Trọng số encoder không được cập nhật (requires_grad = False), chỉ train head; encoder dùng như bộ trích feature cố định.
8. **Fine-tuning thay đổi những tham số nào?** Toàn bộ: embedding, mọi layer attention/FFN/LayerNorm của encoder và classification head (66.4M), với learning rate nhỏ để không phá kiến thức pretrained.
9. **Vì sao pretrained hữu ích khi downstream dataset nhỏ?** Phần lớn kiến thức ngôn ngữ đã có sẵn, dữ liệu nhỏ chỉ cần dạy ánh xạ sang nhãn; trong lab 2,975 review fine-tuned đã đạt F1 0.883.
10. **Pretrained có đảm bảo không bị domain shift không?** Không; accuracy giảm từ 0.914 (IMDb) xuống 0.816 (tweet).
11. **Vì sao tăng sequence length làm chi phí attention tăng?** Mỗi token attend tới mọi token: attention matrix T × T mỗi head mỗi layer, nên T ×2 thì bộ nhớ/tính toán attention ×4.
12. **Confidence cao có đảm bảo prediction đúng không?** Không; 107 lỗi có confidence ≥ 0.95, lỗi tự tin nhất là 0.998 (sarcasm).
