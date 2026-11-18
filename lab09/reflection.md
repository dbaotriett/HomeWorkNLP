# LAB 09 — Reflection

## Kết quả chính (CoNLL-2003 test, entity-level)

| Model | Precision | Recall | F1 | Token accuracy | Train |
|---|--:|--:|--:|--:|--:|
| Baseline: token features (±1 word) + Logistic Regression | 0.722 | 0.778 | 0.749 | 0.959 | 279 s, CPU |
| **BERT (`bert-base-cased`) fine-tuned, Strategy A** | **0.908** | **0.923** | **0.915** | 0.983 | 229 s (3 epoch), Tesla T4 |
| BERT, Strategy B (lan truyền nhãn) | 0.907 | 0.921 | 0.914 | | 228 s |

| Entity (BERT) | PER | ORG | LOC | MISC |
|---|--:|--:|--:|--:|
| F1 | 0.962 | 0.902 | 0.935 | 0.796 |
| Baseline F1 | 0.770 | 0.643 | 0.825 | 0.769 |

| Độ dài entity | 1 | 2 | 3 | 4+ |
|---|--:|--:|--:|--:|
| BERT F1 | 0.912 | 0.928 | 0.880 | 0.886 |
| Baseline F1 | 0.734 | 0.813 | 0.582 | 0.289 |

Mini IE trên 30 câu tin tức tự viết: entity F1 0.944.

## Kiểm chứng prediction (dự đoán gốc giữ nguyên trong `prediction.md`)

| Prediction | Dự đoán | Kết quả | Đúng/Sai | Giải thích |
|---|---|---|---|---|
| 1 | Transformer tăng entity F1 10–15 điểm, nhiều nhất ở entity chưa gặp, nhập nhằng, ranh giới | +16.6 điểm (0.749 → 0.915); entity chưa gặp +29 điểm; nhập nhằng 69.5% → 87.7%; lỗi ranh giới 423 → 87 | Đúng (độ lớn còn hơn dự đoán) | Ngữ cảnh hai chiều và kiến thức pretraining. |
| 2 | Entity càng dài càng khó (1 > 2 > 3 > 4+) | Baseline đúng: 0.73 / 0.81 / 0.58 / 0.29. BERT gần như phẳng: 0.91 / 0.93 / 0.88 / 0.89 | Đúng với baseline, **sai với BERT** | Mình đánh giá thấp vai trò của ngữ cảnh: BERT nhìn cả câu nên giữ được ranh giới cụm dài; baseline phân loại từng word độc lập nên một word sai là hỏng cả entity. Entity 2 word tốt nhất vì đa số là Tên + Họ. Nhóm 4+ chỉ có 62 entity nên kết quả dao động. |
| 3 | Entity hiếm gây lỗi nhiều hơn, khoảng cách lớn với baseline | F1 đã gặp / chưa gặp: BERT 0.964 / 0.863; baseline 0.922 / 0.573 | Đúng | Baseline chủ yếu nhớ từ đã gặp; BERT khái quát tốt hơn nhưng vẫn giảm 10 điểm. |

## 32. Reflection (khoảng 460 từ)
**Tại sao cùng một pretrained Transformer dùng được cho cả sequence classification và token classification chỉ bằng cách thay prediction head?**

Pretrained Transformer như BERT không được huấn luyện cho một task cụ thể. Mục tiêu pretraining (masked language modeling) buộc model tạo ra cho **mỗi token** một vector chứa thông tin về chính token đó **trong ngữ cảnh của cả câu**. Đầu ra của encoder là ma trận H kích thước T × d (ở LAB 09: 8 × 128 × 768 cho một batch), không phải một nhãn. Encoder vì vậy là một **backbone** tạo contextual representation, còn task được quyết định bởi phần đọc H ra như thế nào.

Ở **LAB 08 (sequence classification)**, ta chỉ lấy vector của **[CLS]** (H[:, 0], kích thước B × 768), cho qua Linear(768, 2) để ra một nhãn cho cả review. Nhờ self-attention, vector [CLS] có thể tổng hợp thông tin từ mọi token, và fine-tuning dạy model dồn thông tin sentiment vào vị trí đó.

Ở **LAB 09 (token classification)**, ta dùng **toàn bộ H** và áp dụng **cùng một** Linear(768, 9) cho từng vị trí: Z = HW + b có kích thước B × T × 9, mỗi token một vector logits trên 9 nhãn BIO. Head chỉ có 6,921 tham số so với 108 triệu tham số của encoder. Cái thay đổi thật sự là **cách định nghĩa output và label**: một nhãn cho mỗi word, căn chỉnh xuống subword bằng `word_ids` (Strategy A: subword đầu nhận nhãn, các mảnh sau là −100 nên không vào loss).

Lý do một backbone dùng được cho cả hai là representation của BERT đã chứa sẵn thông tin cả hai task cần. Ví dụ "Barcelona" trong "meeting in Barcelona" và trong "Barcelona ( Spain ) 9 4 5" có hai vector khác nhau, nên model gán LOC và ORG đúng 87.7% với các từ nhập nhằng như vậy, trong khi baseline dựa trên đặc trưng tĩnh chỉ 69.5%. Kiến thức tên riêng từ pretraining giúp nhận diện entity **chưa từng thấy** trong train (F1 0.863 so với 0.573 của baseline).

Tuy vậy, thay head không giải quyết mọi thứ. Token classification có các vấn đề riêng mà sequence classification không có: **alignment** giữa word và subword (entity bị tách nhiều gấp 4 lần từ thường, "WEAH" → W ##EA ##H), **ranh giới** entity, và **cách đánh giá**: token accuracy 0.983 nghe rất cao nhưng 82.5% word là O; entity-level F1 mới cho biết model có trích xuất đúng cả cụm hay không.

Từ đó mình nhìn pretrained Transformer như một khối dùng chung: [CLS] + head cho phân loại câu (LAB 08), mọi token + head cho NER (LAB 09), và sắp tới vị trí start/end + head cho question answering (LAB 10).

## 31. Individual learning check
1. **Sequence classification khác token classification?** Sequence: một nhãn cho cả chuỗi (X → y, dùng [CLS]). Token: một nhãn cho mỗi token (x₁..x_T → y₁..y_T, dùng mọi vector của H).
2. **BIO tagging dùng để làm gì?** Biểu diễn cả **loại** và **ranh giới** entity bằng nhãn từng token: B-X bắt đầu entity X, I-X tiếp tục, O ngoài entity; nhờ vậy phân biệt được một entity nhiều từ với nhiều entity liền nhau.
3. **B-PER khác I-PER?** B-PER là token đầu tiên của một tên người; I-PER là token tiếp theo trong cùng tên đó (Barack = B-PER, Obama = I-PER).
4. **Tại sao tokenizer gây label alignment problem?** Annotation gán theo word nhưng tokenizer tách word thành nhiều subword (Elon → El ##on), nên số token ≠ số nhãn và phải quyết định nhãn cho các mảnh ##.
5. **IGNORE trong loss có ý nghĩa gì?** Token có nhãn −100 không được tính trong cross-entropy; token vẫn là input và vẫn tham gia attention, chỉ là nhãn của nó không dùng để học.
6. **Tại sao NER không nên chỉ đánh giá bằng token accuracy?** Đa số token là O (82.5%): model đoán toàn O có accuracy 82.5% nhưng F1 = 0; một lỗi ranh giới vẫn cho accuracy cao dù entity sai.
7. **Entity-level F1 khác token-level F1?** Entity-level tính TP khi đúng cả loại và toàn bộ ranh giới (New York → chỉ "New" thì TP = 0, thêm 1 FP và 1 FN); token-level chấm từng token riêng nên vẫn cho điểm phần đúng.
8. **Vì sao entity boundary khó?** Phụ thuộc ngữ cảnh và quy ước ("Santiago Bernabeu stadium", "The New York Times", "Pope Francis"); entity dài, hiếm; một cụm có thể chứa entity khác (California trong University of California Berkeley).
9. **Vì sao contextual representation hữu ích cho NER?** Cùng surface form có thể khác loại (Barcelona thành phố hay đội bóng); vector theo ngữ cảnh phân biệt được (87.7% so với 69.5%), và giúp đoán loại của tên chưa gặp.
10. **NER dùng như thế nào trong Information Extraction?** Là bước đầu biến text không cấu trúc thành dữ liệu có cấu trúc: tìm entity → gom theo loại (JSON) → dùng tiếp cho tìm kiếm, liên kết thực thể, trích quan hệ (ai làm gì ở đâu).
