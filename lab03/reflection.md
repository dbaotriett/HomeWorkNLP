# LAB 03 — Reflection: từ Word2Vec đến Transformer

| Representation | Context-dependent? | Sparse/Dense | Một từ có nhiều vector? |
|---|---|---|---|
| TF-IDF | Không (chỉ đếm từ trong tài liệu, không xét từ xung quanh) | Sparse (\|V\| chiều, hầu hết bằng 0) | Không — một từ là một chiều cố định |
| Co-occurrence | Không theo câu cụ thể (tổng hợp context trên toàn corpus) | Sparse (có thể nén thành dense bằng SVD) | Không — một vector/từ |
| Word2Vec | Không (static; vector cố định sau khi train) | Dense (50–300 chiều) | Không — nhiều nghĩa bị trộn vào một vector |
| Contextual embedding (BERT, Transformer) | Có — vector phụ thuộc cả câu | Dense | Có — cùng một từ cho vector khác nhau ở mỗi lần xuất hiện |

## Tại sao `bank` cần contextual representation?
"I deposited money in the bank" và "We sat on the river bank" dùng hai nghĩa khác hẳn. Word2Vec chỉ có một vector cho `bank`; trong thí nghiệm, vector này chỉ gần nghĩa tài chính (credit, citibank, banking, cheque) vì nghĩa đó chiếm đa số trong corpus, còn nghĩa "bờ sông" gần như không còn dấu vết. Một vector duy nhất không thể nằm gần cả "money" lẫn "river", nên dùng nó cho câu thứ hai sẽ sai. Contextual embedding (self-attention nhìn các từ xung quanh như deposited/money hay river/sat) tạo vector riêng cho từng lần xuất hiện nên phân biệt được hai nghĩa. Đó là lý do Transformer là bước tiếp theo.

## Những điều rút ra
- Distributional hypothesis hoạt động thật: doctor–physician gần nhau (0.716) dù **không bao giờ xuất hiện cùng câu**, vì chia sẻ context.
- Dữ liệu quan trọng hơn kích thước vector: tăng corpus từ 3k lên 8k doc cải thiện nhiều hơn tăng dim 50→300.
- Dense không phải lúc nào cũng tốt hơn: ở semantic search, embedding trung bình vượt TF-IDF (P@20 0.90 vs 0.80 cho "medical treatment"), nhưng query expansion bằng neighbor lại giảm P@20 (0.60) vì kéo thêm từ nhiễu như malpractice, veterinary.
- Analogy chỉ là pattern hình học (queen đúng ở king−man+woman nhưng fail nhiều quan hệ khác), không phải bằng chứng "hiểu".

## Ôn tập cho Individual learning check
1. **Distributional hypothesis là gì?** Những từ xuất hiện trong context giống nhau thì có nghĩa gần nhau. Đây là nền tảng của word embedding.
2. **Tại sao doctor và physician có thể gần nhau?** Chúng chia sẻ context (patient, treat, medical, hospital...) nên vector gần nhau, dù có thể không bao giờ đứng cạnh nhau (trong thí nghiệm: 0 câu chứa cả hai nhưng cos = 0.716).
3. **CBOW khác Skip-gram ở đâu?** CBOW: input là các context words (gộp lại), dự đoán target; nhanh hơn, tốt cho từ tần suất cao. Skip-gram: input là target, dự đoán từng context word; chậm hơn nhưng tốt hơn với từ hiếm (trong thí nghiệm 38s so với 16s).
4. **Tại sao tăng context window có thể vừa tốt vừa xấu?** Tốt: bắt được quan hệ chủ đề (doctor–nurse 0.564 → 0.625). Xấu: kéo thêm context xa, nhiễu hơn, và nghiêng về liên quan chủ đề hơn là đồng nghĩa (doctor–physician 0.730 → 0.676); thời gian train tăng (23s → 55s).
5. **Tại sao Word2Vec không phân biệt được hai nghĩa của bank?** Chỉ có một vector cho mỗi từ, được học từ mọi câu chứa từ đó, nên các nghĩa bị trộn thành một trung bình, nghĩa phổ biến lấn át nghĩa hiếm.
6. **Tại sao TF-IDF không phải word embedding?** TF-IDF là biểu diễn sparse theo tài liệu, một chiều cho mỗi từ, không học từ context và không tạo ra quan hệ giữa từ gần nghĩa (doctor và physician là hai chiều độc lập, cosine = 0).
