# LAB 10 — Reflection

## 1. Kiểm chứng Prediction (mục 9)

| # | Dự đoán (viết trước khi chạy) | Kết quả thật | Đúng / sai |
|---|---|---|---|
| 1 | Câu mồi trùng từ làm model nhầm, EM giảm 10–20 điểm; baseline bị nặng hơn | EM 75.7 → **28.7 (−47)**, F1 84.0 → 32.5. Model chọn đáp án giả **58%**, baseline 69%. Baseline mất 78% F1 tương đối, model mất 61% | **Đúng hướng, sai mức độ**: thực tế nặng gấp 2–3 lần dự đoán |
| 2 | Span càng dài thì EM càng thấp; F1 giảm ít hơn EM | EM 79.8 (1 word) → 54.1 (7+), giảm 25.6; F1 84.8 → 75.5, giảm 9.4 | **Đúng** |
| 3 | Câu không liên quan: giảm < 2 F1. Distractor: giảm 10–25 F1 | irrelevant: **+0.2** F1. distractor: **−51.5** F1 | **Đúng** với irrelevant, **sai mức độ** với distractor |
| 4 | Xáo thứ tự câu: ảnh hưởng nhỏ | F1 84.30 → 83.48 (**−0.8**) trên cùng 1,824 câu | **Đúng** |

**Điều mình đánh giá sai.** Mình nghĩ fine-tuning trên 84 nghìn câu hỏi sẽ giúp model "hiểu" câu hỏi đủ để không bị lừa nhiều. Thực tế model chọn cả đáp án giả **sai loại**: câu "Who conceptualized the aeolipile?" được trả lời "New England Patriots". Nghĩa là tín hiệu "câu nào lặp lại câu hỏi" mạnh hơn tín hiệu "loại đáp án phải khớp wh-word". Experiment 4 xác nhận điều này từ phía khác: F1 tăng từ 73 lên 90 khi overlap giữa câu hỏi và câu đáp án tăng. Document QA cũng gặp đúng lỗi này ngoài thực tế: "deadline of LAB 06" được trả lời bằng deadline của LAB 10.

## 2. Reflection — Cùng backbone, ba bài toán (LAB 08, 09, 10)

```
                   Pretrained Transformer (BERT / DistilBERT)
                                    │
             ┌──────────────────────┼──────────────────────┐
             ↓                      ↓                      ↓
     [CLS] representation    Token representations   Token representations
        h_CLS ∈ R^d             h_1..h_T ∈ R^d          h_1..h_T ∈ R^d
             ↓                      ↓                      ↓
      Linear(d, C)              Linear(d, 9)           Linear(d, 2)
   softmax trên C lớp      softmax trên 9 nhãn BIO   softmax trên T vị trí
             ↓                 tại mỗi token          (cột start, cột end)
   1 nhãn cho cả câu        1 nhãn cho mỗi token       1 cặp (start, end)
             ↓                      ↓                      ↓
          LAB 08                 LAB 09                 LAB 10
    "Câu này thuộc lớp nào?"  "Từ nào là entity?"   "Đáp án nằm ở đâu?"
    Accuracy: 0.914          Entity F1: 0.915        EM 75.8 / F1 84.3
```

Ba lab dùng gần như cùng một backbone: hàng chục triệu tham số đã học ngôn ngữ từ pretraining. Thứ khác nhau chỉ là **head rất nhỏ** đặt lên trên và **target** dùng để tính loss. Với DistilBERT ở LAB 10, head chỉ có 1,538 tham số trong tổng 66 triệu.

Lý do nằm ở **output của bài toán có hình dạng gì**.
- **LAB 08** cần một nhãn cho cả văn bản, nên chỉ cần một vector tóm tắt: lấy biểu diễn [CLS], vốn đã "nhìn" toàn câu qua self-attention, rồi phân loại vào C lớp.
- **LAB 09** cần một nhãn cho mỗi từ, nên head áp dụng cùng một Linear lên **từng token**. Target là chuỗi BIO, và phải căn nhãn word ↔ subword: chỉ subword đầu nhận nhãn, các subword còn lại là −100.
- **LAB 10** không có tập nhãn cố định: đáp án là một **vị trí** trong chính input, nên số "lớp" bằng độ dài chuỗi. Linear(d, 2) cho mỗi token hai điểm, "là đầu" và "là cuối". Softmax được tính **theo trục vị trí** chứ không theo trục lớp. Target là hai chỉ số token, phải đổi từ vị trí ký tự bằng offset mapping.

Cả ba đều tận dụng một điều: Transformer trả về **biểu diễn có ngữ cảnh cho mọi token**, và đã học những biểu diễn này trước khi gặp bài toán. Vì vậy cùng một backbone có thể dùng biểu diễn tổng hợp (LAB 08) hoặc biểu diễn từng token (LAB 09, 10). Fine-tuning vừa học head vừa chỉnh nhẹ backbone cho bài toán mới, nên chỉ cần 2–3 epoch.

Theo mình, khó khăn thật sự nằm ở **dữ liệu**, không ở model. Ở LAB 09 là căn nhãn subword. Ở LAB 10 là offset mapping, truncation và sliding window: chỉ cần lệch một ký tự là nhãn sai, và window không chứa đáp án phải trỏ về [CLS]. Kết quả cũng cho thấy giới hạn chung của cả ba: model học rất tốt các **tín hiệu bề mặt** (trùng từ trong QA, tên quen thuộc trong NER) và dễ bị lừa khi các tín hiệu đó sai. Thí nghiệm distractor làm F1 giảm 51 điểm là bằng chứng rõ nhất.

## 3. Individual Learning Check (mục 33)

1. **Extractive QA khác classification thế nào?** Classification chọn một nhãn trong **tập cố định** (positive/negative). Extractive QA chọn một **đoạn trong input**: "lớp" là các vị trí token, thay đổi theo từng context. Output là văn bản trích từ context, không phải một nhãn có sẵn.
2. **Tại sao cần start và end?** Đáp án là một đoạn [i, j], cần hai số để xác định. Coi mỗi span là một lớp thì có T(T+1)/2 lớp (khoảng 74 nghìn với T = 384), quá nhiều. Tách thành hai phân phối P_start, P_end chỉ cần 2T điểm.
3. **Character offset khác token index?** Character offset là vị trí ký tự trong chuỗi context gốc ("Paris" bắt đầu ở ký tự 25). Token index là vị trí trong chuỗi token của **toàn input** [CLS] question [SEP] context [SEP] ("paris" là token 14). Một token có thể gồm nhiều ký tự, và một từ có thể thành nhiều token.
4. **Tại sao cần offset mapping?** Dataset cho đáp án theo ký tự, model dự đoán theo token. Offset mapping là cầu nối hai chiều: khi train, đổi ký tự thành token để làm nhãn; khi dự đoán, đổi token về ký tự để **trích đúng chuỗi gốc**. Decode token thì sẽ mất chữ hoa và dấu, ví dụ "Beyoncé" thành "beyonce".
5. **stride dùng để làm gì?** Stride là số token chồng lấp giữa hai window liên tiếp. Nhờ đó một đáp án nằm ở chỗ cắt vẫn có ít nhất một window chứa trọn, và mỗi window có thêm ngữ cảnh ở hai đầu.
6. **Vì sao context dài phải chia window?** Model có giới hạn độ dài (512 với BERT), và attention tốn O(L²). Chỉ truncate thì mất đáp án: 13.5% ở L = 128, 0.07% ở L = 384. Chia window giữ được khoảng 100% đáp án.
7. **Exact Match đo điều gì?** Prediction có trùng **hoàn toàn** với ít nhất một gold sau khi chuẩn hóa hay không (chữ thường, bỏ dấu câu, mạo từ và khoảng trắng thừa). Kết quả là 0 hoặc 1 cho mỗi câu.
8. **F1 của QA khác accuracy?** Accuracy chấm đúng/sai một nhãn. F1 của QA so **túi từ** giữa prediction và gold: precision = số từ trùng / số từ của prediction, recall = số từ trùng / số từ của gold. F1 cho điểm một phần, và lấy max trên các gold.
9. **Vì sao F1 > 0 mà EM = 0?** Khi prediction trùng một phần với gold: lệch ranh giới, thừa hoặc thiếu đơn vị, giới từ ("13 per cent" vs "13"). Trên test có 1,584 câu như vậy (15%), F1 trung bình 0.56.
10. **Wrong start vs wrong end?** Wrong start: end đúng, start sai (thừa hoặc thiếu ở đầu). Wrong end: start đúng, end sai. Ở đây #wrong start = 654 > #wrong end = 549, vì đầu span có nhiều từ tùy chọn (giới từ, mạo từ) và gold không nhất quán (xem `error_analysis.md`).
11. **Extractive QA khác RAG ở đâu?** Extractive QA **giả định context đã có**, và đáp án là một đoạn trích nguyên văn. RAG thêm bước **retrieval** để tự tìm tài liệu liên quan trong kho lớn, rồi thường dùng **generator** sinh câu trả lời, có thể diễn đạt lại hoặc tổng hợp nhiều đoạn.
12. **Khi nào extractive QA không thể trả lời?**
    - Context không chứa đáp án (cần no-answer như SQuAD 2.0, nếu không model vẫn chọn đại một span).
    - Đáp án không phải một đoạn liên tục: yes/no, phải tính toán, tổng hợp nhiều câu.
    - Context sai hoặc chứa câu mồi (distractor).
    - Không biết context nào là đúng: việc này cần retrieval, sang LAB 11.

## 4. QA và RAG (mục 28)

```
Extractive QA (LAB 10):  Question + Context (đã cho) → QA model → Answer span
RAG (LAB 11–12):         Question → Retriever → Top-K documents → Context → Generator / QA model → Answer
```

Trong `document_qa.py`, tài liệu được đưa thẳng vào model, không có bước tìm kiếm. Cách này chỉ dùng được khi tài liệu ngắn và biết trước. Với 10,000 tài liệu thì không thể chạy QA trên tất cả (khoảng 2 window cho mỗi tài liệu 425 token, tức hàng chục nghìn lượt forward cho mỗi câu hỏi), nên cần retriever lọc ra top-K trước. Thí nghiệm distractor cũng là lời cảnh báo cho RAG: nếu retriever đưa vào đoạn trùng từ nhưng sai, QA model rất dễ trích nhầm đáp án từ đoạn đó.
