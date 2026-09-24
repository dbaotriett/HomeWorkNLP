# Part C — Prediction before experiment


## Prediction 1 — Vocabulary
Với 30K documents, vocabulary khoảng **150,000–200,000 unique terms**, vì corpus lớn, nhiều chủ đề, và CountVectorizer mặc định không loại stopword / không giới hạn `max_features`, nên sẽ giữ nhiều tên riêng, biến thể chính tả, số, ký hiệu, v.v.

Con số này **phụ thuộc mạnh vào tokenizer**. Pipeline A (tách theo khoảng trắng, giữ dấu câu) cho vocabulary lớn hơn nhiều lần so với tokenizer mặc định của CountVectorizer — chi tiết này nối sang Part F.

## Prediction 2 — Sparsity
TF-IDF matrix sẽ **sparse**. Tỉ lệ zero entries khoảng **99.9% trở lên**, vì mỗi document chỉ chứa một tập rất nhỏ các term so với vocabulary hàng trăm nghìn từ.

## Prediction 3 — Search
Top kết quả **không nhất thiết** là các document gần nghĩa nhất.  
TF-IDF chủ yếu dựa trên **lexical overlap**: document trùng nhiều từ với query sẽ đứng cao, kể cả khi khác ngữ cảnh. Nó không hiểu synonym, paraphrase, phủ định hay word order. Vì vậy document đúng nghĩa nhưng dùng từ khác có thể bị bỏ sót.

## (Sau khi chạy) So sánh với kết quả thực

| Prediction | Dự đoán | Thực tế | Đúng/Sai |
|---|---|---|---|
| Vocabulary | 150,000–200,000 | 193,540 (CountVectorizer mặc định) | **Đúng** |
| Sparsity | ~99.9% zero entries | 99.914% | **Đúng** |
| Search | Không nhất thiết gần nghĩa nhất; dễ fail vì lexical gap | Query `heart attack treatment` không đẩy `myocardial infarction therapy` lên top, do hai cụm gần như không chia sẻ token | **Đúng** |

