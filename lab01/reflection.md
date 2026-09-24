# Reflection — LAB 01

1. **Prediction nào của em sai?**  
Trong Exercise 6, dự đoán IDF có thể làm thay đổi thứ hạng giữa các document. Thực tế, với Q = "medical image classification", D1 trùng cả 3 term nên `cos(Q, D1) = 1` dù có IDF hay không; D2 trùng 2 term, D3 trùng 0 term. Thứ hạng D1 > D2 > D3 giữ nguyên. Đã lập luận lại: IDF có thể đảo ranking trong trường hợp tổng quát, nhưng ở ví dụ này quan hệ trùng term quá rõ ràng nên không đổi.

2. **Kết quả nào bất ngờ nhất?**  
Vocabulary phụ thuộc rất mạnh vào tokenizer. Cùng corpus 30K documents, CountVectorizer mặc định cho 193,540 terms, còn pipeline tách theo khoảng trắng giữ dấu câu cho 473,388 terms — gấp ~2.5 lần. Chưa lường trước mức độ ảnh hưởng này. Ngoài ra, việc `implementation.py` dùng `ln` thay vì `log₁₀` cho ra `tf-idf(cat, D1) ≈ 0.1352` thay vì `0.0587`, nhưng tỉ lệ giữa các term vẫn giữ nguyên, `fish` vẫn bằng 0.

3. **Experiment nào cung cấp evidence mạnh nhất?**  
Part F (Preprocessing Ablation) là mạnh nhất. Nó cho thấy preprocessing không chỉ là làm sạch mà là một quyết định mô hình hóa: cùng dữ liệu, chỉ đổi tokenization mà vocabulary thay đổi 2.5 lần. Điều này chứng minh bằng số liệu rằng "preprocessing càng nhiều càng tốt" là giả định sai.

4. **Failure case quan trọng nhất là gì?**  
Query `heart attack treatment` không đẩy được document chứa `myocardial infarction therapy` lên top. Hai cụm gần nghĩa nhưng không chia sẻ token nào, nên TF-IDF thất bại hoàn toàn. Đây là failure điển hình của lexical representation: nó chỉ so khớp từ, không hiểu ngữ nghĩa. Đây chính là động lực chuyển sang word embedding / contextual embedding.

5. **Nếu được xây lại search engine, em sẽ thay đổi điều gì?**  
thêm bước chuẩn hóa từ (stemming/lemmatization) và mở rộng query bằng synonym. Về dài hạn, dùng embedding để bắt được similarity về nghĩa thay vì chỉ trùng token. Ngoài ra, chọn tokenizer cẩn thận hơn vì nó ảnh hưởng trực tiếp đến vocabulary size và sparsity.

6. **AI đã được sử dụng ở những phần nào và đóng góp cụ thể là gì?**  
- **AI contribution:**  
  - Gợi ý cách kiểm tra unit test cho `cosine_similarity`.  
  - Hỗ trợ format `calculations.md` và `prediction.md`.  
