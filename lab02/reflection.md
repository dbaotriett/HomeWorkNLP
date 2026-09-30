# Reflection — LAB 02

## Câu 1
**Nếu tăng n, mô hình nhận thêm thông tin gì?**

Model nhận thêm ngữ cảnh lịch sử dài hơn. Unigram chỉ biết từ hiện tại; bigram biết 1 từ trước; trigram biết 2 từ trước. Context dài hơn giúp model phân biệt được các trường hợp mà bậc thấp không phân biệt được (ví dụ: "the cat" vs "the dog" — bigram phân biệt được, unigram thì không).

## Câu 2
**Tại sao tăng n lại làm sparsity tăng?**

Vì số tổ hợp context–word tăng theo cấp số nhân khi n tăng, trong khi dữ liệu huấn luyện là hữu hạn. Với V = 47,699:
- Số bigram tiềm năng: V² ≈ 2.3 tỉ
- Số trigram tiềm năng: V³ ≈ 1.1 × 10¹⁴

Trong khi training chỉ có 2.9M token → không thể phủ hết. Kết quả: 64% trigram trong valid/test chưa từng thấy trong train.

## Câu 3
**Tại sao smoothing cần thiết?**

Vì MLE gán P = 0 cho mọi n-gram chưa thấy. Chỉ cần 1 token có P = 0 là cả câu P = 0 → PPL = ∞. Điều này làm:
- Không thể so sánh model trên valid/test.
- Model không tổng quát hóa được cho dữ liệu mới.

Smoothing (Laplace, add-k, backoff, interpolation) gán xác suất dương cho unseen n-gram, giúp PPL hữu hạn và model xử lý được dữ liệu mới.

## Câu 4
**Perplexity do điều gì?**

Perplexity đo mức độ "bối rối" của model khi quan sát dữ liệu. Nó phụ thuộc vào:
- Chất lượng model (model càng khớp dữ liệu, PPL càng thấp).
- Bậc n (n cao hơn fit training tốt hơn nhưng dễ overfit).
- Corpus size (corpus lớn → ít sparsity → PPL thấp hơn).
- Smoothing (Laplace làm PPL training tăng nhưng valid/test ổn định hơn).
- Preprocessing (tokenization, min_count, cách xử lý `<unk>`).
- Cách tính (có tính `<s>`, `</s>` hay không).

**Lưu ý:** PPL chỉ có ý nghĩa khi các model được đánh giá trên cùng dữ liệu, cùng preprocessing và cùng cách tính.

## Câu 5
**Một model có perplexity thấp hơn có luôn tạo ra văn bản tốt hơn đối với con người không? Giải thích.**

**Không.** PPL chỉ đo mức độ khớp phân phối xác suất, không đo:
- Tính mạch lạc và ý nghĩa của văn bản.
- Tính sáng tạo và đa dạng.
- Tính phù hợp ngữ cảnh văn hóa, đạo đức.
- Khả năng duy trì mạch văn dài.

Ví dụ: một model overfit có thể gán PPL rất thấp trên test set nếu test set giống train, nhưng sinh ra văn bản lặp lại, nhàm chán. Ngược lại, model có PPL cao hơn một chút có thể sinh văn bản tự nhiên hơn với con người.

## Câu 6
**N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?**

N-gram LM thất bại ở nhiều điểm:
1. **Phụ thuộc dài hạn:** chỉ nhìn n-1 từ trước, không hiểu mạch văn dài.
2. **Ngữ nghĩa:** không hiểu nghĩa của từ, chỉ dựa vào tần suất.
3. **Tương tự từ:** không biết "cat" và "kitten" gần nghĩa.
4. **Cấu trúc cú pháp phức tạp:** không nắm được dependency xa.
5. **Từ ngoài vocabulary:** xử lý kém với `<unk>`.
6. **Sáng tạo:** không thể tạo câu mới hợp lý nếu chưa thấy n-gram tương tự.
7. **Suy luận:** không thể suy ra thông tin ẩn.

Con người hiểu ngôn ngữ qua ngữ nghĩa, ngữ cảnh, và suy luận — những thứ n-gram LM hoàn toàn thiếu.

## Câu 7
**Nếu context dài 100 từ, trigram có sử dụng được thông tin của 97 từ đầu không?**

**Không.** Trigram chỉ dùng **2 từ gần nhất** làm context. 97 từ đầu hoàn toàn bị bỏ qua.

Ví dụ: câu "Hôm qua tôi đi chơi với bạn ở công viên, sau đó chúng tôi ăn tối, rồi về nhà, và tôi đã..." — trigram chỉ biết "và tôi đã" để dự đoán từ tiếp theo, không biết gì về "công viên", "ăn tối", "về nhà".

Đây là **điểm yếu cốt lõi** của n-gram LM và là động lực chính để chuyển sang **neural language models**:
- RNN/LSTM: có hidden state để ghi nhớ context dài.
- Attention/Transformer: có thể attend đến mọi token trong context.

Các model này có thể dùng thông tin của 100 từ, thậm chí hàng nghìn từ, mà không bị sparsity như n-gram.