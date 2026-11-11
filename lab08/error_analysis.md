# LAB 08 — Error analysis

Model: **DistilBERT fine-tuned** (`distilbert-base-uncased` + head [CLS] → Linear(768, 2)), IMDb, max_length 256, test 9,916 review (split 60/20/20, seed 2026). Code và hình nằm trong `fine_tuning.ipynb`.

## Confusion matrix (xem `confusion_matrix.png`)

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Fine-tuned | 4429 | 510 | 341 | 4636 | 0.9142 | 0.9009 | 0.9315 | 0.9159 |
| Frozen [CLS] | 4275 | 664 | 697 | 4280 | 0.8627 | — | — | 0.8628 |
| From scratch | 4549 | 390 | 880 | 4097 | 0.8719 | 0.9131 | 0.8232 | 0.8658 |

Fine-tuned hơi nghiêng positive (FP 510 > FN 341); from scratch nghiêng negative mạnh (FN 880).

## 20. 3 false positive, 3 false negative, 2 sai với confidence rất cao
FP/FN chọn ngẫu nhiên (seed 2026) trong review ≤ 200 từ; 2 mẫu cuối là 2 lỗi có confidence cao nhất toàn test set.

| # | Text (rút gọn) | True | Pred | Confidence | Error type | Hypothesis |
|---|---|---|---|--:|---|---|
| FP1 | "hated, hated, hated it! ... most booooring movie ... stick a fork in a toaster ... Leigh came up with a lot of really worthwhile stuff much later" | neg | pos | 0.635 | rare word / tokenization + context | "booooring" thành boo ##oor ##ing nên mất tín hiệu "boring"; câu khen sự nghiệp về sau của đạo diễn ở cuối kéo về positive. |
| FP2 | "This film may be great, but it is a complete ripoff of ... Comfort and Joy ... sweetest films ... I can recommend almost anything by forsythe" | neg | pos | 0.569 | context (khen phim khác) | Lời khen dành cho phim gốc và đạo diễn khác; model không phân biệt đối tượng được khen. |
| FP3 | "I loved the book ... The actors are to be commended ... smarmy, overwrought script ... insulted" | neg | pos | 0.504 | ambiguous sentiment | Khen sách và diễn viên, chê phim; sát ngưỡng 0.5. |
| FN1 | "Convoluted, infuriating and implausible ... completely ridiculous ... defies description" | pos | neg | 0.927 | label ambiguity | Phần lớn từ ngữ là chê; positive chỉ gián tiếp ("Parker Posey is really the only actress who could take this story"). |
| FN2 | "I just don't understand why this movie is getting beat-up ... mindless, it isn't polished ... The jokes work on more then one level" | pos | neg | 0.988 | negation + context | Review bênh vực phim nên nhắc lại lời chê của người khác; model tính các từ chê mà không hiểu chúng đang bị phản bác. |
| FN3 | "mess of a movie ... mediocre ... slow ... great story of Australian history ... 6/10" | pos | neg | 0.917 | ambiguous sentiment / label ambiguity | Review 6/10 lẫn lộn, chữ nghiêng về chê; nhãn positive dựa theo điểm số. |
| HC1 | "This sleek, sexy movie is a must-see ... masterpiece ... the trainables ... sexual experience" | neg | pos | **0.998** | sarcasm | Khen quá mức một cách mỉa mai; mọi từ trên bề mặt đều là từ khen. |
| HC2 | "Great movie - especially the music - Etta James - 'At Last'..." | neg | pos | **0.996** | label ambiguity / noise | Nội dung hoàn toàn khen; nhãn negative có vẻ sai, model đúng theo text. |

Cả hai lỗi confidence cao nhất **cũng sai ở model from scratch** (p = 0.99 và 0.98): đây là giới hạn của dữ liệu/nhãn (sarcasm, nhãn nhiễu), không riêng của pretrained model.

## Kiểm chứng giả thuyết trên toàn test set

| Nhóm | Tỉ lệ lỗi | n |
|---|--:|--:|
| ≤ 256 token | 0.066 | 5,904 |
| > 256 token (bị truncate) | **0.115** | 4,012 |
| Có từ phủ định | 0.089 | 8,524 |
| Không có phủ định | 0.068 | 1,392 |
| Có từ tương phản | 0.094 | 7,747 |
| Không có tương phản | 0.057 | 2,169 |

- **Truncation** là nguồn lỗi lớn nhất: review dài lỗi gần gấp đôi. Với 462 review dài bị sai, chỉ cần cho model đọc **256 token cuối** thay vì 256 token đầu là đúng **65.4%**: kết luận thường nằm ở cuối review.
- Negation và contrast làm tăng lỗi nhưng không nhiều (model fine-tuned xử lý "not", "but" tốt hơn LAB 07: các câu contrast trong Application đều đúng).

## 21. Confidence analysis (xem `confidence_analysis.png`)

| Confidence | n | % test | Accuracy | Số lỗi |
|---|--:|--:|--:|--:|
| Low (< 0.75) | 1,063 | 10.7 | 0.599 | 426 |
| Medium (0.75–0.95) | 2,262 | 22.8 | 0.859 | 318 |
| High (≥ 0.95) | 6,591 | 66.5 | 0.984 | 107 |

- Model khá calibrated (ECE 0.0125): confidence cao thì thường đúng.
- Nhưng **confidence ≠ correctness**: 107 lỗi (13% tổng số 851 lỗi) có confidence ≥ 0.95, có lỗi với confidence 0.998. Confidence là xác suất model tự gán, không phải bảo đảm đúng; sarcasm và nhãn nhiễu tạo ra lỗi "rất tự tin".
- Application: hai câu sai (sarcasm "Oh great, another two hours of my life I will never get back" → Positive 0.547; tweet "omg this film is lit 🔥🔥..." → Negative 0.461) đều có confidence thấp (~0.55), nhưng câu double negation "It's not that the movie isn't good" sai với confidence 0.93.
