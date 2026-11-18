# LAB 09 — Error analysis

Model: **BERT fine-tuned** (`bert-base-cased` + Linear(768, 9) trên mọi token, Strategy A), CoNLL-2003 test (3,453 câu, 5,648 entity). Code nằm trong `ner_finetuning.ipynb`.

## Tổng quan lỗi (entity-level)

| | Baseline (features + LR) | BERT |
|---|--:|--:|
| Entity F1 | 0.749 | **0.915** |
| Gold entity bị nhận sai loại (đúng ranh giới) | 504 | 225 |
| Gold entity đúng loại nhưng sai ranh giới | 423 | 87 |
| Gold entity bị bỏ sót hoàn toàn | 213 | 60 |
| Sai cả loại lẫn ranh giới | 115 | 65 |

Phân loại các entity **dự đoán** sai của BERT: 143 spurious (FP, không chồng lên gold nào), 225 sai loại, 84 sai ranh giới; 60 gold entity bị bỏ sót hoàn toàn (FN).

Confusion matrix (`confusion_matrix.png`, word-level): nhầm nhiều nhất là O → MISC (124 word), LOC ↔ ORG (54 + 42), MISC → ORG (41). Gần như không nhầm B ↔ I cùng loại.

## 23. 5 false positive, 5 false negative, 5 boundary error (chọn ngẫu nhiên seed 2026) + lỗi sai loại

| # | Text (rút gọn) | Gold | Prediction | Error type | Explanation (hypothesis) |
|---|---|---|---|---|---|
| FP1 | "GENERAL OBLIGATION BONDS" | — | MISC "GENERAL OBLIGATION" | spurious entity | Tiêu đề VIẾT HOA: chữ hoa là tín hiệu tên riêng; "GENERAL" → GE ##NE ##RA ##L. |
| FP2 | "... before the New Year break ..." | — | MISC "New Year" | spurious / ambiguous | Tên ngày lễ viết hoa; CoNLL không gán nhãn ở đây. |
| FP3 | "Italian BTPs offered returns ..." | — | ORG "BTPs" | rare entity, domain-specific | Thuật ngữ trái phiếu Ý hiếm, viết hoa. |
| FP4 | "A meeting with the Pope is also planned" | — | MISC "Pope" | ambiguous entity | Chức danh viết hoa, gold không coi là entity. |
| FP5 | "Public Service Minister David Dofara" | — | ORG "Public Service" | context-dependent | Phần đầu chức danh bị hiểu là tên bộ (tổ chức); "David Dofara" nhận đúng PER. |
| FN1 | "... all kick- off times GMT )" | MISC "GMT" | — | rare / ambiguous | Viết tắt múi giờ, hiếm khi là entity. |
| FN2 | "League game on Thursday ( home team in CAPS )" | LOC "League" | — | annotation noise | Gold LOC cho "League" có vẻ là lỗi annotation. |
| FN3 | "SOCCER - FIFA BOSS HAVELANGE STANDS BY WEAH ." | PER "WEAH" | — | missing, tokenization | "WEAH" → W ##EA ##H trong tiêu đề viết hoa, mất hình dạng tên người. |
| FN4 | "Premier league" | MISC "Premier league" | — | context-dependent | Câu 2 word, không có ngữ cảnh. |
| FN5 | "Association games on Friday ..." | ORG "Association" | — | annotation fragment | Phần còn lại của tên dài bị tách dòng. |
| B1 | "-- New York Commodities Desk , 212-859-1640" | ORG "New York Commodities" | ORG "New York Commodities Desk" | wrong boundary | Lấy thừa "Desk"; ranh giới tên bộ phận không rõ. |
| B2 | "Scorers : Shkvyrin Igor 78" | PER "Shkvyrin Igor" | PER "Shkvyrin" | wrong boundary, rare entity | Tên Nga Họ-Tên đảo ngược, hiếm; cắt mất "Igor". |
| B3 | "10= Florence Masnada ( France ) 51" | PER "Masnada" | PER "Florence Masnada" | wrong boundary (gold đáng ngờ) | "Florence" là tên của vận động viên; dự đoán hợp lý hơn gold. |
| B4 | "the Santiago Bernabeu stadium" | LOC "Santiago Bernabeu stadium" | LOC "Santiago Bernabeu" | wrong boundary | Từ chữ thường "stadium" không được gộp. |
| B5 | "dubbed " The Super Peasants "" | ORG "The Super Peasants" | ORG "The" | wrong boundary, rare | Biệt danh hiếm trong ngoặc kép. |
| T1 | "Czech President Vaclav Havel ..." | LOC "Czech" | MISC "Czech" | wrong entity type | Tính từ quốc gia thường là MISC; gold không nhất quán. |
| T2 | "Fini told state radio RAI ..." | LOC "RAI" | ORG "RAI" | wrong type (gold đáng ngờ) | RAI là đài phát thanh (ORG). |

**Nhận xét:**
- Lỗi tập trung ở **tiêu đề VIẾT HOA** và **dòng ngắn không có ngữ cảnh** (bảng kết quả, tiêu đề cụt): tokenizer cased tách từ viết hoa thành nhiều mảnh lạ, và câu 1–2 word không có ngữ cảnh để phân biệt loại.
- Một phần lỗi là **annotation không nhất quán** (League = LOC, RAI = LOC, Masnada không có Florence): model có thể đúng hơn gold, nên F1 không phải lúc nào cũng phản ánh đúng chất lượng.
- **Sai loại** (225) nhiều hơn **sai ranh giới** (84): sau khi có ngữ cảnh, khó nhất là phân biệt ORG / LOC / MISC.

## Entity hiếm và entity nhập nhằng
- 48% entity test chưa xuất hiện trong train. F1 entity đã gặp / chưa gặp: BERT 0.964 / 0.863, baseline 0.922 / 0.573.
- 64 surface form có ≥ 2 loại trong test (thành phố = đội bóng: Barcelona, Leicester, Tasmania...): BERT đúng 87.7% trong 367 lần xuất hiện, baseline 69.5%.

## Mini IE (30 câu tin tức tự viết)
F1 = 0.944 (67/71). Lỗi: Elon Musk → ORGANIZATION (khi đứng trước "said Tesla"), Wimbledon → LOCATION thay vì MISC, "The New York Times" và "Pope Francis" lấy thừa mạo từ/chức danh (khác quy ước gold).
