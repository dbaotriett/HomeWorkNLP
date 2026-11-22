# LAB 10 — Error Analysis (wrong start vs wrong end)

## Cách thực hiện

- **Model:** DistilBERT fine-tuned, L = 384, stride = 128.
- **Dữ liệu:** test = SQuAD dev, 10,570 câu hỏi. Có **2,555 câu EM = 0** (24.2%).
- **Cách phân loại:** với mỗi câu sai, lấy gold có F1 cao nhất. Bỏ khoảng trắng và dấu câu ở hai đầu cả prediction lẫn gold, rồi so vị trí ký tự:
  - **wrong start**: end đúng, start sai.
  - **wrong end**: start đúng, end sai.
  - **wrong span (chồng lấp)**: sai cả hai đầu nhưng vẫn chồng lấp gold.
  - **wrong span (sai chỗ)**: không chồng lấp, model chọn một cụm khác.

## Thống kê

| Loại lỗi | Số câu | % câu sai | Dài hơn gold | Ngắn hơn gold |
|---|--:|--:|--:|--:|
| wrong start | **654** | 25.6 | 400 | 254 |
| wrong end | **549** | 21.5 | 343 | 206 |
| wrong span (chồng lấp) | 174 | 6.8 | 108 | 66 |
| wrong span (sai chỗ) | 1,177 | 46.1 | 529 | 648 |
| cùng span, khác chuẩn hóa | 1 | 0.0 | — | — |
| **Tổng** | **2,555** | 100 | | |

- **#wrong start = 654 > #wrong end = 549**: start sai nhiều hơn end khoảng 19%.
- Lỗi ranh giới (start, end, chồng lấp) = 1,377 câu (54%). Phần lớn vẫn được F1 cao: 1,584 câu có EM = 0 nhưng F1 > 0, F1 trung bình 0.56.
- Khi sai một đầu, model thường cho span **dài hơn** gold (61% với start, 62% với end).

## 5 lỗi wrong start

| # | Câu hỏi | Gold | Prediction | Nhận xét |
|---|---|---|---|---|
| 1 | How much did the population of Victoria increase in ten years after the discovery of gold? | 76,000 to 540,000 | **sevenfold from** 76,000 to 540,000 | Thừa đầu. "sevenfold" cũng trả lời "how much", có thể coi là đúng |
| 2 | Where did Tesla and Twain hang out? | in his lab | his lab | Thiếu giới từ "in". Gold không nhất quán: có lúc gồm giới từ, có lúc không |
| 3 | What is usual form of oxygen bound compounds? | oxides | **chemical bonds with almost all other elements to give corresponding** oxides | Thừa đầu rất dài: model lấy cả mệnh đề có trùng từ "bound/bonds" |
| 4 | What did Luther think was required to stop the violence? | trust God's word | God's word | Thiếu động từ "trust": model chọn cụm danh từ, gold là cụm động từ |
| 5 | The Rhine and what other river drained the northern flanks of the alps? | Danube | **Rhone and** Danube | Câu hỏi nói "Rhine" nhưng context viết "Rhone", nên model không nhận ra Rhone chính là vế đã có trong câu hỏi và lấy cả cụm "X and Y" |

## 5 lỗi wrong end

| # | Câu hỏi | Gold | Prediction | Nhận xét |
|---|---|---|---|---|
| 1 | At A-level, what percentage of British students attend fee-paying schools? | 13 | 13 **per cent** | Thừa đơn vị. Về nghĩa là đúng |
| 2 | What is the main reason consulting pharmacists are increasingly working directly with patients? | many elderly people are now taking numerous medications but continue to live outside of institutional settings | many elderly people are now taking numerous medications | Dừng sớm ở "but": đáp án dài 16 word |
| 3 | How many miles from the north Sea is Newcastle? | 8.5 mi | 8.5 | Thiếu đơn vị: ngược với ví dụ 1 |
| 4 | How many affiliates does Sinclair Broadcast Group service or own? | 28 ABC affiliates and two additional subchannel-only | 28 | Câu hỏi "how many" nên model chỉ lấy số. Gold dài bất thường |
| 5 | What is the basic pay for a teacher, in Euros? | €27,814 | €27,814 **p.a.** | Thừa "p.a." (per annum) |

## 5 lỗi wrong span

| # | Loại | Câu hỏi | Gold | Prediction | Nhận xét |
|---|---|---|---|---|---|
| 1 | chồng lấp | When did the last glacial end? | end of the Pleistocene (~11,600 BP) | Pleistocene | Lấy đúng phần lõi, bỏ cả hai đầu |
| 2 | chồng lấp | What is unique about a hermaphrodite? | can produce both eggs and sperm, meaning it can fertilize its own egg | **a single animal** can produce … own egg**, not needing a mate** | Thừa cả hai đầu với mệnh đề dài |
| 3 | sai chỗ | Which of the three heavily populated areas has the least number of inhabitants? | the San Diego area | Los Angeles area | Đúng loại (khu vực), sai thực thể: không so sánh được số dân ("least") |
| 4 | sai chỗ | In what year did William Maclure begin the process of creating the first geological map of the U.S.? | 1807 | 1809 | Đúng loại (năm), sai năm. Câu hỏi diễn đạt lại ("commenced" → "begin") |
| 5 | sai chỗ | Who played Doctor Who in the revival series? | Christopher Eccleston | Russell T Davies | Đúng loại (người), sai vai trò: chọn nhà sản xuất thay vì diễn viên |

## Bảng tổng hợp loại lỗi

| Loại | Biểu hiện điển hình | Nguyên nhân chính |
|---|---|---|
| Thừa / thiếu từ chức năng | "in his lab" vs "his lab" | Gold SQuAD không nhất quán về giới từ và mạo từ |
| Thừa / thiếu đơn vị | "13 per cent", "8.5" vs "8.5 mi" | Đơn vị lúc thuộc đáp án, lúc không |
| Span dài bị cắt | đáp án 7+ word, mệnh đề "because / but" | Ranh giới mệnh đề mơ hồ; EM của nhóm 7+ chỉ 54.1 |
| Chọn cả danh sách | "Rhone and Danube" | Không loại được vế đã nêu trong câu hỏi khi context viết khác (Rhine / Rhone) |
| Đúng loại, sai thực thể | 1809 vs 1807, LA vs San Diego | Chọn theo loại câu hỏi + trùng từ, không suy luận so sánh |
| Distractor | 58% chọn đáp án giả trong thí nghiệm distractor | Phụ thuộc mạnh vào lexical overlap |

## Giả thuyết: vì sao wrong start nhiều hơn wrong end?

1. **Phần đầu span có nhiều từ "tùy chọn".** Mạo từ, giới từ, từ chỉ mức độ ("sevenfold from", "in", "a single animal") đứng trước lõi của đáp án. Người gán nhãn lúc giữ, lúc bỏ chúng, nên vị trí start trong dữ liệu train nhiễu hơn. Phần cuối thường kết thúc tự nhiên ở dấu câu hoặc hết cụm danh từ.
2. **Câu hỏi thường lặp lại phần ngay trước đáp án.** Ví dụ "increased … from" và "drained … Rhone and". Model dùng trùng từ để định vị nên dễ kéo start về sát phần trùng, tức là lấy thừa.
3. **Start và end được dự đoán độc lập** (hai head logit riêng), chỉ ràng buộc i ≤ j và độ dài ≤ 30. Không có gì buộc span phải là một cụm ngữ pháp trọn vẹn, nên đầu nào mơ hồ hơn thì sai nhiều hơn. Theo giả thuyết 1, đó là đầu start.
4. **Wrong end tập trung ở đáp án dài và ở đơn vị.** Gold dài 7+ word dừng ở giữa mệnh đề ("…numerous medications" / "…but continue…"). Với đơn vị, gold lúc có lúc không ("13" vs "8.5 mi"), nên model không học được quy tắc nhất quán.

**Hướng cải thiện:**
- Dùng model lớn hơn hoặc nhiều epoch hơn.
- Thêm ràng buộc span phải trùng ranh giới cụm từ (constituent).
- Huấn luyện với nhiều gold (SQuAD dev có 3 gold, train chỉ có 1).
- Thêm adversarial data (câu mồi) vào train để giảm phụ thuộc vào trùng từ.
