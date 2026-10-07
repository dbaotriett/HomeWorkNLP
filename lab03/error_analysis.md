# LAB 03 — Error analysis

Model: Word2Vec skip-gram + negative sampling, dim=100, window=5, min_count=3, epochs=10; corpus = 8,000 document đầu của C4 (≈2.9M token).

## Đúng (3)

**1. doctor → physician** (cos 0.716, hạng 2)
- Observed: physician là neighbor gần thứ 2 của doctor (sau dentist).
- Expected: rất gần.
- Possible explanation: đồng nghĩa, cùng context nghề nghiệp.
- Evidence from corpus: hai từ **không xuất hiện cùng câu lần nào** (0 lần), nhưng cùng đi với các context như patient, medical, surgery → đúng distributional hypothesis (similarity đến từ context chung, không phải từ việc đứng cạnh nhau).

**2. doctor → nurse** (cos 0.619, hạng 22)
- Observed: gần, cùng miền y tế. Expected: gần.
- Possible explanation: cùng bối cảnh bệnh viện/chăm sóc.
- Evidence from corpus: chỉ 2 câu chứa cả hai, vẫn gần nhờ context chung.

**3. cat → banana** (cos 0.094, hạng 32,870)
- Observed: xa. Expected: xa (khác miền).
- Possible explanation: động vật vs thức ăn, context khác hẳn.
- Evidence from corpus: top neighbor của banana là cereals, diced, fennel (nấu ăn); của cat là các từ về thú cưng.

## Sai / bất ngờ (3)

**1. car ~ automobile** (cos 0.592, hạng 24; trực giác 9.5)
- Observed: chỉ xếp hạng 8/20 cặp, thấp hơn computer–software.
- Expected: gần ngang doctor–physician.
- Possible explanation: **frequency / corpus nhỏ** — automobile ít dữ liệu nên vector chưa ổn định.
- Evidence from corpus: automobile chỉ xuất hiện 15 lần, physician 73 lần, car hàng nghìn lần.

**2. doctor ~ disease** (cos 0.401, hạng 6,281)
- Observed: khá xa dù liên quan chủ đề.
- Expected: gần hơn (trực giác 5.0, giữa bảng).
- Possible explanation: **loại quan hệ** — doctor và disease liên quan về chủ đề nhưng không thay thế được cho nhau; Word2Vec ưu tiên từ cùng vai trò (danh từ chỉ nghề).
- Evidence from corpus: neighbor của disease là chronic, autoimmune, syndrome (cùng loại với bệnh), không có doctor; chỉ 2 câu chứa cả hai.

**3. football** (football–stadium cos 0.269; top-5 của football là replica, shirt, shirts, shirtr, thcheap)
- Observed: neighbor là hàng áo bóng đá bán online, không phải soccer, league.
- Expected: soccer, league, team.
- Possible explanation: **domain bias + noisy data** — C4 là web crawl, có nhiều trang bán áo lặp lại (spam/SEO).
- Evidence from corpus: neighbor chứa token rác (`shirtr`, `thcheap`), nghĩa là kết quả bị kéo bởi một số trang lặp.

## Nguyên nhân chung
- Corpus nhỏ và không đều (từ hiếm ít bằng chứng); web text nhiễu; một vector cho mọi nghĩa (polysemy).
- Analogy đúng 8/15 (hit@1): fail ở quan hệ địa lý (tokyo→china, paris→france) và hình thái (walk→swimming) vì cần nhiều dữ liệu hơn.

## Polysemy: bank
Neighbor của `bank`: credit, citibank, td, banking, cheque — **toàn nghĩa "ngân hàng"**. Câu như "load bank is normally sized..." (bank = bộ tải) hay "river bank" bị nghĩa tài chính lấn át vì nghĩa này áp đảo trong corpus. Chỉ có một vector cho `bank`, nó là trung bình theo tần suất các nghĩa nên nghĩa hiếm gần như biến mất.
