# Part B — Calculation Exercises



## Exercise 1 — Count vector (vocab: [cat, dog, eats, fish, likes])

Corpus:
- D1 = "cat eats fish"
- D2 = "dog eats fish"
- D3 = "cat likes fish"

Count vector:

- D1 = `[1, 0, 1, 1, 0]`
- D2 = `[0, 1, 1, 1, 0]`
- D3 = `[1, 0, 0, 1, 1]`

## Exercise 2 — TF của D1 = "cat eats fish"

D1 có 3 từ. Dùng TF chuẩn hóa:

- tf(cat, D1) = 1/3 ≈ 0.3333
- tf(eats, D1) = 1/3 ≈ 0.3333
- tf(fish, D1) = 1/3 ≈ 0.3333
- Kiểm tra tổng = 1

Nếu dùng TF thô:
- tf(cat, D1) = 1
- tf(eats, D1) = 1
- tf(fish, D1) = 1
- Tổng = 3

## Exercise 3 — IDF (N = 3)

| term | df | idf |
|---|---:|---:|
| cat | 2 | log₁₀(3/2) ≈ 0.1761 |
| dog | 1 | log₁₀(3) ≈ 0.4771 |
| eats | 2 | log₁₀(3/2) ≈ 0.1761 |
| fish | 3 | log₁₀(1) = 0 |
| likes | 1 | log₁₀(3) ≈ 0.4771 |

Term có IDF thấp nhất: **fish**.

Vì sao: `fish` xuất hiện trong cả 3 tài liệu, tức `df = N = 3`, nên:

`idf(fish) = log(3/3) = log(1) = 0`

Nó không có khả năng phân biệt giữa các tài liệu.

## Exercise 4 — TF-IDF của D1

Dùng TF chuẩn hóa ở Exercise 2:

- cat: tf-idf = (1/3) × log₁₀(3/2) ≈ 0.0587
- eats: tf-idf = (1/3) × log₁₀(3/2) ≈ 0.0587
- fish: tf-idf = (1/3) × 0 = 0

Nếu dùng TF thô:

- cat: 1 × 0.1761 = 0.1761
- eats: 1 × 0.1761 = 0.1761
- fish: 1 × 0 = 0

Vì sao TF-IDF của `fish` bằng 0:

Vì `fish` có mặt trong mọi tài liệu nên `idf(fish) = log(3/3) = 0`.  
Nhân với TF nào cũng bằng 0.

## Exercise 5 — cos([1,1,1], [1,1,0])

Công thức:

`cos(a, b) = (a · b) / (||a|| × ||b||)`

Tính:

`a · b = 1×1 + 1×1 + 1×0 = 2`

`||a|| = sqrt(1² + 1² + 1²) = sqrt(3)`

`||b|| = sqrt(1² + 1² + 0²) = sqrt(2)`

Vậy:

`cos([1,1,1], [1,1,0]) = 2 / (sqrt(3) × sqrt(2)) = 2 / sqrt(6) ≈ 0.8165`

Vì sao không bằng 2/3:

Vì cosine chia cho **tích độ dài Euclid** của hai vector, tức `sqrt(3) × sqrt(2) = sqrt(6)`,  
không phải chia cho 3 hay chia cho tổng độ dài.

## Exercise 6 — Prediction (Q = "medical image classification")

Cho:
- D1 = "medical image classification"
- D2 = "medical image analysis"
- D3 = "natural language processing"

Tách từ:

- Q = `[medical, image, classification]`
- D1 = `[medical, image, classification]`
- D2 = `[medical, image, analysis]`
- D3 = `[natural, language, processing]`

### 1. Document nào có similarity cao nhất?

Q trùng với:
- D1: 3 từ (`medical`, `image`, `classification`)
- D2: 2 từ (`medical`, `image`)
- D3: 0 từ

Vậy similarity cao nhất là **D1**.

### 2. Document nào có similarity thấp nhất?

**D3** — không trùng từ nào với Q, nên similarity = 0.

### 3. Term nào có IDF thấp?

Trong 3 tài liệu:
- `medical` xuất hiện ở D1, D2 → df = 2
- `image` xuất hiện ở D1, D2 → df = 2
- Các term còn lại như `classification`, `analysis`, `natural`, `language`, `processing` đều có df = 1

Vậy term có IDF thấp là: **`medical`** và **`image`**.

Vì:

`idf(medical) = idf(image) = log(3/2) ≈ 0.1761`  
(thấp hơn `log(3) ≈ 0.4771` của các term chỉ xuất hiện trong 1 tài liệu)

### 4. Bỏ IDF, chỉ dùng count vector thì ranking có đổi không?

**Không đổi.**

Lý do cụ thể:

- `D1` và `Q` có **cùng tập term** (`medical`, `image`, `classification`).  
  Nên dù có IDF hay không, vector của D1 và Q luôn tỉ lệ với nhau, và:

  `cos(Q, D1) = 1`

  D1 luôn đứng rank 1.

- `D2` chỉ trùng 2 term (`medical`, `image`) với Q. Dù có IDF hay không, D2 vẫn đứng sau D1 và trước D3.

- `D3` không trùng term nào với Q, nên `cos(Q, D3) = 0` trong cả hai cách. D3 luôn đứng cuối.

Vậy thứ hạng **D1 > D2 > D3** giữ nguyên khi bỏ IDF.  
IDF có thể đảo ranking trong trường hợp tổng quát, nhưng ở ví dụ này không, vì quan hệ giữa Q với từng document là quá rõ ràng (trùng 3 / trùng 2 / trùng 0).
