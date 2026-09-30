"""
LAB 02 — N-gram language model (mục 14–15).

Không dùng thư viện language model có sẵn (nltk.lm, kenlm, ...); chỉ dùng thư viện chuẩn.

Quy ước:
    - Mỗi câu được thêm (n-1) token <s> ở đầu và một token </s> ở cuối (pad=True).
      </s> được tính là một token cần dự đoán; <s> thì không.
    - Từ ngoài vocabulary được thay bằng <unk>.
    - MLE:     P(w | h) = C(h, w) / C(h)
    - Laplace: P(w | h) = (C(h, w) + 1) / (C(h) + V),  V = số token có thể dự đoán
               (vocabulary + </s> + <unk>, không gồm <s>)
    - log dùng logarit tự nhiên; perplexity = exp(-1/N · Σ log P).

Run:
    python ngram_lm.py            # chạy unit tests
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

Ngram = Tuple[str, ...]

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


# ---------------------------------------------------------------------------
# Tiền xử lý
# ---------------------------------------------------------------------------
def tokenize(sentence: str) -> List[str]:
    """Lowercase, giữ chữ/số (và dạng rút gọn như don't), bỏ dấu câu."""
    return _WORD_RE.findall(sentence.lower())


def split_sentences(document: str) -> List[str]:
    """Tách document thành câu theo . ! ? và xuống dòng."""
    return [s.strip() for s in _SENT_RE.split(document) if s.strip()]


def _as_tokens(corpus: Iterable) -> List[List[str]]:
    """Chấp nhận list câu (str) hoặc list câu đã tách token (list[str])."""
    return [tokenize(s) if isinstance(s, str) else list(s) for s in corpus]


# ---------------------------------------------------------------------------
# Các hàm lõi (mục 14)
# ---------------------------------------------------------------------------
def build_vocabulary(corpus: Iterable, min_count: int = 1) -> set:
    """Tập các từ xuất hiện ít nhất min_count lần (chưa gồm <s>, </s>, <unk>)."""
    counts = Counter(t for sent in _as_tokens(corpus) for t in sent)
    return {w for w, c in counts.items() if c >= min_count}


def pad_sentence(tokens: Sequence[str], n: int) -> List[str]:
    """Thêm (n-1) <s> ở đầu và </s> ở cuối."""
    return [BOS] * (n - 1) + list(tokens) + [EOS]


def count_ngrams(corpus: Iterable, n: int, pad: bool = True) -> Counter:
    """Đếm các n-gram (tuple độ dài n) trong corpus."""
    counts: Counter = Counter()
    for sent in _as_tokens(corpus):
        toks = pad_sentence(sent, n) if pad else sent
        for i in range(len(toks) - n + 1):
            counts[tuple(toks[i:i + n])] += 1
    return counts


def _train_mle(corpus: Iterable, n: int, pad: bool) -> Dict[Ngram, float]:
    """Bảng P(w | h) = C(h, w) / C(h) cho mọi n-gram đã thấy."""
    ngrams = count_ngrams(corpus, n, pad)
    if n == 1:
        total = sum(ngrams.values())
        return {g: c / total for g, c in ngrams.items()}
    context_counts: Counter = Counter()
    for g, c in ngrams.items():
        context_counts[g[:-1]] += c
    return {g: c / context_counts[g[:-1]] for g, c in ngrams.items()}


def train_unigram(corpus: Iterable, pad: bool = False) -> Dict[Ngram, float]:
    """P(w) = C(w) / N."""
    return _train_mle(corpus, 1, pad)


def train_bigram(corpus: Iterable, pad: bool = False) -> Dict[Ngram, float]:
    """P(w_t | w_{t-1}) = C(w_{t-1}, w_t) / C(w_{t-1})."""
    return _train_mle(corpus, 2, pad)


def train_trigram(corpus: Iterable, pad: bool = False) -> Dict[Ngram, float]:
    """P(w_t | w_{t-2}, w_{t-1}) = C(w_{t-2}, w_{t-1}, w_t) / C(w_{t-2}, w_{t-1})."""
    return _train_mle(corpus, 3, pad)


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------
class NGramLanguageModel:
    def __init__(self, n: int, smoothing: str = "mle", min_count: int = 1, pad: bool = True):
        if smoothing not in ("mle", "laplace"):
            raise ValueError("smoothing must be 'mle' or 'laplace'")
        self.n = n
        self.smoothing = smoothing
        self.min_count = min_count
        self.pad = pad
        self.vocab: set = set()
        self.V = 0
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()

    # --- training ---------------------------------------------------------
    def _map_unk(self, tokens: Sequence[str]) -> List[str]:
        return [t if t in self.vocab else UNK for t in tokens]

    def _prepare(self, sentence) -> List[str]:
        tokens = tokenize(sentence) if isinstance(sentence, str) else list(sentence)
        tokens = self._map_unk(tokens)
        return pad_sentence(tokens, self.n) if self.pad else tokens

    def fit(self, corpus: Sequence) -> "NGramLanguageModel":
        corpus = _as_tokens(corpus)
        self.vocab = build_vocabulary(corpus, self.min_count)
        self.V = self._compute_V()
        n = self.n
        for sent in corpus:
            toks = self._prepare(sent)
            for i in range(len(toks) - n + 1):
                g = tuple(toks[i:i + n])
                self.ngram_counts[g] += 1
                self.context_counts[g[:-1]] += 1
        return self

    def _compute_V(self) -> int:
        """Số token có thể dự đoán (dùng trong mẫu số Laplace)."""
        extra = {EOS} if self.pad else set()
        if self.min_count > 1 or UNK in self.vocab:
            extra.add(UNK)
        return len(self.vocab | extra)

    # --- probabilities ------------------------------------------------------
    def probability(self, context: Sequence[str], word: str) -> float:
        """P(word | context). Chỉ dùng (n-1) token cuối của context."""
        h = tuple(context)[-(self.n - 1):] if self.n > 1 else ()
        h = tuple(t if t in self.vocab or t in (BOS, EOS) else UNK for t in h)
        if word not in self.vocab and word != EOS:
            word = UNK
        c_hw = self.ngram_counts.get(h + (word,), 0)
        c_h = self.context_counts.get(h, 0)
        if self.smoothing == "laplace":
            return (c_hw + 1) / (c_h + self.V)
        return c_hw / c_h if c_h else 0.0

    def _token_log_probs(self, sentence) -> List[float]:
        toks = self._prepare(sentence)
        k = self.n - 1
        start = k if self.pad else 0
        out = []
        for i in range(start, len(toks)):
            p = self.probability(toks[max(0, i - k):i], toks[i])
            out.append(math.log(p) if p > 0 else -math.inf)
        return out

    def sentence_log_probability(self, sentence) -> float:
        """log P(S) = Σ_t log P(w_t | context)  (mục 15)."""
        return sum(self._token_log_probs(sentence))

    def sentence_probability(self, sentence) -> float:
        """P(S) = exp(log P(S)). Với câu dài giá trị này underflow về 0.0."""
        return math.exp(self.sentence_log_probability(sentence))

    def evaluate(self, sentences: Sequence) -> Dict[str, float]:
        """Perplexity + số token có xác suất 0."""
        total_lp, n_tok, n_zero = 0.0, 0, 0
        for s in sentences:
            for lp in self._token_log_probs(s):
                n_tok += 1
                if lp == -math.inf:
                    n_zero += 1
                else:
                    total_lp += lp
        ppl = math.inf if n_zero else math.exp(-total_lp / n_tok)
        return {"perplexity": ppl, "tokens": n_tok, "zero_prob_tokens": n_zero,
                "zero_prob_rate": n_zero / n_tok if n_tok else 0.0}

    def perplexity(self, sentences: Sequence) -> float:
        """PP = exp(-1/N · Σ log P(w_i | context_i))  (mục 17)."""
        return self.evaluate(sentences)["perplexity"]

    def next_word_distribution(self, context: Sequence[str], k: int = 5,
                               exclude=(UNK,)) -> List[Tuple[str, float]]:
        """Top-k từ tiếp theo và xác suất (mục 20)."""
        if isinstance(context, str):
            context = tokenize(context)
        h = list(context)
        if self.pad and self.n > 1 and len(h) < self.n - 1:
            h = [BOS] * (self.n - 1 - len(h)) + h
        h = tuple(t if t in self.vocab or t == BOS else UNK for t in h)[-(self.n - 1):] if self.n > 1 else ()
        # Quét các n-gram đã thấy có cùng context (không lưu thêm index để tiết kiệm RAM).
        candidates = [g[-1] for g in self.ngram_counts if g[:-1] == h] or sorted(self.vocab)
        scored = [(w, self.probability(h, w)) for w in candidates if w not in exclude]
        return sorted(scored, key=lambda x: -x[1])[:k]

    def continuation_log_probability(self, context: str, continuation: str) -> float:
        """log P(continuation | context), dùng cho sentence ranking (mục 21)."""
        ctx = self._map_unk(tokenize(context))
        cont = tokenize(continuation)
        hist = ([BOS] * (self.n - 1) if self.pad else []) + ctx
        total = 0.0
        for w in cont:
            p = self.probability(hist[-(self.n - 1):] if self.n > 1 else [], w)
            total += math.log(p) if p > 0 else -math.inf
            hist.append(w if w in self.vocab else UNK)
        return total


# ---------------------------------------------------------------------------
# Unit tests — dùng corpus riêng, khác corpus bài tập tính tay.
# ---------------------------------------------------------------------------
EPS = 1e-9
TEST_CORPUS = ["red fish blue fish", "one fish two fish", "red car"]


def test_tokenize_and_split():
    assert tokenize("Red FISH, blue fish!") == ["red", "fish", "blue", "fish"]
    assert split_sentences("One fish. Two fish!\nRed") == ["One fish.", "Two fish!", "Red"]


def test_build_vocabulary():
    assert build_vocabulary(TEST_CORPUS) == {"red", "fish", "blue", "one", "two", "car"}
    assert build_vocabulary(TEST_CORPUS, min_count=2) == {"red", "fish"}


def test_count_ngrams():
    uni = count_ngrams(TEST_CORPUS, 1, pad=False)
    assert uni[("fish",)] == 4 and sum(uni.values()) == 10
    bi = count_ngrams(TEST_CORPUS, 2, pad=True)
    assert bi[(BOS, "red")] == 2 and bi[("fish", EOS)] == 2
    tri = count_ngrams(TEST_CORPUS, 3, pad=True)
    assert tri[(BOS, BOS, "red")] == 2


def test_train_unigram():
    p = train_unigram(TEST_CORPUS)
    assert abs(p[("fish",)] - 4 / 10) < EPS
    assert abs(sum(p.values()) - 1.0) < EPS


def test_train_bigram():
    p = train_bigram(TEST_CORPUS)
    assert abs(p[("red", "fish")] - 1 / 2) < EPS
    assert abs(p[("fish", "blue")] - 1 / 2) < EPS  # "fish" có 2 từ theo sau (không pad)


def test_train_trigram():
    p = train_trigram(TEST_CORPUS)
    assert abs(p[("red", "fish", "blue")] - 1.0) < EPS


def test_probability_sums_to_one():
    for smoothing in ("mle", "laplace"):
        lm = NGramLanguageModel(2, smoothing).fit(TEST_CORPUS)
        targets = sorted(lm.vocab) + [EOS]
        total = sum(lm.probability(["fish"], w) for w in targets)
        assert abs(total - 1.0) < EPS, (smoothing, total)


def test_mle_zero_and_laplace():
    mle = NGramLanguageModel(2, "mle").fit(TEST_CORPUS)
    lap = NGramLanguageModel(2, "laplace").fit(TEST_CORPUS)
    assert mle.probability(["car"], "fish") == 0.0
    # C(car)=1 (car </s>), C(car fish)=0, V = 6 từ + </s> = 7
    assert abs(lap.probability(["car"], "fish") - 1 / 8) < EPS
    assert mle.sentence_log_probability("red car fish") == -math.inf


def test_sentence_log_probability():
    lm = NGramLanguageModel(2, "mle").fit(TEST_CORPUS)
    # P(red|<s>) P(car|red) P(</s>|car) = 2/3 * 1/2 * 1
    expected = (2 / 3) * (1 / 2) * 1.0
    assert abs(lm.sentence_probability("red car") - expected) < EPS
    assert abs(lm.sentence_log_probability("red car") - math.log(expected)) < EPS


def test_perplexity():
    lm = NGramLanguageModel(2, "mle").fit(TEST_CORPUS)
    lp = lm.sentence_log_probability("red car")
    assert abs(lm.perplexity(["red car"]) - math.exp(-lp / 3)) < EPS  # 3 token: red car </s>


def test_unk_mapping():
    lm = NGramLanguageModel(2, "laplace", min_count=2).fit(TEST_CORPUS)
    assert UNK in {g[-1] for g in lm.ngram_counts}
    assert lm.probability(["red"], "zebra") == lm.probability(["red"], UNK)


def test_next_word_distribution():
    lm = NGramLanguageModel(2, "mle").fit(TEST_CORPUS)
    top = lm.next_word_distribution(["red"], k=2)
    assert {w for w, _ in top} == {"fish", "car"}
    assert abs(top[0][1] - 0.5) < EPS


def run_tests() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
