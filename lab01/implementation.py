"""
LAB 01 - Part E: Core implementation of TF-IDF and cosine similarity.

Only the Python standard library is used for the core functions.
scikit-learn is imported *only* inside `compare_with_sklearn()` (Part 8.5),
after the implementation has been verified by the unit tests.

Conventions used here (the same as the lab handout, section 4):
    tf(t, d)   = c(t, d) / sum_t' c(t', d)
    df(t)      = |{d : t in d}|
    idf(t)     = ln(N / df(t))           (natural log)
    tfidf(t,d) = tf(t, d) * idf(t)
    cos(x, y)  = x.y / (||x||_2 * ||y||_2)

Run:
    python implementation.py            # run unit tests + sklearn comparison
    python -m pytest implementation.py  # same tests with pytest
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, List, Sequence

Vector = List[float]

TOY_CORPUS = [
    "cat eats fish",
    "dog eats fish",
    "cat likes fish",
]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase + keep alphanumeric runs (a minimal tokenizer)."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def build_vocabulary(
    corpus: Sequence[str], tokenizer: Callable[[str], List[str]] = tokenize
) -> Dict[str, int]:
    """Return a mapping term -> column index, with terms sorted alphabetically."""
    terms = set()
    for doc in corpus:
        terms.update(tokenizer(doc))
    return {term: idx for idx, term in enumerate(sorted(terms))}


def compute_counts(
    doc: str, vocab: Dict[str, int], tokenizer: Callable[[str], List[str]] = tokenize
) -> Vector:
    """Count vector c(t, d) of length |V|. Out-of-vocabulary tokens are ignored."""
    vec = [0.0] * len(vocab)
    for term, c in Counter(tokenizer(doc)).items():
        idx = vocab.get(term)
        if idx is not None:
            vec[idx] = float(c)
    return vec


def compute_tf(counts: Sequence[float]) -> Vector:
    """Relative term frequency: each count divided by the document length."""
    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    return [c / total for c in counts]


def compute_df(count_matrix: Sequence[Sequence[float]]) -> Vector:
    """Document frequency of each column: number of rows with a non-zero entry."""
    n_terms = len(count_matrix[0]) if count_matrix else 0
    df = [0.0] * n_terms
    for row in count_matrix:
        for j, c in enumerate(row):
            if c > 0:
                df[j] += 1
    return df


def compute_idf(count_matrix: Sequence[Sequence[float]]) -> Vector:
    """idf(t) = ln(N / df(t)). Terms with df = 0 get idf = 0."""
    n_docs = len(count_matrix)
    return [math.log(n_docs / d) if d > 0 else 0.0 for d in compute_df(count_matrix)]


def compute_tfidf(tf: Sequence[float], idf: Sequence[float]) -> Vector:
    """Element-wise product tf * idf."""
    if len(tf) != len(idf):
        raise ValueError(f"length mismatch: tf={len(tf)} idf={len(idf)}")
    return [a * b for a, b in zip(tf, idf)]


def cosine_similarity(x: Sequence[float], y: Sequence[float]) -> float:
    """Cosine of the angle between x and y. Returns 0.0 if either is a zero vector."""
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    dot = sum(a * b for a, b in zip(x, y))
    nx = math.sqrt(sum(a * a for a in x))
    ny = math.sqrt(sum(b * b for b in y))
    if nx == 0 or ny == 0:
        return 0.0
    return dot / (nx * ny)


# ---------------------------------------------------------------------------
# Small end-to-end helper
# ---------------------------------------------------------------------------
class TinyTfidf:
    """Fit/transform wrapper around the functions above (for small corpora)."""

    def __init__(self, tokenizer: Callable[[str], List[str]] = tokenize):
        self.tokenizer = tokenizer
        self.vocab: Dict[str, int] = {}
        self.idf: Vector = []

    def fit(self, corpus: Sequence[str]) -> "TinyTfidf":
        self.vocab = build_vocabulary(corpus, self.tokenizer)
        counts = [compute_counts(d, self.vocab, self.tokenizer) for d in corpus]
        self.idf = compute_idf(counts)
        return self

    def transform(self, docs: Sequence[str]) -> List[Vector]:
        return [
            compute_tfidf(compute_tf(compute_counts(d, self.vocab, self.tokenizer)), self.idf)
            for d in docs
        ]

    def fit_transform(self, corpus: Sequence[str]) -> List[Vector]:
        return self.fit(corpus).transform(corpus)

    def search(self, query: str, corpus_vectors: Sequence[Vector], k: int = 5):
        q = self.transform([query])[0]
        scores = [(i, cosine_similarity(q, v)) for i, v in enumerate(corpus_vectors)]
        return sorted(scores, key=lambda p: -p[1])[:k]


# ---------------------------------------------------------------------------
# Unit tests (at least one per function, section 8.4)
# ---------------------------------------------------------------------------
EPS = 1e-9


def _toy():
    vocab = build_vocabulary(TOY_CORPUS)
    counts = [compute_counts(d, vocab) for d in TOY_CORPUS]
    return vocab, counts


def test_tokenize():
    assert tokenize("Cat, EATS fish!") == ["cat", "eats", "fish"]


def test_build_vocabulary():
    vocab, _ = _toy()
    assert list(vocab) == ["cat", "dog", "eats", "fish", "likes"]
    assert vocab["fish"] == 3


def test_compute_counts():
    vocab, counts = _toy()
    assert counts == [
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1],
    ]
    # repeated term + OOV term
    assert compute_counts("fish fish bird", vocab) == [0, 0, 0, 2, 0]


def test_compute_tf():
    vocab, counts = _toy()
    tf = compute_tf(counts[0])
    tf_cat = tf[vocab["cat"]]
    assert abs(tf_cat - 1 / 3) < EPS
    assert abs(sum(tf) - 1.0) < EPS
    assert compute_tf([0, 0]) == [0.0, 0.0]


def test_compute_df():
    _, counts = _toy()
    assert compute_df(counts) == [2, 1, 2, 3, 1]


def test_compute_idf():
    vocab, counts = _toy()
    idf = compute_idf(counts)
    assert abs(idf[vocab["cat"]] - math.log(3 / 2)) < EPS
    assert abs(idf[vocab["dog"]] - math.log(3)) < EPS
    assert abs(idf[vocab["fish"]]) < EPS  # appears in every doc -> idf = 0


def test_compute_tfidf():
    vocab, counts = _toy()
    idf = compute_idf(counts)
    tfidf_d1 = compute_tfidf(compute_tf(counts[0]), idf)
    assert abs(tfidf_d1[vocab["cat"]] - (1 / 3) * math.log(3 / 2)) < EPS
    assert abs(tfidf_d1[vocab["eats"]] - (1 / 3) * math.log(3 / 2)) < EPS
    assert abs(tfidf_d1[vocab["fish"]]) < EPS
    assert tfidf_d1[vocab["dog"]] == 0.0


def test_cosine_similarity():
    assert abs(cosine_similarity([1, 1, 1], [1, 1, 0]) - 2 / math.sqrt(6)) < EPS
    assert abs(cosine_similarity([1, 2], [2, 4]) - 1.0) < EPS  # same direction
    assert cosine_similarity([1, 0], [0, 1]) == 0.0  # orthogonal
    assert cosine_similarity([0, 0], [1, 1]) == 0.0  # zero vector


def test_tiny_tfidf_search():
    model = TinyTfidf()
    X = model.fit_transform(TOY_CORPUS)
    top_doc, _ = model.search("dog", X, k=1)[0]
    assert top_doc == 1


def run_tests() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"All {len(tests)} tests passed.")


# ---------------------------------------------------------------------------
# Section 8.5 - compare with the reference implementation (scikit-learn)
# ---------------------------------------------------------------------------
def compare_with_sklearn(corpus: Sequence[str] = TOY_CORPUS) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    ours = TinyTfidf().fit(corpus)
    X_ours = ours.transform(corpus)

    sk_default = TfidfVectorizer().fit(corpus)
    X_sk = sk_default.transform(corpus).toarray()

    print("\nVocabulary (ours)   :", list(ours.vocab))
    print("Vocabulary (sklearn):", list(sk_default.get_feature_names_out()))
    print("\nD1 ours    :", [round(v, 4) for v in X_ours[0]])
    print("D1 sklearn :", [round(float(v), 4) for v in X_sk[0]])

    # Reproduce sklearn's default convention with our own building blocks:
    #   raw counts (not relative tf), idf = ln((1+N)/(1+df)) + 1, then L2-normalise.
    n = len(corpus)
    counts = [compute_counts(d, ours.vocab) for d in corpus]
    df = compute_df(counts)
    idf_smooth = [math.log((1 + n) / (1 + d)) + 1 for d in df]
    rebuilt = []
    for row in counts:
        v = [c * w for c, w in zip(row, idf_smooth)]
        norm = math.sqrt(sum(a * a for a in v))
        rebuilt.append([a / norm for a in v])

    max_diff = max(abs(a - float(b)) for r1, r2 in zip(rebuilt, X_sk) for a, b in zip(r1, r2))
    print("\nD1 ours, sklearn convention:", [round(v, 4) for v in rebuilt[0]])
    print(f"max |ours(sklearn convention) - sklearn| = {max_diff:.2e}")
    assert max_diff < 1e-9
    print("-> The difference comes only from conventions (tf, idf smoothing, L2 norm).")


if __name__ == "__main__":
    print("Running unit tests...")
    run_tests()
    compare_with_sklearn()
