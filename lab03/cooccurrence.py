"""LAB 03 - Co-occurrence word vectors (tự implement, không dùng Word2Vec).

context -> co-occurrence -> vector -> similarity
"""
import re
from collections import Counter

import numpy as np
from scipy import sparse


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def build_vocabulary(sentences, min_count=1, max_size=None):
    """Trả về (vocab: list[str], word2id: dict). Sắp xếp theo tần suất giảm dần."""
    counts = Counter(w for s in sentences for w in s)
    items = [(w, c) for w, c in counts.items() if c >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_size is not None:
        items = items[:max_size]
    vocab = [w for w, _ in items]
    return vocab, {w: i for i, w in enumerate(vocab)}


def build_cooccurrence_matrix(sentences, word2id, window=1):
    """Ma trận X[target, context] (scipy CSR). Context = các từ cách target <= window,
    không vượt qua ranh giới câu/document. Từ ngoài vocabulary bị bỏ qua
    (nhưng vẫn chiếm vị trí, tức là window đo theo vị trí gốc)."""
    n = len(word2id)
    rows, cols = [], []
    for s in sentences:
        ids = np.array([word2id.get(w, -1) for w in s], dtype=np.int64)
        for d in range(1, window + 1):
            if len(ids) <= d:
                break
            a, b = ids[:-d], ids[d:]
            ok = (a >= 0) & (b >= 0)
            a, b = a[ok], b[ok]
            rows += [a, b]  # đối xứng: (a->b) và (b->a)
            cols += [b, a]
    if not rows:
        return sparse.csr_matrix((n, n))
    r, c = np.concatenate(rows), np.concatenate(cols)
    X = sparse.coo_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n))
    return X.tocsr()  # tocsr cộng dồn các entry trùng


def ppmi(X):
    """Positive PMI: log(P(w,c)/(P(w)P(c))), cắt âm về 0. Giữ sparse."""
    X = sparse.csr_matrix(X, dtype=np.float64)
    total = X.sum()
    rs = np.asarray(X.sum(axis=1)).ravel()
    cs = np.asarray(X.sum(axis=0)).ravel()
    coo = X.tocoo()
    pmi = np.log(coo.data * total / (rs[coo.row] * cs[coo.col]))
    keep = pmi > 0
    return sparse.csr_matrix((pmi[keep], (coo.row[keep], coo.col[keep])), shape=X.shape)


def cosine_similarity(u, v):
    """Cosine của 2 vector (dense hoặc sparse 1 hàng). Trả 0 nếu một vector bằng 0."""
    if sparse.issparse(u):
        u = u.toarray().ravel()
    if sparse.issparse(v):
        v = v.toarray().ravel()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def most_similar(word, matrix, vocabulary, top_k=5):
    """Top-k từ gần `word` nhất theo cosine (loại chính nó). `vocabulary` là list từ."""
    w2i = {w: i for i, w in enumerate(vocabulary)}
    if word not in w2i:
        raise KeyError(word)
    M = sparse.csr_matrix(matrix, dtype=np.float64)
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    q = M[w2i[word]]
    sims = np.asarray((M @ q.T).todense()).ravel() / norms / norms[w2i[word]]
    sims[w2i[word]] = -np.inf
    top = np.argsort(-sims)[:top_k]
    return [(vocabulary[i], float(sims[i])) for i in top]


# ---------------------------------------------------------------- tests
def _tests():
    corpus = [s.split() for s in [
        "the cat eats fish", "the cat likes milk", "the dog eats meat", "the dog likes fish"]]
    vocab, w2i = build_vocabulary(corpus)
    assert set(vocab) == {"the", "cat", "dog", "eats", "likes", "fish", "milk", "meat"}
    X = build_cooccurrence_matrix(corpus, w2i, window=1).toarray()
    assert np.allclose(X, X.T)
    row = lambda w: {vocab[j]: int(v) for j, v in enumerate(X[w2i[w]]) if v}
    assert row("cat") == {"the": 2, "eats": 1, "likes": 1}
    assert row("dog") == {"the": 2, "eats": 1, "likes": 1}
    assert row("eats") == {"cat": 1, "dog": 1, "fish": 1, "meat": 1}
    assert cosine_similarity(X[w2i["cat"]], X[w2i["dog"]]) > 0.999
    assert abs(cosine_similarity(np.array([1, 2, 1]), np.array([2, 4, 2])) - 1) < 1e-9
    assert cosine_similarity(np.array([1, 0]), np.array([0, 1])) == 0
    assert most_similar("cat", X, vocab, 1)[0][0] == "dog"
    # window lớn hơn -> nhiều nonzero hơn
    X2 = build_cooccurrence_matrix(corpus, w2i, window=3)
    assert X2.nnz > sparse.csr_matrix(X).nnz
    # PPMI không âm
    assert ppmi(X).min() >= 0
    print("all tests passed")


if __name__ == "__main__":
    _tests()
