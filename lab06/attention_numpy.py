"""LAB 06 — Scaled dot-product attention và self-attention tự implement bằng NumPy.

Không dùng torch.nn.MultiheadAttention hay keras.layers.Attention; torch chỉ dùng trong _tests()
để đối chiếu kết quả.
"""
import numpy as np


def softmax(x, axis=-1):
    """Softmax ổn định số: trừ max trước khi exp (kết quả không đổi vì e^(x-m) = e^x / e^m)."""
    x = np.asarray(x, dtype=float)
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_naive(x, axis=-1):
    """Softmax tính thẳng np.exp(x): tràn số (inf/inf = nan) khi score lớn. Chỉ dùng để so sánh."""
    e = np.exp(np.asarray(x, dtype=float))
    return e / np.sum(e, axis=axis, keepdims=True)


def entropy(p, axis=-1):
    """Entropy (nat) của phân phối attention: thấp = tập trung, cao = phẳng (tối đa ln T)."""
    p = np.asarray(p, dtype=float)
    return -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0.0), axis=axis)


def scaled_dot_product_attention(Q, K, V, mask=None, scale=True, temperature=1.0):
    """S = QKᵀ / √d_k,  A = softmax(S),  C = AV.

    Q: (..., T_q, d_k), K: (..., T_k, d_k), V: (..., T_k, d_v).
    mask: (..., T_q, T_k) bool, True = được attend (False, ví dụ PAD, nhận weight 0).
    scale=False cho dot-product attention không chia √d_k; temperature chia thêm S cho T.
    Trả về (scores, attention_weights, context).
    """
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2)
    if scale:
        scores = scores / np.sqrt(d_k)
    scores = scores / temperature
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    context = weights @ V
    return scores, weights, context


def self_attention(X, W_Q, W_K, W_V, mask=None):
    """Self-attention: Q, K, V cùng tạo từ một sequence X (T, d). Trả về (Q, K, V, scores, weights, output)."""
    X = np.asarray(X, dtype=float)
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    scores, weights, out = scaled_dot_product_attention(Q, K, V, mask=mask)
    return Q, K, V, scores, weights, out


# Dữ liệu các bài tính tay (calculations.md)
BAI21_X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)          # I, like, cats
BAI21_WQ = np.array([[1, 0], [1, 1]], dtype=float)
BAI21_WK = np.array([[1, 1], [0, 1]], dtype=float)
BAI21_WV = np.array([[1, 0], [0, 2]], dtype=float)


def _tests():
    # Bài 1: scores
    q = np.array([1, 2]); K = np.array([[1, 0], [0, 1], [1, 1]])
    assert np.allclose(K @ q, [1, 2, 3])
    # Bài 2: softmax([1,2,3]) và tổng = 1
    a = softmax([1, 2, 3]); assert np.allclose(a, [0.0900, 0.2447, 0.6652], atol=1e-4) and np.isclose(a.sum(), 1)
    # Bài 3: context
    V3 = np.array([[1, 0], [0, 1], [2, 2]]); assert np.allclose(np.array([0.1, 0.2, 0.7]) @ V3, [1.5, 1.6])
    # Bài 4: toàn bộ (dot-product, không scale)
    s, w, c = scaled_dot_product_attention([[1, 0]], [[1, 0], [0, 1], [1, 1]], [[1, 0], [0, 2], [2, 1]], scale=False)
    assert np.allclose(s, [[1, 0, 1]]) and np.allclose(w, [[0.4223, 0.1554, 0.4223]], atol=1e-4)
    assert np.allclose(c, [[1.2669, 0.7331]], atol=1e-3)
    # Mục 14: scaling, d_k = 4
    assert np.allclose(softmax(np.array([4, 2, 0]) / 2), [0.6652, 0.2447, 0.0900], atol=1e-4)
    assert np.allclose(softmax([4, 2, 0]), [0.8668, 0.1173, 0.0159], atol=1e-4)
    # Mục 17: numerical stability
    x = np.array([1000, 1001, 1002])
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.isnan(softmax_naive(x)).all()
    assert np.allclose(softmax(x), softmax([0, 1, 2]))
    # Mục 21: self-attention
    Q, K_, _, _, A, _ = self_attention(BAI21_X, BAI21_WQ, BAI21_WK, BAI21_WV)
    assert np.allclose(Q @ K_.T, [[1, 0, 1], [2, 1, 3], [3, 1, 4]])
    assert np.allclose(A.sum(-1), 1)
    # Mask: vị trí bị mask nhận weight 0
    _, w, _ = scaled_dot_product_attention([[1, 0]], [[1, 0], [0, 1], [1, 1]], np.eye(3)[:, :2], mask=np.array([[True, True, False]]))
    assert w[0, 2] < 1e-12
    # Đối chiếu với PyTorch
    try:
        import torch
        import torch.nn.functional as F
        rng = np.random.default_rng(0)
        Qr, Kr, Vr = rng.normal(size=(2, 5, 8)), rng.normal(size=(2, 7, 8)), rng.normal(size=(2, 7, 4))
        ref = F.scaled_dot_product_attention(torch.tensor(Qr), torch.tensor(Kr), torch.tensor(Vr)).numpy()
        assert np.allclose(scaled_dot_product_attention(Qr, Kr, Vr)[2], ref, atol=1e-10)
        print("khớp torch.nn.functional.scaled_dot_product_attention: OK")
    except ImportError:
        pass
    print("all tests passed")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    _tests()
    print("Bài 4 weights:", scaled_dot_product_attention([[1, 0]], [[1, 0], [0, 1], [1, 1]], [[1, 0], [0, 2], [2, 1]], scale=False)[1])
    Q, K, V, S, A, O = self_attention(BAI21_X, BAI21_WQ, BAI21_WK, BAI21_WV)
    print("Q=\n", Q, "\nK=\n", K, "\nV=\n", V, "\nQK^T/sqrt2=\n", S, "\nA=\n", A, "\nAV=\n", O)
    for T in (0.5, 1.0, 2.0):
        p = softmax(np.array([2, 1, 0]) / T); print("T", T, p, "max", p.max().round(4), "H", entropy(p).round(4))
