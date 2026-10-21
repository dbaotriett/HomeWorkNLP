"""LAB 05 - Simple RNN cell tự implement bằng NumPy.

    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)

x_t -> W_xh -+
             +--(+)--> tanh --> h_t
h_{t-1} -> W_hh -+
"""
import numpy as np


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.RandomState(seed)
        k = 1.0 / np.sqrt(hidden_size)  # cùng thang khởi tạo với torch.nn.RNNCell
        self.input_size, self.hidden_size = input_size, hidden_size
        self.W_xh = rng.uniform(-k, k, (hidden_size, input_size))
        self.W_hh = rng.uniform(-k, k, (hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        """x_t: (input_size,), h_prev: (hidden_size,) -> h_t: (hidden_size,)."""
        a_t = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h   # pre-activation
        return np.tanh(a_t)

    def forward_sequence(self, xs, h0=None):
        """xs: (T, input_size) -> (list h_1..h_T). Trạng thái cuối h_T tóm tắt cả chuỗi."""
        h = np.zeros(self.hidden_size) if h0 is None else h0
        out = []
        for x_t in xs:
            h = self.forward(x_t, h)
            out.append(h)
        return out


def _tests():
    # 1) Bài 1 của đề (tính tay): h_t ≈ [0.585, 0.291]
    cell = SimpleRNNCell(2, 2)
    cell.W_xh = np.array([[0.5, 0.2], [0.1, 0.4]])
    cell.W_hh = np.array([[0.3, 0.1], [0.2, 0.5]])
    cell.b_h = np.zeros(2)
    h = cell.forward(np.array([1.0, 0.0]), np.array([0.5, 0.2]))
    assert np.allclose(h, [0.58497, 0.29131], atol=1e-4), h
    # 2) Khớp với torch.nn.RNNCell
    import torch
    t = torch.nn.RNNCell(2, 2, nonlinearity="tanh")
    with torch.no_grad():
        t.weight_ih.copy_(torch.tensor(cell.W_xh)); t.weight_hh.copy_(torch.tensor(cell.W_hh))
        t.bias_ih.zero_(); t.bias_hh.zero_()
        ht = t(torch.tensor([[1.0, 0.0]], dtype=torch.float64).float(), torch.tensor([[0.5, 0.2]]))
    assert np.allclose(h, ht.numpy()[0], atol=1e-5)
    # 3) Chuỗi: thứ tự quan trọng (dog bites man != man bites dog)
    c = SimpleRNNCell(3, 4, seed=1)
    dog, bites, man = np.eye(3)
    a = c.forward_sequence([dog, bites, man])[-1]
    b = c.forward_sequence([man, bites, dog])[-1]
    assert not np.allclose(a, b)
    # 4) Cộng thêm: torch sequence khớp forward_sequence
    tc = torch.nn.RNN(3, 4, batch_first=True, nonlinearity="tanh")
    with torch.no_grad():
        tc.weight_ih_l0.copy_(torch.tensor(c.W_xh, dtype=torch.float32)); tc.weight_hh_l0.copy_(torch.tensor(c.W_hh, dtype=torch.float32))
        tc.bias_ih_l0.zero_(); tc.bias_hh_l0.zero_()
        _, hn = tc(torch.tensor(np.stack([dog, bites, man]), dtype=torch.float32)[None])
    assert np.allclose(a, hn.numpy()[0, 0], atol=1e-5)
    print("all tests passed")


if __name__ == "__main__":
    _tests()
