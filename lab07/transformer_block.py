"""LAB 07 — Transformer Encoder block tối giản, tự ghép từ Linear / softmax / LayerNorm / Dropout.

Không dùng nn.TransformerEncoderLayer hay nn.MultiheadAttention trong phần core (chỉ dùng trong _tests
để đối chiếu kết quả).

Forward (post-norm, như bài báo gốc):
    x -> Multi-Head Self-Attention -> Add & Norm -> Feed-Forward -> Add & Norm -> output
"""
import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.d_model, self.h, self.d_k = d_model, n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)          # gộp W_i^Q của mọi head thành một ma trận (d_model, h·d_k)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)          # W^O sau khi concat các head
        self.drop = nn.Dropout(dropout)

    def split_heads(self, x):                           # (B, T, d_model) -> (B, h, T, d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, x, pad_mask=None, shapes=None):
        """x: (B, T, d_model); pad_mask: (B, T) True = PAD. Trả về (output (B, T, d_model), weights (B, h, T, T))."""
        B, T, _ = x.shape
        Q, K, V = self.split_heads(self.W_q(x)), self.split_heads(self.W_k(x)), self.split_heads(self.W_v(x))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)            # (B, h, T, T)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        A = torch.softmax(scores, dim=-1)
        heads = self.drop(A) @ V                                           # (B, h, T, d_k)
        concat = heads.transpose(1, 2).reshape(B, T, self.d_model)         # (B, T, h·d_k) = Concat(head_1..head_h)
        out = self.W_o(concat)
        if shapes is not None:
            shapes.update({"X": x.shape, "Q": Q.shape, "K": K.shape, "V": V.shape, "Attention scores": scores.shape,
                           "Attention weights": A.shape, "Heads output": heads.shape, "Concat": concat.shape,
                           "Attention output (sau W^O)": out.shape})
        return out, A


class FeedForward(nn.Module):
    """FFN(x) = W2 · ReLU(W1 x + b1) + b2, áp dụng độc lập tại từng vị trí."""
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.lin1, self.lin2, self.drop = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model), nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.drop(torch.relu(self.lin1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, pad_mask=None, shapes=None):
        a, A = self.attn(x, pad_mask, shapes)
        x = self.norm1(x + self.drop1(a))               # residual + LayerNorm
        f = self.ffn(x)
        if shapes is not None: shapes["FFN hidden"] = self.ffn.lin1(x).shape; shapes["FFN output"] = f.shape
        x = self.norm2(x + self.drop2(f))               # residual + LayerNorm
        if shapes is not None: shapes["Output"] = x.shape
        return x, A


def sinusoidal_pe_torch(T, d):
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    div = torch.pow(10000.0, torch.arange(0, d, 2, dtype=torch.float32) / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos / div), torch.cos(pos / div)
    return pe


class TransformerClassifier(nn.Module):
    """Embedding (+ positional encoding) -> N Transformer block -> pooling (mean hoặc CLS) -> linear."""
    pe: torch.Tensor

    def __init__(self, vocab, d_model=128, n_heads=4, d_ff=256, n_layers=2, max_len=512, dropout=0.1,
                 use_pe=True, pooling="mean"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.register_buffer("pe", sinusoidal_pe_torch(max_len, d_model), persistent=False)
        self.blocks = nn.ModuleList(TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.use_pe, self.pooling, self.scale = use_pe, pooling, math.sqrt(d_model)
        self.drop, self.fc = nn.Dropout(dropout), nn.Linear(d_model, 1)

    def forward(self, x, lens=None, return_attn=False):
        pad = x == 0
        h = self.emb(x) * self.scale
        if self.use_pe: h = h + self.pe[: x.size(1)][None]
        h = self.drop(h); attns = []
        for blk in self.blocks:
            h, A = blk(h, pad); attns.append(A)
        if self.pooling == "cls":
            z = h[:, 0]                                                   # vector của token [CLS] ở vị trí 0
        else:
            keep = (~pad).unsqueeze(-1).float()
            z = (h * keep).sum(1) / keep.sum(1).clamp(min=1)              # mean pooling, bỏ PAD
        logit = self.fc(self.drop(z)).squeeze(-1)
        return (logit, attns, h) if return_attn else logit


def _tests():
    torch.manual_seed(0)
    B, T, d, h = 8, 100, 128, 4
    blk = TransformerBlock(d, h, 512, dropout=0.0).eval()
    shapes = {}
    y, A = blk(torch.randn(B, T, d), shapes=shapes)
    assert shapes["Q"] == (8, 4, 100, 32) and shapes["Attention scores"] == (8, 4, 100, 100) and y.shape == (8, 100, 128)
    assert torch.allclose(A.sum(-1), torch.ones(1))
    # Khớp nn.TransformerEncoderLayer (post-norm, ReLU) khi chép cùng trọng số
    ref = nn.TransformerEncoderLayer(d, h, 512, dropout=0.0, batch_first=True).eval()
    with torch.no_grad():
        ref.self_attn.in_proj_weight.copy_(torch.cat([blk.attn.W_q.weight, blk.attn.W_k.weight, blk.attn.W_v.weight]))
        ref.self_attn.in_proj_bias.copy_(torch.cat([blk.attn.W_q.bias, blk.attn.W_k.bias, blk.attn.W_v.bias]))
        ref.self_attn.out_proj.weight.copy_(blk.attn.W_o.weight); ref.self_attn.out_proj.bias.copy_(blk.attn.W_o.bias)
        ref.linear1.load_state_dict(blk.ffn.lin1.state_dict()); ref.linear2.load_state_dict(blk.ffn.lin2.state_dict())
        ref.norm1.load_state_dict(blk.norm1.state_dict()); ref.norm2.load_state_dict(blk.norm2.state_dict())
        x = torch.randn(2, 7, d); pad = torch.zeros(2, 7, dtype=torch.bool); pad[1, 5:] = True
        ours = blk(x, pad)[0]; theirs = ref(x, src_key_padding_mask=pad)
        assert torch.allclose(ours[0], theirs[0], atol=1e-5) and torch.allclose(ours[1, :5], theirs[1, :5], atol=1e-5)
        # Không positional encoding: hoán vị input -> output hoán vị theo (permutation equivariant)
        perm = torch.tensor([2, 0, 6, 1, 3, 5, 4]); z = blk(x[:1])[0]
        assert torch.allclose(blk(x[:1, perm])[0], z[:, perm], atol=1e-5)
    print("transformer block tests passed (shape, khớp nn.TransformerEncoderLayer, permutation equivariance)")


if __name__ == "__main__":
    _tests()
