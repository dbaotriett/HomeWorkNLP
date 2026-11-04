"""LAB 07 — Sinusoidal positional encoding.

PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
"""
import numpy as np


def sinusoidal_pe(T, d):
    """Ma trận positional encoding (T, d) cho các vị trí 0..T-1."""
    pos = np.arange(T)[:, None]                      # (T, 1)
    i = np.arange(0, d, 2)[None, :]                  # (1, d/2) — chỉ số chiều chẵn 2i
    angle = pos / np.power(10000.0, i / d)           # (T, d/2)
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, : d // 2])
    return pe


def plot_pe(T=64, d=128, path="positional_encoding.png"):
    import matplotlib.pyplot as plt
    pe = sinusoidal_pe(T, d)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[2, 1]))
    im = ax[0].imshow(pe, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax[0].set_xlabel("dimension"); ax[0].set_ylabel("position"); ax[0].set_title(f"Sinusoidal PE (T={T}, d={d})")
    fig.colorbar(im, ax=ax[0])
    sim = pe @ pe.T / d
    ax[1].imshow(sim, cmap="viridis"); ax[1].set_xlabel("position"); ax[1].set_ylabel("position")
    ax[1].set_title("PE(i)·PE(j) / d (gần nhau → giống nhau)")
    plt.tight_layout(); plt.savefig(path, dpi=130)
    return fig


def _tests():
    pe = sinusoidal_pe(3, 4)
    expected = np.array([[0, 1, 0, 1],
                         [np.sin(1), np.cos(1), np.sin(0.01), np.cos(0.01)],
                         [np.sin(2), np.cos(2), np.sin(0.02), np.cos(0.02)]])
    assert np.allclose(pe, expected)
    big = sinusoidal_pe(512, 128)
    assert len({tuple(r.round(6)) for r in big}) == 512          # mỗi vị trí một vector khác nhau
    assert np.all(np.abs(big) <= 1)
    print("positional encoding tests passed")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    _tests()
    print(sinusoidal_pe(3, 4))
