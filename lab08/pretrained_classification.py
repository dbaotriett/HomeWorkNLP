"""LAB 08 — Pretrained Transformer + classification head tự viết.

Input text -> tokenizer -> input_ids, attention_mask -> pretrained encoder -> contextual representations
-> [CLS] representation (vị trí 0) -> Linear -> class logits -> softmax -> prediction.

Chế độ:
- fine-tuning: encoder + head đều được train;
- frozen (feature extraction): encoder đóng băng, chỉ train head;
- from scratch: cùng kiến trúc, trọng số khởi tạo ngẫu nhiên (không pretraining).
"""
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

MODEL_NAME = "distilbert-base-uncased"


class PretrainedClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=2, pretrained=True, dropout=0.1, config=None):
        super().__init__()
        if config is not None:                      # dùng cho test (config nhỏ, không cần tải model)
            self.encoder = AutoModel.from_config(config)
        elif pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:                                       # from scratch: cùng kiến trúc, trọng số ngẫu nhiên
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_labels))   # task-specific

    def cls_representation(self, input_ids, attention_mask):
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, T, hidden)
        return H[:, 0]                                                                          # [CLS]

    def forward(self, input_ids, attention_mask):
        return self.head(self.cls_representation(input_ids, attention_mask))                   # (B, num_labels)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def n_params(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def batches(tokenizer, texts, max_length, batch_size, labels=None, order=None):
    """Sinh batch đã tokenize (padding động theo câu dài nhất trong batch, truncation ở max_length)."""
    idx = np.arange(len(texts)) if order is None else order
    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size]
        enc = tokenizer([texts[k] for k in j], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        yield enc, (None if labels is None else torch.as_tensor(np.asarray(labels)[j], dtype=torch.long))


@torch.no_grad()
def predict_proba(model, tokenizer, texts, max_length, batch_size=64, device="cpu"):
    """Xác suất P(positive) = softmax(logits)[:, 1]."""
    model.eval(); out = []
    for enc, _ in batches(tokenizer, list(texts), max_length, batch_size):
        with torch.autocast(device_type=device.type if isinstance(device, torch.device) else str(device), enabled=str(device).startswith("cuda")):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def train_classifier(model, tokenizer, train_texts, train_labels, val_texts, val_labels, max_length=256,
                     epochs=2, lr=2e-5, batch_size=32, device="cpu", seed=2026, warmup=0.1, log=print):
    """AdamW + linear warmup/decay, chọn epoch có validation loss thấp nhất. Trả về dict lịch sử."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(train_texts) / batch_size)); warm = int(warmup * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lossf = nn.CrossEntropyLoss(); train_texts = list(train_texts); y_val = np.asarray(val_labels)
    if use_amp: torch.cuda.reset_peak_memory_stats()
    hist = dict(train_loss=[], val_loss=[], val_acc=[], epoch_s=[]); best = (1e9, None, 0)
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl, n = 0.0, 0
        for enc, yb in batches(tokenizer, train_texts, max_length, batch_size, train_labels, rng.permutation(len(train_texts))):
            assert yb is not None
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = lossf(model(enc["input_ids"].to(device), enc["attention_mask"].to(device)).float(), yb.to(device))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(yb); n += len(yb)
        if use_amp: torch.cuda.synchronize()
        dt = time.time() - t0
        pv = predict_proba(model, tokenizer, val_texts, max_length, device=device)
        vl = float(-np.mean(np.log(np.clip(np.where(y_val == 1, pv, 1 - pv), 1e-7, 1))))
        hist["train_loss"].append(tl / n); hist["val_loss"].append(vl); hist["val_acc"].append(float(((pv >= .5) == y_val).mean())); hist["epoch_s"].append(dt)
        log(f"ep{ep+1}: train {tl/n:.4f} val {vl:.4f} val_acc {hist['val_acc'][-1]:.4f} {dt:.0f}s")
        if vl < best[0]: best = (vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    return dict(hist=hist, best_epoch=best[2], train_time=float(sum(hist["epoch_s"])),
                peak_mem_mb=(torch.cuda.max_memory_allocated() / 1e6 if use_amp else float("nan")))


@torch.no_grad()
def extract_cls(model, tokenizer, texts, max_length, batch_size=128, device="cpu"):
    """Feature extraction: vector [CLS] từ encoder đóng băng."""
    model.eval(); out = []
    for enc, _ in batches(tokenizer, list(texts), max_length, batch_size):
        with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
            out.append(model.cls_representation(enc["input_ids"].to(device), enc["attention_mask"].to(device)).float().cpu())
    return torch.cat(out)


def _tests():
    from transformers import DistilBertConfig
    torch.manual_seed(0)
    cfg = DistilBertConfig(vocab_size=100, dim=32, hidden_dim=64, n_layers=1, n_heads=2, max_position_embeddings=64)
    m = PretrainedClassifier(config=cfg).eval()
    ids = torch.randint(1, 100, (3, 10)); mask = torch.ones(3, 10, dtype=torch.long); mask[2, 6:] = 0
    logits = m(ids, mask)
    assert logits.shape == (3, 2)
    # attention mask: thay token ở vị trí bị mask không làm đổi kết quả
    ids2 = ids.clone(); ids2[2, 6:] = 7
    assert torch.allclose(m(ids2, mask)[2], logits[2], atol=1e-5)
    total = m.n_params(); m.freeze_encoder()
    assert m.n_params(trainable_only=True) == 32 * 2 + 2 and m.n_params() == total
    print("pretrained_classification tests passed (shape, attention mask, freeze)")


if __name__ == "__main__":
    _tests()
