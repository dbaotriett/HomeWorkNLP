"""LAB 09 — Hàm dùng chung: BIO → entity spans, entity-level / token-level metrics, label alignment.

Entity-level: một entity chỉ được tính đúng (TP) khi **đúng cả loại lẫn ranh giới** (start, end).
"""
from collections import Counter

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]   # thứ tự của CoNLL-2003 (HF)
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
TYPES = ["PER", "ORG", "LOC", "MISC"]
IGNORE = -100            # nhãn bị bỏ qua trong CrossEntropyLoss (ignore_index mặc định của PyTorch)


def get_entities(tags):
    """Danh sách tag BIO -> danh sách (type, start, end) với end không bao gồm.

    Như conlleval/seqeval (chế độ mặc định): I-X đứng sau O hoặc sau loại khác cũng mở một entity mới.
    """
    ents, cur = [], None                       # cur = [type, start]
    for i, t in enumerate(list(tags) + ["O"]):
        prefix, typ = (t.split("-", 1) + [None])[:2] if t != "O" else ("O", None)
        if cur is not None and (prefix != "I" or typ != cur[0]):
            ents.append((cur[0], cur[1], i)); cur = None
        if prefix == "B" or (prefix == "I" and cur is None):
            cur = [typ, i]
    return ents


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def entity_metrics(gold_seqs, pred_seqs):
    """Entity-level micro P/R/F1 và theo từng loại. gold_seqs/pred_seqs: list các list tag BIO."""
    tp, npred, ngold = Counter(), Counter(), Counter()
    for g, p in zip(gold_seqs, pred_seqs):
        G, P = set(get_entities(g)), set(get_entities(p))
        for e in G: ngold[e[0]] += 1
        for e in P: npred[e[0]] += 1
        for e in G & P: tp[e[0]] += 1
    out = {"micro": dict(zip(("precision", "recall", "f1"), prf(sum(tp.values()), sum(npred.values()), sum(ngold.values())))) |
           {"tp": sum(tp.values()), "n_pred": sum(npred.values()), "n_gold": sum(ngold.values())}}
    for t in TYPES:
        out[t] = dict(zip(("precision", "recall", "f1"), prf(tp[t], npred[t], ngold[t]))) | {"tp": tp[t], "n_pred": npred[t], "n_gold": ngold[t]}
    return out


def token_metrics(gold_seqs, pred_seqs):
    """Token-level: accuracy trên mọi token và P/R/F1 trên các token không phải O (đúng khi tag trùng khớp)."""
    n = correct = tp = npred = ngold = 0
    for g, p in zip(gold_seqs, pred_seqs):
        for a, b in zip(g, p):
            n += 1; correct += a == b
            npred += b != "O"; ngold += a != "O"; tp += (a == b) and a != "O"
    pr, rc, f = prf(tp, npred, ngold)
    return {"accuracy": correct / n, "precision": pr, "recall": rc, "f1": f}


def align_labels(word_ids, word_labels, strategy="first"):
    """Căn nhãn word-level (id) sang subword token.

    word_ids: từ tokenizer(...).word_ids(), None cho [CLS]/[SEP]/[PAD].
    strategy "first" (A): subword đầu tiên của mỗi word nhận nhãn, các subword sau = IGNORE.
    strategy "all"   (B): mọi subword nhận nhãn; subword sau của B-X đổi thành I-X.
    Special tokens luôn = IGNORE.
    """
    out, prev = [], None
    for w in word_ids:
        if w is None:
            out.append(IGNORE)
        elif w != prev:
            out.append(word_labels[w])
        elif strategy == "first":
            out.append(IGNORE)
        else:
            lab = LABELS[word_labels[w]]
            out.append(LABEL2ID["I-" + lab[2:]] if lab.startswith("B-") else word_labels[w])
        prev = w
    return out


def word_predictions(word_ids, token_pred_ids):
    """Lấy dự đoán của subword đầu tiên mỗi word làm nhãn của word (dùng cho cả hai strategy khi đánh giá)."""
    out, prev = {}, None
    for w, p in zip(word_ids, token_pred_ids):
        if w is not None and w != prev:
            out[w] = LABELS[p]
        prev = w
    return [out[i] for i in range(len(out))]


def _tests():
    assert get_entities(["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]) == [("PER", 0, 2), ("LOC", 3, 5)]
    assert get_entities(["I-ORG", "I-ORG", "B-ORG", "O"]) == [("ORG", 0, 2), ("ORG", 2, 3)]
    assert get_entities(["B-PER", "I-LOC"]) == [("PER", 0, 1), ("LOC", 1, 2)]
    # Ví dụ "New York": token đúng 2/3 nhưng entity sai hoàn toàn
    gold, pred = [["B-LOC", "I-LOC", "O"]], [["B-LOC", "O", "O"]]
    assert token_metrics(gold, pred)["accuracy"] == 2 / 3
    m = entity_metrics(gold, pred)["micro"]
    assert m["tp"] == 0 and m["f1"] == 0.0
    # Alignment: "Washington playing" -> [CLS] Washington play ##ing [SEP]
    wid, lab = [None, 0, 1, 1, None], [LABEL2ID["B-LOC"], LABEL2ID["B-PER"]]
    assert align_labels(wid, lab, "first") == [IGNORE, 5, 1, IGNORE, IGNORE]
    assert align_labels(wid, lab, "all") == [IGNORE, 5, 1, 2, IGNORE]
    assert word_predictions(wid, [0, 5, 1, 2, 0]) == ["B-LOC", "B-PER"]
    try:                                       # đối chiếu với seqeval nếu có cài
        from seqeval.metrics import f1_score
        g = [["B-PER", "I-PER", "O", "B-LOC"], ["B-ORG", "O", "B-MISC", "I-MISC"]]
        p = [["B-PER", "I-PER", "O", "B-ORG"], ["B-ORG", "O", "B-MISC", "O"]]
        assert abs(f1_score(g, p) - entity_metrics(g, p)["micro"]["f1"]) < 1e-9
        print("khớp seqeval: OK")
    except ImportError:
        pass
    print("ner_utils tests passed")


if __name__ == "__main__":
    _tests()
