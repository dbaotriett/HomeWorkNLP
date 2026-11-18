"""LAB 09 — Baseline NER: token features + Logistic Regression (không có Transformer).

Mỗi word được phân loại độc lập thành một trong 9 nhãn BIO dựa trên đặc trưng của chính nó và
hai word bên cạnh (cửa sổ ±1): dạng chữ thường, viết hoa, hình dạng (Xxxx, dd), tiền tố/hậu tố...
"""
import json, os, re, time
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from ner_utils import LABELS, entity_metrics, token_metrics


def shape(w):
    s = re.sub("[A-Z]", "X", w); s = re.sub("[a-z]", "x", s); s = re.sub("[0-9]", "d", s)
    return re.sub(r"(.)\1+", r"\1\1", s)                 # rút gọn: "Xxxxxx" -> "Xxx"


def word_features(w, prefix=""):
    return {f"{prefix}lower": w.lower(), f"{prefix}shape": shape(w), f"{prefix}title": w.istitle(), f"{prefix}upper": w.isupper(),
            f"{prefix}digit": any(c.isdigit() for c in w), f"{prefix}suf3": w[-3:].lower()}


def sent_features(words):
    feats = []
    for i, w in enumerate(words):
        f = {"bias": 1.0, **word_features(w), "pre2": w[:2].lower(), "pre3": w[:3].lower(), "suf2": w[-2:].lower(),
             "hyphen": "-" in w, "len": min(len(w), 10), "first": i == 0}
        f.update(word_features(words[i - 1], "-1:") if i > 0 else {"BOS": True})
        f.update(word_features(words[i + 1], "+1:") if i < len(words) - 1 else {"EOS": True})
        feats.append(f)
    return feats


class BaselineNER:
    def __init__(self, C=1.0, max_iter=1000):
        self.vec = DictVectorizer()
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, sents, tag_seqs):
        X = self.vec.fit_transform([f for s in sents for f in sent_features(s)])
        self.clf.fit(X, [t for ts in tag_seqs for t in ts])
        return self

    def predict(self, sents):
        out = []
        for s in sents:
            out.append(list(self.clf.predict(self.vec.transform(sent_features(s)))) if s else [])
        return out


def load_conll():
    from datasets import load_dataset
    ds = load_dataset("eriktks/conll2003", revision="refs/convert/parquet")
    return {sp: ([list(x) for x in ds[sp]["tokens"]], [[LABELS[i] for i in x] for x in ds[sp]["ner_tags"]]) for sp in ("train", "validation", "test")}


if __name__ == "__main__":
    D = load_conll()
    t0 = time.time(); model = BaselineNER().fit(*D["train"]); tt = time.time() - t0
    print(f"train {tt:.0f}s | số feature: {len(model.vec.vocabulary_):,}")
    res = {"train_time_s": tt, "n_features": len(model.vec.vocabulary_)}
    for sp in ("validation", "test"):
        pred = model.predict(D[sp][0]); em = entity_metrics(D[sp][1], pred); tm = token_metrics(D[sp][1], pred)
        print(f"{sp}: entity P {em['micro']['precision']:.4f} R {em['micro']['recall']:.4f} F1 {em['micro']['f1']:.4f} | token acc {tm['accuracy']:.4f}")
        res[sp] = {"pred": pred, "entity": em, "token": tm}
    os.makedirs("cache", exist_ok=True)
    json.dump(res, open("cache/baseline.json", "w"))
