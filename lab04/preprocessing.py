"""LAB 04 - Preprocessing cho IMDb: load, làm sạch, bỏ trùng, chia train/val/test (60/20/20, seed 2026).

Pipeline đúng (không leakage):
    dataset -> split train/val/test -> FIT vectorizer trên TRAIN -> TRANSFORM val/test
"""
import re
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ZIP = Path(__file__).parent / "data" / "imdb.zip"
SEED = 2026


def load_imdb(zip_path=DATA_ZIP):
    """Đọc IMDB Dataset.csv trong zip -> DataFrame(review, sentiment)."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open("IMDB Dataset.csv") as f:
            return pd.read_csv(f)


def clean_text(text):
    """Bỏ thẻ HTML (<br />), chuẩn hóa khoảng trắng. KHÔNG bỏ stopword/phủ định, giữ nguyên chữ
    để vectorizer quyết định lowercase/tokenize."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def inspect(df):
    """Thống kê dataset: số mẫu, class, phân phối, độ dài, trùng lặp, thiếu dữ liệu."""
    lengths = df["review"].astype(str).str.split().str.len()
    dist = df["sentiment"].value_counts()
    return {
        "n_samples": len(df),
        "n_classes": df["sentiment"].nunique(),
        "class_counts": dist.to_dict(),
        "class_pct": (dist / len(df) * 100).round(2).to_dict(),
        "length_words": lengths.describe().round(1).to_dict(),
        "duplicate_reviews": int(df["review"].duplicated().sum()),
        "missing_text": int(df["review"].isna().sum()),
        "missing_label": int(df["sentiment"].isna().sum()),
    }


def prepare(df):
    """Làm sạch, bỏ review trùng (tránh cùng một review nằm ở cả train lẫn test), nhãn -> 0/1."""
    df = df.dropna().copy()
    df["text"] = df["review"].map(clean_text)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df[["text", "label"]]


def split(df, val_size=0.20, test_size=0.20, seed=SEED):
    """Chia stratified 60/20/20. Trả về (train, val, test) DataFrame."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    rel = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=rel, stratify=train_val["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _tests():
    assert clean_text("a<br /><br />b  c") == "a b c"
    df = pd.DataFrame({"text": [f"t{i}" for i in range(200)], "label": [0, 1] * 100})
    tr, va, te = split(df)
    assert len(tr) + len(va) + len(te) == 200
    assert not (set(tr.text) & set(te.text)) and not (set(tr.text) & set(va.text)) and not (set(va.text) & set(te.text))
    assert abs(tr.label.mean() - 0.5) < 0.02 and abs(te.label.mean() - 0.5) < 0.05
    assert tokenize("It's NOT good!") == ["it's", "not", "good"]
    print("all tests passed")


if __name__ == "__main__":
    _tests()
