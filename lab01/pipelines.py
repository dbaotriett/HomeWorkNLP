"""Shared helpers for experiments.ipynb: corpus loading, the three preprocessing
pipelines (Part F), and TF-IDF search (Part G)."""

from __future__ import annotations

import gzip
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Callable, List

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# Override with the LAB01_CORPUS environment variable if the file is elsewhere.
DEFAULT_CORPUS = Path(os.environ.get(
    "LAB01_CORPUS", Path(__file__).parent / "data" / "c4-train.00000-of-01024-30K.json.gz"))


def load_corpus(path: str | Path = DEFAULT_CORPUS, limit: int = 30_000) -> List[str]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    docs = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            if len(docs) >= limit:
                break
    return docs


# --- Pipeline A: lowercase + whitespace tokenization -----------------------
def tokenize_a(text: str) -> List[str]:
    return text.lower().split()


# --- Pipeline B: lowercase + punctuation normalization + stopword removal ---
_ALNUM = re.compile(r"[a-z0-9]+")


def tokenize_b(text: str) -> List[str]:
    return [t for t in _ALNUM.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


# --- Pipeline C: unicode normalization + subword (WordPiece) tokenization ---
def normalize_c(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def train_subword_tokenizer(docs: List[str], vocab_size: int = 30_000):
    """Train a WordPiece tokenizer on the corpus itself (no external download)."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence(
        [normalizers.NFKD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    tok.train_from_iterator(docs, trainer=trainer)
    return tok


def make_tokenize_c(subword_tokenizer) -> Callable[[str], List[str]]:
    def tokenize_c(text: str) -> List[str]:
        return subword_tokenizer.encode(text).tokens

    return tokenize_c


def pretokenize_c(subword_tokenizer, docs: List[str]) -> List[List[str]]:
    """Batch-encode documents (much faster than encoding one by one)."""
    return [e.tokens for e in subword_tokenizer.encode_batch(docs)]


def make_vectorizer(tokenizer: Callable[[str], List[str]]) -> TfidfVectorizer:
    # lowercase / punctuation / stopwords are handled inside each tokenizer,
    # so the vectorizer itself must not preprocess anything.
    return TfidfVectorizer(tokenizer=tokenizer, lowercase=False, token_pattern=None)


# --- Search ----------------------------------------------------------------
class TfidfSearch:
    def __init__(self, vectorizer: TfidfVectorizer, X, prep: Callable = None):
        self.vectorizer = vectorizer
        # prep: applied to the raw query before vectorizer.transform
        # (Pipeline C feeds pre-tokenized subwords to the vectorizer).
        self.prep = prep or (lambda q: q)
        self.X = X  # rows are L2-normalised, so X @ q = cosine similarity
        self.vocab = vectorizer.get_feature_names_out()

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([self.prep(query)])
        return (self.X @ q.T).toarray().ravel()

    def search(self, query: str, k: int = 5):
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in top]

    def contributions(self, query: str, doc_id: int, n: int = 5):
        """Terms that contribute most to cos(q, d) = sum_t q_t * d_t."""
        q = self.vectorizer.transform([self.prep(query)]).toarray().ravel()
        d = self.X[doc_id].toarray().ravel()
        prod = q * d
        idx = [i for i in np.argsort(-prod)[:n] if prod[i] > 0]
        return [(self.vocab[i], round(float(prod[i]), 4)) for i in idx]


def preview(text: str, n: int = 90) -> str:
    return " ".join(text.split())[:n]
