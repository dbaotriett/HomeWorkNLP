"""LAB 10 — Extractive QA: alignment char ↔ token, model start/end, loss, hậu xử lý, metric EM/F1, baseline.

(question, context) -> tokenizer -> pretrained encoder -> H (B, T, d) -> Linear(d, 2) -> start logits, end logits
-> chọn (i, j) hợp lệ có điểm cao nhất -> offset_mapping -> answer = context[char_start : char_end]
"""
import collections, re, string
import numpy as np
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "distilbert-base-uncased"


# ---------------------------------------------------------------- metric (theo script đánh giá chính thức của SQuAD v1.1)
def normalize_answer(s):
    """Chữ thường, bỏ dấu câu, bỏ mạo từ a/an/the, gộp khoảng trắng."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g); overlap = sum(common.values())
    if overlap == 0: return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def best_over_golds(fn, pred, golds):
    return max(fn(pred, g) for g in golds)


def squad_metrics(preds, golds):
    """preds: {id: text}, golds: {id: [text, ...]} -> EM, F1 (%) lấy max trên các gold answer."""
    em = [best_over_golds(exact_match, preds[i], golds[i]) for i in golds]
    f1 = [best_over_golds(f1_score, preds[i], golds[i]) for i in golds]
    return {"exact_match": 100 * float(np.mean(em)), "f1": 100 * float(np.mean(f1))}


# ---------------------------------------------------------------- alignment character span -> token span
def char_to_token_span(offsets, sequence_ids, ans_start, ans_end):
    """offsets: list (s, e) theo ký tự (e không bao gồm); sequence_ids: 0 = question, 1 = context, None = special.
    ans_start: ký tự đầu của answer; ans_end: ký tự ngay sau ký tự cuối (end exclusive).
    Trả về (start_token, end_token). Nếu answer không nằm trọn trong phần context của window -> (0, 0) = [CLS]."""
    ctx = [k for k, s in enumerate(sequence_ids) if s == 1]
    if not ctx or offsets[ctx[0]][0] > ans_start or offsets[ctx[-1]][1] < ans_end:
        return 0, 0                                          # answer bị truncation / nằm ngoài window
    start_tok = end_tok = None
    for k in ctx:
        s, e = offsets[k]
        if start_tok is None and s <= ans_start < e: start_tok = k          # answer bắt đầu giữa token -> vẫn lấy token chứa nó
        if s < ans_end <= e: end_tok = k                                     # answer kết thúc giữa token
    if start_tok is None:                                     # ans_start rơi vào khoảng trắng giữa hai token
        start_tok = next(k for k in ctx if offsets[k][0] >= ans_start)
    if end_tok is None:
        end_tok = max(k for k in ctx if offsets[k][1] <= ans_end)
    return start_tok, end_tok


def prepare_train_features(tokenizer, questions, contexts, answers, max_length=384, stride=128):
    """Tokenize (question, context) với sliding window; gán start/end token cho từng window."""
    enc = tokenizer([q.lstrip() for q in questions], list(contexts), truncation="only_second", max_length=max_length, stride=stride,
                    return_overflowing_tokens=True, return_offsets_mapping=True, padding=False)
    answers = list(answers); starts, ends = [], []
    for k, offsets in enumerate(enc["offset_mapping"]):
        ex = enc["overflow_to_sample_mapping"][k]; a = answers[ex]
        s_char = a["answer_start"][0]; e_char = s_char + len(a["text"][0])
        s, e = char_to_token_span(offsets, enc.sequence_ids(k), s_char, e_char)
        starts.append(s); ends.append(e)
    enc["start_positions"], enc["end_positions"] = starts, ends
    return enc


def prepare_eval_features(tokenizer, questions, contexts, max_length=384, stride=128):
    """Như trên nhưng giữ offset (đặt None cho token không thuộc context) và id ví dụ gốc của mỗi window."""
    enc = tokenizer([q.lstrip() for q in questions], list(contexts), truncation="only_second", max_length=max_length, stride=stride,
                    return_overflowing_tokens=True, return_offsets_mapping=True, padding=False)
    enc["offset_mapping"] = [[o if enc.sequence_ids(k)[j] == 1 else None for j, o in enumerate(offs)]
                             for k, offs in enumerate(enc["offset_mapping"])]
    enc["example_idx"] = list(enc["overflow_to_sample_mapping"])
    return enc


# ---------------------------------------------------------------- model + loss
class QAModel(nn.Module):
    """Pretrained encoder + hai classifier start/end dùng chung một Linear(d, 2)."""
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.qa_outputs = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   # (B, T, d)
        logits = self.qa_outputs(H)                                                           # (B, T, 2)
        start_logits, end_logits = logits.unbind(-1)
        return start_logits, end_logits                                                       # (B, T) mỗi cái


def qa_loss(start_logits, end_logits, start_pos, end_pos):
    """L = (CE(start) + CE(end)) / 2 với CE(start) = −log P_start(i_đúng)."""
    ce = nn.CrossEntropyLoss()
    return (ce(start_logits.float(), start_pos) + ce(end_logits.float(), end_pos)) / 2


# ---------------------------------------------------------------- hậu xử lý: chọn span tốt nhất cho mỗi câu hỏi
def postprocess(contexts, features, all_start, all_end, n_best=20, max_answer_len=30):
    """contexts: list context theo example; features: output của prepare_eval_features; all_start/all_end: list logits mỗi window.
    Ràng buộc: i ≤ j, j − i + 1 ≤ max_answer_len, i và j thuộc context. Điểm = start_logit + end_logit (logit thô, so sánh được
    giữa các window). Trả về {example_idx: dict(text, char_start, char_end, score, start_tok, end_tok, feature)}."""
    by_ex = collections.defaultdict(list)
    for k, ex in enumerate(features["example_idx"]): by_ex[ex].append(k)
    out = {}
    for ex, feats in by_ex.items():
        best = None
        for k in feats:
            offs, sl, el = features["offset_mapping"][k], np.asarray(all_start[k]), np.asarray(all_end[k])
            for i in np.argsort(sl)[-n_best:][::-1]:
                for j in np.argsort(el)[-n_best:][::-1]:
                    if i >= len(offs) or j >= len(offs) or offs[i] is None or offs[j] is None or j < i or j - i + 1 > max_answer_len:
                        continue
                    score = float(sl[i] + el[j])
                    if best is None or score > best["score"]:
                        best = dict(score=score, char_start=offs[i][0], char_end=offs[j][1], start_tok=int(i), end_tok=int(j), feature=k)
        if best is None: best = dict(score=-1e9, char_start=0, char_end=0, start_tok=0, end_tok=0, feature=feats[0])
        best["text"] = contexts[ex][best["char_start"]:best["char_end"]]
        out[ex] = best
    return out


# ---------------------------------------------------------------- baseline: keyword overlap -> câu ứng viên -> span ứng viên
STOP = set("a an the of in on at to for from by with and or is are was were be been being do does did what which who whom whose when "
           "where why how that this these those it its as into than then there their they he she his her i you we our your not no "
           "can could would should may might will shall has have had about after before during over under between many much".split())
WORD_RE = re.compile(r"\w+(?:[-']\w+)*")


def keyword_baseline(question, context):
    """1) Chọn câu có nhiều từ khóa của question nhất. 2) Trong câu đó, các cụm liên tiếp gồm từ KHÔNG xuất hiện trong
    question là ứng viên; ưu tiên theo loại câu hỏi (số cho when/how many, chữ hoa cho who/where), rồi theo khoảng cách
    gần nhất tới các từ khóa đã khớp. Trả về chuỗi trong context."""
    q_words = {w.lower() for w in WORD_RE.findall(question)}; q_key = q_words - STOP
    sents = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]*", context)]
    def overlap(span):
        ws = {w.lower() for w in WORD_RE.findall(context[span[0]:span[1]])}
        return len(ws & q_key)
    s0, s1 = max(sents, key=overlap) if sents else (0, len(context))
    toks = [(m.start() + s0, m.end() + s0, m.group()) for m in WORD_RE.finditer(context[s0:s1])]
    if not toks: return ""
    match_pos = [k for k, (_, _, w) in enumerate(toks) if w.lower() in q_key]
    runs, cur = [], []
    for k, (_, _, w) in enumerate(toks):
        if w.lower() in q_words: (runs.append(cur) if cur else None); cur = []
        else: cur.append(k)
    if cur: runs.append(cur)
    runs = [[k for k in r if not (toks[k][2].lower() in STOP and (k == r[0] or k == r[-1]))] or r for r in runs]
    runs = [r for r in runs if r]
    if not runs: return toks[0][2]
    ql = question.lower()
    def prefer(r):
        text = " ".join(toks[k][2] for k in r)
        if re.search(r"\b(when|year|how many|how much|how long|how old|what percentage)\b", ql): return any(c.isdigit() for c in text)
        if re.search(r"\b(who|whom|where|which)\b", ql): return any(toks[k][2][0].isupper() for k in r)
        return True
    def dist(r):
        return min((abs(k - m) for k in r for m in match_pos), default=0)
    best = max(runs, key=lambda r: (prefer(r), -dist(r), -len(r)))
    best = best[:6] if len(best) > 6 else best                 # baseline giữ span ngắn
    return context[toks[best[0]][0]:toks[best[-1]][1]]


def _tests():
    assert exact_match("Paris", "paris") == 1.0
    assert exact_match("the city of Paris", "Paris") == 0.0 and abs(f1_score("the city of Paris", "Paris") - 0.5) < 1e-9
    assert abs(f1_score("France", "Paris France") - 2 / 3) < 1e-9
    assert squad_metrics({"a": "Denver Broncos"}, {"a": ["Broncos", "Denver Broncos"]})["exact_match"] == 100.0
    # alignment: context "The capital of France is Paris." trong cặp với câu hỏi
    offs = [(0, 0), (0, 4), (5, 7), (0, 0), (0, 3), (4, 11), (12, 14), (15, 21), (22, 24), (25, 30), (30, 31), (0, 0)]
    seq = [None, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, None]
    assert char_to_token_span(offs, seq, 25, 30) == (9, 9)                      # "Paris"
    assert char_to_token_span(offs, seq, 4, 21) == (5, 7)                       # "capital of France"
    assert char_to_token_span(offs, seq, 27, 30) == (9, 9)                      # bắt đầu giữa token "Paris"
    assert char_to_token_span(offs, seq, 25, 28) == (9, 9)                      # kết thúc giữa token
    assert char_to_token_span(offs, seq, 40, 45) == (0, 0)                      # nằm ngoài window -> [CLS]
    # postprocess: chọn span hợp lệ (i ≤ j, trong context)
    feats = {"offset_mapping": [[None, (0, 3), (4, 11), (12, 14)]], "example_idx": [0]}
    r = postprocess(["The capital of"], feats, [[9.0, 0.1, 5.0, 0.2]], [[9.0, 0.3, 0.1, 4.0]])[0]
    assert r["text"] == "capital of"                                             # [CLS] (None) bị loại dù điểm cao
    assert keyword_baseline("What is the capital of France?", "Paris is big. The capital of France is Paris.") == "Paris"
    print("qa_model tests passed")


if __name__ == "__main__":
    _tests()
