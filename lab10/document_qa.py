"""LAB 10 — Mini Document QA: user question + tài liệu dài -> sliding window -> QA model -> answer span.

Context được cung cấp trực tiếp (không có retrieval, khác RAG ở LAB 11–12).
"""
import torch
from transformers import AutoTokenizer
from qa_model import MODEL_NAME, QAModel, prepare_eval_features, postprocess, squad_metrics, best_over_golds, exact_match, f1_score

# Tài liệu tự viết (tóm tắt thông tin các lab của môn học), tiếng Anh vì model được fine-tune trên SQuAD tiếng Anh.
COURSE_FAQ = (
    "Natural Language Processing and Applications is taught in the first semester of 2026 at VNU University of Science. "
    "The lecturer and teaching assistant is Pham Ngoc Hai. The course is organised as a sequence of weekly labs, and each lab "
    "has a written assignment, a prediction section that students fill in before running any experiment, and a reflection. "
    "LAB 05 compares RNN, LSTM and GRU models for sentiment classification on the IMDb movie review dataset. "
    "LAB 06 introduces the attention mechanism, and its deadline was 23:59 on 28 October 2026. "
    "LAB 07 builds a Transformer encoder block from scratch, including multi-head attention and sinusoidal positional encoding; "
    "it was due on 4 November 2026. "
    "LAB 08 fine-tunes a pretrained Transformer such as BERT for sentiment classification and compares freezing the encoder with "
    "full fine-tuning. LAB 09 moves from sequence classification to token classification: students train a named entity "
    "recognition model on the CoNLL-2003 dataset, whose entity types are person, organization, location and miscellaneous. "
    "LAB 10 is about extractive question answering on the Stanford Question Answering Dataset, known as SQuAD. In this lab the "
    "model predicts a start position and an end position, and the answer is the span of the context between them. "
    "Answers are evaluated with Exact Match and token-level F1. The deadline for LAB 10 is 23:59 on 25 November 2026. "
    "In the LAB 10 rubric, tokenization and span alignment are worth 15 points, the QA model implementation is worth 15 points, "
    "and the document QA application is worth 5 points. The reflection must be between 300 and 500 words. "
    "After LAB 10, LAB 11 studies semantic retrieval, because in real systems the context that contains the answer is not known "
    "in advance. LAB 12 then combines a retriever with a generator to build retrieval-augmented generation, usually called RAG. "
    "Students may use AI tools for documentation, debugging and boilerplate code, but they must explain the character and token "
    "alignment, the metric calculation and the error analysis in their own words."
)

FAQ_EVAL = [
    ("Who is the lecturer of the course?", ["Pham Ngoc Hai"]),
    ("At which university is the course taught?", ["VNU University of Science"]),
    ("Which dataset does LAB 05 use?", ["IMDb movie review dataset", "IMDb"]),
    ("When was the deadline of LAB 06?", ["23:59 on 28 October 2026", "28 October 2026"]),
    ("What kind of positional encoding is used in LAB 07?", ["sinusoidal positional encoding", "sinusoidal"]),
    ("When was LAB 07 due?", ["4 November 2026"]),
    ("What does LAB 08 compare full fine-tuning with?", ["freezing the encoder"]),
    ("Which dataset is used for named entity recognition?", ["CoNLL-2003 dataset", "CoNLL-2003"]),
    ("What are the entity types of CoNLL-2003?", ["person, organization, location and miscellaneous"]),
    ("What is SQuAD?", ["Stanford Question Answering Dataset"]),
    ("Which metrics are used to evaluate answers?", ["Exact Match and token-level F1"]),
    ("What is the deadline for LAB 10?", ["23:59 on 25 November 2026", "25 November 2026"]),
    ("How many points is the document QA application worth?", ["5 points", "5"]),
    ("What does LAB 11 study?", ["semantic retrieval"]),
    ("What does LAB 12 build?", ["retrieval-augmented generation", "RAG"]),
]


class DocumentQA:
    def __init__(self, weights, model_name=MODEL_NAME, device="cpu", max_length=384, stride=128):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = QAModel(model_name)
        self.model.load_state_dict({k: v.float() for k, v in torch.load(weights, map_location="cpu").items()})
        self.model.to(device).eval(); self.device, self.L, self.stride = device, max_length, stride

    @torch.no_grad()
    def answer_many(self, questions, document):
        feats = prepare_eval_features(self.tok, questions, [document] * len(questions), self.L, self.stride)
        S, E = [], []
        for k in range(len(feats["input_ids"])):
            ids = torch.tensor([feats["input_ids"][k]], device=self.device)
            s, e = self.model(ids, torch.ones_like(ids))
            S.append(s[0].float().cpu().numpy()); E.append(e[0].float().cpu().numpy())
        best = postprocess([document] * len(questions), feats, S, E)
        n_windows = len(feats["input_ids"]) // max(1, len(questions))
        return [best[i] for i in range(len(questions))], n_windows

    def answer(self, question, document):
        return self.answer_many([question], document)[0][0]

    def evaluate(self, document, qa_pairs):
        preds, n_windows = self.answer_many([q for q, _ in qa_pairs], document)
        rows = [dict(question=q, gold=g[0], prediction=p["text"], EM=best_over_golds(exact_match, p["text"], g),
                     F1=round(best_over_golds(f1_score, p["text"], g), 3), window=p["feature"] % max(1, n_windows)) for (q, g), p in zip(qa_pairs, preds)]
        m = squad_metrics({str(k): p["text"] for k, p in enumerate(preds)}, {str(k): g for k, (_, g) in enumerate(qa_pairs)})
        return dict(rows=rows, exact_match=m["exact_match"], f1=m["f1"], n_windows=n_windows, doc_tokens=len(self.tok(document)["input_ids"]))


def _tests():
    assert len(FAQ_EVAL) == 15
    assert all(any(g in COURSE_FAQ for g in golds) for _, golds in FAQ_EVAL), "mọi gold answer phải nằm nguyên văn trong tài liệu"
    print("document_qa tests passed")


if __name__ == "__main__":
    _tests()
