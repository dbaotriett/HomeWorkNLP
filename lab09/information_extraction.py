"""LAB 09 — Token classification model và pipeline Information Extraction.

Raw text -> tách word -> tokenizer(is_split_into_words=True) -> pretrained encoder -> logits mỗi token
-> nhãn của subword đầu tiên mỗi word -> BIO spans -> {"PERSON": [...], "ORGANIZATION": [...], ...}
"""
import re
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from ner_utils import LABELS, get_entities, word_predictions, prf

MODEL_NAME = "bert-base-cased"
TYPE_NAMES = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION", "MISC": "MISC"}


class TokenClassifier(nn.Module):
    """Pretrained encoder + head Z = HW + b áp dụng cho **mọi** token (không chỉ [CLS])."""
    def __init__(self, model_name=MODEL_NAME, num_labels=len(LABELS), dropout=0.1, pretrained=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name) if pretrained else AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   # (B, T, d)
        return self.head(self.drop(H))                                                          # (B, T, C)


def split_words(text):
    """Tách word giống kiểu CoNLL: từ (có thể chứa - ' .) và dấu câu riêng."""
    return re.findall(r"(?:[A-Z]\.){2,}|\w+(?:[-'&]\w+)*|[^\w\s]", text)


class NERExtractor:
    def __init__(self, weights=None, model_name=MODEL_NAME, device="cpu", model=None, tokenizer=None):
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert weights is not None, "cần đường dẫn trọng số hoặc model"
            model = TokenClassifier(model_name)
            state = torch.load(weights, map_location="cpu")
            model.load_state_dict({k: v.float() for k, v in state.items()})
        self.model = model.to(device).eval(); self.device = device

    @torch.no_grad()
    def predict_words(self, words, max_length=256):
        enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=max_length, return_tensors="pt")
        logits = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))[0]
        probs = torch.softmax(logits.float(), -1).cpu()
        tags = word_predictions(enc.word_ids(), probs.argmax(-1).tolist())
        return tags + ["O"] * (len(words) - len(tags))          # word bị truncate -> O

    def extract(self, text):
        words = split_words(text); tags = self.predict_words(words)
        out = {v: [] for v in TYPE_NAMES.values()}
        for typ, s, e in get_entities(tags):
            span = " ".join(words[s:e])
            if span not in out[TYPE_NAMES[typ]]: out[TYPE_NAMES[typ]].append(span)
        return {k: v for k, v in out.items() if v}


# Mini IE task (Option A — News): 30 câu tự viết (không lấy từ CoNLL-2003), gold do mình gán nhãn theo
# quy ước CoNLL (quốc tịch, sự kiện, giải đấu = MISC).
IE_EVAL = [
    ("Barack Obama visited Paris on Monday.", [("PERSON", "Barack Obama"), ("LOCATION", "Paris")]),
    ("Elon Musk said Tesla will open a new factory in Berlin.", [("PERSON", "Elon Musk"), ("ORGANIZATION", "Tesla"), ("LOCATION", "Berlin")]),
    ("Apple and Microsoft reported strong quarterly earnings.", [("ORGANIZATION", "Apple"), ("ORGANIZATION", "Microsoft")]),
    ("The United Nations held an emergency meeting in Geneva.", [("ORGANIZATION", "United Nations"), ("LOCATION", "Geneva")]),
    ("Angela Merkel met Emmanuel Macron in Brussels to discuss the European Union budget.",
     [("PERSON", "Angela Merkel"), ("PERSON", "Emmanuel Macron"), ("LOCATION", "Brussels"), ("ORGANIZATION", "European Union")]),
    ("Real Madrid beat Barcelona 2-1 in the Spanish league.", [("ORGANIZATION", "Real Madrid"), ("ORGANIZATION", "Barcelona"), ("MISC", "Spanish")]),
    ("Japanese automaker Toyota recalled 500,000 vehicles in the United States.", [("MISC", "Japanese"), ("ORGANIZATION", "Toyota"), ("LOCATION", "United States")]),
    ("Serena Williams won her first match at Wimbledon.", [("PERSON", "Serena Williams"), ("MISC", "Wimbledon")]),
    ("The World Health Organization warned about a new virus outbreak in Brazil.", [("ORGANIZATION", "World Health Organization"), ("LOCATION", "Brazil")]),
    ("Google announced that Sundar Pichai will speak at a conference in Tokyo.", [("ORGANIZATION", "Google"), ("PERSON", "Sundar Pichai"), ("LOCATION", "Tokyo")]),
    ("Police in London arrested two men after the robbery.", [("LOCATION", "London")]),
    ("Amazon plans to hire 10,000 workers in India next year.", [("ORGANIZATION", "Amazon"), ("LOCATION", "India")]),
    ("Vladimir Putin and Xi Jinping signed an energy agreement in Moscow.", [("PERSON", "Vladimir Putin"), ("PERSON", "Xi Jinping"), ("LOCATION", "Moscow")]),
    ("The New York Times reported that Boeing delayed deliveries.", [("ORGANIZATION", "New York Times"), ("ORGANIZATION", "Boeing")]),
    ("Lionel Messi joined Inter Miami after leaving Paris Saint-Germain.", [("PERSON", "Lionel Messi"), ("ORGANIZATION", "Inter Miami"), ("ORGANIZATION", "Paris Saint-Germain")]),
    ("Thousands of protesters gathered in Hong Kong on Sunday.", [("LOCATION", "Hong Kong")]),
    ("Samsung Electronics unveiled a new smartphone in Seoul.", [("ORGANIZATION", "Samsung Electronics"), ("LOCATION", "Seoul")]),
    ("NASA said the Artemis mission will return astronauts to the Moon.", [("ORGANIZATION", "NASA"), ("MISC", "Artemis"), ("LOCATION", "Moon")]),
    ("German Chancellor Olaf Scholz arrived in Washington for talks with Joe Biden.",
     [("MISC", "German"), ("PERSON", "Olaf Scholz"), ("LOCATION", "Washington"), ("PERSON", "Joe Biden")]),
    ("The International Monetary Fund lowered its growth forecast for Argentina.", [("ORGANIZATION", "International Monetary Fund"), ("LOCATION", "Argentina")]),
    ("Taylor Swift will perform three concerts in Singapore.", [("PERSON", "Taylor Swift"), ("LOCATION", "Singapore")]),
    ("Shares of Nvidia rose 5 percent on Wall Street.", [("ORGANIZATION", "Nvidia"), ("LOCATION", "Wall Street")]),
    ("The Olympic Games will be held in Los Angeles in 2028.", [("MISC", "Olympic Games"), ("LOCATION", "Los Angeles")]),
    ("Ukrainian President Volodymyr Zelensky addressed the European Parliament.", [("MISC", "Ukrainian"), ("PERSON", "Volodymyr Zelensky"), ("ORGANIZATION", "European Parliament")]),
    ("Heavy rain caused flooding across Vietnam and Thailand.", [("LOCATION", "Vietnam"), ("LOCATION", "Thailand")]),
    ("Manchester United appointed Ruben Amorim as head coach.", [("ORGANIZATION", "Manchester United"), ("PERSON", "Ruben Amorim")]),
    ("OpenAI released a new model, according to Reuters.", [("ORGANIZATION", "OpenAI"), ("ORGANIZATION", "Reuters")]),
    ("Pope Francis celebrated Mass at the Vatican.", [("PERSON", "Francis"), ("LOCATION", "Vatican")]),
    ("The Federal Reserve kept interest rates unchanged, said Jerome Powell.", [("ORGANIZATION", "Federal Reserve"), ("PERSON", "Jerome Powell")]),
    ("Vietnam Airlines launched direct flights from Hanoi to Sydney.", [("ORGANIZATION", "Vietnam Airlines"), ("LOCATION", "Hanoi"), ("LOCATION", "Sydney")]),
]


def evaluate_ie(extractor, data=IE_EVAL):
    """Entity-level P/R/F1: một entity đúng khi trùng cả loại và chuỗi (per câu)."""
    tp = npred = ngold = 0; rows = []
    for text, gold in data:
        pred = {(k, s) for k, v in extractor.extract(text).items() for s in v}; G = set(gold)
        tp += len(G & pred); npred += len(pred); ngold += len(G)
        rows.append(dict(text=text, gold=sorted(G), pred=sorted(pred), missing=sorted(G - pred), spurious=sorted(pred - G)))
    p, r, f = prf(tp, npred, ngold)
    return dict(precision=p, recall=r, f1=f, tp=tp, n_pred=npred, n_gold=ngold, rows=rows)


def _tests():
    assert split_words("Barack Obama visited Paris.") == ["Barack", "Obama", "visited", "Paris", "."]
    assert split_words("Saint-Germain won 2-1, U.S. said.") == ["Saint-Germain", "won", "2-1", ",", "U.S.", "said", "."]
    assert len(IE_EVAL) == 30 and all(t in TYPE_NAMES.values() for _, g in IE_EVAL for t, _ in g)
    assert all(s in text for text, g in IE_EVAL for _, s in g)
    print("information_extraction tests passed")


if __name__ == "__main__":
    _tests()
