"""Sinh word_embedding.ipynb (rồi chạy bằng nbconvert --execute)."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("# LAB 03 — Word Representations and Embeddings\nCo-occurrence → PPMI → Word2Vec (skip-gram / CBOW) → similarity, analogy, semantic search.\nChạy từ trên xuống. Corpus: `../lab02/data/c4-train.00000-of-01024-30K.json.gz` (dùng 8,000 documents đầu của C4 cho nhanh).")

code('''
import gzip, json, re, time, random, csv
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from gensim.models import Word2Vec
from cooccurrence import (tokenize, build_vocabulary, build_cooccurrence_matrix,
                          cosine_similarity, most_similar, ppmi)
random.seed(0); np.random.seed(0)
RESULTS = []   # (experiment, config, metric, value)
def log(exp, cfg, metric, val): RESULTS.append((exp, cfg, metric, val))

docs = [json.loads(l)["text"] for l in gzip.open("../lab02/data/c4-train.00000-of-01024-30K.json.gz","rt",encoding="utf8")][:8000]
# mỗi câu là một "context unit": window không vượt qua ranh giới câu
sentences = [tokenize(s) for d in docs for s in re.split(r"[.!?\\n]+", d.lower())]
sentences = [s for s in sentences if len(s) >= 3]
print(len(docs), "docs;", len(sentences), "sentences;", sum(map(len, sentences)), "tokens")
''')

md("## Experiment 1 — Word-context matrix\nTrước hết kiểm tra trên corpus nhỏ trong đề (window = 1), rồi chạy trên C4 với window 1/2/5.")
code('''
toy = [s.split() for s in ["the cat eats fish","the cat likes milk","the dog eats meat","the dog likes fish"]]
tv, tw2i = build_vocabulary(toy)
Xt = build_cooccurrence_matrix(toy, tw2i, window=1)
vocab_order = ["cat","dog","eats","likes","fish","milk","meat"]
print(pd.DataFrame(Xt.toarray()[[tw2i[w] for w in vocab_order]][:, [tw2i[w] for w in vocab_order]],
                   index=vocab_order, columns=vocab_order))
print("cos(cat,dog) =", cosine_similarity(Xt[tw2i["cat"]], Xt[tw2i["dog"]]))
''')
code('''
vocab, w2i = build_vocabulary(sentences, min_count=5, max_size=10000)
print("vocab size:", len(vocab))
rows, mats = [], {}
for w in (1, 2, 5):
    t = time.time()
    X = build_cooccurrence_matrix(sentences, w2i, window=w); mats[w] = X
    P = ppmi(X)
    rows.append(dict(window=w, vocab=len(vocab), matrix=f"{X.shape[0]}x{X.shape[1]}", nonzero=X.nnz,
                     density=f"{X.nnz/X.shape[0]/X.shape[1]:.2%}", build_s=round(time.time()-t,1),
                     cos_doctor_physician_count=round(cosine_similarity(X[w2i['doctor']],X[w2i['physician']]),3),
                     cos_doctor_physician_ppmi=round(cosine_similarity(P[w2i['doctor']],P[w2i['physician']]),3),
                     cos_doctor_banana_ppmi=round(cosine_similarity(P[w2i['doctor']],P[w2i['banana']]),3),
                     cos_cat_dog_ppmi=round(cosine_similarity(P[w2i['cat']],P[w2i['dog']]),3)))
    for k in ("cos_doctor_physician_count","cos_doctor_physician_ppmi","cos_cat_dog_ppmi"):
        log("exp1_cooc", f"window={w}", k, rows[-1][k])
    log("exp1_cooc", f"window={w}", "nonzero", X.nnz)
pd.DataFrame(rows).set_index("window").T
''')
code('''
P5 = ppmi(mats[5])
for name, M in (("raw count, window=5", mats[5]), ("PPMI, window=5", P5)):
    print("==", name)
    for w in ("doctor","banana"):
        print(f"  {w:8s}", [(a, round(s,2)) for a, s in most_similar(w, M, vocab, 5)])
''')

md("## Experiment 2 — Word2Vec\nTraining objective: **skip-gram + negative sampling** (`sg=1, negative=5`): cực đại hoá log σ(v_c·v_w) cho cặp (target, context) thật và log σ(−v_n·v_w) cho 5 negative lấy theo phân phối tần suất^0.75.")
code('''
CFG = dict(vector_size=100, window=5, min_count=3, epochs=10, sg=1, negative=5, workers=8, seed=42)
def train(**kw):
    c = {**CFG, **kw}; t = time.time()
    m = Word2Vec(sentences, **c); m.train_time = time.time() - t
    return m
base = train()
print("corpus: C4 8k docs |", len(sentences), "sentences |", sum(map(len,sentences)), "tokens")
print("vocabulary:", len(base.wv), "| dim:", base.vector_size, "| window:", base.window,
      "| min_count: 3 | epochs: 10 | objective: skip-gram + negative sampling (neg=5)")
print("train time: %.1fs" % base.train_time)
''')
code('''
inspect_words = ["doctor","hospital","patient","disease","computer","football","banana"]
for w in inspect_words:
    top = base.wv.most_similar(w, topn=5)
    print(f"{w:9s}", [(a, round(s,2)) for a, s in top])
    log("exp2_w2v", "base(d100,w5,sg)", f"top5_{w}", " ".join(a for a,_ in top))
''')
code('''
# Evidence từ corpus: các từ gần "doctor" có thực sự xuất hiện chung context không?
def cooc_count(a, b, win=5):
    n = 0
    for s in sentences:
        if a in s and b in s:
            ia = [i for i,x in enumerate(s) if x==a]; ib = [i for i,x in enumerate(s) if x==b]
            n += any(abs(i-j) <= win for i in ia for j in ib)
    return n
for b in ["physician","nurse","hospital","patient","disease","banana"]:
    same = sum(1 for s in sentences if 'doctor' in s and b in s)
    print(f"doctor ~ {b:10s} cùng câu: {same:4d} | trong 5 token: {cooc_count('doctor', b):4d} | cos = {base.wv.similarity('doctor', b):.3f}")
''')

md("### CBOW vs Skip-gram")
code('''
cbow = train(sg=0)
for name, m in (("skip-gram", base), ("CBOW", cbow)):
    print(f"{name:9s} time={m.train_time:5.1f}s | doctor→", [a for a,_ in m.wv.most_similar('doctor', topn=5)])
    log("exp2_w2v", name, "train_time_s", round(m.train_time,1))
''')

md("## Experiment 3 — Context window (2 / 5 / 10)")
code('''
pairs3 = [("doctor","physician"),("doctor","hospital"),("cat","dog"),("doctor","nurse"),("car","road")]
wmodels = {2: train(window=2), 5: base, 10: train(window=10)}
tab = pd.DataFrame({f"window {w}": [m.wv.similarity(a,b) for a,b in pairs3] for w, m in wmodels.items()},
                   index=[f"{a}–{b}" for a,b in pairs3]).round(3)
for w, m in wmodels.items():
    for (a,b) in pairs3: log("exp3_window", f"window={w}", f"sim_{a}-{b}", round(float(m.wv.similarity(a,b)),3))
    log("exp3_window", f"window={w}", "train_time_s", round(m.train_time,1))
display(tab)
for w, m in wmodels.items():
    print(f"window {w:2d} time={m.train_time:5.1f}s doctor→", [a for a,_ in m.wv.most_similar('doctor', topn=6)])
''')

md("## Evaluation datasets (dùng chung cho Experiment 4 và mục Evaluation)\nWord-similarity: điểm 0–10 do mình tự gán theo trực giác (không phải benchmark chuẩn như WordSim-353/SimLex-999, nên chỉ là kiểm tra thô). Đo bằng Spearman ρ.")
code('''
SIM = [("doctor","physician",9.5),("car","automobile",9.5),("king","queen",8.0),("cat","dog",7.5),
       ("doctor","nurse",8.0),("doctor","hospital",7.0),("football","soccer",9.0),("computer","software",7.5),
       ("man","woman",7.5),("big","large",9.0),("happy","glad",8.5),("car","truck",8.0),
       ("doctor","disease",5.0),("money","bank",6.5),("football","stadium",6.0),("cat","banana",1.0),
       ("computer","banana",0.5),("doctor","banana",0.5),("car","football",1.0),("king","computer",0.5)]
def sim_eval(m):
    ok = [(a,b,h) for a,b,h in SIM if a in m.wv and b in m.wv]
    sims = [m.wv.similarity(a,b) for a,b,_ in ok]
    return spearmanr(sims, [h for *_,h in ok]).correlation, ok, sims
ANA = [("man","king","woman","queen"),("king","man","queen","woman"),("boy","girl","brother","sister"),
       ("paris","france","london","england"),("london","england","paris","france"),("good","better","bad","worse"),
       ("big","bigger","small","smaller"),("walk","walking","swim","swimming"),("he","his","she","her"),
       ("brother","sister","son","daughter"),("father","mother","uncle","aunt"),("tokyo","japan","beijing","china"),
       ("france","french","germany","german"),("quick","quickly","slow","slowly"),("great","greater","high","higher")]
# a : b :: c : d  -> b - a + c ≈ d
def ana_eval(m, topn=1):
    hit = n = 0
    for a,b,c,d in ANA:
        if all(x in m.wv for x in (a,b,c,d)):
            n += 1
            hit += d in [w for w,_ in m.wv.most_similar(positive=[b,c], negative=[a], topn=topn)]
    return hit / max(n,1), n
CATS = {"medical":"doctor nurse hospital patient disease surgery therapy clinic medicine cancer".split(),
        "sports":"football soccer basketball tennis baseball player coach stadium league tournament".split(),
        "food":"banana apple bread cheese chicken pizza coffee sugar rice butter".split(),
        "vehicle":"car truck bus train bicycle airplane motorcycle engine driver highway".split(),
        "tech":"computer software internet laptop server website database network keyboard programming".split()}
def downstream(m):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X = np.array([m.wv[w] for ws in CATS.values() for w in ws if w in m.wv])
    y = [c for c, ws in CATS.items() for w in ws if w in m.wv]
    return cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5).mean()
rho, *_ = sim_eval(base); acc, n = ana_eval(base)
print(f"base: Spearman={rho:.3f} | analogy top-1 {acc:.2f} (n={n}) | word-category CV acc={downstream(base):.2f}")
''')

md("## Experiment 4 — Embedding dimension (50 / 100 / 300)")
code('''
import os, tempfile
dmodels = {50: train(vector_size=50), 100: base, 300: train(vector_size=300)}
rows = []
for d, m in dmodels.items():
    rho, *_ = sim_eval(m); a1, n = ana_eval(m, 1); a5, _ = ana_eval(m, 5)
    p = os.path.join(tempfile.gettempdir(), f"w2v_{d}.kv"); m.wv.save(p)
    size = sum(os.path.getsize(os.path.join(tempfile.gettempdir(), f)) for f in os.listdir(tempfile.gettempdir()) if f.startswith(f"w2v_{d}.kv"))
    rows.append(dict(dim=d, train_s=round(m.train_time,1), size_MB=round(size/1e6,1), spearman=round(rho,3),
                     analogy_top1=round(a1,2), analogy_top5=round(a5,2), category_acc=round(downstream(m),2),
                     doctor_physician=round(float(m.wv.similarity("doctor","physician")),3)))
    for k, v in rows[-1].items():
        if k != "dim": log("exp4_dim", f"dim={d}", k, v)
display(pd.DataFrame(rows).set_index("dim"))
''')

md("## Evaluation — Word similarity (quantitative)")
code('''
rho, ok, sims = sim_eval(base)
ev = pd.DataFrame([(a,b,h,round(float(s),3)) for (a,b,h),s in zip(ok,sims)], columns=["w1","w2","human","model_cos"])
ev["rank_human"] = ev.human.rank(ascending=False); ev["rank_model"] = ev.model_cos.rank(ascending=False)
ev["rank_gap"] = (ev.rank_human - ev.rank_model).abs()
display(ev.sort_values("model_cos", ascending=False).reset_index(drop=True))
print("Spearman rho (model vs intuition) = %.3f" % rho)
log("eval_sim", "base", "spearman", round(rho,3))
print("\\nBất ngờ nhất (chênh hạng lớn):"); display(ev.sort_values("rank_gap", ascending=False).head(5)[["w1","w2","human","model_cos","rank_gap"]])
''')

md("## Evaluation — Word analogy\nĐây là *pattern trong không gian vector*, không phải bằng chứng embedding “hiểu” quan hệ như con người.")
code('''
print("king − man + woman →", [(w, round(s,3)) for w,s in base.wv.most_similar(positive=["king","woman"], negative=["man"], topn=5)])
res = []
for a,b,c,d in ANA:
    if all(x in base.wv for x in (a,b,c,d)):
        top = [w for w,_ in base.wv.most_similar(positive=[b,c], negative=[a], topn=5)]
        res.append((f"{a}:{b} :: {c}:?", d, top[0], d == top[0], d in top))
df = pd.DataFrame(res, columns=["query","expected","top1","hit@1","hit@5"]); display(df)
print("analogy hit@1 = %.2f, hit@5 = %.2f (n=%d)" % (df["hit@1"].mean(), df["hit@5"].mean(), len(df)))
log("eval_analogy", "base", "hit@1", round(df["hit@1"].mean(),2)); log("eval_analogy", "base", "hit@5", round(df["hit@5"].mean(),2))
''')
code('''
# Bài tập tính analogy (đề bài): vector giả định
king, man, woman = np.array([8,2,7]), np.array([5,1,5]), np.array([5,3,5])
v = king - man + woman
print("king − man + woman =", v, "| cos(v, king) = %.3f | cos(v, woman) = %.3f" % (cosine_similarity(v,king), cosine_similarity(v,woman)))
''')

md("## Application — Semantic search\nSo sánh với LAB 01 (TF-IDF): (a) TF-IDF thuần, (b) query expansion bằng từ gần trong Word2Vec rồi TF-IDF, (c) document = trung bình word vector có trọng số IDF.")
code('''
from sklearn.feature_extraction.text import TfidfVectorizer
rng = random.Random(1)
corpus_docs = [d for d in docs if 60 <= len(d.split()) <= 400]
pool = rng.sample(corpus_docs, min(1500, len(corpus_docs)))
tf = TfidfVectorizer(stop_words="english", min_df=2); T = tf.fit_transform(pool)
idf = dict(zip(tf.get_feature_names_out(), tf.idf_))
wv = base.wv
def doc_vec(text):
    ws = [w for w in tokenize(text) if w in wv and w in idf]
    if not ws: return np.zeros(wv.vector_size)
    v = np.average([wv[w] for w in ws], axis=0, weights=[idf[w] for w in ws]); return v / (np.linalg.norm(v) or 1)
D = np.array([doc_vec(d) for d in pool])
def expand(query, k=4):
    out = list(tokenize(query))
    for w in tokenize(query):
        if w in wv: out += [a for a,_ in wv.most_similar(w, topn=k)]
    return out
def search_tfidf(q, k=5):
    s = (T @ tf.transform([q]).T).toarray().ravel(); return np.argsort(-s)[:k], s
def search_expanded(q, k=5): return search_tfidf(" ".join(expand(q)), k)
def search_emb(q, k=5):
    s = D @ doc_vec(q); return np.argsort(-s)[:k], s
snip = lambda i: " ".join(pool[i].split()[:22]) + " …"
q = "medical treatment"
print("Query:", q, "| mở rộng:", expand(q))
for name, fn in (("TF-IDF", search_tfidf), ("TF-IDF + expansion", search_expanded), ("Embedding avg", search_emb)):
    print("\\n==", name)
    for i in fn(q)[0][:3]: print("  -", snip(i))
''')
code('''
# Đánh giá định lượng thô: tài liệu "đúng chủ đề" = chứa >= 3 từ trong bộ từ khóa y tế rộng
MED = set("medical doctor hospital patient patients disease therapy clinical treatment health surgery physician nurse medicine drug cancer symptoms diagnosis".split())
rel = np.array([len(MED & set(tokenize(d))) >= 3 for d in pool]); print("relevant docs:", rel.sum(), "/", len(pool))
for q in ["medical treatment", "doctor hospital", "cancer therapy"]:
    for name, fn in (("TF-IDF", search_tfidf), ("TF-IDF+expansion", search_expanded), ("Embedding avg", search_emb)):
        idx = fn(q, 20)[0]; p = rel[idx].mean()
        print(f"{q:18s} {name:17s} P@20 = {p:.2f}"); log("application_search", f"{q}|{name}", "P@20", round(float(p),2))
''')

md("## Error analysis & polysemy")
code('''
for a, b in [("doctor","nurse"),("doctor","hospital"),("doctor","disease"),("doctor","physician"),("car","automobile"),("cat","banana")]:
    nb_list = [w for w,_ in base.wv.most_similar(a, topn=len(base.wv))]
    print(f"{a:7s}~{b:10s} cos = {base.wv.similarity(a,b):.3f} | rank của {b} trong neighbors({a}) = {nb_list.index(b)+1}")
for w in ("physician","automobile","doctor"): print(w, "freq:", base.wv.get_vecattr(w, "count"))
print("\\nbank →", [(a, round(s,2)) for a,s in base.wv.most_similar("bank", topn=10)])
def ctx(w, k=4, seed=3):
    r = random.Random(seed); ss = [s for s in sentences if w in s]; return [" ".join(s[:18]) for s in r.sample(ss, k)]
print("\\nVí dụ câu chứa 'bank':"); [print("  -", s) for s in ctx("bank", 6)]
''')

md("## Lưu kết quả")
code('''
with open("results.csv", "w", newline="", encoding="utf8") as f:
    wr = csv.writer(f); wr.writerow(["experiment","config","metric","value"]); wr.writerows(RESULTS)
print(len(RESULTS), "rows → results.csv")
''')
nb.cells = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "word_embedding.ipynb")
