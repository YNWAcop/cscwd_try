import hashlib
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def vecfit(rows):
    text = [t["content"] for _, doc in rows for t in doc["meeting_transcripts"]]
    return [
        TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True, min_df=2).fit(
            text
        ),
        TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), min_df=8, max_features=24000, sublinear_tf=True
        ).fit(text),
    ]


def build(rows, vec):
    queries = []
    for name, doc in rows:
        lines = [s["content"] for s in doc["meeting_transcripts"]]
        chunks = [" ".join(lines[i : i + 8]) for i in range(0, len(lines), 8)]
        extended = [" ".join(chunks[max(0, i - 1) : i + 2]) for i in range(len(chunks))]
        size = len(chunks)
        matrices = [(v.transform(chunks), v.transform(extended), v.transform(lines)) for v in vec]
        for i, query in enumerate(doc["specific_query_list"]):
            if not query.get("relevant_text_span"):
                continue
            hit = np.zeros(len(lines), bool)
            for lo, hi in query["relevant_text_span"]:
                hit[int(lo) : int(hi) + 1] = True
            y = np.array([int(hit[j : j + 8].any()) for j in range(0, len(hit), 8)])
            if not y.any():
                continue
            local = []
            context = []
            for v, (l, e, t) in zip(vec, matrices):
                q = v.transform([query["query"]]).T
                x = (l @ q).toarray().ravel()
                s = (e @ q).toarray().ravel()
                per = (t @ q).toarray().ravel()
                peak = np.array([per[j : j + 8].max() for j in range(0, len(per), 8)])
                rank = lambda a: np.argsort(np.argsort(-a, kind="stable"), kind="stable") / max(
                    1, len(a) - 1
                )
                local.extend([x, peak, rank(x)])
                context.extend(
                    [s, np.maximum.reduce([peak, np.r_[0, peak[:-1]], np.r_[peak[1:], 0]]), rank(s)]
                )
            for z in (local, context):
                z.extend(
                    [
                        np.arange(size) / max(1, size - 1),
                        np.array([np.log1p(len(c.split())) for c in chunks]),
                    ]
                )
            queries.append(
                dict(
                    meeting=name, id=f"{name}_{i}", y=y, l=np.array(local).T, e=np.array(context).T
                )
            )
    return queries


class Encoder:
    def __init__(self, assets):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(str(assets / "minilm"), local_files_only=True)
        self.net = (
            AutoModel.from_pretrained(str(assets / "minilm"), local_files_only=True)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def __call__(self, txt):
        order = np.argsort([len(v) for v in txt])
        out = []
        for sub in np.array_split(order, max(1, int(np.ceil(len(order) / 32)))):
            d = self.tok(
                [txt[i] for i in sub],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            last = self.net(**d).last_hidden_state
            mask = d.attention_mask.unsqueeze(-1)
            u = (last * mask).sum(1) / mask.sum(1).clamp_min(1)
            out.append(torch.nn.functional.normalize(u, dim=1).cpu().numpy())
        return np.concatenate(out)[np.argsort(order)]


def enrich(records, docs, enc, assets):
    idx = {q["id"]: q for q in records}
    for m, doc in docs:
        turns = doc["meeting_transcripts"]
        queries = [
            (j, q["query"]) for j, q in enumerate(doc["specific_query_list"]) if f"{m}_{j}" in idx
        ]
        if not queries:
            continue
        text = [t["speaker"] + ": " + t["content"] for t in turns]
        chunks = [" ".join(text[i : i + 8]) for i in range(0, len(text), 8)]
        pairs = [" ".join(text[i : i + 2]) for i in range(0, len(text), 2)]
        cache = assets / "embeddings"
        cache.mkdir(exist_ok=True)
        f = cache / (
            m + "_" + hashlib.sha256(repr((queries, chunks)).encode()).hexdigest()[:12] + ".npz"
        )
        if not f.exists():
            np.savez_compressed(f, C=enc(chunks), P=enc(pairs), Q=enc([q for _, q in queries]))
        with np.load(f) as z:
            local = z["Q"] @ z["C"].T
            detail = z["Q"] @ z["P"].T
        for i, (j, q) in enumerate(queries):
            ar = idx[f"{m}_{j}"]
            spread = [np.sort(detail[i, k : k + 4]) for k in range(0, len(pairs), 4)]
            peak = np.array([x[-1] for x in spread])
            two = np.array([x[-2:].mean() for x in spread])
            whole = local[i]
            role = np.array(
                [
                    np.mean([t["speaker"].lower() in q.lower() for t in turns[k : k + 8]])
                    for k in range(0, len(turns), 8)
                ]
            )
            ar["cosine"] = whole
            ar["short"] = peak
            ar["lex"] = ar["l"][:, 0]
            base = ar["l"]
            ar["l"] = np.column_stack([base, whole, role])
            ar["e"] = np.column_stack([base, whole, role, peak, two, peak - whole])
