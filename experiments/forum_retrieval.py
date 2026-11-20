"""First-stage corpus control and independent recovery of the design score mapping."""
from pathlib import Path
import csv
import hashlib
import json
import sys
from collections import defaultdict
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import argparse
import pyarrow.parquet as pq
from .benchmark_inputs import title_records
from .paths import OUTPUT


def dump(name, data):
    (OUTPUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")


def corpus_control(forum_dir, model_path):
    with_verifier = model_path is not None
    old = title_records()
    rows = []
    if with_verifier:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from controller.index import indexed, realize

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        net = (
            AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            .to(device)
            .eval()
        )
        torch.set_num_threads(2)

        @torch.inference_mode()
        def verifier_scores(query, texts):
            if not texts:
                return np.array([])
            features = tokenizer(
                [query] * len(texts),
                texts,
                padding=True,
                truncation=True,
                max_length=192,
                return_tensors="pt",
            ).to(device)
            return expit(net(**features).logits.flatten().float().cpu().numpy().astype(float))

    for folder in sorted(forum_dir.iterdir()):
        if not folder.is_dir():
            continue

        def read(name):
            return pq.read_table(next((folder / name).glob("*.parquet"))).to_pylist()

        corpus = read("corpus")
        qs = {str(q["_id"]): q["text"] for q in read("queries")}
        rel = defaultdict(set)
        with (folder / "test.tsv").open(encoding="utf8") as f:
            for a in csv.DictReader(f, delimiter="\t"):
                if float(a["score"]) > 0:
                    rel[a["query-id"]].add(a["corpus-id"])
        ids = np.array([str(doc["_id"]) for doc in corpus])
        titles = [doc["title"] or "" for doc in corpus]
        docs = [title + " " + (doc["text"] or "") for title, doc in zip(titles, corpus)]
        tf = TfidfVectorizer(dtype=np.float64)
        full = tf.fit_transform(docs)
        title = tf.transform(titles)
        for q in [a for a in old if a["domain"] == folder.name]:
            uid = q["query"]
            vec = tf.transform([qs[uid]]).T
            z = dict(
                forum=folder.name,
                id=uid,
                positive=q["positive"],
                center=q["arms"]["0"]["p"],
                protected=q["arms"]["1"]["p"],
                probes=q["arms"]["1"]["pairs"],
            )
            for key, mt in [("title", title), ("body", full)]:
                score = (mt @ vec).toarray().ravel()
                score[ids == uid] = -np.inf
                ix = np.argsort(-score, kind="stable")[:24]
                hits = np.isin(ids[ix], list(rel[uid]))
                if key == "title":
                    np.testing.assert_array_equal(hits, q["y"])
                found = int(hits.sum())
                z[key] = dict(
                    hits=found,
                    recall=found / q["positive"],
                    p3=float(hits[:3].mean()),
                    ceiling=min(found, 3) / 3,
                )
            if with_verifier:
                # ix still indexes the body-generated pool; both downstream readouts stay fixed.
                b = (title[ix] @ vec).toarray().ravel().clip(0, 1)
                r = ((b + (full[ix] @ vec).toarray().ravel()) / 2).clip(0, 1)
                c, e, ctx = indexed(r, b)
                vertices = sorted({i for edge in e for i in edge})
                docs24 = [docs[k] for k in ix]
                p = verifier_scores(qs[uid], docs24)
                s = realize(c, e, ctx, p)
                sparse = realize(
                    c,
                    e,
                    ctx,
                    dict(zip(vertices, verifier_scores(qs[uid], [docs24[k] for k in vertices]))),
                )
                center = set(np.argsort(-c, kind="stable")[:3])
                new = set(np.argsort(-s, kind="stable")[:3])
                z["body_controller"] = dict(
                    center=float(hits[list(center)].mean()),
                    protected=float(hits[list(new)].mean()),
                    add=int(hits[list(new - center)].sum()),
                    lost=int(hits[list(center - new)].sum()),
                    probes=len(vertices),
                    equal=bool(np.array_equal(s, sparse)),
                    ids=ids[ix].tolist(),
                    r=r.tolist(),
                    b=b.tolist(),
                    p=p.tolist(),
                    y=hits.astype(int).tolist(),
                )
            rows.append(z)
        print(folder.name, len(rows), flush=True)

    def summary(v):
        result = {k: float(np.mean([q[k] for q in v])) for k in ("center", "protected")}
        result["probes"] = sum(q["probes"] for q in v)
        for key in ("title", "body"):
            result[key] = {
                k: float(np.mean([q[key][k] for q in v])) for k in ("recall", "p3", "ceiling")
            }
            result[key]["empty"] = sum(q[key]["hits"] == 0 for q in v)
        if with_verifier:
            result["body_controller"] = {
                k: float(np.mean([q["body_controller"][k] for q in v]))
                for k in ("center", "protected")
            }
            result["body_controller"].update(
                {
                    k: sum(q["body_controller"][k] for q in v)
                    for k in ("add", "lost", "probes", "equal")
                }
            )
        return result

    assert len(rows) == 768
    stem = "body_rerun" if with_verifier else "first_stage"
    dump(stem + "_queries.json", rows)
    dump(
        stem + ".json",
        {
            "overall": summary(rows),
            "forums": {
                f: summary([q for q in rows if q["forum"] == f])
                for f in sorted({q["forum"] for q in rows})
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild title/body first-stage controls; optionally run the frozen verifier."
    )
    parser.add_argument("forum_dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    corpus_control(args.forum_dir, args.model_dir)
