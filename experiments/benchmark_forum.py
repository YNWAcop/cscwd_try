"""Replay text-input latency, including candidate generation, on each forum."""
import json, time
from collections import defaultdict
import numpy as np
import torch
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits
import argparse
from pathlib import Path
import pyarrow.parquet as pq
from controller.index import indexed, realize
from controller.adaptive import requests
from .benchmark_inputs import CrossEncoder, title_records
from .paths import OUTPUT


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forum_dir", type=Path)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA device is needed for the timed protocol.")
    model = CrossEncoder(args.model_dir)
    records = title_records()
    out = []
    for domain in sorted({r["domain"] for r in records}):
        folder = args.forum_dir / domain
        read = lambda name: pq.read_table(next((folder / name).glob("*.parquet"))).to_pylist()
        cor = read("corpus")
        titles = [c["title"] or "" for c in cor]
        docs = [t + " " + (c["text"] or "") for t, c in zip(titles, cor)]
        qbank = {str(q["_id"]): q["text"] for q in read("queries")}
        ids = np.array([str(c["_id"]) for c in cor])
        tf = TfidfVectorizer(dtype=np.float64).fit(docs)
        mt = tf.transform(titles)
        mf = tf.transform(docs)

        def one(row, mode):
            q = row["query"]
            text = qbank[q]
            t = time.perf_counter()
            vec = tf.transform([text]).T
            sim = (mt @ vec).toarray().ravel()
            sim[ids == q] = -np.inf
            i = np.argsort(-sim, kind="stable")[:24]
            b = sim[i].clip(0, 1)
            r = ((b + (mf[i] @ vec).toarray().ravel()) / 2).clip(0, 1)
            a = time.perf_counter()
            c, e, ctx = indexed(r, b)
            pr = {}
            tokens = gpu = 0.0
            count = 0
            pending = (
                range(24)
                if mode == "dense"
                else sorted({j for edge in e for j in edge})
                if mode == "batch"
                else requests(e, pr, 16, c)
            )
            while len(pending):
                start = time.perf_counter()
                enc = model.tok(
                    [text] * len(pending),
                    [docs[i[j]] for j in pending],
                    padding=True,
                    truncation=True,
                    max_length=192,
                    return_tensors="pt",
                ).to(model.device)
                torch.cuda.synchronize()
                tokens += time.perf_counter() - start
                start = time.perf_counter()
                p = expit(model.net(**enc).logits.flatten().float().cpu().numpy().astype(float))
                gpu += time.perf_counter() - start
                pr.update(zip(pending, p))
                count += len(pending)
                pending = requests(e, pr, 16, c) if mode == "adaptive" else []
            s = realize(c, [(j, k) for j, k in e if j in pr and k in pr], ctx, pr)
            end = time.perf_counter()
            p0 = np.array(row["p"])
            match = np.array_equal(s, realize(c, e, ctx, p0))
            return [end - t, a - t, end - a - tokens - gpu, tokens, gpu, count, int(match)]

        rs = [r for r in records if r["domain"] == domain][:16]
        one(rs[0], "dense")
        for rep in range(3):
            for h, r in enumerate(rs):
                methods = ["batch", "adaptive", "dense"]
                k = (h + rep) % 3
                for mode in methods[k:] + methods[:k]:
                    out.append([domain, r["query"], rep, mode, one(r, mode)])
        print(domain, flush=True)
    (OUTPUT / "forum_latency.json").write_text(json.dumps(out))


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
