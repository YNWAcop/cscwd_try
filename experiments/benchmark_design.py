"""Same indexed graph, same dynamic GPU batching, different query schedules."""
import json, time
from collections import defaultdict
import numpy as np
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from pathlib import Path
import argparse
from controller.index import indexed, realize
from controller.adaptive import requests
from .paths import OUTPUT
from .benchmark_inputs import CrossEncoder, design_timing_inputs


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qmsum_zip", type=Path)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA device is needed for the timed protocol.")
    ranker = CrossEncoder(args.model_dir)
    test, dev, scale, bias = design_timing_inputs(args.qmsum_zip)

    def once(rows, mode):
        torch.cuda.synchronize()
        start = time.perf_counter()
        groups = []
        todo = []
        tkn = net = 0.0
        n = waves = 0
        for h, (q, docs, r, b, p, uid) in enumerate(rows):
            c, e, ctx = indexed(r, b)
            known = {}
            ix = (
                range(len(r))
                if mode == "dense"
                else sorted({v for z in e for v in z})
                if mode == "batch"
                else requests(e, known, mode, c)
            )
            groups.append((c, e, ctx, known, q, docs))
            todo.extend((h, j) for j in ix)
        setup = time.perf_counter() - start
        while todo:
            waves += 1
            n += len(todo)
            for begin in range(0, len(todo), 32):
                pairs = todo[begin : begin + 32]
                t = time.perf_counter()
                x = ranker.tok(
                    [groups[h][4] for h, j in pairs],
                    [groups[h][5][j] for h, j in pairs],
                    padding=True,
                    truncation=True,
                    max_length=192,
                    return_tensors="pt",
                ).to(ranker.device)
                torch.cuda.synchronize()
                tkn += time.perf_counter() - t
                t = time.perf_counter()
                vals = expit(
                    ranker.net(**x).logits.flatten().float().cpu().numpy().astype(float) * scale
                    + bias
                )
                net += time.perf_counter() - t
                for (h, j), v in zip(pairs, vals):
                    groups[h][3][j] = v
            todo = []
            if isinstance(mode, int):
                for h, (c, e, ctx, known, q, docs) in enumerate(groups):
                    todo.extend((h, j) for j in requests(e, known, mode, c))
        out = []
        for c, e, ctx, known, _, _ in groups:
            out.append(realize(c, [(i, j) for i, j in e if i in known and j in known], ctx, known))
        elapsed = time.perf_counter() - start
        for s, row, (c, e, ctx, *_) in zip(out, rows, groups):
            np.testing.assert_array_equal(s, realize(c, e, ctx, row[4]))
        return dict(
            total=elapsed,
            setup=setup,
            tokenize=tkn,
            model=net,
            other=elapsed - setup - tkn - net,
            n=n,
            waves=waves,
        )

    once(dev[:2], "dense")
    result = {}
    cal = {}
    for size in (1, 8, 129):
        devsub = dev[:32]
        records = defaultdict(list)
        for step in range(3):
            modes = (1, 4, 16)
            modes = modes[step:] + modes[:step]
            for m in modes:
                records[m].append(
                    sum(once(devsub[i : i + size], m)["total"] for i in range(0, len(devsub), size))
                )
        pick = min(records, key=lambda m: np.median(records[m]))
        cal[size] = [pick, dict(records)]
        runs = defaultdict(list)
        for passno in range(3):
            modes = ("batch", pick, "dense")
            modes = modes[passno:] + modes[:passno]
            for m in modes:
                runs[str(m)].append(
                    [once(test[i : i + size], m) for i in range(0, len(test), size)]
                )
        result[size] = dict(runs)
        print(size, pick, flush=True)
    (OUTPUT / "waves.json").write_text(json.dumps(dict(cal=cal, test=result)))


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
