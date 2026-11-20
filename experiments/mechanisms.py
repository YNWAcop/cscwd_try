"""Recompute the matched design mechanisms and perturbation audit from frozen endpoints."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import sys
import numpy as np
from scipy.special import expit, logit
from scipy.stats import spearmanr, t
from controller.protection import order, movement, allocate_center, envelope
from controller.index import indexed, realize

from .paths import AUDIT, OUTPUT


def emit(name, data):
    (OUTPUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")


def edges(c, lo, hi, budget):
    top = order(c)[:3]
    outside = np.setdiff1d(np.arange(len(c)), top)
    actions = []
    for i in top:
        for j in outside[(hi[outside] >= lo[i] - 1e-12) & (c[i] - c[outside] <= budget + 1e-12)]:
            displayed = sorted((set(top) - {i}) | {j})
            s = movement(c, lo, hi, displayed)
            if s is not None:
                cost = float(np.abs(s - c).sum())
                if cost <= budget + 1e-12:
                    actions.append((int(i), int(j), cost, s))
    return actions


def decide(acts, p, c, threshold=0.0):
    best = (threshold, 0.0)
    winner = None
    for i, j, cost, s in acts:
        key = (p[j] - p[i], -cost)
        if key > best and key[0] > threshold:
            best, winner = key, (i, j, s)
    return (None, None, c) if winner is None else winner


def clustered(rows):
    groups = np.array([v["team"] for v in rows])
    d = np.array([(v["add"] - v["lost"]) / 3 for v in rows])
    omit = np.array([d[groups != g].mean() for g in np.unique(groups)])
    rad = t.ppf(0.975, len(omit) - 1) * np.sqrt(
        (len(omit) - 1) / len(omit) * ((omit - omit.mean()) ** 2).sum()
    )
    return [d.mean(), d.mean() - rad, d.mean() + rad]


def audit():
    z = np.load(AUDIT / "endpoints.npz")
    arms = defaultdict(list)
    all_gain = []
    selected_gain = []
    padding = []
    checks = []
    vectors = []
    offsets = [0]
    for index, (dom, meeting, a, b0) in enumerate(
        zip(z["domain"], z["group"], z["offset"][:-1], z["offset"][1:])
    ):
        if dom != "design":
            continue
        r, b, p, y = z["values"][a:b0].T
        u = r + 0.05 * (b - r)
        c, graph, (lo, hi, top) = indexed(r, b)
        B = abs(r - u).sum()
        old = set(top)
        ctx = old - set(order(b)[:3])
        row_vectors = []
        for floor, budget in [(True, True), (False, True), (True, False), (False, False)]:
            bounds = (lo, hi) if floor else (np.zeros(len(r)), np.ones(len(r)))
            acts = edges(c, *bounds, B if budget else np.inf)
            if floor and budget:
                assert [(i, j) for i, j, _, _ in acts] == graph
                protected = acts
                assert np.array_equal(decide(acts, p, c)[2], realize(c, graph, (lo, hi, top), p))
            for lock in (False, True):
                name = (
                    "+".join(
                        k for k, on in [("floor", floor), ("budget", budget), ("lock", lock)] if on
                    )
                    or "free_one"
                )
                moves = [a for a in acts if not lock or a[0] not in ctx]
                i, j, s = decide(moves, p, c)
                display = set(order(s)[:3])
                row_vectors.append(s)
                labels = np.array([0.0, 1.0])
                db = abs(b[:, None] - labels)
                gap = np.maximum(
                    0,
                    0.9 * np.maximum(0, db - abs(r[:, None] - labels))
                    - np.maximum(0, db - abs(s[:, None] - labels)),
                )
                bad = gap > 1e-12
                if floor:
                    assert not bad.any()
                if budget:
                    assert abs(s - c).sum() <= B + 1e-12
                if lock:
                    assert ctx <= display
                if i is not None:
                    assert display == (old - {i}) | {j}
                arms[name].append(
                    dict(
                        index=index,
                        team=str(meeting)[:-1],
                        out=i,
                        enter=j,
                        add=int(sum(y[list(display - old)])),
                        lost=int(sum(y[list(old - display)])),
                        hits=int(sum(y[list(display)])),
                        kept=int(sum(y[list(ctx & display)])),
                        probes=len({v for a in moves for v in a[:2]}),
                        bad=int(bad.sum()),
                        observed=int(bad[np.arange(len(r)), y.astype(int)].sum()),
                        max_gap=float(gap.max()),
                    )
                )
        vectors.append(np.column_stack(row_vectors))
        offsets.append(offsets[-1] + len(r))
        all_gain.extend((p[j] - p[i], y[j] - y[i]) for i, j, _, _ in protected)
        i, j, s = decide(protected, p, c)
        if i is not None:
            selected_gain.append((p[j] - p[i], y[j] - y[i]))
        changes = []
        for scale in (0.5, 1.0, 2.0):
            ps = p if scale == 1 else expit(scale * logit(p))
            ii, jj, ss = decide(protected, ps, c)
            changes.append([int((ii, jj) != (i, j)), int(sum(y[order(ss)[:3]]) - sum(y[top]))])
        checks.append(changes)
        fourth = np.sort(lo)[-4]
        if fourth <= 0:
            continue
        h = min(1, 4 * fourth)
        rp = np.r_[r, np.zeros(32)]
        bp = np.r_[b, np.full(32, h)]
        up = rp + 0.05 * (bp - rp)
        pp = np.r_[p, np.ones(32)]
        rebuilt = allocate_center(rp, bp, up, up)
        lp, hp = envelope(rp, bp, np.array([0, 1]))
        fixed = np.r_[c, up[len(c) :]]
        top8 = order(c)[:8]
        factor = 5.402338282289333
        local = factor * abs(r - u)[top8].sum()
        base = {"absolute": s, "global": s, "local": decide(edges(c, lo, hi, local), p, c)[2]}
        row = {
            "index": index,
            "center_change": float(np.max(abs(c - rebuilt[: len(c)]))),
            "top8_changed": set(top8) != set(order(rebuilt)[:8]),
        }
        for tag, cp in [("fixed", fixed), ("rebuilt", rebuilt)]:
            caps = {
                "global": abs(rp - up).sum(),
                "absolute": B,
                "local": factor * abs(rp - up)[order(cp)[:8]].sum(),
            }
            family = edges(cp, lp, hp, max(caps.values()))
            for key, cap in caps.items():
                displayed = set(
                    order(decide([e for e in family if e[2] <= cap + 1e-12], pp, cp)[2])[:3]
                )
                assert max(displayed) < len(r)
                row[tag + "_" + key] = int(displayed != set(order(base[key])[:3]))
        padding.append(row)
    report = {}
    for k, rows in arms.items():
        row = {
            s: sum(a[s] for a in rows) for s in ("add", "lost", "kept", "probes", "bad", "observed")
        }
        row.update(
            precision=sum(a["hits"] for a in rows) / (3 * len(rows)),
            bad_queries=sum(a["bad"] > 0 for a in rows),
            max_gap=max(a["max_gap"] for a in rows),
            ci=clustered(rows),
        )
        report[k] = row
    assert [report["floor+budget"][k] for k in ("add", "lost", "probes", "kept")] == [
        81,
        60,
        1765,
        77,
    ]
    assert len(padding) == 832 and sum(q["fixed_global"] for q in padding) == 605
    summed = {
        k: sum(int(q[k]) for q in padding)
        for k in padding[0]
        if k not in ("index", "center_change")
    }
    summed.update(
        eligible=len(padding),
        center_changed=sum(q["center_change"] > 1e-12 for q in padding),
        max_shift=max(q["center_change"] for q in padding),
    )

    def rho(v):
        return [len(v), float(spearmanr(*np.array(v).T).statistic)]

    out = dict(
        arms=report,
        padding=summed,
        all_edges=rho(all_gain),
        chosen_edges=rho(selected_gain),
        slope_diagnostics=np.sum(checks, axis=0).tolist(),
    )
    emit("mechanisms.json", out)
    emit("query_actions.json", arms)
    emit("padding_queries.json", padding)
    np.savez_compressed(
        OUTPUT / "policy_scores.npz", values=np.vstack(vectors), offset=offsets, arms=list(arms)
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    audit()
