"""Matched query outcomes and a retrospective score-consuming workflow illustration."""
from pathlib import Path
import json
import sys
import zipfile
import numpy as np
from scipy.stats import t
from controller.index import indexed
from controller.protection import order

from .paths import AUDIT, DESIGN, OUTPUT

read = lambda name: json.loads((DESIGN / name).read_text())


def main(transcript_zip=None):
    ix = read("query_index.json")
    with np.load(AUDIT / "endpoints.npz") as z:
        endpoints = z["values"]
        offset = z["offset"]
    with np.load(OUTPUT / "policy_scores.npz") as z:
        scores = z["values"]
        arms = list(z["arms"])
        assert np.array_equal(z["offset"], offset[: len(ix) + 1])
    stats = []
    cases = []
    crossing = 0
    flagged = 0
    for i, item in enumerate(ix):
        a, e = offset[i : i + 2]
        r, b, p, y = endpoints[a:e].T
        c, _, (lo, hi, T) = indexed(r, b)
        sl = scores[a:e, arms.index("budget+lock")]
        sf = scores[a:e, arms.index("floor+budget+lock")]
        L = order(sl)[:3]
        F = order(sf)[:3]
        promoted = set(T) - set(order(b)[:3])
        stats.append(
            dict(
                **item,
                baseline_hits=int(y[L].sum()),
                protected_hits=int(y[F].sum()),
                same_set=set(L) == set(F)
            )
        )
        # A proposed queue reuses the stored score of a locked item; the fixed cutoff is illustrative.
        for j in sorted(promoted):
            if r[j] > b[j] and sl[j] < 0.20 <= lo[j] - 1e-12 and sf[j] >= 0.20:
                crossing += 1
        flagged_indices = np.flatnonzero((sl < lo - 1e-12) | (sl > hi + 1e-12))
        flagged += int(len(flagged_indices) > 0)
        assert np.all(sf >= lo - 1e-12) and np.all(sf <= hi + 1e-12)
        for j in flagged_indices:
            helps = (r[j] > b[j]) == bool(y[j])
            cases.append(
                dict(
                    **item,
                    index=int(j),
                    y=int(y[j]),
                    helps=bool(helps),
                    common=bool(j in set(L) & set(F)),
                    locked=bool(j in promoted),
                    r=r[j],
                    b=b[j],
                    c=c[j],
                    interval=[lo[j], hi[j]],
                    base=sl[j],
                    guard=sf[j],
                    same_set=set(L) == set(F),
                    lists=[L.tolist(), F.tolist()],
                    list_scores=[sl[L].tolist(), sf[F].tolist()]
                )
            )
    d = np.array([q["protected_hits"] - q["baseline_hits"] for q in stats]) / 3
    groups = np.array([q["meeting"][:-1] for q in stats])
    leave = np.array([d[groups != g].mean() for g in np.unique(groups)])
    G = len(leave)
    radius = t.ppf(0.975, G - 1) * np.sqrt((G - 1) / G * ((leave - leave.mean()) ** 2).sum())
    out = dict(
        better=int((d > 0).sum()),
        equal=int((d == 0).sum()),
        worse=int((d < 0).sum()),
        point=d.mean(),
        ci=[d.mean() - radius, d.mean() + radius],
        same_set=sum(q["same_set"] for q in stats),
        groups=G,
        fixed_020_examples=crossing,
        export_flags=[flagged, 0],
    )
    (OUTPUT / "paired_rows.json").write_text(json.dumps(stats))
    (OUTPUT / "paired_summary.json").write_text(json.dumps(out, indent=2))
    print(out)
    if transcript_zip is not None and cases:
        selected = sorted(
            cases,
            key=lambda q: (
                -q["same_set"],
                -q["common"],
                -q["helps"],
                -q["locked"],
                q["id"],
                q["index"],
            ),
        )[0]
        with zipfile.ZipFile(transcript_zip) as z:
            f = next(
                f
                for f in z.namelist()
                if "/Product/" in f and f.endswith("/" + selected["meeting"] + ".json")
            )
            doc = json.loads(z.read(f))
        j = selected["index"]
        qnum = int(selected["id"].rsplit("_", 1)[1])
        ask = doc["specific_query_list"][qnum]
        selected.update(
            question=ask["query"],
            spans=ask["relevant_text_span"],
            turns=[
                dict(turn=t, **doc["meeting_transcripts"][t])
                for t in range(j * 8, min((j + 1) * 8, len(doc["meeting_transcripts"])))
            ],
        )
        (OUTPUT / "meeting_case.json").write_text(
            json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf8"
        )
        print(selected)
