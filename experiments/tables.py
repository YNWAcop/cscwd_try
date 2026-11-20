"""Reconstruct the three paper tables from predictions and recorded service times."""
import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import t

from controller.index import indexed, realize
from controller.protection import order
from .paths import AUDIT, DATA, OUTPUT, RESULTS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(name, rows):
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def meeting_table():
    with np.load(AUDIT / "endpoints.npz") as archive:
        domains, values, offsets = (archive[k] for k in ("domain", "values", "offset"))
    promoted_hits = 0
    original_hits = 0
    for i in np.flatnonzero(domains == "design"):
        r, b, _, y = values[offsets[i] : offsets[i + 1]].T
        center, _, (_, _, top) = indexed(r, b)
        original_hits += y[top].sum()
        promoted_hits += sum(y[j] for j in set(top) - set(order(b)[:3]))
    queried = int(np.sum(domains == "design"))
    rows = [
        dict(
            policy="center",
            added=0,
            lost=0,
            kept=int(promoted_hits),
            precision=original_hits / (3 * queried),
            probes=0,
            violating_queries=0,
        )
    ]
    arms = read(OUTPUT / "mechanisms.json")["arms"]
    for name in (
        "free_one",
        "lock",
        "floor",
        "floor+lock",
        "budget",
        "floor+budget",
        "budget+lock",
        "floor+budget+lock",
    ):
        arm = arms[name]
        p = (original_hits + arm["add"] - arm["lost"]) / (3 * queried)
        np.testing.assert_allclose(p, arm["precision"])
        rows.append(
            dict(
                policy=name,
                added=arm["add"],
                lost=arm["lost"],
                kept=arm["kept"],
                precision=p,
                probes=arm["probes"],
                violating_queries=arm["bad_queries"],
            )
        )
    write_table("table_1_mechanisms.csv", rows)


def timing_tables():
    design = read(DATA / "timing" / "waves.json")
    rows = []
    for size, methods in design["test"].items():
        for method, trials in methods.items():
            seconds = [r["total"] for trial in trials for r in trial]
            rows.append(
                dict(
                    group_size=int(size),
                    method=method,
                    trials=len(seconds),
                    median_ms=float(1000 * np.median(seconds)),
                    p95_ms=float(1000 * np.quantile(seconds, 0.95)) if int(size) < 129 else "",
                )
            )
    write_table("table_2_timing.csv", rows)
    forum = read(DATA / "timing" / "forum_latency.json")
    latency = []
    for mode in ("dense", "batch", "adaptive"):
        secs = [x[4][0] for x in forum if x[3] == mode]
        assert len(secs) == 576
        latency.append(
            dict(
                method=mode, median_ms=np.median(secs) * 1000, p95_ms=np.quantile(secs, 0.95) * 1000
            )
        )
    write_table("forum_service_time.csv", latency)


def forum_table():
    records = read(DATA / "forum" / "body_rerun_queries.json")
    with np.load(AUDIT / "forum.npz") as archive:
        titles = archive["values"]
    assert len(records) == len(titles) == 768
    by_forum = defaultdict(list)
    for q, title_values in zip(records, titles):
        for tag, values in (
            ("title", title_values.T),
            ("body", [np.array(q["body_controller"][k]) for k in ("r", "b", "p", "y")]),
        ):
            r, b, p, y = values
            c, edges, ctx = indexed(r, b)
            vertices = sorted({v for edge in edges for v in edge})
            dense = realize(c, edges, ctx, p)
            sparse = realize(c, edges, ctx, {v: p[v] for v in vertices})
            np.testing.assert_array_equal(dense, sparse)
            base = y[order(c)[:3]].mean()
            precision = y[order(sparse)[:3]].mean()
            ref = q if tag == "title" else q["body_controller"]
            assert (
                base == ref["center"]
                and precision == ref["protected"]
                and len(vertices) == ref["probes"]
            )
        by_forum[q["forum"]].append(q)
    rows = []
    differences = []
    for name, group in sorted(by_forum.items()):
        assert len(group) == 64
        change = [q["body_controller"]["protected"] - q["body_controller"]["center"] for q in group]
        differences.append(np.mean(change))
        rows.append(
            dict(
                forum=name,
                title_recall=np.mean([q["title"]["recall"] for q in group]),
                body_recall=np.mean([q["body"]["recall"] for q in group]),
                title_net=round(sum(q["protected"] - q["center"] for q in group) * 3),
                body_net=round(sum(change) * 3),
            )
        )
    assert sum(r["title_net"] for r in rows) == 14 and sum(r["body_net"] for r in rows) == 27
    write_table("table_3_forums.csv", rows)
    radius = t.ppf(0.975, 11) * np.std(differences, ddof=1) / np.sqrt(12)
    assert round(np.mean(differences) - radius, 5) == 0.00301


def reproduce_tables():
    meeting_table()
    timing_tables()
    forum_table()
