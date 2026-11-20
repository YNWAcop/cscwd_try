"""Replay graph enumeration, LP checks, fixed heads and matched query outcomes."""
import hashlib
from collections import Counter, defaultdict
from itertools import combinations
import json
import numpy as np
from scipy.special import expit
from .audit_methods import actions, choose, enumerate_all, lp
from controller.index import indexed
from controller.protection import envelope
from .paired import main as paired

from .paths import AUDIT, DESIGN, OUTPUT, RESULTS, ROOT


def verify_inputs():
    manifest = ROOT / "provenance" / "replay_inputs.sha256"
    for line in manifest.read_text().splitlines():
        digest, relative_path = line.split("  ", 1)
        path = ROOT / relative_path
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input checksum mismatch: {relative_path}")


def floor_interval(r, b, labels, rho, enabled=True):
    if not 0 <= rho <= 1:
        raise ValueError("rho outside [0,1]")
    return envelope(r, b, np.asarray(labels), rho=rho if enabled else 0.0)


def local_multiplier(dev_masses):
    ratios = [total / local for total, local in dev_masses if local > 0]
    return float(np.median(ratios)) if ratios else 0.0


def stratified_lp(records):
    # Five length ranks per domain. For each distance check the nearest boundary
    # swap, an extreme swap and the cheapest affordable swap when one exists.
    blocks = defaultdict(list)
    for i, (domain, v) in enumerate(records):
        blocks[domain].append((len(v), i))
    output = []
    for domain, queries in blocks.items():
        ordered = sorted(queries)
        for k in np.linspace(0, len(ordered) - 1, 5, dtype=int):
            n, i = ordered[k]
            r, b, _, _ = records[i][1].T
            c, _, (lo, hi, T) = indexed(r, b)
            budget = 0.05 * abs(r - b).sum()
            fast = actions(c, lo, hi, budget, 2)
            cost = {tuple(sorted((set(T) - set(A)) | set(J))): v for A, J, v, _ in fast}
            outsiders = np.argsort(-c, kind="stable")[3:]
            labels = set()
            for d in (1, 2):
                if len(outsiders) < d:
                    continue
                for A, J in ((T[-d:], outsiders[:d]), (T[:d], outsiders[-d:])):
                    labels.add(tuple(sorted((set(T) - set(A)) | set(J))))
                options = [(v, S) for S, v in cost.items() if len(set(S) - set(T)) == d]
                if options:
                    labels.add(min(options)[1])
            for S in sorted(labels):
                val = lp(c, lo, hi, S)
                assert (val <= budget + 1e-12) == (S in cost)
                if S in cost:
                    assert abs(cost[S] - val) < 1e-9
                output.append(
                    dict(
                        domain=domain,
                        index=i,
                        n=n,
                        target=[int(s) for s in S],
                        affordable=S in cost,
                        lp_cost=float(val) if np.isfinite(val) else None,
                        budget=float(budget),
                    )
                )
    return output


def main():
    from .mechanisms import audit
    from .tables import reproduce_tables
    from .violations import check_violations

    verify_inputs()
    audit()
    paired()
    check_violations()
    with np.load(AUDIT / "endpoints.npz") as z:
        v = z["values"]
        off = z["offset"]
        dom = z["domain"]
        group = z["group"]
    records = [(str(d), v[a:b]) for d, a, b in zip(dom, off[:-1], off[1:])]
    with np.load(AUDIT / "forum.npz") as z:
        records.extend(("external", x) for x in z["values"])
    counts = Counter()
    lp_counts = Counter()
    checked = []
    accepted = 0
    rejected = 0
    max_error = 0.0
    for idx, (domain, row) in enumerate(records):
        r, b, p, y = row.T
        c, _, (lo, hi, T) = indexed(r, b)
        B = 0.05 * abs(r - b).sum()
        full, _ = enumerate_all(c, lo, hi, B)
        fast = actions(c, lo, hi, B, 2)
        assert [(i, j, cost) for i, j, cost, s in full] == [(i, j, cost) for i, j, cost, s in fast]
        for prob in (p, y):
            np.testing.assert_array_equal(choose(fast, prob, c), choose(full, prob, c))
        counts[domain] += 1
        if idx >= len(dom) or len(r) > 10 or accepted + rejected >= 900:
            continue
        costs = {tuple(sorted((set(T) - set(A)) | set(J))): cost for A, J, cost, _ in fast}
        n = 0
        for S in combinations(range(len(r)), min(3, len(r))):
            if not 1 <= len(set(S) - set(T)) <= 2:
                continue
            d = lp(c, lo, hi, S)
            assert (d <= B + 1e-12) == (S in costs)
            if S in costs:
                err = abs(costs[S] - d)
                assert err < 1e-9
                max_error = max(err, max_error)
                accepted += 1
            else:
                rejected += 1
            n += 1
        lp_counts[domain] += n
        checked.append(dict(index=idx, domain=domain, group=str(group[idx]), n=len(r), actions=n))
    assert sum(counts.values()) == 2761 and accepted + rejected == 924
    extra = stratified_lp(records)
    (OUTPUT / "stratified_lp.json").write_text(json.dumps(extra, indent=1))
    idx = json.loads((DESIGN / "query_index.json").read_text())
    params = json.loads((DESIGN / "endpoint_fits.json").read_text())["folds"]
    with np.load(DESIGN / "heldout_features.npz") as z:
        X = z["X"]
    worst = 0.0
    for i, record in enumerate(idx):
        a, b = off[i : i + 2]
        x = X[a:b]
        r0, b0, p, y = v[a:b].T
        readings = []
        for key, target, design in [("r", r0, x), ("b", b0, x[:, :10])]:
            head = params[record["fold"]]["parameters"][key]
            s = expit(((design - head["mean"]) / head["scale"]) @ head["coef"] + head["intercept"])
            err = np.max(abs(target - s))
            assert err < 1e-9
            worst = max(worst, err)
            readings.append(s)
        assert indexed(*readings)[1] == indexed(r0, b0)[1]
    assert local_multiplier([]) == 0 and local_multiplier([(1, 0), (6, 2)]) == 3
    dev = json.loads((DESIGN / "development_masses.json").read_text())
    assert len(dev) == 125 and local_multiplier([(g, l) for _, g, l in dev]) == 5.402338282289333
    test = np.array([0.2, 0.8])
    for enabled, rho in [(False, 0.9), (True, 0.0)]:
        low, high = floor_interval(test, test[::-1], [0, 1], rho, enabled)
        assert np.all(low == 0) and np.all(high == 1)
    report = dict(
        joint_queries=counts,
        LP_domain=lp_counts,
        LP_queries=checked,
        LP_affordable=accepted,
        LP_infeasible_or_overbudget=rejected,
        LP_max_affordable_cost_error=max_error,
        stratified_count=len(extra),
        stratified_by_domain=Counter(q["domain"] for q in extra),
        stratified_affordable=sum(q["affordable"] for q in extra),
        reconstructed_graphs=len(idx),
        feature_max_probability_error=worst,
    )
    (OUTPUT / "replay_verification.json").write_text(json.dumps(report, indent=2))
    for name in ("mechanisms.json", "paired_summary.json", "replay_verification.json"):
        current = json.loads((OUTPUT / name).read_text())
        expected = json.loads((RESULTS / name).read_text())

        def check(actual, reference):
            if isinstance(reference, dict):
                assert actual.keys() == reference.keys(), name
                for key in reference:
                    check(actual[key], reference[key])
            elif isinstance(reference, list):
                assert len(actual) == len(reference), name
                for x, y in zip(actual, reference):
                    check(x, y)
            elif isinstance(reference, float):
                np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-12)
            else:
                assert actual == reference, name

        check(current, expected)
    reproduce_tables()
    print(
        f"Verified {len(records)} lists, {accepted+rejected} short-list LP tests and {len(extra)} stratified LP tests."
    )
    print(f"Reproduced tables and diagnostics: {OUTPUT}")


if __name__ == "__main__":
    main()
