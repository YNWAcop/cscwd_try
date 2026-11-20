"""Exact protected single-exchange retrieval and grouped ordinal allocation. Requires NumPy."""

import numpy as np

LABELS = np.array([-1.0, 0.0, 1.0])
EPS = 1e-8


def envelope(r, b, labels=LABELS, rho=0.90):
    if not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0,1]")
    r, b = np.broadcast_arrays(r, b)
    if not (
        np.all((r >= labels.min()) & (r <= labels.max()))
        and np.all((b >= labels.min()) & (b <= labels.max()))
    ):
        raise ValueError("End scores must already be on the declared score scale.")
    if rho == 0:
        return (
            np.full(r.shape, labels.min(), dtype=float),
            np.full(r.shape, labels.max(), dtype=float),
        )
    db = np.abs(b[..., None] - labels)
    dr = np.abs(r[..., None] - labels)
    helpful = db > dr
    radius = (1 - rho) * db + rho * dr
    lo = np.maximum(labels.min(), np.where(helpful, labels - radius, -np.inf).max(-1))
    hi = np.minimum(labels.max(), np.where(helpful, labels + radius, np.inf).min(-1))
    assert np.all(lo <= r + 1e-12) and np.all(hi >= r - 1e-12)
    return np.minimum(lo, r), np.maximum(hi, r)


def order(scores):
    return np.argsort(-scores, kind="stable")


def allocate_center(r, b, u, pp):
    prob = np.where(r > b, pp, 1 - pp)
    direc = np.sign(r - b)
    k = min(3, len(r))
    top = order(u)[:k]
    cap = abs(u - r).copy()
    for i in range(len(r)):
        competitors = np.arange(len(r)) if i in top else top
        dist = abs(u[i] - u[competitors[competitors != i]]).min(initial=np.inf)
        cap[i] = min(cap[i], 0.25 * dist)
    # Fill capacity in descending posterior order; one threshold item may use a partial allocation.
    t = -cap.copy()
    left = cap.sum()
    for i in order(prob):
        add = min(2 * cap[i], left)
        t[i] += add
        left -= add
        if left <= 0:
            break
    s = u + direc * t
    assert abs(t.sum()) < 1e-10 and np.array_equal(order(s)[:k], top)
    return s


def movement(reference, lo, hi, indices):
    k = len(indices)
    inside = np.zeros(len(lo), bool)
    inside[list(indices)] = True
    left = lo[~inside].max(initial=-np.inf)
    right = hi[inside].min() - EPS
    if left > right:
        return None
    adjusted = reference - EPS * inside
    t = np.clip(np.partition(adjusted, len(lo) - k)[len(lo) - k], left, right)
    s = np.where(inside, np.maximum(reference, t + EPS), np.minimum(reference, t))
    assert np.all((s >= lo - 1e-12) & (s <= hi + 1e-12)) and set(order(s)[:k]) == set(indices)
    return s


def geometry(r, b, f):
    u = r + 0.05 * (b - r)
    c = allocate_center(r, b, u, u)
    top = order(c)[:3]
    lo, hi = envelope(r, b, np.array([0, 1]))
    budget = f * abs(r - u).sum()
    swaps = []
    need = set()
    outs = np.setdiff1d(np.arange(len(r)), top)
    for i in top:
        # Entering j must overtake i; their score gap lower-bounds L1 expense.
        possible = outs[(hi[outs] >= lo[i] - 1e-12) & (c[i] - c[outs] <= budget + 1e-12)]
        for j in possible:
            idx = tuple(sorted((set(top) - {i}) | {j}))
            s = movement(c, lo, hi, idx)
            if s is None:
                continue
            cost = abs(s - c).sum()
            if cost <= budget + 1e-12:
                need.update((i, j))
                swaps.append((i, j, cost, s))
    return c, swaps, np.array(sorted(need), dtype=int)


def infer(r, b, f, score_subset):
    c, swaps, ids = geometry(r, b, f)
    if not swaps:
        return c, ids
    values = dict(zip(ids, score_subset(ids)))
    best = (0.0, 0.0)
    out = c
    for i, j, cost, s in swaps:
        key = (values[j] - values[i], -cost)
        if key > best:
            best = key
            out = s
    return out, ids


def allocate_sentiment(r, b, probability, groups, labels=LABELS):
    r, b = np.asarray(r, dtype=float), np.asarray(b, dtype=float)
    groups = np.asarray(groups)
    if r.ndim != 2 or r.shape != b.shape or r.shape[1] != len(groups):
        raise ValueError("Use [views, utterances] endpoints and one source ID per utterance.")
    u = r + 0.05 * (b - r)
    gain = abs(b[..., None] - labels) > abs(r[..., None] - labels)
    p = (gain * probability).sum(-1)
    gap = ~np.any(
        (labels > np.minimum(r, b)[..., None]) & (labels < np.maximum(r, b)[..., None]), axis=-1
    )
    cap = 0.05 * abs(r - b) * gap
    moved = np.zeros_like(r)

    for g in np.unique(groups):
        for view in range(len(r)):
            s = np.flatnonzero((groups == g) & (cap[view] > 0))
            if len(s) < 2:
                continue
            s = s[np.argsort(p[view, s], kind="stable")]
            weights = cap[view, s]
            # Split total capacity at its weighted median. Exact ties share a rate.
            ranks, ix = np.unique(p[view, s], return_inverse=True)
            caps = np.bincount(ix, weights=weights)
            negative = np.clip(weights.sum() / 2 - (caps.cumsum() - caps), 0, caps)
            moved[view, s] = weights * (1 - 2 * (negative / caps)[ix])
            assert abs(moved[view, s].sum()) < 1e-11
    return u + moved * np.sign(r - b)
