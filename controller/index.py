"""Implicit single-exchange geometry; materialize a score vector only for contenders."""
import numpy as np
from .protection import envelope
from .protection import order
from .protection import movement, EPS


def indexed(r, b, f=1.0):
    r = np.asarray(r, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    u = r + 0.05 * (b - r)
    rank = order(u)
    top = rank[:3]
    n = len(u)
    k = len(top)
    gaps = np.abs(u[:, None] - u[top])
    gaps[top, np.arange(k)] = np.inf
    gap = gaps.min(1)
    gap[top] = np.min(
        np.where(np.arange(n)[:, None] == top, np.inf, np.abs(u[:, None] - u[top])), axis=0
    )
    cap = np.minimum(np.abs(u - r), 0.25 * gap)
    t = -cap.copy()
    left = cap.sum()
    for i in order(np.where(r > b, u, 1 - u)):
        add = min(2 * cap[i], left)
        t[i] += add
        left -= add
        if left <= 0:
            break
    c = u + np.sign(r - b) * t
    rank = order(c)
    top = rank[:k]
    outs = np.sort(rank[k:])
    lo, hi = envelope(r, b, np.array([0, 1]))
    B = f * np.abs(r - u).sum()
    edges = []
    # Removing k selected items leaves at least one of the largest k+1 lower bounds.
    highlo = order(lo)[: k + 1]
    bank = rank[: 2 * k]
    asc = np.sort(c)
    prefix = np.r_[0.0, np.cumsum(asc)]
    # A conservative absolute roundoff band for sums of at most n unit scores.
    guard = 32 * (n + 1) ** 2 * np.finfo(float).eps
    for i in top:
        js = outs[(hi[outs] >= lo[i] - 1e-12) & (c[i] - c[outs] <= B + 1e-12)]
        if not len(js):
            continue
        kept = top[top != i]
        sets = np.column_stack((np.broadcast_to(kept, (len(js), k - 1)), js))
        left = np.where(
            np.any(highlo[None, :, None] == sets[:, None, :], axis=2), -np.inf, lo[highlo]
        ).max(1)
        right = hi[sets].min(1) - EPS
        adjusted = c[bank] - EPS * np.any(bank[None, :, None] == sets[:, None, :], axis=2)
        pivot = np.partition(adjusted, len(bank) - k, axis=1)[:, len(bank) - k]
        v = np.clip(pivot, left, right)
        cut = np.searchsorted(asc, v, side="right")
        excess = prefix[-1] - prefix[cut] - (n - cut) * v
        cost = (
            excess
            - np.maximum(c[sets] - v[:, None], 0).sum(1)
            + np.maximum((v + EPS)[:, None] - c[sets], 0).sum(1)
        )
        for j, T, ok, d in zip(js, sets, left <= right, cost):
            if not ok or d > B + 1e-12 + guard:
                continue
            if abs(d - B - 1e-12) <= guard:
                s = movement(c, lo, hi, T)
                if s is None or abs(s - c).sum() > B + 1e-12:
                    continue
            edges.append((i, j))
    return c, edges, (lo, hi, top)


def realize(c, edges, context, p):
    lo, hi, top = context
    gain = 0.0
    ties = []
    for i, j in edges:
        diff = p[j] - p[i]
        if diff > gain:
            gain = diff
            ties = [(i, j)]
        elif diff == gain and gain > 0:
            ties.append((i, j))
    best = np.inf
    result = c
    for i, j in ties:
        s = movement(c, lo, hi, tuple(sorted((set(top) - {i}) | {j})))
        cost = np.abs(s - c).sum()
        if cost < best:
            best = cost
            result = s
    return result
