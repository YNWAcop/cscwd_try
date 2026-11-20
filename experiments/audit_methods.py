import numpy as np
from scipy.optimize import linprog
from itertools import combinations
from controller.protection import order, EPS, movement


def actions(c, lo, hi, B, m, prune=True):
    top = tuple(order(c)[:3])
    out = sorted(set(range(len(c))) - set(top))
    acts = []
    for d in range(1, min(m, len(top), len(out)) + 1):
        pool = [j for j in out if not prune or min(c[list(top)]) - c[j] + EPS <= B + 1e-12]
        for A in combinations(top, d):
            keep = set(top) - set(A)
            for J in combinations(pool, d):
                S = tuple(sorted(keep | set(J)))
                if prune:
                    if sum(c[i] for i in A) - sum(c[j] for j in J) + d * EPS > B + 1e-12:
                        continue
                    if any(x not in S for x in np.flatnonzero(lo > min(hi[j] for j in J) - EPS)):
                        continue
                s = movement(c, lo, hi, S)
                if s is None:
                    continue
                cost = abs(s - c).sum()
                if cost <= B + 1e-12:
                    acts.append((A, J, cost, s))
    return acts


def choose(acts, p, c, threshold=0):
    best = (threshold, 0)
    chosen = c
    for A, J, cost, s in acts:
        gain = sum(p[j] for j in J) - sum(p[i] for i in A)
        if (gain, -cost) > best and gain > threshold:
            best = (gain, -cost)
            chosen = s
    return chosen


def enumerate_all(c, lo, hi, B):
    ranking = order(c)
    top = ranking[:3]
    k = len(top)
    outs = sorted(set(range(len(c))) - set(top))
    bank = ranking[: 2 * k]
    high = order(lo)[: k + 1]
    asc = np.sort(c)
    pref = np.r_[0.0, np.cumsum(asc)]
    n = len(c)
    result = []
    nsets = 0
    for d in (1, 2):
        J = np.array(list(combinations(outs, d)), int).reshape(-1, d)
        if len(J) == 0:
            continue
        for A in combinations(top, d):
            retained = sorted(set(top) - set(A))
            S = np.column_stack([np.broadcast_to(retained, (len(J), k - d)), J]).astype(int)
            nsets += len(J)
            low = np.where(np.any(high[None, :, None] == S[:, None, :], 2), -np.inf, lo[high]).max(
                1
            )
            up = hi[S].min(1) - EPS
            adj = c[bank] - EPS * np.any(bank[None, :, None] == S[:, None, :], 2)
            v = np.clip(np.partition(adj, len(bank) - k, axis=1)[:, len(bank) - k], low, up)
            cut = np.searchsorted(asc, v, side="right")
            cost = (
                pref[-1]
                - pref[cut]
                - (n - cut) * v
                - np.maximum(c[S] - v[:, None], 0).sum(1)
                + np.maximum((v + EPS)[:, None] - c[S], 0).sum(1)
            )
            for h in np.flatnonzero(
                (low <= up) & (cost <= B + 1e-12 + 32 * (n + 1) ** 2 * np.finfo(float).eps)
            ):
                s = movement(c, lo, hi, tuple(sorted(S[h])))
                co = abs(c - s).sum()
                if co <= B + 1e-12:
                    result.append((A, tuple(J[h]), co, s))
    return result, nsets


def lp(center, low, high, T):
    n = len(center)
    rows = []
    rhs = []
    for i in T:
        for j in set(range(n)) - set(T):
            a = np.zeros(2 * n)
            a[i] = -1
            a[j] = 1
            rows.append(a)
            rhs.append(-1e-8)
    for i in range(n):
        for sign in (-1, 1):
            a = np.zeros(2 * n)
            a[i] = sign
            a[n + i] = -1
            rows.append(a)
            rhs.append(sign * center[i])
    fit = linprog(
        np.r_[np.zeros(n), np.ones(n)],
        A_ub=rows,
        b_ub=rhs,
        bounds=list(zip(low, high)) + [(0, None)] * n,
        method="highs",
        options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
    )
    assert fit.success or fit.status == 2
    return fit.fun if fit.success else np.inf
