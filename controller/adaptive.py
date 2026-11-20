"""Select endpoints of unresolved edges with largest potential gain."""


def requests(edges, known, width, c):
    full = [p for i, j in edges if i in known and j in known for p in [known[j] - known[i]]]
    best = max([0.0, *full])
    pending = []
    for i, j in edges:
        if i in known and j in known:
            continue
        ub = known.get(j, 1.0) - known.get(i, 0.0)
        if ub >= best and ub > 0:
            pending.append((ub, c[j] - c[i], i, j))
    chosen = set()
    for _, _, i, j in sorted(pending, reverse=True)[:width]:
        chosen.update(v for v in (i, j) if v not in known)
    return sorted(chosen)
