"""Can the selected unprotected lists be realized another way under the floor?"""
from pathlib import Path
import json
import numpy as np
from controller.protection import movement
from controller.index import indexed

from .paths import AUDIT, OUTPUT, RESULTS


def check_violations():
    with np.load(AUDIT / "endpoints.npz") as archive:
        z = {k: archive[k] for k in ("values", "offset")}
    rows = json.loads((OUTPUT / "query_actions.json").read_text())
    out = {}
    for arm in ("budget", "budget+lock", "lock", "free_one"):
        counts = {
            "queries": 0,
            "floor_infeasible": 0,
            "over_budget_under_floor": 0,
            "alternative_realization": 0,
        }
        for q in rows[arm]:
            if not q["bad"]:
                continue
            counts["queries"] += 1
            idx = q["index"]
            a, b0 = z["offset"][idx : idx + 2]
            r, b, p, y = z["values"][a:b0].T
            c, _, (lo, hi, top) = indexed(r, b)
            displayed = sorted((set(top) - {q["out"]}) | {q["enter"]})
            s = movement(c, lo, hi, displayed)
            limit = abs(r - (r + 0.05 * (b - r))).sum() if "budget" in arm else np.inf
            if s is None:
                key = "floor_infeasible"
            elif abs(s - c).sum() > limit + 1e-12:
                key = "over_budget_under_floor"
            else:
                key = "alternative_realization"
            counts[key] += 1
        out[arm] = counts
    (OUTPUT / "violation_causes.json").write_text(json.dumps(out, indent=2))
    assert out == json.loads((RESULTS / "violation_causes.json").read_text())
