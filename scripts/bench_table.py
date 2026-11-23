"""usage: bench_table.py <episodes.jsonl> [<episodes.jsonl> ...]

The main benchmark in one table: every agent on the 3 stated and the 4 free-choice families,
seeds 2000-2039, camera right (none) and one bad camera (persistent).  Later files win when
the same (agent, condition, family, seed) appears twice.  Differences are paired with BASE
(default A9) over the episodes both have.
"""
import json
import os
import sys

import numpy as np

STATED = ["ordered_stack", "blocked_rearrange", "constrained_place"]
FREE = ["free_two_zones", "free_three_zones", "free_stack", "free_two_stacks"]
ORDER = ["naive", "oracle", "A2", "A5", "A9", "A10r-none", "A10-none", "A11-none", "A11-lessons"]
BASE = os.environ.get("BASE", "A9")

rows = {}
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        if r.get("info", "none") == "none" and 2000 <= r["seed"] < 2040 and r["family"] in STATED + FREE and not r.get("error"):
            rows.setdefault((r["agent"], r["perception"]), {})[(r["family"], r["seed"])] = r
rng = np.random.default_rng(0)


def cell(d, fams):
    rs = [r for (fam, _), r in d.items() if fam in fams]
    return f"{sum(bool(r['success']) for r in rs):>3}/{len(rs):<3}" if rs else "   -   "


def diff(d, b, fams):
    keys = sorted(k for k in d if k in b and k[0] in fams)
    if len(keys) < 20:
        return "         -         "
    x = np.array([float(bool(d[k]["success"])) - float(bool(b[k]["success"])) for k in keys])
    bt = rng.choice(x, size=(10000, len(x))).mean(axis=1)
    return f"{100 * x.mean():+5.1f} [{100 * np.quantile(bt, .025):+5.1f},{100 * np.quantile(bt, .975):+5.1f}]"


for per in ("none", "persistent"):
    print(f"\n=== camera: {'right' if per == 'none' else 'one bad camera'}   (successes / episodes; differences to {BASE}, paired)")
    print(f"{'agent':<12} " + " ".join(f"{f[:9]:>9}" for f in STATED + FREE) + f"  {'stated':>8} {'free':>8} {'all':>8}   {'stated vs ' + BASE:<20} {'free vs ' + BASE:<20} calls  tokens")
    agents = [a for a in ORDER if (a, per) in rows] + sorted(a for a, p in rows if p == per and a not in ORDER)
    for a in agents:
        d, b = rows[(a, per)], rows.get((BASE, per), {})
        calls = np.mean([r["calls"] for r in d.values()])
        toks = np.mean([r.get("total_tokens") or 0 for r in d.values()])
        print(f"{a:<12} " + " ".join(f"{cell(d, [f]):>9}" for f in STATED + FREE)
              + f"  {cell(d, STATED):>8} {cell(d, FREE):>8} {cell(d, STATED + FREE):>8}   "
              + (f"{diff(d, b, STATED):<20} {diff(d, b, FREE):<20}" if a != BASE else " " * 41) + f" {calls:5.1f} {toks:7.0f}")
