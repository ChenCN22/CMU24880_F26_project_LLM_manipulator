"""A9 on the hold-out seeds against the other agents, the scripts and its own ablations (paired by seed).

    python results/holdout_pairs.py <holdout dir> <ablation dir>
"""
import json
import os
import sys

import numpy as np

rng = np.random.default_rng(0)
H, A = sys.argv[1], sys.argv[2]
PERC = ("none", "transient", "persistent", "vlm")


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def cells(rows):
    c = {}
    for r in rows:
        c.setdefault(r["agent"], {})[(r["perception"], r["family"], r["seed"])] = r
    return c


full = cells(load(f"{H}/A9/episodes.jsonl"))["A9"]
others = {}
for a in ("A2", "A5"):
    others.update(cells(load(f"{H}/{a}/episodes.jsonl")))
others.update(cells(load(f"{H}/base/episodes.jsonl")))
for p in ("single", "prior", "stale", "doubt", "fingers", "guards"):
    others.update(cells(load(f"{A}/{p}/episodes.jsonl")))


def pd(a, b, keys, key="success"):
    d = np.array([float(a[k][key]) - float(b[k][key]) for k in keys])
    bt = rng.choice(d, size=(10000, len(d))).mean(axis=1)
    s = 100 if key == "success" else 1
    return f"{d.mean() * s:+.1f} [{np.quantile(bt, .025) * s:+.1f}, {np.quantile(bt, .975) * s:+.1f}]"


for p in PERC:
    ks = [k for k in full if k[0] == p]
    if ks:
        print(f"A9 {p:<10} {sum(full[k]['success'] for k in ks)}/{len(ks)}  calls {np.mean([full[k]['calls'] for k in ks]):.1f}")
for name, o in others.items():
    print(f"\n{name}: n={len(o)}")
    for p in PERC + ("text",):
        ks = sorted(k for k in set(full) & set(o) if k[0] == p or (p == "text" and k[0] != "vlm"))
        if not ks:
            continue
        print(f"  {p:<10} n={len(ks):<4} other {sum(o[k]['success'] for k in ks):>3}  A9 {sum(full[k]['success'] for k in ks):>3}"
              f"  A9-other {pd(full, o, ks)}  calls {pd(full, o, ks, 'calls')}")
