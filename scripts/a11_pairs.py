"""usage: a11_pairs.py <folder> [<folder> ...], each with sub-folders that hold an episodes.jsonl (scripts/a11_eval.sh).

What the agents chose and what it did to the other blocks, from the true positions kept with
every call, and the paired differences to the first agent listed.  Camera right.

  near path   share of carries with another block (on the table) within 6 cm of the straight line
              from where the block was picked up to its target
  knock       share of picks / carries after which a block the call was not about lay 2 cm or
              more from where it was
  undone      placements that were on their target and are not at the end, per episode
"""
import json
import os
import sys

import numpy as np

ZONES = {"zone_A": (0.11, -0.11), "zone_B": (0.11, 0.0), "zone_C": (0.11, 0.11)}
ORDER = ["A10-none", "A11-none", "A11-episodic", "A11-lessons", "A11-map", "A11-told"]


def line_dist(p, a, b) -> float:
    p, a, b = (np.asarray(v[:2], dtype=float) for v in (p, a, b))
    ab = b - a
    t = 0.0 if ab @ ab < 1e-9 else float(np.clip((p - a) @ ab / (ab @ ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def measure(r):
    """Per episode: counts read off the true positions around every pick and carry."""
    m = dict(picks=0, pick_ok=0, pick_knock=0, carries=0, near=0, carry_knock=0, carry_cm=0.0, replans=max(0, len(r.get("plans", [])) - 1))
    tr = [t for t in r["transcript"] if t.get("pos")]
    prev, start, held = None, None, None
    for t in tr:
        pos = {n: np.asarray(p, dtype=float) for n, p in t["pos"].items()}
        if prev is not None and t["tool"] in ("pick", "place_on") and t["visible"].get("executed"):
            about = t["args"].get("obj") if t["tool"] == "pick" else held
            target = t["args"].get("target") if t["tool"] == "place_on" else None
            knocked = any(n not in (about, target) and np.linalg.norm(pos[n] - prev[n]) >= 0.02 for n in pos if n in prev)
            if t["tool"] == "pick" and about in prev:
                m["picks"] += 1
                m["pick_ok"] += bool(t["truth"])
                m["pick_knock"] += knocked
                start = prev[about] if t["truth"] else None
            elif t["tool"] == "place_on" and about and start is not None:
                end = ZONES.get(target, prev.get(target))
                if end is not None:
                    m["carries"] += 1
                    m["carry_knock"] += knocked
                    m["carry_cm"] += 100 * float(np.linalg.norm(start[:2] - np.asarray(end[:2])))
                    m["near"] += any(n not in (about, target) and p[2] < 0.035 and line_dist(p, start, end) < 0.06 for n, p in prev.items())
                start = None
        prev, held = pos, t.get("held")
    return m


rows = {}
for root in sys.argv[1:]:
    for a in sorted(os.listdir(root)):
        f = os.path.join(root, a, "episodes.jsonl")
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if r["perception"] == "none" and not r.get("error"):
                    rows.setdefault(r["agent"], {})[(root, r["family"], r["episode"])] = (r, measure(r))
agents = [a for a in ORDER if a in rows] + sorted(a for a in rows if a not in ORDER)
base = os.environ.get("BASE") or agents[0]                 # BASE=A11-none: differences to that agent instead
keys_all = sorted(set.intersection(*(set(rows[a]) for a in agents)))
rng = np.random.default_rng(0)


def diff(x, y):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    bt = rng.choice(d, size=(10000, len(d))).mean(axis=1)
    return f"{100 * d.mean():+5.1f} [{100 * np.quantile(bt, .025):+.1f}, {100 * np.quantile(bt, .975):+.1f}]"


share = lambda m, k: m[k] / m["carries"] if m["carries"] else 0.0   # noqa: E731
for fam in [None, "zones"] + sorted({k[1] for k in keys_all}):
    keys = [k for k in keys_all if fam is None or (fam == "zones" and k[1].endswith("zones")) or fam == k[1]]
    print(f"\n--- {'all families' if fam is None else 'the two zone families' if fam == 'zones' else fam} (n={len(keys)} episodes per agent, paired)")
    for a in agents:
        rs, ms = [rows[a][k][0] for k in keys], [rows[a][k][1] for k in keys]
        tot = {k: sum(m[k] for m in ms) for k in ms[0]}
        ok = [bool(r["success"]) for r in rs]
        ep_near = [m["near"] / m["carries"] if m["carries"] else 0.0 for m in ms]
        line = (f"{a:<13} success {sum(ok):>3} ({100 * np.mean(ok):4.1f}%)  carries near path {100 * tot['near'] / max(tot['carries'], 1):4.1f}%  "
                f"carry {tot['carry_cm'] / max(tot['carries'], 1):4.1f} cm  knock: pick {100 * tot['pick_knock'] / max(tot['picks'], 1):4.1f}% "
                f"carry {100 * tot['carry_knock'] / max(tot['carries'], 1):4.1f}%  picks {tot['picks'] / len(keys):4.2f}/ep worked {100 * tot['pick_ok'] / max(tot['picks'], 1):4.1f}%  "
                f"calls {np.mean([r['calls'] for r in rs]):4.1f}  replans {tot['replans'] / len(keys):4.2f}")
        if a != base:
            b_ok = [bool(rows[base][k][0]["success"]) for k in keys]
            b_near = [rows[base][k][1]["near"] / rows[base][k][1]["carries"] if rows[base][k][1]["carries"] else 0.0 for k in keys]
            line += (f"\n{'':<13} vs {base}: success {diff(ok, b_ok)}   carries near path (per episode) {diff(ep_near, b_near)}"
                     f"   carries that knock (per episode) {diff([share(m, 'carry_knock') for m in ms], [share(rows[base][k][1], 'carry_knock') for k in keys])}")
        print(line)
