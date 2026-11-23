"""Numbers behind the two failure modes on the free-choice families (camera right, true positions from the log).
  undo      the agent picks up a block that at that moment is part of what the task asks for
            (inside a wanted zone, or in a stack), per episode
  retry     picks on a block whose previous pick failed: how often they work, against first picks
  spare     episodes that end with a block picked 4 times and a never-picked block left: where the spare lies
"""
import json, sys
import os
ROOT = os.environ.get("BENCH", "results/bench/qwen2.5_14b")   # one model's benchmark folder; arguments = agent folders in it
from collections import Counter, defaultdict
import numpy as np
FREE = ["free_two_zones", "free_three_zones", "free_stack", "free_two_stacks"]
ZONES = {"zone_A": (0.11, -0.11), "zone_B": (0.11, 0.0), "zone_C": (0.11, 0.11)}
HALF, SIZE, FENCE = 0.045, 0.04, 0.19


def zone_of(p):
    for z, c in ZONES.items():
        if abs(p[0] - c[0]) <= HALF and abs(p[1] - c[1]) <= HALF and p[2] < 0.035:
            return z
    return None


def stacked(pos):
    out = set()
    for a, pa in pos.items():
        for b, pb in pos.items():
            if a != b and abs(pa[2] - pb[2] - SIZE) < 0.012 and np.hypot(pa[0] - pb[0], pa[1] - pb[1]) < 0.025:
                out |= {a, b}
    return out


def where(p):
    gx, gy = (FENCE - SIZE / 2 - abs(float(p[i])) for i in (0, 1))
    return "corner" if gx < 0.02 and gy < 0.02 else "rim" if min(gx, gy) < 0.02 else "inside"


for ag in sys.argv[1:]:
    S = defaultdict(Counter)
    for l in open(f"{ROOT}/{ag}/episodes.jsonl"):
        r = json.loads(l)
        fam = r["family"]
        if r["perception"] != "none" or fam not in FREE:
            continue
        want = [w for w in ZONES if w in r["instruction"]]
        s, picks, lastfail, prev = S[fam], Counter(), {}, None
        s["n"] += 1
        pairs = set()
        held = None
        for t in r["transcript"]:
            a = t.get("args") or {}
            pos = t.get("pos")
            if t["tool"] == "pick" and (t.get("visible") or {}).get("executed"):
                b = a.get("obj")
                if prev and b in prev:
                    good = (zone_of(prev[b]) in want) if want else (b in stacked(prev))
                    s["undo"] += bool(good)
                kind = "retry" if lastfail.get(b) else "first" if picks[b] == 0 else "again"
                s[kind] += 1
                s[kind + "_ok"] += t.get("truth") is True
                picks[b] += 1
                lastfail[b] = t.get("truth") is False
            if t["tool"] == "place_on" and (t.get("visible") or {}).get("executed") and held:
                pairs.add((held, a.get("target")))
                s["places"] += 1
            if pos:
                prev = {n: np.asarray(p, dtype=float) for n, p in pos.items()}
            held = t.get("held", held)
        s["pairs"] += len(pairs)
        stuck = [b for b, n in picks.items() if n >= 4]
        if not r["success"] and r["calls"] < 40 and stuck and len(picks) < 4 and prev:
            for n, p in prev.items():
                if n not in picks:
                    s["spare_" + where(p)] += 1
    print(f"##### {ag}")
    for fam in FREE:
        s = S[fam]
        pc = lambda k: f"{100 * s[k + '_ok'] / s[k]:3.0f}% of {s[k]:<3}" if s[k] else "   -      "   # noqa: E731
        print(f"  {fam:<17} undo/ep {s['undo'] / s['n']:4.2f}  placements/ep {s['places'] / s['n']:4.1f}  distinct (block,target)/ep {s['pairs'] / s['n']:4.2f}  "
              f"pick works: first {pc('first')} after a failed pick {pc('retry')} | spare block when it gave up: "
              + ", ".join(f"{k[6:]} {v}" for k, v in sorted(s.items()) if k.startswith("spare_")))
