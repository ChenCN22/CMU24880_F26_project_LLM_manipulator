"""usage: a10_pooled.py <folder> [<folder> ...], each with one sub-folder per A10 form (scripts/a10_eval.sh).
Paired differences to A10-none over all the folders together, camera right, in total and per task family."""
import json, sys
import numpy as np
sets = {root: root for root in sys.argv[1:]}
agents = ["A10-none", "A10-episodic", "A10-lessons", "A10-map"]
rows = {a: {} for a in agents}
for s, root in sets.items():
    for a in agents:
        for l in open(f"{root}/{a}/episodes.jsonl"):
            r = json.loads(l)
            if r["perception"] == "none":
                rows[a][(s, r["family"], r["episode"])] = r
rng = np.random.default_rng(0)
fams = sorted({k[1] for k in rows["A10-none"]})
for fam in [None] + fams:
    keys = [k for k in rows["A10-none"] if fam in (None, k[1])]
    print(f"--- {fam or 'all four families'} (n={len(keys)})")
    for a in agents:
        s = np.array([rows[a][k]["success"] for k in keys], dtype=float)
        picks = [t for k in keys for t in rows[a][k]["transcript"] if t["tool"] == "pick"]
        first = [next((t for t in rows[a][k]["transcript"] if t["tool"] == "pick"), None) for k in keys]
        first = [t for t in first if t]
        line = (f"{a:<13} success {int(s.sum()):>3} ({100*s.mean():4.1f}%)  picks {len(picks)/len(keys):4.2f}/ep, worked {100*np.mean([bool(t['truth']) for t in picks]):4.1f}%"
                f"  first pick worked {100*np.mean([bool(t['truth']) for t in first]):4.1f}%  out of calls {sum(not rows[a][k]['declared_done'] for k in keys):>3}")
        if a != "A10-none":
            d = np.array([rows[a][k]["success"] - rows["A10-none"][k]["success"] for k in keys], dtype=float)
            bt = rng.choice(d, size=(10000, len(d))).mean(axis=1)
            line += f"  | vs none {100*d.mean():+5.1f} [{100*np.quantile(bt,.025):+.1f}, {100*np.quantile(bt,.975):+.1f}]"
        print(line)
