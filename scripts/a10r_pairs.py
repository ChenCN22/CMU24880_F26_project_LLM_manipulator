"""usage: scripts/a10r_pairs.py <folder with one sub-folder per agent> [reference agent, default A9] [family]
Success, pushes, and paired differences to the reference agent; with "family", one block per task family too."""
import json, os, sys
import numpy as np
root = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else "A9"
rows = {a: [json.loads(l) for l in open(f"{root}/{a}/episodes.jsonl")] for a in sorted(os.listdir(root))
        if os.path.exists(f"{root}/{a}/episodes.jsonl")}
key = lambda r: (r["perception"], r["family"], r["episode"])
rng = np.random.default_rng(0)
groups = [(pm, None) for pm in sorted({r["perception"] for rs in rows.values() for r in rs})]
if "family" in sys.argv[3:]:
    groups += [(pm, f) for pm, _ in list(groups) for f in sorted({r["family"] for rs in rows.values() for r in rs})]
for pm, fam in groups:
    print(f"--- perception {pm}" + (f", {fam}" if fam else ""))
    base = {key(r): r for r in rows.get(REF, []) if r["perception"] == pm}
    for a, rs in rows.items():
        rs = [r for r in rs if r["perception"] == pm and fam in (None, r["family"])]
        if not rs:
            continue
        retry = push = after_fail = 0
        for r in rs:
            last, failed = {}, set()
            calls = [t for t in r["transcript"] if t["tool"] in ("pick", "place_on", "push")]
            held = None
            for t in calls:
                o = t["args"].get("obj")
                if t["tool"] == "pick":
                    if o in failed:
                        retry += 1
                        after_fail += last.get(o) == "push"
                    held = o if t["truth"] else None
                    if not t["truth"]:
                        failed.add(o)
                    last[o] = "pick"
                elif t["tool"] == "push":
                    push += 1
                    last[o] = "push"
                elif held:
                    if not t["truth"]:
                        failed.add(held)
                    last[held] = "place"
                    held = None
        s = np.array([r["success"] for r in rs], dtype=float)
        line = (f"{a:<14} n={len(rs):<4} success {int(s.sum()):>3} ({100 * s.mean():3.0f}%)  calls {np.mean([r['calls'] for r in rs]):4.1f}  "
                f"tokens {np.mean([r.get('total_tokens', 0) for r in rs]):6.0f}  out of calls {sum(not r['declared_done'] for r in rs):>2}  "
                f"pushes {push:>3}  retries {retry:>3}, of them right after a push {after_fail:>3}")
        if a != REF and base:
            d = np.array([r["success"] - base[key(r)]["success"] for r in rs if key(r) in base], dtype=float)
            if len(d):
                bt = rng.choice(d, size=(10000, len(d))).mean(axis=1)
                line += f"  | vs {REF} {100 * d.mean():+5.1f} [{100 * np.quantile(bt, .025):+.1f}, {100 * np.quantile(bt, .975):+.1f}]"
        print(line)
