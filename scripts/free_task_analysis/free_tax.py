"""Why episodes on the free-choice families fail, per agent: read off the logged calls (camera right)."""
import json, sys
import os
ROOT = os.environ.get("BENCH", "results/bench/qwen2.5_14b")   # one model's benchmark folder; arguments = agent folders in it
from collections import Counter, defaultdict
FREE = ["free_two_zones", "free_three_zones", "free_stack", "free_two_stacks"]
NEED = {"free_two_zones": 2, "free_three_zones": 3, "free_stack": 1, "free_two_stacks": 2}   # blocks that have to be moved
for ag in sys.argv[1:]:
    cat, stat = defaultdict(Counter), defaultdict(Counter)
    for l in open(f"{ROOT}/{ag}/episodes.jsonl"):
        r = json.loads(l)
        fam = r["family"]
        if r["perception"] != "none" or fam not in FREE:
            continue
        picks, fails, empty_place, held = Counter(), Counter(), 0, None
        for t in r["transcript"]:
            a = t.get("args") or {}
            if t["tool"] == "pick" and (t.get("visible") or {}).get("executed"):
                picks[a.get("obj")] += 1
                fails[a.get("obj")] += t.get("truth") is False
            if t["tool"] == "place_on" and t.get("truth") is False and not held:
                empty_place += 1
            held = t.get("held", held)
        s = stat[fam]
        s["n"] += 1
        s["ok"] += bool(r["success"])
        s["picks"] += sum(picks.values())
        s["pick_fail"] += sum(fails.values())
        s["blocks"] += len(picks)
        s["more_blocks_than_needed"] += len(picks) > NEED[fam]
        if r["success"]:
            continue
        stuck = [b for b, n in picks.items() if n >= 4]
        if r["calls"] >= 40:
            c = "ran out of calls"
        elif stuck and len(picks) < 4:
            c = "gave up on a block (4 picks) with an untouched block left"
        elif stuck:
            c = "gave up on a block (4 picks), all blocks touched"
        else:
            c = "called done with the task unmet, picks left"
        cat[fam][c] += 1
    print(f"##### {ag}")
    for fam in FREE:
        s = stat[fam]
        print(f"  {fam:<17} success {s['ok']:>2}/{s['n']}  picks/ep {s['picks'] / s['n']:4.1f}  picks that fail {100 * s['pick_fail'] / max(s['picks'], 1):3.0f}%  "
              f"distinct blocks picked/ep {s['blocks'] / s['n']:.2f}  episodes that touch more blocks than needed {s['more_blocks_than_needed']:>2}")
        for c, n in cat[fam].most_common():
            print(f"        {n:>2}  {c}")
