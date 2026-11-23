"""How A9 / A10r read the open instructions: the goal parse of every episode, looked up in the run's answer cache."""
import hashlib, json, sys
import os
ROOT = os.environ.get("BENCH", "results/bench/qwen2.5_14b")   # one model's benchmark folder; arguments = agent folders in it
from collections import Counter, defaultdict
sys.path.insert(0, ".")
from agents.evidence import GOALS_FORMAT
names = ["red_block", "green_block", "blue_block", "yellow_block"]
zones = ["zone_A", "zone_B", "zone_C"]
schema = {"type": "object", "required": ["goals"], "properties": {"goals": {
    "type": "array", "items": {"type": "object", "required": ["block", "target"], "properties": {
        "block": {"type": "string", "enum": names},
        "target": {"type": "string", "enum": zones + names}}}}}}
FREE = ["free_two_zones", "free_three_zones", "free_stack", "free_two_stacks"]


def shape(goals):
    ids = {}
    def nm(x):
        if x in zones:
            return "Z"
        ids.setdefault(x, f"b{len(ids) + 1}")
        return ids[x]
    return ", ".join(f"{nm(b)}>{nm(t)}" for b, t in goals)


for ag in sys.argv[1:]:
    d = f"{ROOT}/{ag}"
    cache = {json.loads(l)["key"]: json.loads(l)["out"] for l in open(f"{d}/llm_cache.jsonl") if l.strip()}
    by = defaultdict(Counter)
    ok = defaultdict(Counter)
    miss = 0
    for l in open(f"{d}/episodes.jsonl"):
        r = json.loads(l)
        if r["perception"] != "none" or r["family"] not in FREE:
            continue
        msgs = [{"role": "system", "content": "You read robot task instructions.\n\n" + GOALS_FORMAT},
                {"role": "user", "content": f"INSTRUCTION: {r['instruction']}"}]
        key = hashlib.sha256(json.dumps(["qwen2.5:14b", 0.0, 0, 8192, 256, schema, msgs], sort_keys=True).encode()).hexdigest()
        out = cache.get(key)
        if out is None:
            miss += 1
            continue
        goals = []
        for g in out.get("goals", []):
            pair = (g.get("block"), g.get("target"))
            if pair[0] in names and pair[1] in zones + names and pair[0] != pair[1] and pair not in goals:
                goals.append(pair)
        s = shape(goals)
        by[r["family"]][s] += 1
        ok[r["family"]][s] += bool(r["success"])
    print(f"##### {ag}  (cache misses {miss})")
    for fam in FREE:
        print(" ", fam)
        for s, n in by[fam].most_common():
            print(f"     {n:>3} episodes, {ok[fam][s]:>2} succeeded   parsed goals: {s or '(none)'}")
