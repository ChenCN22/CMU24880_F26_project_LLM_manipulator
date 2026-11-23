"""What does the arm do to the blocks it is not supposed to move?

Episodes of a finished sweep are played again from its answer cache (same seed, same answers,
so the same episode), and this time the true positions of all blocks are kept before and
after every pick, place_on and push.  Counted per call:

  toppled   a block that stood on another block no longer does (and was not the one picked)
  knocked   a block the call was not about moved by 2 cm or more
  thrown    a block ends 10 cm or more from where it was (a carried block: from its target)
  to_rim    a block that was 2 cm or more from the rim ends within 2 cm of it (corner: both sides)

and for every pick where the block lay (inside / at one rim / in a corner) and whether the
pick worked.  Nothing here is seen by an agent.

    python tests/disturbance_audit.py --runs results/a10_holdout/A10-none results/a10_holdout/A10-map \
        --out results/disturbance_audit
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, ".")
from agent_iface.critic import load as load_calib             # noqa: E402
from agent_iface.executor import FrozenSkillExecutor          # noqa: E402
from agent_iface.perception import PerceptionModel            # noqa: E402
from agent_iface.tools import ToolInterface                   # noqa: E402
from envs.scene_graph import scene_graph                      # noqa: E402
from envs.tabletop import OBJECT_SIZE, ZONES, MultiObjectTabletopEnv  # noqa: E402
from envs.task_specs import sample                            # noqa: E402

FENCE, NEAR, MOVED, THROWN = 0.19, 0.02, 0.02, 0.10
A = None
_llm, _memory = {}, {}


def where(p) -> str:
    gx, gy = (FENCE - OBJECT_SIZE / 2 - abs(float(p[i])) for i in (0, 1))
    return "corner" if gx < NEAR and gy < NEAR else "rim" if min(gx, gy) < NEAR else "inside"


def llm_for(folder: str):
    if folder not in _llm:
        from agents.llm import LLMBackbone
        copy = os.path.join(A.tmp, f"{os.getpid()}_{folder.strip('/').replace('/', '_')}.jsonl")
        shutil.copy(os.path.join(folder, "llm_cache.jsonl"), copy)       # the run's own cache is never written to
        _llm[folder] = LLMBackbone(model=A.model, host=A.host, think=None, effort="none", cache_path=copy)
    return _llm[folder]


def run(job):
    folder, row = job
    from agents import graphs
    from agents.nodes import SYSTEM_V2
    agent, seed = row["agent"], row["seed"]
    env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=FENCE)
    env.reset(seed=seed)
    spec = sample(row["family"], env.object_names, seed)
    ex = FrozenSkillExecutor(env, policy_path=A.policy, max_steps=100, seed=seed, push_policy_path=A.push_policy)
    calib = load_calib(A.calib) if os.path.exists(A.calib) else None
    iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=A.max_calls, hide_claims=True, critic_calib=calib,
                          info=row["info"],
                          perception=PerceptionModel(spec, seed, row["perception"], 0.30, 0.10, bad=(0.90, 0.30), good=(0.05, 0.02)))
    pos = lambda: {n: np.asarray(env.sim.get_base_position(n), dtype=float) for n in env.object_names}   # noqa: E731
    start, events, raw = pos(), [], iface.call

    def call(name, args=None):
        args = args or {}
        if name not in ("pick", "place_on", "push"):
            return raw(name, args)
        sg0, p0 = scene_graph(env), pos()
        out = raw(name, args)
        sg1, p1 = scene_graph(env), pos()
        about = args.get("obj") if name != "place_on" else sg0["held"]
        target = args.get("target") if name == "place_on" else None
        ok = (sg1["held"] == about) if name == "pick" else \
             (about is not None and (sg1["in_zone"].get(about) == target or sg1["on_top_of"].get(about) == target)) if name == "place_on" else None
        d = {n: float(np.linalg.norm(p1[n] - p0[n])) for n in p0}
        if name == "place_on" and about:                     # a carried block: how far from its target it ended
            goal = np.asarray(ZONES[target], dtype=float)[:2] if target in ZONES else p1[target][:2] if target in p1 else p1[about][:2]
            d[about] = float(np.linalg.norm(p1[about][:2] - goal))
        events.append({
            "i": iface.calls, "tool": name, "about": about, "target": target, "ok": ok,
            "from": where(p0[about]) if about in p0 else None,
            "toppled": [n for n, b in sg0["on_top_of"].items() if b and sg1["on_top_of"].get(n) != b and n != about],
            "knocked": [n for n in d if n not in (about, target) and d[n] >= MOVED],
            "base_moved": bool(target in d and d[target] >= MOVED),
            "thrown": [n for n in d if d[n] >= THROWN and sg1["held"] != n],
            "to": where(p1[about]) if about in p1 else None,
            "to_rim": {n: where(p1[n]) for n in d if where(p0[n]) == "inside" and where(p1[n]) != "inside" and sg1["held"] != n
                       and not (n == about and name == "place_on" and ok)},      # a zone may lie at the rim
            "off_table": list(sg1.get("off_table", [])),
            "about_moved": round(d.get(about, 0.0), 3)})
        return out
    iface.call = call

    err = None
    try:
        memory = None
        if agent.startswith("A10"):
            if "m" not in _memory:
                from agents.skill_memory import SkillMemory
                _memory["m"] = SkillMemory(A.memory, learn=False)
            memory = _memory["m"]
        graphs.build(agent, iface, llm_for(folder), memory, SYSTEM_V2).invoke(
            {"instruction": spec["instruction"], "history": [], "reflections": [], "memories": [], "steps": 0, "stop": False},
            {"recursion_limit": A.max_calls * 4 + 20})
    except Exception as e:                                   # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    s, end = iface.score(), pos()
    same = [(t["tool"], t.get("args")) for t in iface.transcript] == row["calls_logged"] and bool(s["success"]) == row["success"]
    env.close()
    return {"run": folder, "agent": agent, "perception": row["perception"], "family": row["family"], "seed": seed,
            "success": bool(s["success"]), "calls": s["calls"], "same_as_logged": same, "error": err, "events": events,
            "start": {n: where(p) for n, p in start.items()}, "end": {n: where(p) for n, p in end.items()}}


def report(rows, out) -> None:
    lines = []
    say = lambda s="": (lines.append(s), print(s))            # noqa: E731
    pct = lambda a, b: f"{a:>4}/{b:<4} ({100 * a / b:3.0f}%)" if b else "   0/0"   # noqa: E731
    say(f"{len(rows)} episodes played again; {sum(r['same_as_logged'] for r in rows)} with the same calls and outcome as logged; "
        f"{sum(bool(r['error']) for r in rows)} errors")
    rows = [r for r in rows if r["same_as_logged"]]
    kinds = {"a stack toppled": lambda e: bool(e["toppled"]),
             "another block knocked 2 cm or more": lambda e: bool(e["knocked"]),
             "a block thrown 10 cm or more": lambda e: bool(e["thrown"]),
             "a block sent to the rim or a corner": lambda e: bool(e["to_rim"]),
             "a block sent into a corner": lambda e: "corner" in e["to_rim"].values(),
             "any of these": lambda e: bool(e["toppled"] or e["knocked"] or e["thrown"] or e["to_rim"])}
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["family"]].append(r)
    for name, rs in groups.items():
        fail, good = [r for r in rs if not r["success"]], [r for r in rs if r["success"]]
        say(f"\n{name}: {len(rs)} episodes, {len(fail)} failed")
        say(f"  {'episodes with at least one call in which':<44} {'failed':<18} {'successful':<18}")
        for k, f in kinds.items():
            say(f"  {k:<44} {pct(sum(any(f(e) for e in r['events']) for r in fail), len(fail)):<18} "
                f"{pct(sum(any(f(e) for e in r['events']) for r in good), len(good)):<18}")
        ev = [e for r in rs for e in r["events"]]
        for tool in ("pick", "place_on", "push"):
            es = [e for e in ev if e["tool"] == tool]
            if es:
                say(f"  {tool:<9} {len(es):>5} calls: toppled a stack {pct(sum(bool(e['toppled']) for e in es), len(es))}, "
                    f"knocked another block {pct(sum(bool(e['knocked']) for e in es), len(es))}, "
                    f"threw a block {pct(sum(bool(e['thrown']) for e in es), len(es))}")
        picks = [e for e in ev if e["tool"] == "pick" and e["from"]]
        say("  picks by where the block lay: " + ";  ".join(
            f"{w} {pct(sum(bool(e['ok']) for e in picks if e['from'] == w), sum(e['from'] == w for e in picks))}"
            for w in ("inside", "rim", "corner")))
        pushes = [e for e in ev if e["tool"] == "push" and e["from"]]
        if pushes:
            say("  pushes by where the block lay, ending inside: " + ";  ".join(
                f"{w} {pct(sum(e['to'] == 'inside' for e in pushes if e['from'] == w), sum(e['from'] == w for e in pushes))}"
                for w in ("inside", "rim", "corner")))
        moved_there = Counter()
        for r in rs:
            for n, w in r["end"].items():
                if w != "inside":
                    moved_there[(w, "was there at the start" if r["start"][n] == w else "got there during the episode", r["success"])] += 1
        for w in ("rim", "corner"):
            say(f"  blocks at the end in/at the {w}: " + ", ".join(
                f"{how} {moved_there[(w, how, True)]} in successful / {moved_there[(w, how, False)]} in failed episodes"
                for how in ("was there at the start", "got there during the episode")))
        stuck = sum(any(w == "corner" and r["start"][n] != "corner" for n, w in r["end"].items()) for r in fail)
        say(f"  failed episodes that end with a block in a corner it did not start in: {pct(stuck, len(fail))}")
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run folders with episodes.jsonl and llm_cache.jsonl")
    ap.add_argument("--perception", default="none", help="which rows of the runs to play again")
    ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
    ap.add_argument("--push-policy", default=None)
    ap.add_argument("--calib", default="checkpoints/critic_calib.json")
    ap.add_argument("--memory", default="checkpoints/a10r_memory.json")
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tmp", default="/tmp/disturbance_audit")
    ap.add_argument("--out", default="results/disturbance_audit")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    os.makedirs(A.tmp, exist_ok=True)
    jobs = []
    for folder in A.runs:
        with open(os.path.join(folder, "episodes.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["perception"] == A.perception and not r.get("error"):
                    jobs.append((folder, {k: r[k] for k in ("agent", "perception", "info", "family", "seed", "success")}
                                 | {"calls_logged": [(t["tool"], t.get("args")) for t in r["transcript"]]}))
    with Pool(A.workers) as pool:
        rows = pool.map(run, jobs, chunksize=4)
    with open(os.path.join(A.out, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    report(rows, A.out)
