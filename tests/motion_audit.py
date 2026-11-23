"""How the hand's motion around the skill calls changes what happens to the other blocks.

Scripted episodes that read the true scene (no camera, no agent): the stated blocks, pick and
place_on, at most four picks per block, goals that were knocked out of place are done again.
The same seeds are played with each motion of the executor (agent_iface/executor.py, MOTION),
and for every call the true positions of all blocks are compared before and after:

  toppled   a block that stood on another block no longer does (and was not the one picked)
  knocked   a block the call was not about moved by 2 cm or more; with the part of the call
            (home, grasp, lift, carry, align, release, retreat) in which it first moved by 1 cm
  thrown    a block ends 10 cm or more from where it was (a carried block: from its target)

    python tests/motion_audit.py --motions low,clear --episodes 100 --seed-base 4000 --out results/motion_audit
"""
import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, ".")
from agent_iface.executor import FrozenSkillExecutor          # noqa: E402
from agent_iface.tools import ToolInterface                   # noqa: E402
from envs.scene_graph import scene_graph                      # noqa: E402
from envs.tabletop import ZONES, MultiObjectTabletopEnv       # noqa: E402
from envs.task_specs import sample                            # noqa: E402
from tests.disturbance_audit import FENCE, MOVED, THROWN, where   # noqa: E402

MAX_PICKS = 4
FAMILY = {"stack": "ordered_stack", "two_stacks": "ordered_stack", "two_zones": "constrained_place",
          "three_zones": "color_sort", "rearrange": "blocked_rearrange"}
A = None


def run(job):
    task, seed, motion = job
    env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=FENCE)
    env.reset(seed=seed)
    spec = sample(FAMILY[task], env.object_names, seed)
    goals = [(g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])
             for g in spec["goals"] if g["type"] in ("in_zone", "on_top_of")]
    if task == "two_stacks":
        perm = [str(n) for n in np.random.default_rng(seed).permutation(list(env.object_names))]
        goals = [(perm[0], perm[1]), (perm[2], perm[3])]
    ex = FrozenSkillExecutor(env, policy_path=A.policy, max_steps=100, seed=seed, motion=motion)
    iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=200, hide_claims=True)
    pos = lambda: {n: np.asarray(env.sim.get_base_position(n), dtype=float) for n in env.object_names}   # noqa: E731

    def met(g) -> bool:
        sg = scene_graph(env)
        return sg["in_zone"].get(g[0]) == g[1] or sg["on_top_of"].get(g[0]) == g[1]

    watch = {"p0": None, "about": None, "first": {}}
    step = env.sim.step

    def watched_step():
        step()
        if watch["p0"] is not None:
            for n, p in watch["p0"].items():
                if n != watch["about"] and n not in watch["first"] and \
                        np.linalg.norm(np.asarray(env.sim.get_base_position(n), dtype=float) - p) >= 0.01:
                    watch["first"][n] = ex.phase
                    ee = np.asarray(env.robot.get_ee_position(), dtype=float)
                    ab = np.asarray(env.sim.get_base_position(watch["about"]), dtype=float)
                    watch.setdefault("ctx", {})[n] = [round(float(ee[2]), 3), round(float(env.robot.get_fingers_width()), 3),
                                                      round(float(np.linalg.norm(ab - watch["p0"][watch["about"]])), 3),
                                                      round(float(np.linalg.norm(ee[:2] - ab[:2])), 3)]
    env.sim.step = watched_step
    events = []

    def call(name, about, target=None):
        sg0, p0 = scene_graph(env), pos()
        watch.update(p0=p0, about=about, first={}, ctx={})
        iface.call(name, {"obj": about} if name == "pick" else {"target": target})
        watch["p0"] = None
        sg1, p1 = scene_graph(env), pos()
        ok = (sg1["held"] == about) if name == "pick" else met((about, target))
        d = {n: float(np.linalg.norm(p1[n] - p0[n])) for n in p0}
        if name == "place_on":
            goal = np.asarray(ZONES[target], dtype=float)[:2] if target in ZONES else p1[target][:2]
            d[about] = float(np.linalg.norm(p1[about][:2] - goal))
        knocked = [n for n in d if n not in (about, target) and d[n] >= MOVED]
        events.append({"tool": name, "about": about, "target": target, "ok": bool(ok), "from": where(p0[about]),
                       "kind": None if name == "pick" else "zone" if target in ZONES else "stack",
                       "toppled": [n for n, b in sg0["on_top_of"].items() if b and sg1["on_top_of"].get(n) != b and n != about],
                       "knocked": knocked, "phase": {n: watch["first"].get(n, "?") for n in knocked},
                       "ctx": {n: watch["ctx"].get(n) for n in knocked},
                       "rel": {n: [round(float(v), 3) for v in (p0[n] - p0[about])] for n in p0 if n != about},
                       "p0": {n: [round(float(v), 3) for v in p0[n]] for n in p0},
                       "moved": {n: round(d[n], 3) for n in d if n != about},
                       "about_moved": round(float(np.linalg.norm(p1[about] - p0[about])), 3), "about_z": round(float(p0[about][2]), 3),
                       "base_moved": bool(target in d and d[target] >= MOVED),
                       "thrown": [n for n in d if d[n] >= THROWN and sg1["held"] != n],
                       "to_rim": [n for n in d if where(p0[n]) == "inside" and where(p1[n]) != "inside" and sg1["held"] != n
                                  and not (n == about and name == "place_on" and ok)]})
        return ok

    picks = Counter()
    for _ in range(3):                                       # later sweeps redo goals that were knocked
        if all(met(g) or picks[g[0]] >= MAX_PICKS for g in goals):
            break
        for g in goals:
            while not met(g) and picks[g[0]] < MAX_PICKS:
                picks[g[0]] += 1
                if call("pick", g[0]):
                    call("place_on", g[0], g[1])
    out = {"task": task, "seed": seed, "motion": motion, "success": all(met(g) for g in goals),
           "calls": iface.calls, "events": events}
    env.close()
    return out


def report(rows, motions, tasks, out) -> None:
    lines = []
    say = lambda s="": (lines.append(s), print(s))            # noqa: E731
    pct = lambda a, b: f"{a:>4}/{b:<4} ({100 * a / b:3.0f}%)" if b else "     0/0      "   # noqa: E731
    rng = np.random.default_rng(0)
    by = {(r["task"], r["seed"], r["motion"]): r for r in rows}
    for name in ["all"] + tasks:
        keys = sorted({(t, s) for t, s, _ in by if name in ("all", t)})
        say(f"\n{name}  ({len(keys)} episodes per motion)")
        base = np.array([by[(*k, motions[0])]["success"] for k in keys], dtype=float)
        for m in motions:
            rs = [by[(*k, m)] for k in keys]
            ok = np.array([r["success"] for r in rs], dtype=float)
            ev = [e for r in rs for e in r["events"]]
            pk, pl = [e for e in ev if e["tool"] == "pick"], [e for e in ev if e["tool"] == "place_on"]
            line = f"  {m:<11} success {pct(int(ok.sum()), len(ok))}  calls/episode {np.mean([r['calls'] for r in rs]):4.1f}"
            if m != motions[0]:
                bt = rng.choice(ok - base, size=(10000, len(ok))).mean(axis=1)
                line += f"  vs {motions[0]} {100 * (ok - base).mean():+5.1f} [{100 * np.quantile(bt, .025):+.1f}, {100 * np.quantile(bt, .975):+.1f}]"
            say(line)
            say(f"      picks that hold the block {pct(sum(e['ok'] for e in pk), len(pk))}   "
                + "  ".join(f"{w} {pct(sum(e['ok'] for e in pk if e['from'] == w), sum(e['from'] == w for e in pk))}"
                            for w in ("inside", "rim", "corner")))
            say("      placements that end on the target: " + "   ".join(
                f"{k} {pct(sum(e['ok'] for e in pl if e['kind'] == k), sum(e['kind'] == k for e in pl))}" for k in ("zone", "stack")))
            for tool, es in (("pick", pk), ("place_on", pl)):
                ph = Counter(p for e in es for p in e["phase"].values())
                say(f"      {tool:<9} knocked another block {pct(sum(bool(e['knocked']) for e in es), len(es))}  "
                    f"toppled a stack {pct(sum(bool(e['toppled']) for e in es), len(es))}  "
                    f"threw a block {pct(sum(bool(e['thrown']) for e in es), len(es))}  "
                    f"sent a block to the rim {pct(sum(bool(e['to_rim']) for e in es), len(es))}"
                    + (f"   first moved during: {dict(ph.most_common())}" if ph else ""))
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", default="low,clear")
    ap.add_argument("--tasks", default="stack,two_stacks,two_zones,three_zones,rearrange")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=4000)
    ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/motion_audit")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    motions, tasks = A.motions.split(","), A.tasks.split(",")
    jobs = [(t, A.seed_base + i, m) for t in tasks for i in range(A.episodes) for m in motions]
    with Pool(A.workers) as pool:
        rows = pool.map(run, jobs, chunksize=4)
    with open(os.path.join(A.out, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    report(rows, motions, tasks, A.out)
