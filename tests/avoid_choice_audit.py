"""The same question as tests/avoid_audit.py for the tasks that leave the choice of blocks open.

Scripted episodes that read the true scene, motion "low".  At every step all (block, target)
pairs that would bring the task forward are listed, and one is chosen:

  first      the first pair in the order of the names
  near       the pair with the shortest way to carry
  rim        a block that does not lie at the rim if there is one (what the skill map of A10
             says about picks), then the shortest way
  risk       the pair with the lowest predicted knock risk (tables of tests/avoid_audit.py)
  risk+rim   rim first, then the lowest predicted knock risk
  guard      rim first, then the lowest predicted risk to blocks that already are where the
             task wants them (to 0.1), then the shortest way

    python tests/avoid_choice_audit.py --episodes 200 --seed-base 10000 --out results/avoid_choice_audit
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
from envs.scene_graph import _goal_met, scene_graph           # noqa: E402
from envs.tabletop import ZONES, MultiObjectTabletopEnv       # noqa: E402
from envs.task_specs import sample                            # noqa: E402
from tests.avoid_audit import risks                           # noqa: E402
from tests.disturbance_audit import FENCE, MOVED, where       # noqa: E402

MAX_PICKS = 4
A = None


def run(job):
    fam, seed, how = job
    env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=FENCE)
    env.reset(seed=seed)
    spec = sample(fam, env.object_names, seed)
    ex = FrozenSkillExecutor(env, policy_path=A.policy, max_steps=100, seed=seed, motion="low")
    iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=200, hide_claims=True)
    pos = lambda: {n: np.asarray(env.sim.get_base_position(n), dtype=float) for n in env.object_names}   # noqa: E731
    n_met = lambda: sum(_goal_met(g, scene_graph(env)) for g in spec["goals"])                            # noqa: E731
    stats = Counter()

    def call(name, args, about, target=None):
        p0, m0 = pos(), n_met()
        iface.call(name, args)
        p1, sg1 = pos(), scene_graph(env)
        stats[name] += 1
        stats[name + "_knocked"] += any(n not in (about, target) and np.linalg.norm(p1[n] - p0[n]) >= MOVED for n in p0)
        stats["goal_undone"] += max(0, m0 - n_met())
        return sg1["held"] == about if name == "pick" else None

    def pairs(sg):
        on = sg["on_top_of"]
        in_stack = {t for t, b in on.items() if b} | {b for b in on.values() if b}
        loose = [n for n in env.object_names if n not in in_stack and n not in sg["off_table"]]
        if spec["goals"][0]["type"] == "zone_filled":
            want = [g["zone"] for g in spec["goals"]]
            open_ = [z for z in want if z not in sg["in_zone"].values()]
            return [(b, z) for b in loose if sg["in_zone"].get(b) not in want for z in open_]
        return [(t, b) for t in loose for b in loose if t != b]

    picks = Counter()
    while n_met() < len(spec["goals"]):
        sg, p = scene_graph(env), pos()
        cand = [c for c in pairs(sg) if picks[c[0]] < MAX_PICKS]
        if not cand:
            break
        way = lambda c: float(np.linalg.norm(p[c[0]][:2] - (ZONES[c[1]] if c[1] in ZONES else p[c[1]][:2])))   # noqa: E731
        rim = lambda c: {"inside": 0, "rim": 1, "corner": 2}[where(p[c[0]])]                                    # noqa: E731
        risk = lambda c: sum(risks(p, c).values())                                                               # noqa: E731
        on = sg["on_top_of"]
        done = {t for t, b in on.items() if b} | {b for b in on.values() if b} | \
               {n for n, z in sg["in_zone"].items() if z in [g.get("zone") for g in spec["goals"]]}
        harm = lambda c: round(sum(v for n, v in risks(p, c).items() if n in done), 1)                          # noqa: E731
        key = {"first": lambda c: 0, "near": way, "rim": lambda c: (rim(c), way(c)), "risk": risk,
               "risk+rim": lambda c: (rim(c), risk(c)), "guard": lambda c: (rim(c), harm(c), way(c))}[how]
        c = min(cand, key=key)
        picks[c[0]] += 1
        stats["risk_sum"] += risk(c)
        if call("pick", {"obj": c[0]}, c[0]):
            call("place_on", {"target": c[1]}, c[0], c[1])
    out = {"task": fam, "seed": seed, "how": how, "success": n_met() == len(spec["goals"]), "calls": iface.calls, **stats}
    env.close()
    return out


if __name__ == "__main__":
    from tests.avoid_audit import report
    ap = argparse.ArgumentParser()
    ap.add_argument("--hows", default="first,near,rim,risk,risk+rim")
    ap.add_argument("--tasks", default="free_two_zones,free_three_zones,free_stack,free_two_stacks")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=10000)
    ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/avoid_choice_audit")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    hows, tasks = A.hows.split(","), A.tasks.split(",")
    jobs = [(t, A.seed_base + i, h) for t in tasks for i in range(A.episodes) for h in hows]
    with Pool(A.workers) as pool:
        rows = pool.map(run, jobs, chunksize=4)
    with open(os.path.join(A.out, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    report(rows, hows, tasks, A.out)
