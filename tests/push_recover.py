"""After the pick skill has failed on a block, is pushing the block first better than picking again?

Scripted episodes on the fixed task families, seeds 6400+ (used for nothing else).  Each runs
the usual pick / place loop until a block is first left in a failed state (a pick that did not
take it, or a placement that missed).  From that same state, replayed exactly, three calls are
spent in two ways:

  repick   pick, pick, pick
  push     push the block to a point at least 8 cm from the rim, then pick, pick
           (the push skill called directly with a point: the first measurement)
  center   the same through the agent's tool: push(obj, "center", 0.08), then pick, pick
  axis     the same through the tool, along one axis: 8 cm straight away from the nearest rim

and the question is how often the block is in the hand after two and after three calls.

    python tests/push_recover.py --episodes 200 --out results/push_recover
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, ".")
from agent_iface.executor import FrozenSkillExecutor          # noqa: E402
from agent_iface.tools import ToolInterface                   # noqa: E402
from envs.scene_graph import _goal_met, scene_graph           # noqa: E402
from envs.tabletop import OBJECT_SIZE, MultiObjectTabletopEnv  # noqa: E402
from envs.task_specs import sample                            # noqa: E402
from tests.skill_limits import FENCE, before, state           # noqa: E402

INSIDE = FENCE - OBJECT_SIZE / 2 - 0.08        # |x|, |y| of a point 8 cm from the rim
A = None


def push_goal(env, ex, obj) -> np.ndarray:
    """A point 8 cm or more from the rim; a block that is already there is moved 6 cm away from its nearest neighbour."""
    s = state(env, ex)
    p = s[obj]["pos"][:2]
    goal = np.clip(p, -INSIDE, INSIDE)
    if np.linalg.norm(goal - p) < 0.04:
        near = min((s[n]["pos"][:2] for n in s if n != obj), key=lambda q: np.linalg.norm(q - p))
        away = p - near
        away = away / (np.linalg.norm(away) + 1e-9)
        goal = np.clip(p + 0.06 * away, -INSIDE, INSIDE)
    return goal


def run(job):
    family, seed, arm = job
    env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=FENCE)
    env.reset(seed=seed)
    spec = sample(family, env.object_names, seed)
    goals = [g for g in spec["goals"] if g["type"] in ("in_zone", "on_top_of")]
    ex = FrozenSkillExecutor(env, policy_path=A.policy, push_policy_path=A.push_policy, max_steps=100, seed=seed)
    iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=200, hide_claims=True)
    pair = lambda g: (g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])   # noqa: E731
    met = lambda g: _goal_met(g, scene_graph(env))                                                   # noqa: E731
    held = lambda o: scene_graph(env)["held"] == o                                                   # noqa: E731

    def branch(obj, why):
        if held(obj):                       # a placement that never let go: not a state to recover from
            return None
        row = {"family": family, "seed": seed, "arm": arm, "obj": obj, "why": why, **before(env, ex, obj), "calls": []}
        if arm == "push":
            r = ex.push_to(obj, push_goal(env, ex, obj))
            row["calls"].append({"tool": "push", "ok": bool(r["_truth"]), "moved": r["_info"]["moved"],
                                 "after": before(env, ex, obj)})
        elif arm in ("center", "axis"):                 # what an agent can ask for
            direction = "center"
            if arm == "axis":
                ax = "x" if row["rim_gap_x"] <= row["rim_gap_y"] else "y"
                direction = ("-" if row[ax] > 0 else "+") + ax
            iface.call("push", {"obj": obj, "direction": direction, "distance": 0.08})
            t = iface.transcript[-1]
            row["calls"].append({"tool": "push", "ok": bool(t["truth"]), "moved": t["info"]["moved"],
                                 "direction": direction, "after": before(env, ex, obj)})
        while len(row["calls"]) < 3 and not held(obj):
            iface.call("pick", {"obj": obj})
            row["calls"].append({"tool": "pick", "ok": bool(held(obj))})
        row["held_after"] = next((i + 1 for i, c in enumerate(row["calls"]) if c["tool"] == "pick" and c["ok"]), None)
        return row

    out, done = None, False
    for g in goals:
        obj, tgt = pair(g)
        for _ in range(4):
            if met(g):
                break
            iface.call("pick", {"obj": obj})
            if not held(obj):
                out, done = branch(obj, "failed pick"), True
                break
            iface.call("place_on", {"target": tgt})
            if not met(g):
                out, done = branch(obj, "failed place"), True
                break
        if done:
            break
    env.close()
    return out


def rate(xs):
    xs = list(xs)
    return f"{sum(xs):>3}/{len(xs):<3} ({100 * np.mean(xs):3.0f}%)" if xs else "       -      "


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="ordered_stack,blocked_rearrange,constrained_place")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=6400)
    ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
    ap.add_argument("--push-policy", default="checkpoints/tqc_her_push_350000_steps.zip")
    ap.add_argument("--arms", default="repick,push")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/push_recover")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    jobs = [(f, A.seed_base + i, arm) for f in A.families.split(",") for i in range(A.episodes) for arm in A.arms.split(",")]
    with Pool(A.workers) as pool:
        rows = [r for r in pool.map(run, jobs, chunksize=4) if r]
    with open(os.path.join(A.out, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    by = {}
    for r in rows:
        by.setdefault((r["family"], r["seed"]), {})[r["arm"]] = r
    arms = A.arms.split(",")
    pairs = [v for v in by.values() if len(v) == len(arms)]
    print(f"{len(pairs)} failed states out of {len(jobs) // len(arms)} episodes")
    within = lambda r, k: r["held_after"] is not None and r["held_after"] <= k    # noqa: E731
    rng = np.random.default_rng(0)

    def show(name, sel):
        if not sel:
            return
        print(f"{name}  (n={len(sel)})")
        a = {k: np.array([within(v["repick"], k) for v in sel], dtype=float) for k in (2, 3)}
        print(f"    {'repick':<8}" + "".join(f" | {k} calls: {100 * a[k].mean():3.0f}%" + " " * 22 for k in (2, 3)))
        for arm in arms[1:]:
            line = f"    {arm:<8}"
            for k in (2, 3):
                b = np.array([within(v[arm], k) for v in sel], dtype=float)
                bt = rng.choice(b - a[k], size=(10000, len(b))).mean(axis=1)
                line += (f" | {k} calls: {100 * b.mean():3.0f}%  {100 * (b - a[k]).mean():+5.1f} "
                         f"[{100 * np.quantile(bt, .025):+.1f}, {100 * np.quantile(bt, .975):+.1f}]")
            print(line)

    show("all failed states", pairs)
    for why in ("failed pick", "failed place"):
        show(why, [v for v in pairs if v["repick"]["why"] == why])
    show("within 2 cm of the rim", [v for v in pairs if v["repick"]["rim_gap"] < 0.02])
    show("2 cm or more from the rim", [v for v in pairs if v["repick"]["rim_gap"] >= 0.02])
    show("tilted", [v for v in pairs if v["repick"]["tilt"] >= 0.05])
    show("flat, within 2 cm", [v for v in pairs if v["repick"]["tilt"] < 0.05 and v["repick"]["rim_gap"] < 0.02])
    print(f"first call: pick again works {rate(v['repick']['calls'][0]['ok'] for v in pairs)}")
    for arm in arms[1:]:
        ps = [v[arm]["calls"][0] for v in pairs]
        print(f"  {arm}: the push leaves the block 4 cm or more from the rim {rate(c['after']['rim_gap'] >= 0.04 for c in ps)}, "
              f"flat {rate(c['after']['tilt'] < 0.05 for c in ps)}, moved {100 * np.mean([c['moved'] for c in ps]):.1f} cm on average; "
              f"first pick after it works {rate(v[arm]['calls'][1]['ok'] for v in pairs if len(v[arm]['calls']) > 1)}")
