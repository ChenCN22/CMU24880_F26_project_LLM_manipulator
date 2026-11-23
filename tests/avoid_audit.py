"""Can the knocks of the old motion be avoided with the tools an agent already has?

Scripted episodes that read the true scene (no camera, no agent), motion "low".  Three ways
to choose the next step, same seeds:

  fixed        the goals in the order of the task, as in tests/motion_audit.py
  order        of the goals that can be done now, the one with the lowest predicted knock risk
  guard        the goal with the lowest predicted risk to blocks that already are where the task
               wants them (to 0.1), then as order
  order+push   as order; if even that goal is risky because of one free block, that block is
               pushed out of the way first (the push skill), at most PUSHES times per episode

The risk is read off two tables measured with motion "low" on seeds 4000-4199
(results/motion_audit_low_pos): the chance that a pick moves another block, by its distance
from the block picked, and the chance that carrying a block moves another one, by its
distance from the straight line start -> target.  Run on other seeds than those.

    python tests/avoid_audit.py --episodes 200 --seed-base 10000 --out results/avoid_audit
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
from envs.tabletop import OBJECT_SIZE, ZONES, MultiObjectTabletopEnv  # noqa: E402
from envs.task_specs import sample                            # noqa: E402
from tests.disturbance_audit import FENCE, MOVED              # noqa: E402
from tests.motion_audit import FAMILY, MAX_PICKS              # noqa: E402

# distance (m) -> chance that the other block is moved by 2 cm or more
PICK_RISK = ([0.045, 0.0525, 0.07, 0.09, 0.115, 0.165, 0.25], [0.54, 0.44, 0.27, 0.14, 0.09, 0.04, 0.015])
CARRY_RISK = ([0.01, 0.03, 0.05, 0.07, 0.09, 0.125, 0.20], [0.85, 0.71, 0.41, 0.16, 0.08, 0.026, 0.01])
PUSHES, PUSH_DIST = 3, 0.08
A = None


def line_dist(p, a, b) -> float:
    p, a, b = (np.asarray(v[:2], dtype=float) for v in (p, a, b))
    ab = b - a
    t = 0.0 if ab @ ab < 1e-9 else float(np.clip((p - a) @ ab / (ab @ ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def risks(pos, goal):
    """Per other block: the chance that doing `goal` (pick, then carry) moves it."""
    obj, tgt = goal
    end = ZONES[tgt] if tgt in ZONES else pos[tgt]
    out = {}
    for n, p in pos.items():
        if n == obj:
            continue
        r_pick = float(np.interp(np.linalg.norm(p[:2] - pos[obj][:2]), *PICK_RISK)) if abs(p[2] - pos[obj][2]) < 0.03 else 0.0
        r_carry = float(np.interp(line_dist(p, pos[obj], end), *CARRY_RISK)) if n != tgt and p[2] < 0.035 else 0.0
        out[n] = 1 - (1 - r_pick) * (1 - r_carry)
    return out


def run(job):
    task, seed, how = job
    env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=FENCE)
    env.reset(seed=seed)
    spec = sample(FAMILY[task], env.object_names, seed)
    goals = [(g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])
             for g in spec["goals"] if g["type"] in ("in_zone", "on_top_of")]
    if task == "two_stacks":
        perm = [str(n) for n in np.random.default_rng(seed).permutation(list(env.object_names))]
        goals = [(perm[0], perm[1]), (perm[2], perm[3])]
    ex = FrozenSkillExecutor(env, policy_path=A.policy, max_steps=100, seed=seed, motion="low",
                             push_policy_path=A.push_policy)
    iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=200, hide_claims=True)
    pos = lambda: {n: np.asarray(env.sim.get_base_position(n), dtype=float) for n in env.object_names}   # noqa: E731

    def met(g) -> bool:
        sg = scene_graph(env)
        return sg["in_zone"].get(g[0]) == g[1] or sg["on_top_of"].get(g[0]) == g[1]

    stats = Counter()

    def call(name, args, about, target=None):
        p0, sg0 = pos(), scene_graph(env)
        was_met = [g for g in goals if met(g)]
        iface.call(name, args)
        p1, sg1 = pos(), scene_graph(env)
        stats[name] += 1
        stats[name + "_knocked"] += any(n not in (about, target) and np.linalg.norm(p1[n] - p0[n]) >= MOVED for n in p0)
        stats["goal_undone"] += sum(not met(g) for g in was_met if g[0] != about)
        return (sg1["held"] == about) if name == "pick" else met((about, target)) if name == "place_on" else None

    picks, pushes = Counter(), 0
    while True:
        unmet = [g for g in goals if not met(g) and picks[g[0]] < MAX_PICKS]
        # a block that still has to go somewhere else is not built on yet
        ready = [g for g in unmet if not any(h[0] == g[1] for h in unmet)] or unmet
        if not ready:
            break
        p = pos()
        if how == "guard":                                    # first of all spare what is already done
            done = {h[0] for h in goals if met(h)} | {h[1] for h in goals if met(h)}
            g = min(ready, key=lambda q: (round(sum(v for n, v in risks(p, q).items() if n in done), 1), sum(risks(p, q).values())))
        else:
            g = ready[0] if how == "fixed" else min(ready, key=lambda q: sum(risks(p, q).values()))
        if how == "order+push" and pushes < PUSHES:
            r = risks(p, g)
            n = max(r, key=r.get)
            sg = scene_graph(env)
            free = (p[n][2] < 0.035 and n not in sg["on_top_of"].values() and n != g[1]
                    and not any(h[0] == n and met(h) for h in goals))
            if r[n] >= A.push_at and free:
                lim = FENCE - OBJECT_SIZE / 2 - 0.02
                best = None
                for d, u in (("+x", (1, 0)), ("-x", (-1, 0)), ("+y", (0, 1)), ("-y", (0, -1))):
                    q = dict(p)
                    q[n] = p[n] + np.array([u[0] * PUSH_DIST, u[1] * PUSH_DIST, 0.0])
                    if np.any(np.abs(q[n][:2]) > lim) or any(np.linalg.norm(q[n][:2] - z) < 0.07 for z in ZONES.values()) or \
                            any(m != n and np.linalg.norm(q[n][:2] - q[m][:2]) < 0.07 for m in q):
                        continue
                    after = risks(q, g)[n]
                    if best is None or after < best[0]:
                        best = (after, d)
                if best and best[0] < r[n] / 2:
                    pushes += 1
                    call("push", {"obj": n, "direction": best[1], "distance": PUSH_DIST}, n)
                    continue
        picks[g[0]] += 1
        stats["risk_sum"] += sum(risks(p, g).values())
        if call("pick", {"obj": g[0]}, g[0]):
            call("place_on", {"target": g[1]}, g[0], g[1])
    out = {"task": task, "seed": seed, "how": how, "success": all(met(g) for g in goals), "calls": iface.calls, **stats}
    env.close()
    return out


def report(rows, hows, tasks, out) -> None:
    lines = []
    say = lambda s="": (lines.append(s), print(s))            # noqa: E731
    rng = np.random.default_rng(0)
    by = {(r["task"], r["seed"], r["how"]): r for r in rows}
    for name in ["all"] + tasks:
        keys = sorted({(t, s) for t, s, _ in by if name in ("all", t)})
        say(f"\n{name}  ({len(keys)} episodes per strategy)")
        base = np.array([by[(*k, hows[0])]["success"] for k in keys], dtype=float)
        for h in hows:
            rs = [by[(*k, h)] for k in keys]
            ok = np.array([r["success"] for r in rs], dtype=float)
            tot = Counter()
            for r in rs:
                tot.update({k: v for k, v in r.items() if isinstance(v, (int, float)) and k not in ("seed", "success", "calls")})
            line = f"  {h:<11} success {int(ok.sum()):>4}/{len(ok):<4} ({100 * ok.mean():3.0f}%)"
            if h != hows[0]:
                bt = rng.choice(ok - base, size=(10000, len(ok))).mean(axis=1)
                line += f"  vs {hows[0]} {100 * (ok - base).mean():+5.1f} [{100 * np.quantile(bt, .025):+.1f}, {100 * np.quantile(bt, .975):+.1f}]"
            else:
                line += " " * 31
            pk, pl = max(tot["pick"], 1), max(tot["place_on"], 1)
            line += (f"  calls {np.mean([r['calls'] for r in rs]):4.1f}  picks that knock {100 * tot['pick_knocked'] / pk:3.0f}%  "
                     f"carries that knock {100 * tot['place_on_knocked'] / pl:3.0f}%  goals undone/episode {tot['goal_undone'] / len(rs):.2f}  "
                     f"pushes/episode {tot['push'] / len(rs):.2f}  predicted risk/attempt {tot['risk_sum'] / pk:.2f}")
            say(line)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hows", default="fixed,order,order+push")
    ap.add_argument("--tasks", default="stack,two_stacks,two_zones,three_zones,rearrange")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=10000)
    ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
    ap.add_argument("--push-policy", default="checkpoints/tqc_her_push_350000_steps.zip")
    ap.add_argument("--push-at", type=float, default=0.35, help="push a free block away when its predicted risk is at least this")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/avoid_audit")
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
