"""What is a push worth over a whole episode, for a policy that knows the true scene?

tests/push_recover.py looks at one block from one failed state.  Here whole episodes are
played on the fixed task families, through the agent's tools and with the agent's limit on
calls, by a scripted policy that reads the true scene (so nothing here depends on cameras):

  never    pick, place_on; after a failure pick again            (at most 4 picks per block)
  rim      the same, but after a failure of a block that lies within 2 cm of the rim,
           push(obj, "center", 0.08) before the next pick
  always   the same, after every failure

A push is a call like any other.  Counted: task success, and what the pushes did to the
other blocks (moved one by 2 cm or more; undid a goal that was already met).

    python tests/push_episode.py --episodes 200 --out results/push_episode
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
from envs.tabletop import MultiObjectTabletopEnv              # noqa: E402
from envs.task_specs import sample                            # noqa: E402
from tests.skill_limits import FENCE, MAX_PICKS, before       # noqa: E402

A = None


def run(job):
    family, seed, policy = job
    env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=FENCE)
    env.reset(seed=seed)
    spec = sample(family, env.object_names, seed)
    goals = [g for g in spec["goals"] if g["type"] in ("in_zone", "on_top_of")]
    ex = FrozenSkillExecutor(env, policy_path=A.policy, push_policy_path=A.push_policy, max_steps=100, seed=seed)
    iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=A.max_calls, hide_claims=True)
    pair = lambda g: (g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])   # noqa: E731
    met = lambda g: _goal_met(g, scene_graph(env))                                                   # noqa: E731
    held = lambda o: scene_graph(env)["held"] == o                                                   # noqa: E731
    xy = lambda: {n: np.asarray(env.sim.get_base_position(n), dtype=float)[:2] for n in env.object_names}   # noqa: E731
    pushes = []

    def recover(obj):
        gap = before(env, ex, obj)["rim_gap"]
        if policy == "never" or (policy == "rim" and gap >= 0.02) or held(obj):
            return
        was, ok = xy(), [met(g) for g in goals]
        iface.call("push", {"obj": obj, "direction": "center", "distance": 0.08})
        now = xy()
        pushes.append({"obj": obj, "rim_gap": gap,
                       "moved": round(float(np.linalg.norm(now[obj] - was[obj])), 3),
                       "others_moved": sum(float(np.linalg.norm(now[n] - was[n])) >= 0.02 for n in now if n != obj),
                       "undid": sum(a and not met(g) for a, g in zip(ok, goals))})

    for g in goals:
        obj, tgt = pair(g)
        for k in range(MAX_PICKS):
            if met(g) or iface.finished:
                break
            iface.call("pick", {"obj": obj})
            if held(obj):
                iface.call("place_on", {"target": tgt})
            if not met(g) and k < MAX_PICKS - 1:
                recover(obj)
    iface.call("done")
    s = iface.score()
    env.close()
    return {"family": family, "seed": seed, "policy": policy, "success": bool(s["success"]), "progress": s["progress"],
            "calls": s["calls"], "failed": s["failed_actions"], "violations": len(s["violations"]), "pushes": pushes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="ordered_stack,blocked_rearrange,constrained_place")
    ap.add_argument("--policies", default="never,rim,always")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=6400)
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
    ap.add_argument("--push-policy", default="checkpoints/tqc_her_push_350000_steps.zip")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/push_episode")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    pols, fams = A.policies.split(","), A.families.split(",")
    jobs = [(f, A.seed_base + i, p) for f in fams for i in range(A.episodes) for p in pols]
    with Pool(A.workers) as pool:
        rows = pool.map(run, jobs, chunksize=4)
    with open(os.path.join(A.out, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    by = {}
    for r in rows:
        by.setdefault((r["family"], r["seed"]), {})[r["policy"]] = r
    rng = np.random.default_rng(0)

    def show(name, keys):
        base = np.array([by[k][pols[0]]["success"] for k in keys], dtype=float)
        print(f"{name}  (n={len(keys)})")
        for p in pols:
            b = np.array([by[k][p]["success"] for k in keys], dtype=float)
            line = f"    {p:<7} success {int(b.sum()):>3} ({100 * b.mean():3.0f}%)  calls {np.mean([by[k][p]['calls'] for k in keys]):4.1f}"
            if p != pols[0]:
                bt = rng.choice(b - base, size=(10000, len(b))).mean(axis=1)
                ps = [q for k in keys for q in by[k][p]["pushes"]]
                line += (f"  vs {pols[0]} {100 * (b - base).mean():+5.1f} [{100 * np.quantile(bt, .025):+.1f}, {100 * np.quantile(bt, .975):+.1f}]"
                         f"  | {len(ps)} pushes: moved another block {sum(q['others_moved'] > 0 for q in ps)}, undid a met goal {sum(q['undid'] > 0 for q in ps)}"
                         f", new violations {sum(by[k][p]['violations'] for k in keys) - sum(by[k][pols[0]]['violations'] for k in keys):+d}")
            print(line)

    show("all episodes", list(by))
    for fam in fams:
        show(fam, [k for k in by if k[0] == fam])
    show(f"episodes with a failed action under '{pols[0]}'", [k for k in by if by[k][pols[0]]["failed"]])
