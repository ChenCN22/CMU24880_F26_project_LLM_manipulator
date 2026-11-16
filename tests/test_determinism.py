"""Is an episode reproducible?  Same seed + same tool calls must give the same physics.

    python tests/test_determinism.py --policy <ckpt> [--reuse]

Runs the never-looking script over the experiment's seeds three times, the second
and third time in a different order, and compares the final block positions.
  default   a new simulator per episode (what the sweep does)
  --reuse   one simulator for everything (what the sweep used to do)
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from agent_iface.executor import FrozenSkillExecutor             # noqa: E402
from agent_iface.tools import ToolInterface                      # noqa: E402
from envs.tabletop import MultiObjectTabletopEnv                 # noqa: E402
from envs.task_specs import sample                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
ap.add_argument("--families", default="ordered_stack,blocked_rearrange,constrained_place")
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--reuse", action="store_true")
a = ap.parse_args()

make = lambda: MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=0.19)   # noqa: E731
shared = make() if a.reuse else None


def episode(family: str, seed: int):
    env = shared or make()
    env.reset(seed=seed)
    spec = sample(family, env.object_names, seed)
    ex = FrozenSkillExecutor(env, policy_path=a.policy, max_steps=100, seed=seed)
    iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=40, hide_claims=True)
    for g in spec["goals"]:
        obj, tgt = (g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])
        iface.call("pick", {"obj": obj})
        iface.call("place_on", {"target": tgt})
    iface.call("done")
    pos = np.concatenate([env.sim.get_base_position(n) for n in env.object_names])
    ok = bool(iface.score()["success"])
    if not a.reuse:
        env.close()
    return ok, pos


keys = [(f, 1000 + e) for f in a.families.split(",") for e in range(a.episodes)]
orders = [keys, keys[::-1], [keys[i] for i in np.random.default_rng(0).permutation(len(keys))]]
runs = [{k: episode(*k) for k in order} for order in orders]
for i in (1, 2):
    flips = sum(runs[0][k][0] != runs[i][k][0] for k in keys)
    drift = max(float(np.abs(runs[0][k][1] - runs[i][k][1]).max()) for k in keys)
    exact = sum(np.array_equal(runs[0][k][1], runs[i][k][1]) for k in keys)
    print(f"run 0 vs run {i}: success differs in {flips}/{len(keys)} episodes | "
          f"bit-identical final positions {exact}/{len(keys)} | max position difference {drift:.2e} m")
print("success rate per run:", [round(float(np.mean([r[k][0] for k in keys])), 3) for r in runs])
same = all(np.array_equal(runs[0][k][1], runs[i][k][1]) for i in (1, 2) for k in keys)
print("DETERMINISTIC" if same else "NOT deterministic")
sys.exit(0 if same or a.reuse else 1)
