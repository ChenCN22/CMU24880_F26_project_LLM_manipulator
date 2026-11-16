"""Per-call calibration of a frozen checkpoint THROUGH the executor.

diagnose_divergence.py answers "how often does a claim lie".  This answers the
question that comes first: how often does each tool actually work, and is the
task solvable at all with this skill?  Two scripted (no-LLM) drivers:

  naive   do what the goals say once, verify nothing          -> lower reference
  oracle  retry each step up to K times using GROUND TRUTH    -> upper reference
          (what a perfect, free verifier could get out of this skill)

Prints a claimed x truth table per tool and writes one JSON row per call.
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_iface.executor import FrozenSkillExecutor
from agent_iface.tools import ToolInterface
from envs.tabletop import MultiObjectTabletopEnv
from envs.task_specs import sample

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default=None)
ap.add_argument("--episodes", type=int, default=30)
ap.add_argument("--n-objects", type=int, default=4)
ap.add_argument("--families", default="color_sort,ordered_stack,blocked_rearrange,constrained_place")
ap.add_argument("--mode", choices=["naive", "oracle"], default="naive")
ap.add_argument("--retries", type=int, default=3)
ap.add_argument("--max-steps", type=int, default=100)
ap.add_argument("--gain", type=float, default=0.5)
ap.add_argument("--pick-gain", type=float, default=0.7)
ap.add_argument("--pick-hold", type=int, default=3)
ap.add_argument("--stack-standoff", type=float, default=0.08)
ap.add_argument("--fence", type=float, default=0.19)
ap.add_argument("--no-align", dest="align", action="store_false")
ap.add_argument("--no-retract", dest="retract", action="store_false")
ap.add_argument("--seed-base", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

env = MultiObjectTabletopEnv(n_objects=a.n_objects, render_mode="rgb_array", fence=a.fence)
calls, eps = [], []


def act(iface, tool, args):
    iface.call(tool, args)
    t = iface.transcript[-1]
    return t["truth"]


for fam in a.families.split(","):
    for ep in range(a.seed_base, a.seed_base + a.episodes):
        env.reset(seed=ep)
        spec = sample(fam, env.object_names, ep)
        ex = FrozenSkillExecutor(env, policy_path=a.policy, max_steps=a.max_steps, seed=ep, retract=a.retract, gain=a.gain, align=a.align, pick_gain=a.pick_gain, pick_hold=a.pick_hold, stack_standoff=a.stack_standoff)
        iface = ToolInterface(env, ex, spec, observe_budget=8, max_calls=80)
        for g in spec["goals"]:
            obj, tgt = (g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])
            tries = 1 if a.mode == "naive" else a.retries
            for _ in range(tries):
                picked = act(iface, "pick", {"obj": obj})
                if a.mode == "oracle" and not picked:
                    continue
                placed = act(iface, "place_on", {"target": tgt})
                if placed or a.mode == "naive":
                    break
        iface.call("done")
        s = iface.score()
        eps.append({"family": fam, "ep": ep, "success": bool(s["success"]),
                    "progress": s["progress"], "action_calls": s["action_calls"],
                    "off_table": len(s["objects_off_table"])})
        for t in iface.transcript:
            if t["truth"] is None:
                continue
            calls.append({"family": fam, "ep": ep, "tool": t["tool"],
                          "claimed": bool(t["visible"].get("claimed_success")),
                          "truth": bool(t["truth"]), **(t["info"] or {})})
env.close()

print(f"\npolicy {a.policy or 'stub'}   mode {a.mode}   retract {a.retract}   objects {a.n_objects}   episodes/family {a.episodes}")
print(f"{'family':20s} {'succ':>6s} {'prog':>6s} {'calls':>6s} {'off':>4s}")
for fam in a.families.split(","):
    r = [e for e in eps if e["family"] == fam]
    n = len(r)
    print(f"{fam:20s} {sum(e['success'] for e in r)/n:6.2f} {sum(e['progress'] for e in r)/n:6.2f} "
          f"{sum(e['action_calls'] for e in r)/n:6.1f} {sum(e['off_table'] for e in r):4d}")
print(f"\n{'tool':10s} {'n':>5s} {'true':>6s} {'claim':>6s} {'C&T':>5s} {'C&!T':>5s} {'!C&T':>5s} {'!C&!T':>6s}"
      f" {'P(T|C)':>7s} {'P(T|!C)':>8s} {'steps':>6s}")
for tool in ("pick", "place_on", "push"):
    r = [c for c in calls if c["tool"] == tool]
    if not r:
        continue
    n = len(r)
    k = Counter((c["claimed"], c["truth"]) for c in r)
    c1 = k[(True, True)] + k[(True, False)]
    c0 = k[(False, True)] + k[(False, False)]
    print(f"{tool:10s} {n:5d} {sum(c['truth'] for c in r)/n:6.2f} {c1/n:6.2f} {k[(True,True)]:5d} "
          f"{k[(True,False)]:5d} {k[(False,True)]:5d} {k[(False,False)]:6d} "
          f"{(k[(True,True)]/c1 if c1 else float('nan')):7.2f} {(k[(False,True)]/c0 if c0 else float('nan')):8.2f} "
          f"{sum(c['steps'] for c in r)/n:6.1f}")
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"args": vars(a), "episodes": eps, "calls": calls}, f)
