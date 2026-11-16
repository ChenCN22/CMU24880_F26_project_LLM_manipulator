"""Fit and test the critic early-warning band.

    python tests/calibrate_critic.py --policy <ckpt> --out results/critic_calib.json

Runs a scripted pick/place driver (no LLM, truthful camera) on two disjoint seed
ranges, neither of which overlaps the experiment's seeds (1000+).  The band is
fitted on successful calls from the first range and scored on the second:
how many truly failed calls it flags, how many successful ones it flags, and how
many control steps before the end of the call the flag goes up.
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from agent_iface.critic import fit, warning                      # noqa: E402
from agent_iface.executor import FrozenSkillExecutor             # noqa: E402
from agent_iface.tools import ToolInterface                      # noqa: E402
from envs.tabletop import ZONES, MultiObjectTabletopEnv          # noqa: E402
from envs.task_specs import sample                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
ap.add_argument("--families", default="ordered_stack,blocked_rearrange,constrained_place")
ap.add_argument("--fit-episodes", type=int, default=40)
ap.add_argument("--test-episodes", type=int, default=40)
ap.add_argument("--out", default="checkpoints/critic_calib.json")
a = ap.parse_args()

env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=0.19)


def collect(seed0: int, n: int):
    rows = []
    for fam in a.families.split(","):
        for ep in range(n):
            seed = seed0 + ep
            env.reset(seed=seed)
            spec = sample(fam, env.object_names, seed)
            ex = FrozenSkillExecutor(env, policy_path=a.policy, max_steps=100, seed=seed)
            iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=40)
            for g in spec["goals"]:
                obj, tgt = (g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])
                for _ in range(2):
                    iface.call("pick", {"obj": obj})
                    if not iface.transcript[-1]["truth"]:
                        continue
                    iface.call("place_on", {"target": tgt})
                    if iface.transcript[-1]["truth"]:
                        break
            for t in iface.transcript:
                q = (t["info"] or {}).get("q_trace")
                if t["truth"] is None or not q:
                    continue
                kind = "pick" if t["tool"] == "pick" else (
                    "place_zone" if t["args"].get("target") in ZONES else "place_stack")
                rows.append({"kind": kind, "truth": bool(t["truth"]),
                             "claimed": bool(t["visible"].get("claimed_success")), "q": q})
    return rows


fit_rows, test_rows = collect(2000, a.fit_episodes), collect(3000, a.test_episodes)
calib, report = {}, {}
for kind in ("pick", "place_zone", "place_stack"):
    good = [r["q"] for r in fit_rows if r["kind"] == kind and r["truth"]]
    if len(good) < 15:
        print(f"{kind}: only {len(good)} successful calls, no band fitted")
        continue
    calib[kind] = fit(good)
    c = defaultdict(int)
    lead, lead_frac = [], []
    silent = defaultdict(int)
    for r in (x for x in test_rows if x["kind"] == kind):
        fired, at = warning(calib, kind, r["q"])
        c[("tp" if fired else "fn") if not r["truth"] else ("fp" if fired else "tn")] += 1
        if not r["truth"] and r["claimed"]:                 # the silent failures specifically
            silent["flagged" if fired else "missed"] += 1
        if fired and not r["truth"]:
            lead.append(len(r["q"]) - at)
            lead_frac.append(1 - at / len(r["q"]))
    fail, ok = c["tp"] + c["fn"], c["fp"] + c["tn"]
    report[kind] = {
        "fit_successes": len(good), "test_failures": fail, "test_successes": ok,
        "tpr": round(c["tp"] / fail, 3) if fail else None,
        "tnr": round(c["tn"] / ok, 3) if ok else None,
        "silent_failures_flagged": silent["flagged"], "silent_failures_missed": silent["missed"],
        "lead_steps_median": float(np.median(lead)) if lead else None,
        "lead_fraction_median": round(float(np.median(lead_frac)), 2) if lead_frac else None,
    }
    print(kind, json.dumps(report[kind]))

calib["_report"] = report
with open(a.out, "w") as f:
    json.dump(calib, f)
print("wrote", a.out)
env.close()
