"""Where does the frozen skill fail?

Scripted episodes on their own seeds (6000+, used for nothing else).  For every
pick and place_on the log keeps what could be known BEFORE the call (where the
block is, how close the rim and the other blocks are, what happened to this
block last time) and what the call did.  Two-goal tasks are run in both goal
orders, so the effect of the order can be read from paired episodes.

The policy is the one A9 follows when its evidence is right: pick, check the
fingers, place, check the goal, at most four picks per block, and goals that
were knocked out of place are done again.

    python tests/skill_limits.py --episodes 200 --out results/skill_limits
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, ".")
from agent_iface.executor import FrozenSkillExecutor          # noqa: E402
from agent_iface.tools import ToolInterface                   # noqa: E402
from envs.scene_graph import _goal_met, scene_graph           # noqa: E402
from envs.tabletop import OBJECT_SIZE, ZONES, MultiObjectTabletopEnv   # noqa: E402
from envs.task_specs import sample                            # noqa: E402

FENCE, MAX_PICKS = 0.19, 4


def seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to the segment a-b (table plane)."""
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def state(env, ex) -> Dict[str, Dict[str, Any]]:
    out = {}
    for n in env.object_names:
        p = np.asarray(env.sim.get_base_position(n), dtype=float)
        rot = ex._canonical_rotation(n)
        out[n] = {"pos": p, "tilt": float(np.abs(rot[:2]).max()), "yaw": float(rot[2])}
    return out


def before(env, ex, obj: str) -> Dict[str, Any]:
    """What a camera with positions could say about a block before it is handled."""
    s = state(env, ex)
    p = s[obj]["pos"]
    others = [np.linalg.norm(p[:2] - s[n]["pos"][:2]) for n in s if n != obj]
    return {"x": round(p[0], 3), "y": round(p[1], 3), "z": round(p[2], 3),
            "tilt": round(s[obj]["tilt"], 2), "yaw": round(s[obj]["yaw"], 2),
            "rim_gap": round(FENCE - OBJECT_SIZE / 2 - float(np.abs(p[:2]).max()), 3),
            "rim_gap_x": round(FENCE - OBJECT_SIZE / 2 - abs(float(p[0])), 3),
            "rim_gap_y": round(FENCE - OBJECT_SIZE / 2 - abs(float(p[1])), 3),
            "neighbour": round(float(min(others)), 3)}


def plan_of(env, ex, obj: str, tgt: str) -> Dict[str, Any]:
    """What is known about a placement before its block is picked up."""
    s = state(env, ex)
    origin = s[obj]["pos"][:2]
    tpos = np.asarray(ZONES[tgt], dtype=float) if tgt in ZONES else s[tgt]["pos"][:2]
    blockers = [seg_dist(s[n]["pos"][:2], origin, tpos) for n in s if n not in (obj, tgt)]
    tb = before(env, ex, tgt) if tgt not in ZONES else {}
    return {"kind": "zone" if tgt in ZONES else "stack", "target": tgt,
            "to_x": round(float(tpos[0]), 3), "to_y": round(float(tpos[1]), 3),
            "travel": round(float(np.linalg.norm(tpos - origin)), 3),
            "path_gap": round(min(blockers), 3) if blockers else None,
            "target_rim_gap": tb.get("rim_gap"), "target_neighbour": tb.get("neighbour")}


def run(family: str, seed: int, order: int, policy: str, choose=None) -> Optional[Dict[str, Any]]:
    """choose(env, ex, spec) -> goals lets a caller replace the task's goals after seeing the
    start layout (tests/skill_choice.py); the episode is then scored on those goals."""
    env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=FENCE)
    env.reset(seed=seed)
    spec = sample(family, env.object_names, seed)
    goals = [g for g in spec["goals"] if g["type"] in ("in_zone", "on_top_of")]
    if order == 1:
        if len(goals) < 2:
            env.close()
            return None
        goals = goals[::-1]
    ex = FrozenSkillExecutor(env, policy_path=policy, max_steps=100, seed=seed)
    if choose is not None:
        goals = choose(env, ex, spec)
        spec = dict(spec, goals=goals, constraints=[
            {"type": "keep_on_table", "obj": o} for g in goals for o in ([g["obj"]] if g["type"] == "in_zone"
                                                                        else [g["top"], g["bottom"]])])
    iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=200, hide_claims=True)
    pair = lambda g: (g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])   # noqa: E731
    met = lambda g: _goal_met(g, scene_graph(env))                                                   # noqa: E731
    picks: Dict[str, int] = {}
    last: Dict[str, str] = {}
    calls: List[Dict[str, Any]] = []
    for sweep in range(3):                                   # later sweeps redo goals that were knocked
        todo = [g for g in goals if not met(g) and picks.get(pair(g)[0], 0) < MAX_PICKS]
        if not todo:
            break
        for gi, g in enumerate(goals):
            obj, tgt = pair(g)
            while not met(g) and picks.get(obj, 0) < MAX_PICKS:
                picks[obj] = picks.get(obj, 0) + 1
                was = {i: met(h) for i, h in enumerate(goals)}
                b = before(env, ex, obj)
                plan = plan_of(env, ex, obj, tgt)
                origin = np.array([b["x"], b["y"]])
                iface.call("pick", {"obj": obj})
                held = scene_graph(env)["held"] == obj
                now = {i: met(h) for i, h in enumerate(goals)}
                calls.append({"tool": "pick", "obj": obj, "n": picks[obj], "sweep": sweep, "goal": gi,
                              "prev": last.get(obj, "first"), **b, **plan, "ok": bool(held),
                              "knocked": [i for i in was if was[i] and not now[i] and goals[i] is not g]})
                if not held:
                    last[obj] = "failed pick"
                    continue
                s = state(env, ex)
                tpos = (np.asarray(ZONES[tgt], dtype=float) if tgt in ZONES else s[tgt]["pos"][:2])
                blockers = [seg_dist(s[n]["pos"][:2], origin, tpos) for n in s if n not in (obj, tgt)]
                tb = before(env, ex, tgt) if tgt not in ZONES else {}
                was = {i: met(h) for i, h in enumerate(goals)}
                iface.call("place_on", {"target": tgt})
                ok = met(g)
                now = {i: met(h) for i, h in enumerate(goals)}
                a = before(env, ex, obj)
                calls.append({"tool": "place_on", "obj": obj, "target": tgt, "n": picks[obj], "sweep": sweep,
                              "goal": gi, "kind": "zone" if tgt in ZONES else "stack",
                              "from_x": b["x"], "from_y": b["y"], "to_x": round(float(tpos[0]), 3),
                              "to_y": round(float(tpos[1]), 3),
                              "travel": round(float(np.linalg.norm(tpos - origin)), 3),
                              "path_gap": round(min(blockers), 3) if blockers else None,
                              "target_rim_gap": tb.get("rim_gap"), "target_neighbour": tb.get("neighbour"),
                              "ok": bool(ok), "after": a,
                              "knocked": [i for i in was if was[i] and not now[i] and goals[i] is not g]})
                last[obj] = "placed" if ok else "failed place"
    score = iface.score()
    met_at_end = [bool(met(g)) for g in goals]
    env.close()
    return {"family": family, "seed": seed, "order": order, "instruction": spec["instruction"],
            "goals": [list(pair(g)) for g in goals], "success": bool(score["success"]),
            "goals_met": score.get("goals_met"), "met": met_at_end, "calls": calls}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="ordered_stack,blocked_rearrange,constrained_place")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=6000)
    ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
    ap.add_argument("--out", default="results/skill_limits")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    with open(os.path.join(a.out, "episodes.jsonl"), "w") as f:
        for family in a.families.split(","):
            for ep in range(a.episodes):
                for order in (0, 1):
                    row = run(family, a.seed_base + ep, order, a.policy)
                    if row:
                        f.write(json.dumps(row, default=float) + "\n")
                        n += 1
            print(f"[skill_limits] {family} done ({n} episodes so far)", flush=True)
