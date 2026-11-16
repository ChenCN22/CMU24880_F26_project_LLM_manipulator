"""What is knowing the skill's limits worth when the task leaves a choice?

The fixed tasks name the blocks, so an agent that knows where the skill fails
can only stop retrying sooner.  Here the task leaves the blocks open:

  two_zones    "put one block in zone X and another block in zone Y"   any blocks; two are spare
  three_zones  "put one block in each zone"                            any blocks; one is spare
  stack        "stack one block on another"                            any two blocks
  two_stacks   "build two stacks of two"                               all four blocks are needed, so
                                                                       a hard block cannot be left out,
                                                                       only given the job it can do

and the only thing that differs between the runs is how the next block is chosen:

  stated        the blocks the task generator names, retried when they fail:
                what a fixed task amounts to
  model_once    the best start choice by the skill model, then no change
  switch_blind  the stated blocks first; after a failure, another block at random:
                alternatives without knowledge of the skill
  switch_rule   one sentence a memory could hold: move the blocks farthest from the
                rim, and not one whose pick has just failed if there is another
  switch_model  before every attempt, the option the skill model rates highest

The skill model is two logistic regressions, "this pick works" and "this placement
works", fitted to the calls that tests/skill_limits.py logged on other seeds.
All runs have the same budget: four picks per goal.  Nothing here uses an LLM,
and every chooser sees only what a camera with positions would give it.

    python tests/skill_choice.py --fit results/skill_limits/episodes.jsonl --out results/skill_choice
"""
import argparse
import itertools
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
from agent_iface.executor import FrozenSkillExecutor          # noqa: E402
from agent_iface.tools import ToolInterface                   # noqa: E402
from envs.scene_graph import scene_graph                      # noqa: E402
from envs.tabletop import MultiObjectTabletopEnv              # noqa: E402
from envs.task_specs import sample                            # noqa: E402
from skill_limits import FENCE, before, plan_of               # noqa: E402

PICK_F = ["rim_gap_x", "rim_gap_y", "neighbour", "tilt", "yaw", "raised", "after_fail_pick"]
PLACE_F = ["stack", "travel", "path_gap", "target_rim_gap", "target_neighbour"]
CLIP = {"rim_gap_x": 0.08, "rim_gap_y": 0.08, "neighbour": 0.10, "tilt": 0.5, "yaw": 0.8, "path_gap": 0.10,
        "target_rim_gap": 0.08, "target_neighbour": 0.10, "travel": 0.35}
DEFAULT = {"target_rim_gap": 0.06, "target_neighbour": 0.10, "path_gap": 0.10}   # a zone pad; an empty path


def vec(f: Dict[str, Any], names: List[str]) -> List[float]:
    f = dict(f)
    f["stack"] = float(f.get("kind") == "stack")
    f["raised"] = float(f.get("z", 0.02) > 0.045)
    f["yaw"] = abs(float(f.get("yaw", 0.0)))
    f["after_fail_pick"] = float(f.get("prev") == "failed pick")
    out = []
    for k in names:
        v = f.get(k)
        v = DEFAULT[k] if v is None else float(v)
        out.append(min(v, CLIP[k]) if k in CLIP else v)
    return out


def fit(path: str, n_episodes=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if n_episodes:
        rows = rows[:n_episodes]
    models, notes = {}, []
    for tool, names in (("pick", PICK_F), ("place_on", PLACE_F)):
        calls = [(c, r["seed"]) for r in rows for c in r["calls"] if c["tool"] == tool]
        X = np.array([vec(c, names) for c, _ in calls])
        y = np.array([int(c["ok"]) for c, _ in calls])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
        note = f"{tool}: {len(y)} calls, {y.mean():.2f} worked"
        if len({s for _, s in calls}) >= 5 and 0 < y.sum() < len(y):
            p = cross_val_predict(m, X, y, cv=GroupKFold(5), groups=np.array([s for _, s in calls]),
                                  method="predict_proba")[:, 1]
            note += f", cross-validated AUC {roc_auc_score(y, p):.2f}"
        note += "; weights " + ", ".join(f"{k} {w:+.2f}" for k, w in zip(names, m[-1].coef_[0]))
        models[tool], _ = m, notes.append(note)
    return models, notes, len(rows)


def episode(task: str, seed: int, how: str, models, policy: str) -> Dict[str, Any]:
    family = {"two_zones": "constrained_place", "three_zones": "color_sort",
              "stack": "ordered_stack", "two_stacks": "ordered_stack"}[task]
    env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=FENCE)
    env.reset(seed=seed)
    spec = sample(family, env.object_names, seed)
    names = list(env.object_names)
    stated = [(g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])
              for g in spec["goals"] if g["type"] in ("in_zone", "on_top_of")]
    rng = np.random.default_rng(seed)
    if task == "two_stacks":
        perm = [str(n) for n in rng.permutation(names)]
        stated = [(perm[0], perm[1]), (perm[2], perm[3])]
    zones = [t for _, t in stated] if task.endswith("zones") else []
    n_goals = len(stated)
    ex = FrozenSkillExecutor(env, policy_path=policy, max_steps=100, seed=seed)
    iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=200, hide_claims=True)
    last: Dict[str, str] = {}
    budget = 4 * n_goals

    def stacks(g) -> List[Tuple[str, str]]:
        return [(t, b) for t, b in g["on_top_of"].items() if b]

    def n_open() -> int:
        g = scene_graph(env)
        if zones:
            return sum(z not in g["in_zone"].values() for z in zones)
        return max(0, n_goals - len({b for _, b in stacks(g)}))

    def met(opt: Tuple[str, str]) -> bool:
        g = scene_graph(env)
        return (g["in_zone"].get(opt[0]) == opt[1]) if zones else (g["on_top_of"].get(opt[0]) == opt[1])

    def plans() -> List[List[Tuple[str, str]]]:
        """Every complete way of finishing what is still open, with the blocks that are free."""
        g = scene_graph(env)
        on = [n for n in names if n not in g["off_table"]]
        if zones:
            todo = [z for z in zones if z not in g["in_zone"].values()]
            free = [n for n in on if g["in_zone"].get(n) not in zones]
            k = min(len(todo), len(free))
            return [list(zip(bs, todo)) for bs in itertools.permutations(free, k)] if k else []
        used = {x for tb in stacks(g) for x in tb}
        single = [n for n in on if n not in used]
        r = n_open()
        if r >= 2 and len(single) >= 4:
            return [[(a, b), (c, d)] for a, b, c, d in itertools.permutations(single, 4) if a < c]
        return [[(a, b)] for a, b in itertools.permutations(single, 2)] if r else []

    def feats(opt: Tuple[str, str]) -> Dict[str, Any]:
        return {**before(env, ex, opt[0]), **plan_of(env, ex, *opt), "prev": last.get(opt[0], "first")}

    def rate(opt: Tuple[str, str]) -> float:
        f = feats(opt)
        return float(models["pick"].predict_proba([vec(f, PICK_F)])[0, 1]
                     * models["place_on"].predict_proba([vec(f, PLACE_F)])[0, 1])

    def best_plan(ps):
        return max(ps, key=lambda plan: float(np.prod([rate(o) for o in plan])))

    def by_rule(ps):
        """Move the blocks that are far from the rim and have not just failed; leave the others where they are."""
        gap = {n: before(env, ex, n)["rim_gap"] - (1.0 if last.get(n) == "failed pick" else 0.0) for n in names}
        return max(ps, key=lambda plan: (min(gap[o[0]] for o in plan), sum(gap[o[0]] for o in plan)))

    fixed = list(stated)
    if how == "model_once":                                # the best start plan, kept to the end
        fixed = best_plan(plans())
    picks, log = 0, []
    while picks < budget and n_open():
        ps = plans()
        if not ps:
            break
        opts = {o for plan in ps for o in plan}
        if how in ("stated", "model_once", "switch_blind"):
            left = [o for o in fixed if not met(o)]
            if not left:
                break
            opt = left[0]
            if how == "switch_blind" and (last.get(opt[0], "first") != "first" or opt not in opts):
                others = [plan for plan in ps if opt not in plan]              # it failed: any other way
                if others:
                    fixed = [o for o in fixed if met(o)] + others[int(rng.integers(len(others)))]
                    opt = [o for o in fixed if not met(o)][0]
            if opt not in opts and how != "switch_blind":      # e.g. its block landed in another target zone
                mine = [o for o in opts if o[0] in {x[0] for x in left}]
                if not mine:
                    break
                opt = mine[0]
        elif how == "switch_rule":
            plan = by_rule(ps)
            opt = max(plan, key=lambda o: before(env, ex, o[0])["rim_gap"])
        else:                                              # switch_model: replan before every attempt
            opt = max(best_plan(ps), key=rate)
        f = feats(opt)
        picks += 1
        iface.call("pick", {"obj": opt[0]})
        held = scene_graph(env)["held"] == opt[0]
        rec = {"opt": list(opt), "pick_ok": bool(held), "rim_gap": f["rim_gap"], "prev": f["prev"]}
        if held:
            iface.call("place_on", {"target": opt[1]})
            rec["place_ok"] = bool(met(opt))
            last[opt[0]] = "placed" if rec["place_ok"] else "failed place"
        else:
            last[opt[0]] = "failed pick"
        log.append(rec)
    success = n_open() == 0
    env.close()
    return {"task": task, "chooser": how, "seed": seed, "success": bool(success), "picks": picks, "log": log}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="results/skill_limits/episodes.jsonl")
    ap.add_argument("--fit-episodes", type=int, default=None, help="fit on the first N logged episodes only")
    ap.add_argument("--tasks", default="two_zones,three_zones,stack,two_stacks")
    ap.add_argument("--choosers", default="stated,model_once,switch_blind,switch_rule,switch_model")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=6200)
    ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
    ap.add_argument("--out", default="results/skill_choice")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models, notes, n_fit = fit(a.fit, a.fit_episodes)
    print(f"[skill_choice] skill model fitted on {n_fit} episodes")
    for n in notes:
        print("   " + n, flush=True)
    tasks, choosers = a.tasks.split(","), a.choosers.split(",")
    rows = []
    with open(os.path.join(a.out, "episodes.jsonl"), "w") as f:
        for task in tasks:
            for how in choosers:
                sub = [episode(task, a.seed_base + ep, how, models, a.policy) for ep in range(a.episodes)]
                for r in sub:
                    f.write(json.dumps(r, default=float) + "\n")
                rows += sub
                print(f"[skill_choice] {task:<10}{how:<13} success {sum(r['success'] for r in sub)}/{len(sub)}"
                      f"   picks per episode {np.mean([r['picks'] for r in sub]):.2f}", flush=True)
    rng = np.random.default_rng(0)
    for task in tasks:
        cell = {h: {r["seed"]: r for r in rows if r["task"] == task and r["chooser"] == h} for h in choosers}
        for base in ("stated", "switch_blind"):
            for h in choosers:
                if h == base or base not in cell or (base == "switch_blind" and not h.startswith("switch")):
                    continue
                d = np.array([float(cell[h][s]["success"]) - float(cell[base][s]["success"]) for s in sorted(cell[h])])
                bt = rng.choice(d, size=(10000, len(d))).mean(axis=1)
                print(f"[skill_choice] {task}: {h} - {base} {100 * d.mean():+.1f} points "
                      f"[{100 * np.quantile(bt, .025):+.1f}, {100 * np.quantile(bt, .975):+.1f}]")
