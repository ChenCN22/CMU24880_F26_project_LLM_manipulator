"""Checks on the perception-error layer (scripted stub, no checkpoint, no LLM).

    python tests/test_perception.py
"""
import sys
from collections import Counter

sys.path.insert(0, ".")
from agent_iface.executor import FrozenSkillExecutor            # noqa: E402
from agent_iface.perception import VIEWS, PerceptionModel       # noqa: E402
from agent_iface.tools import ToolInterface                     # noqa: E402
from envs.scene_graph import scene_graph                        # noqa: E402
from envs.tabletop import MultiObjectTabletopEnv                # noqa: E402
from envs.task_specs import sample                              # noqa: E402

env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=0.19)
FAMS = ("ordered_stack", "blocked_rearrange", "constrained_place")

# 1. mode none never alters the report; error rates match their definition; persistent repeats
def with_goal_met(graph, goal):
    """The same scene with one goal made true."""
    import copy
    g = copy.deepcopy(graph)
    PerceptionModel._set(None, g, goal, True)
    return g


n = Counter()
for fam in FAMS:
    for seed in range(1000, 1400):
        env.reset(seed=seed)
        spec, g = sample(fam, env.object_names, seed), scene_graph(env)
        assert PerceptionModel(spec, seed, "none").perceive(g, "front") == (g, None)
        for mode in ("transient", "persistent"):
            pm = PerceptionModel(spec, seed, mode)
            shown = {}
            for view in VIEWS:
                out, fault = pm.perceive(g, view)
                assert (fault is None) == (out == g), (mode, fault)
                shown[view] = out
                if mode == "persistent":
                    for _ in range(3):
                        assert pm.perceive(g, view)[0] == out, "persistent error did not repeat"
            for goal in pm.goals:                     # nothing has been done yet: every goal is unmet
                votes = [pm._met(goal, shown[v]) for v in VIEWS]
                n[mode, "fs", "n"] += 1
                n[mode, "fs", "one view"] += votes[seed % 3]          # an arbitrary single camera
                n[mode, "fs", "majority"] += sum(votes) >= 2
            goal = pm.goals[0]                         # and the first goal, once it has been achieved
            done = with_goal_met(g, goal)
            votes = [not pm._met(goal, pm.perceive(done, v)[0]) for v in VIEWS]
            n[mode, "ff", "n"] += 1
            n[mode, "ff", "one view"] += votes[seed % 3]
            n[mode, "ff", "majority"] += sum(votes) >= 2
            held = dict(done, held=goal.get("obj") or goal.get("top"))
            for v in VIEWS:
                fault = pm.perceive(held, v)[1]
                assert fault is None or all(e["goal"] != goal for e in fault["errors"]), "a held block was misreported"
EXPECT = {("transient", "fs"): (0.30, 0.216), ("transient", "ff"): (0.10, 0.028),
          ("persistent", "fs"): (0.333, 0.088), ("persistent", "ff"): (0.113, 0.012)}
for (mode, kind), (one, maj) in EXPECT.items():
    N = n[mode, kind, "n"]
    r1, r3 = n[mode, kind, "one view"] / N, n[mode, kind, "majority"] / N
    name = "false success" if kind == "fs" else "false failure"
    print(f"{mode:<11}{name:<14} one view {r1:.3f} (expect {one:.3f})   majority of three {r3:.3f} "
          f"(expect {maj:.3f})   n={N}")
    assert abs(r1 - one) < 0.04 and abs(r3 - maj) < 0.03, (mode, kind, r1, r3)

# 2. through the tool interface: views, labels, receipts, nothing hidden leaks
env.reset(seed=1003)
spec = sample("blocked_rearrange", env.object_names, 1003)
ex = FrozenSkillExecutor(env, policy_path=None, seed=1003)
iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=60,
                      perception=PerceptionModel(spec, 1003, "persistent"), info="monitor",
                      hide_claims=True)
obj, zone = spec["goals"][0]["obj"], spec["goals"][0]["zone"]
r = iface.call("pick", {"obj": obj})
assert r == {"ok": True, "executed": True}, r          # no success claim is shown
iface.call("place_on", {"target": zone})
assert iface.transcript[0]["claimed"] is not None
seen = {v: iface.call("observe", {"query": "zones", "view": v}) for v in VIEWS}
for v, r in seen.items():
    assert r["ok"] and r["view"] == v and "reliability" in r and "conflicts" in r["monitor"]
    assert not any(k.startswith("_") for k in r), r
    assert iface.call("observe", {"query": "zones", "view": v})["result"] == r["result"]
assert not iface.call("observe", {"query": "zones", "view": "behind"})["ok"]
for i in range(30):                                   # no perception meter any more
    assert iface.call("observe", {"query": "held"})["ok"]
shown = [t for t in iface.transcript if t["tool"] == "observe" and t["visible"].get("ok")]
assert all((t["fault"] is None) == (t["visible"]["result"] == t["true_answer"]) for t in shown)

# 3. a conflict is reported when the camera contradicts a clean placement
iface.handled[obj] = {"tool": "place_on", "target": zone, "claimed": True, "warned": False, "i": 2}
rec = iface._receipt({"in_zone": {obj: None}})
assert rec["conflicts"] and rec["options"], rec
assert iface._receipt({"in_zone": {obj: zone}}) == {"conflicts": []}
iface.handled[obj]["claimed"] = False
assert iface._receipt({"in_zone": {obj: zone}})["conflicts"]

# 4. the original interface is unchanged when no perception model is given
old = ToolInterface(env, ex, spec, observe_budget=2, max_calls=10)
r = old.call("observe", {"query": "zones"})
assert set(r) == {"ok", "result", "budget_left"}, r
old.call("observe", {"query": "held"})
assert not old.call("observe", {"query": "held"})["ok"]
env.close()
print("OK")
