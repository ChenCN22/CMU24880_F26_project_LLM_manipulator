"""Watch episodes in the PyBullet GUI.

    python demo_gui.py --driver oracle                      # scripted, ground-truth retry
    python demo_gui.py --driver A2 --model qwen2.5:14b      # an LLM agent
    python demo_gui.py --driver A5 --model qwen2.5:14b --perception persistent --info monitor
                                                            # the perception-error study

The window shows the instruction, the current tool call and, after each call,
what the executor CLAIMED next to what actually happened (the agent only ever
sees the claim).  Same seeds as the sweep, so episode k here is episode k there.
"""
import argparse
import time

import numpy as np

from agent_iface.executor import FrozenSkillExecutor
from agent_iface.tools import ToolInterface
from envs.tabletop import MultiObjectTabletopEnv
from envs.task_specs import FAMILIES, sample

ap = argparse.ArgumentParser()
ap.add_argument("--driver", default="oracle", help="naive | oracle | A1 ... A8")
ap.add_argument("--families", default=",".join(FAMILIES))
ap.add_argument("--episodes", type=int, default=2)
ap.add_argument("--seed-base", type=int, default=1000)
ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
ap.add_argument("--model", default="qwen2.5:14b")
ap.add_argument("--max-calls", type=int, default=40)
ap.add_argument("--perception", default=None, help="none | transient | persistent | random "
                                                  "(omit for the original truthful, metered observe)")
ap.add_argument("--info", default="none", help="none | label | monitor")
ap.add_argument("--calib", default="checkpoints/critic_calib.json")
ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 2 = twice as fast")
ap.add_argument("--pause", type=float, default=1.2, help="seconds to hold after each tool call")
a = ap.parse_args()

env = MultiObjectTabletopEnv(n_objects=4, render_mode="human", fence=0.19)
pc = env.sim.physics_client
pc.configureDebugVisualizer(pc.COV_ENABLE_GUI, 0)
env.sim.place_visualizer(np.array([-0.05, 0.0, 0.0]), 0.75, 60, -35)

_step = env.sim.step
def slow_step():
    _step()
    time.sleep(0.04 / a.speed)
env.sim.step = slow_step

_ids = {}
def show(slot, text, color=(0, 0, 0), size=1.4):
    pos = {"task": [-0.25, -0.42, 0.52], "call": [-0.25, -0.42, 0.44],
           "result": [-0.25, -0.42, 0.36], "score": [-0.25, -0.42, 0.28]}[slot]
    kw = {"replaceItemUniqueId": _ids[slot]} if slot in _ids else {}
    _ids[slot] = pc.addUserDebugText(text, pos, textColorRGB=list(color), textSize=size, **kw)

GREEN, RED, GREY = (0, 0.5, 0), (0.85, 0, 0), (0.3, 0.3, 0.3)


def watched(iface):
    """Wrap iface.call so every call is drawn before and after it runs."""
    raw = iface.call
    def call(name, args=None):
        arg = ", ".join(str(v) for v in (args or {}).values())
        show("call", f"[{iface.calls + 1}] {name}({arg}) ...", GREY)
        show("result", "")
        out = raw(name, args)
        t = iface.transcript[-1] if iface.transcript else {}
        if name in ("pick", "place_on", "push") and t.get("truth") is not None:
            c, tr = bool(t.get("claimed")), bool(t["truth"])
            tag = "" if c == tr else ("   <-- SILENT FAILURE" if c else "   <-- claim too pessimistic")
            show("result", f"claimed={c}   truth={tr}{tag}", GREEN if c == tr and tr else RED)
        elif t.get("fault"):
            f = t["fault"]
            show("result", f"CAMERA ERROR ({out.get('view')} view): {f['type']} {f.get('direction', '')}", RED)
        else:
            show("result", str(out)[:110], GREY)
        print(f"  {iface.calls:2d} {name}({arg}) -> {out}  truth={t.get('truth')}", flush=True)
        time.sleep(a.pause / a.speed)
        return out
    iface.call = call
    return iface


llm = memory = None
if a.driver.startswith("A"):
    from agents import graphs
    from agents.llm import LLMBackbone
    llm = LLMBackbone(model=a.model)
    if a.driver == "A4":
        from agents.memory import LessonMemory
        memory = LessonMemory(path="/tmp/demo_gui_memory")

for fam in a.families.split(","):
    for ep in range(a.episodes):
        seed = a.seed_base + ep
        env.reset(seed=seed)
        spec = sample(fam, env.object_names, seed)
        ex = FrozenSkillExecutor(env, policy_path=a.policy, max_steps=100, seed=seed)
        if a.perception:
            import os
            from agent_iface.critic import load
            from agent_iface.perception import PerceptionModel
            iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=a.max_calls,
                                  perception=PerceptionModel(spec, seed, a.perception), info=a.info,
                                  critic_calib=load(a.calib) if os.path.exists(a.calib) else None,
                                  hide_claims=True)
        else:
            iface = ToolInterface(env, ex, spec, observe_budget=8, max_calls=a.max_calls)
        iface = watched(iface)
        who = a.driver if llm is None else f"{a.driver} / {a.model}"
        print(f"\n== {fam} seed {seed} [{who}]: {spec['instruction']}", flush=True)
        show("task", f"{who} | {fam} #{ep}: {spec['instruction']}"[:120], (0, 0, 0.6), 1.2)
        show("call", ""); show("result", ""); show("score", "")
        time.sleep(1.5 / a.speed)

        if llm is None:
            for g in spec["goals"]:
                obj, tgt = (g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])
                for _ in range(1 if a.driver == "naive" else 3):
                    iface.call("pick", {"obj": obj})
                    if a.driver == "oracle" and not iface.transcript[-1]["truth"]:
                        continue
                    iface.call("place_on", {"target": tgt})
                    if iface.transcript[-1]["truth"] or a.driver == "naive":
                        break
            iface.call("done")
        else:
            llm.reset_usage()
            try:
                from agents.nodes import SYSTEM_V2
                graphs.build(a.driver, iface, llm, memory, SYSTEM_V2 if a.perception else None).invoke(
                    {"instruction": spec["instruction"], "history": [], "reflections": [],
                     "memories": [], "steps": 0, "stop": False},
                    {"recursion_limit": a.max_calls * 4 + 20})
            except Exception as e:                      # noqa: BLE001
                print("  agent error:", e, flush=True)

        s = iface.score()
        msg = (f"{'SUCCESS' if s['success'] else 'FAILED'}   progress {s['progress']:.2f}   "
               f"calls {s['calls']}   silent failures {s['silent_failures']}   done() {s['declared_done']}")
        show("score", msg, GREEN if s["success"] else RED)
        print("  " + msg, flush=True)
        time.sleep(4 / a.speed)

env.close()
