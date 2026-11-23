"""Watch episodes in the PyBullet GUI.

    python demo_gui.py --driver oracle                      # scripted, ground-truth retry
    python demo_gui.py --driver A2 --model qwen2.5:14b      # an LLM agent
    python demo_gui.py --driver A5 --model qwen2.5:14b --perception persistent --info monitor
                                                            # the perception-error study
    python demo_gui.py --driver A10-none,A10-map --perception none --families free_two_stacks \
        --seeds 2007,2010 --llm-cache /tmp/cache_copy.jsonl
                                                            # block choice without / with the skill map, same scenes

The window shows the instruction, the current tool call and, after each call, what really
happened (the agent only ever sees "executed").  For A10 it also shows the plan the model
chose and every change of plan.  Same seeds and a new simulator per episode as in the sweep,
so with --llm-cache pointing at a COPY of a run's answer cache the episode of the sweep is replayed.
"""
import argparse
import os
import time

import numpy as np

from agent_iface.executor import FrozenSkillExecutor
from agent_iface.tools import ToolInterface
from envs.tabletop import MultiObjectTabletopEnv
from envs.task_specs import FAMILIES, sample

ap = argparse.ArgumentParser()
ap.add_argument("--driver", default="oracle", help="naive | oracle | an agent name; several separated by commas")
ap.add_argument("--families", default=",".join(FAMILIES))
ap.add_argument("--episodes", type=int, default=2)
ap.add_argument("--seed-base", type=int, default=1000)
ap.add_argument("--seeds", default=None, help="seeds separated by commas, instead of --seed-base and --episodes")
ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
ap.add_argument("--push-policy", default=None, help="push skill checkpoint (A10r)")
ap.add_argument("--motion", default=None, help="how the hand moves around the skill: low or clear")
ap.add_argument("--model", default="qwen2.5:14b")
ap.add_argument("--host", default="http://localhost:11434")
ap.add_argument("--llm-cache", default=None, help="answer cache to replay from; give a copy, new answers are appended")
ap.add_argument("--memory", default="checkpoints/a10r_memory.json", help="A10: the frozen skill memory")
ap.add_argument("--avoid-memory", default="checkpoints/a11_memory.json", help="A11: the frozen memory of knocked blocks")
ap.add_argument("--max-calls", type=int, default=40)
ap.add_argument("--perception", default=None, help="none | transient | persistent | random "
                                                  "(omit for the original truthful, metered observe)")
ap.add_argument("--info", default="none", help="none | label | monitor")
ap.add_argument("--calib", default="checkpoints/critic_calib.json")
ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 2 = twice as fast")
ap.add_argument("--pause", type=float, default=1.2, help="seconds to hold after each tool call")
ap.add_argument("--repeat", type=int, default=1, help="play the whole list this many times")
a = ap.parse_args()

GREEN, RED, GREY, BLUE = (0, 0.5, 0), (0.85, 0, 0), (0.3, 0.3, 0.3), (0, 0, 0.6)
SLOTS = {"task": 0.60, "plan": 0.52, "call": 0.44, "result": 0.36, "score": 0.28}
short = lambda s: str(s).replace("_block", "")                    # noqa: E731

llm = memory = None
drivers = a.driver.split(",")
if any(d.startswith("A") for d in drivers):
    from agents import graphs
    from agents.llm import LLMBackbone
    from agents.nodes import SYSTEM_V2
    llm = LLMBackbone(model=a.model, host=a.host, think=None, effort="none", cache_path=a.llm_cache)
    if any(d.startswith("A10") for d in drivers):
        from agents.skill_memory import SkillMemory
        if not os.path.exists(a.memory):                          # without the file every form would act like A10-none
            raise SystemExit(f"skill memory not found: {a.memory}")
        memory = SkillMemory(a.memory, learn=False)
        print(f"skill memory: {len(memory.records)} records, {len(memory.lessons)} lessons", flush=True)
    elif any(d.startswith("A11") for d in drivers):
        from agents.disturb_memory import DisturbMemory
        if any(d.split("-")[1] in ("episodic", "lessons", "map") for d in drivers) and not os.path.exists(a.avoid_memory):
            raise SystemExit(f"disturbance memory not found: {a.avoid_memory}")
        memory = DisturbMemory(a.avoid_memory, learn=False)
        print(f"disturbance memory: {len(memory.records)} attempts, {len(memory.lessons)} lessons", flush=True)
    elif "A4" in drivers:
        from agents.memory import LessonMemory
        memory = LessonMemory(path="/tmp/demo_gui_memory")

seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed_base + i for i in range(a.episodes)]
jobs = [(fam, seed, d) for fam in a.families.split(",") for seed in seeds for d in drivers] * a.repeat

for k, (fam, seed, driver) in enumerate(jobs):
    # a new simulator for every episode, as in the sweep: (seed, tool calls) fixes the physics
    env = MultiObjectTabletopEnv(n_objects=4, render_mode="human", fence=0.19)
    pc = env.sim.physics_client
    pc.configureDebugVisualizer(pc.COV_ENABLE_GUI, 0)
    env.sim.place_visualizer(np.array([-0.05, 0.0, 0.0]), 0.75, 60, -35)
    _step = env.sim.step

    def slow_step():
        _step()
        time.sleep(0.04 / a.speed)
    env.sim.step = slow_step
    ids = {}

    def show(slot, text, color=(0, 0, 0), size=1.4):
        kw = {"replaceItemUniqueId": ids[slot]} if slot in ids else {}
        ids[slot] = pc.addUserDebugText(text, [-0.25, -0.42, SLOTS[slot]], textColorRGB=list(color), textSize=size, **kw)

    env.reset(seed=seed)
    spec = sample(fam, env.object_names, seed)
    ex = FrozenSkillExecutor(env, policy_path=a.policy, max_steps=100, seed=seed, push_policy_path=a.push_policy,
                             motion=a.motion)
    if a.perception:
        from agent_iface.critic import load
        from agent_iface.perception import PerceptionModel
        iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=a.max_calls,
                              perception=PerceptionModel(spec, seed, a.perception, 0.30, 0.10, bad=(0.90, 0.30), good=(0.05, 0.02)),
                              info=a.info, critic_calib=load(a.calib) if os.path.exists(a.calib) else None, hide_claims=True)
    else:
        iface = ToolInterface(env, ex, spec, observe_budget=8, max_calls=a.max_calls)
    raw, seen = iface.call, {"plans": 0}

    def plan_line():
        plans = getattr(iface, "plans", None) or []
        if len(plans) == seen["plans"]:
            return
        seen["plans"] = len(plans)
        p = plans[-1]
        text = "; ".join(f"{b} on {t}" for b, t in p["goals"]) or "(none)"
        why = "" if p["why"] == "start" else f"   [{p['why']}]"
        if p.get("refused"):
            show("plan", short(f"new plan REFUSED: {text}{why}")[:120], RED, 1.2)
        else:
            show("plan", short(f"plan {len(plans)}: {text}{why}")[:120], BLUE, 1.2)
        print(f"  plan {len(plans)}: {text}{why}" + (f"  REFUSED ({p['refused']})" if p.get("refused") else ""), flush=True)
        time.sleep(2.5 / a.speed)                                 # time to read the plan

    def call(name, args=None):
        args = args or {}
        arg = ", ".join(str(v) for key, v in args.items() if key != "query" or v != "scene")
        plan_line()
        show("call", short(f"[{iface.calls + 1}] {name}({arg}) ..."), GREY)
        show("result", "")
        out = raw(name, args)
        t = iface.transcript[-1] if iface.transcript else {}
        res = out.get("result") if isinstance(out, dict) else None
        wait = a.pause
        if name in ("pick", "place_on", "push") and t.get("truth") is not None:
            ok = bool(t["truth"])
            show("result", f"agent is told \"executed\"   |   in fact: {'it worked' if ok else 'it FAILED'}", GREEN if ok else RED)
        elif t.get("fault"):
            f = t["fault"]
            show("result", f"CAMERA ERROR ({out.get('view')} view): {f['type']} {f.get('direction', '')}", RED)
        elif name == "observe" and isinstance(res, dict) and list(res) == ["held"]:
            show("result", short(f"gripper sensor: {'holding ' + str(res['held']) if res['held'] else 'empty'}"), GREY)
            wait = a.pause / 2
        elif name == "observe":
            show("result", f"{args.get('view', 'front')} camera report received", GREY)
            wait = a.pause / 2
        else:
            show("result", str(out)[:110], GREY)
        print(f"  {iface.calls:2d} {name}({arg})  truth={t.get('truth')}", flush=True)
        plan_line()
        time.sleep(wait / a.speed)
        return out
    iface.call = call

    is_agent = driver.startswith("A")
    who = f"{driver} / {a.model}" if is_agent else driver
    note = {"A10-none": " (no memory of the skill)", "A10-map": " (shown the skill's success map)",
            "A10-episodic": " (shown past pick records)", "A10-lessons": " (shown its own written rules)",
            "A11-none": " (no memory of knocked blocks)", "A11-told": " (told the designer's rule about knocks)",
            "A11-episodic": " (shown past attempts and which blocks they moved)",
            "A11-lessons": " (shown its own written rules about knocks)",
            "A11-map": " (shown how often such carries moved a block)"}.get(driver, "")
    print(f"\n== [{k + 1}/{len(jobs)}] {fam} seed {seed} [{who}]: {spec['instruction']}", flush=True)
    show("task", f"[{k + 1}/{len(jobs)}] {who}{note} | motion {ex.motion} | seed {seed}: {spec['instruction']}"[:160], (0, 0, 0), 1.1)
    for slot in ("plan", "call", "result", "score"):
        show(slot, "")
    time.sleep(2.5 / a.speed)

    if not is_agent:
        for g in spec["goals"]:
            obj, tgt = (g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])
            for _ in range(1 if driver == "naive" else 3):
                iface.call("pick", {"obj": obj})
                if driver == "oracle" and not iface.transcript[-1]["truth"]:
                    continue
                iface.call("place_on", {"target": tgt})
                if iface.transcript[-1]["truth"] or driver == "naive":
                    break
        iface.call("done")
    else:
        llm.reset_usage()
        try:
            graphs.build(driver, iface, llm, memory if driver.startswith(("A10", "A11")) or driver == "A4" else None,
                         SYSTEM_V2 if a.perception else None).invoke(
                {"instruction": spec["instruction"], "history": [], "reflections": [],
                 "memories": [], "steps": 0, "stop": False},
                {"recursion_limit": a.max_calls * 4 + 20})
        except Exception as e:                      # noqa: BLE001
            print("  agent error:", e, flush=True)

    s = iface.score()
    msg = (f"{'SUCCESS' if s['success'] else 'FAILED'}   goals met {s['n_goals_met']}/{s['n_goals']}   "
           f"calls {s['calls']}/{a.max_calls}   done() {s['declared_done']}")
    show("score", msg, GREEN if s["success"] else RED, 1.6)
    print("  " + msg, flush=True)
    time.sleep(6 / a.speed)
    env.close()
