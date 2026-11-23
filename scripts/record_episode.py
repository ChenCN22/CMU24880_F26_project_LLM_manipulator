"""Record one episode as an MP4, seen by one of the cameras, with every tool call written under the image.

Under the image: the instruction, the call being made, and what the call really did.  The
second line is what the agent never gets from the skill: whether a pick or a placement
worked, and whether a camera report was right.

    # a scripted policy that reads the true scene: after a failure it pushes the block, then picks again
    python scripts/record_episode.py --driver always --family constrained_place --seed 6412 --out push.mp4
    # an LLM agent; with --llm-cache pointing at a COPY of a run's answer cache the episode is replayed
    python scripts/record_episode.py --driver A10r-none --perception persistent --family ordered_stack \
        --seed 2003 --llm-cache /tmp/cache_copy.jsonl --memory results/a10r_learn/memory.json --out agent.mp4

Needs ffmpeg on the path.  Rendering does not change the physics, so a recorded episode is the
same episode as in the sweep with that seed.
"""
import argparse
import os
import subprocess
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
from agent_iface.critic import load as load_calib                 # noqa: E402
from agent_iface.executor import FrozenSkillExecutor              # noqa: E402
from agent_iface.perception import PerceptionModel                # noqa: E402
from agent_iface.tools import ToolInterface                       # noqa: E402
from agent_iface.vlm_perception import FONT, render               # noqa: E402
from envs.scene_graph import _goal_met, scene_graph               # noqa: E402
from envs.tabletop import MultiObjectTabletopEnv                  # noqa: E402
from envs.task_specs import sample                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--driver", default="always", help="never | always (scripts that read the true scene) or an agent name")
ap.add_argument("--family", default="constrained_place")
ap.add_argument("--seed", type=int, default=6400)
ap.add_argument("--perception", default="none")
ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
ap.add_argument("--push-policy", default="checkpoints/tqc_her_push_350000_steps.zip")
ap.add_argument("--motion", default=None, help="how the hand moves around the skill: low or clear")
ap.add_argument("--calib", default="checkpoints/critic_calib.json")
ap.add_argument("--model", default="qwen2.5:14b")
ap.add_argument("--host", default="http://localhost:11434")
ap.add_argument("--llm-cache", default=None, help="answer cache to replay from; give a copy, new answers are appended")
ap.add_argument("--memory", default=None)
ap.add_argument("--max-calls", type=int, default=40)
ap.add_argument("--view", default="front")
ap.add_argument("--size", default="800x600")
ap.add_argument("--every", type=int, default=1, help="one frame per this many simulation steps (25 steps = 1 s)")
ap.add_argument("--pause", type=float, default=0.9, help="seconds the picture holds after each call")
ap.add_argument("--title", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

W, H = map(int, a.size.split("x"))
BAR, FPS = 132, 25
INK, MUTED, GOOD, BAD = (20, 20, 20), (95, 94, 90), (17, 120, 80), (190, 45, 25)
font = lambda s: ImageFont.truetype(FONT, s)                      # noqa: E731
F_SMALL, F_CALL = font(15), font(19)

env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=0.19)
env.reset(seed=a.seed)
spec = sample(a.family, env.object_names, a.seed)
goals = [g for g in spec["goals"] if g["type"] in ("in_zone", "on_top_of")]
pair = lambda g: (g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])   # noqa: E731
ex = FrozenSkillExecutor(env, policy_path=a.policy, max_steps=100, seed=a.seed, push_policy_path=a.push_policy, motion=a.motion)
calib = load_calib(a.calib) if os.path.exists(a.calib) else None
iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=a.max_calls, hide_claims=True, critic_calib=calib,
                      perception=PerceptionModel(spec, a.seed, a.perception, 0.30, 0.10, bad=(0.90, 0.30), good=(0.05, 0.02)))

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H + BAR}",
                       "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                       "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
title = a.title or {"never": "Scripted policy: pick again after a failure",
                    "always": "Scripted policy: push the block after a failure, then pick"}.get(a.driver, f"LLM agent {a.driver}")
state = {"call": "", "result": "", "color": MUTED, "n": 0, "frames": 0}


def frame() -> bytes:
    img = Image.new("RGB", (W, H + BAR), (250, 249, 245))
    img.paste(render(env, a.view, labels=True, size=(W, H)), (0, 0))
    d = ImageDraw.Draw(img)
    d.text((14, H + 8), title, fill=MUTED, font=F_SMALL)
    for i, line in enumerate(textwrap.wrap("Task: " + spec["instruction"], 82)[:2]):
        d.text((14, H + 29 + 18 * i), line, fill=INK, font=F_SMALL)
    d.text((14, H + 72), state["call"], fill=INK, font=F_CALL)
    size = 19
    while size > 12 and d.textlength(state["result"], font=font(size)) > W - 28:
        size -= 1                                                 # a long camera report still fits on its line
    d.text((14, H + 100), state["result"], fill=state["color"], font=font(size))
    return img.tobytes()


def put(n: int = 1) -> None:
    buf = frame()
    for _ in range(n):
        ff.stdin.write(buf)
    state["frames"] += n


_step = env.sim.step


def step() -> None:
    _step()
    state["n"] += 1
    if state["n"] % a.every == 0:
        put()


env.sim.step = step
_call = iface.call


def on(obj: str, tgt: str) -> bool:
    sg = scene_graph(env)
    return sg["in_zone"].get(obj) == tgt or sg["on_top_of"].get(obj) == tgt


def call(name, args=None):
    args = args or {}
    held = scene_graph(env)["held"]
    was = {n: np.asarray(env.sim.get_base_position(n), dtype=float)[:2] for n in env.object_names}
    shown = ", ".join(str(v) for k, v in args.items() if k in ("obj", "target", "view")) \
        or ("gripper" if args.get("query") == "held" else "")
    state.update(call=f"call {iface.calls + 1}:  {name}({shown})", result="", color=MUTED)
    r = _call(name, args)
    res = r.get("result") if isinstance(r, dict) else None
    text, good = "", None
    if name == "pick":
        good = scene_graph(env)["held"] == args.get("obj")
        text = "skill returns \"executed\"   |   in fact: " + ("the block is in the gripper" if good else "the pick FAILED")
    elif name == "place_on" and held:
        good = on(held, str(args.get("target")))
        text = "skill returns \"executed\"   |   in fact: " + (f"{held} is on {args.get('target')}" if good
                                                              else "the placement FAILED")
    elif name == "push" and args.get("obj") in was:
        now = np.asarray(env.sim.get_base_position(args["obj"]), dtype=float)[:2]
        text = f"the block was slid {100 * float(np.linalg.norm(now - was[args['obj']])):.0f} cm toward the middle"
    elif name == "observe" and isinstance(res, dict) and "held" in res and len(res) == 1:
        text = "gripper sensor: " + (f"holding {res['held']}" if res["held"] else "empty")
    elif name == "observe" and isinstance(res, dict):
        from agents.evidence import reading
        parts, good = [], True
        for g in goals:
            said, true = reading(pair(g), res), _goal_met(g, scene_graph(env))
            if said is None:
                continue
            good = good and said == true
            parts.append(f"{pair(g)[0]} on {pair(g)[1]}: {'yes' if said else 'no'}".replace("_block", "")
                         + ("" if said == true else " (WRONG)"))
        text = f"{args.get('view', 'front')} camera:   " + ";   ".join(parts)
    elif name == "done":
        good = bool(iface.score()["success"])
        text = "episode ends: the task is " + ("complete" if good else "NOT complete")
    state.update(result=text, color=MUTED if good is None else GOOD if good else BAD)
    print(f"  {state['call']:<34} {text}")
    put(int(a.pause * FPS * (2 if name == "done" else 1)))
    return r


iface.call = call
put(int(1.2 * FPS))                                               # the scene before anything moves

if a.driver in ("never", "always"):
    met = lambda g: _goal_met(g, scene_graph(env))                # noqa: E731
    for g in goals:
        obj, tgt = pair(g)
        for k in range(4):
            if met(g) or iface.finished:
                break
            iface.call("pick", {"obj": obj})
            if scene_graph(env)["held"] == obj:
                iface.call("place_on", {"target": tgt})
            if not met(g) and k < 3 and a.driver == "always" and scene_graph(env)["held"] != obj:
                iface.call("push", {"obj": obj, "direction": "center", "distance": 0.08})
    iface.call("done")
else:
    from agents import graphs
    from agents.llm import LLMBackbone
    from agents.nodes import SYSTEM_V2
    llm = LLMBackbone(model=a.model, host=a.host, think=None, effort="none", cache_path=a.llm_cache)
    memory = None
    if a.driver.startswith("A10"):
        from agents.skill_memory import SkillMemory
        memory = SkillMemory(a.memory, learn=False)
    graphs.build(a.driver, iface, llm, memory, SYSTEM_V2).invoke(
        {"instruction": spec["instruction"], "history": [], "reflections": [], "memories": [], "steps": 0, "stop": False},
        {"recursion_limit": a.max_calls * 4 + 20})
    print("LLM calls:", len(llm.trace))

ff.stdin.close()
ff.wait()
s = iface.score()
print(" ".join(t["tool"] + ("" if t.get("truth") in (True, None) else "[failed]") for t in iface.transcript if t["tool"] != "observe"))
print(f"{a.out}: {state['frames'] / FPS:.1f} s, {os.path.getsize(a.out) / 1e6:.1f} MB | success {s['success']} calls {s['calls']}")
