"""How well does a vision-language model read the scene, per camera?

    python tests/eval_vlm.py --model qwen2.5vl:7b --episodes 10 --out results/vlm_eval

Runs a scripted pick-and-place (up to three attempts per goal) and, after every
placement, renders the three cameras and asks the VLM where each cube is.  The
answer is scored against the true scene:
  goal level   false success  P(report says met | goal is not met)
               false failure  P(report says not met | goal is met)
               per camera, and for the majority of the three cameras
  cube level   share of cubes whose reported place equals the true place
Images and one JSON line per look are written to --out.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")
from agent_iface.executor import FrozenSkillExecutor             # noqa: E402
from agent_iface.perception import VIEWS                         # noqa: E402
from agent_iface.tools import ToolInterface                      # noqa: E402
from agent_iface.vlm_perception import VLMPerception, render     # noqa: E402
from envs.scene_graph import scene_graph                         # noqa: E402
from envs.tabletop import MultiObjectTabletopEnv                 # noqa: E402
from envs.task_specs import sample                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5vl:7b")
ap.add_argument("--host", default="http://127.0.0.1:11435")
ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
ap.add_argument("--families", default="ordered_stack,blocked_rearrange,constrained_place")
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--seed-base", type=int, default=7000)
ap.add_argument("--attempts", type=int, default=3)
ap.add_argument("--save-images", type=int, default=30, help="how many looks to save as PNG")
ap.add_argument("--style", default="scene")
ap.add_argument("--no-retract", action="store_true", help="leave the arm where the last action ended")
ap.add_argument("--replay", default=None, help="jsonl of an earlier run made with --save-images all: re-ask its images")
ap.add_argument("--out", default="results/vlm_eval")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.model.replace(":", "_").replace("/", "_") + "_" + a.style
log = open(os.path.join(a.out, f"{tag}.jsonl"), "w")


def place_of(graph, n):
    if n not in graph["in_zone"]:
        return "not_visible" if n in graph.get("hidden", []) else "off_table"
    if graph["on_top_of"].get(n):
        return "on_" + graph["on_top_of"][n]
    return graph["in_zone"][n] or "table"


rows, saved, t0 = [], 0, time.time()


class _Names:                                                    # stands in for the env in replay mode
    object_names = ["red_block", "green_block", "blue_block", "yellow_block"]


def record(vlm, truth, img, view, base, said=None):
    if said is None:
        said = vlm.ask(img, view, skip=truth.get("held"))
    seen = vlm.to_graph(truth, said)
    rec = dict(base, view=view, said=said, held=truth.get("held"),
               truth={k: truth[k] for k in ("in_zone", "on_top_of", "clear", "off_table", "held")},
               goal_specs=vlm.goals,
               cubes={n: [place_of(truth, n), place_of(seen, n)] for n in truth["in_zone"]},
               goals=[[vlm._met(q, truth), vlm._met(q, seen)] for q in vlm.goals])
    rows.append(rec)
    log.write(json.dumps(rec) + "\n")
    log.flush()


if a.replay:
    from PIL import Image
    src = [json.loads(l) for l in open(a.replay)]
    vlm = VLMPerception(_Names(), {"goals": []}, 0, model=a.model, host=a.host, style=a.style,
                        cache_path=os.path.join(a.out, f"{tag}_cache.jsonl"))
    for i, r in enumerate(src):
        vlm.goals = r["goal_specs"]
        img = Image.open(os.path.join(os.path.dirname(a.replay), r["image"]))
        record(vlm, r["truth"], img, r["view"], {k: r[k] for k in ("family", "seed", "step", "image")})
        if i % 30 == 29:
            print(f"[vlm] {i+1}/{len(src)} looks, {time.time()-t0:.0f}s", flush=True)
else:
    for family in a.families.split(","):
        for e in range(a.episodes):
            seed = a.seed_base + e
            env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=0.19)
            env.reset(seed=seed)
            spec = sample(family, env.object_names, seed)
            ex = FrozenSkillExecutor(env, policy_path=a.policy, max_steps=100, seed=seed)
            iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=200, hide_claims=True)
            vlm = VLMPerception(env, spec, seed, model=a.model, host=a.host, style=a.style,
                                retract_first=not a.no_retract,
                                cache_path=os.path.join(a.out, f"{tag}_cache.jsonl"))
            step = 0
            for g in vlm.goals:
                obj, tgt = (g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])
                for _ in range(a.attempts):
                    iface.call("pick", {"obj": obj})
                    iface.call("place_on", {"target": tgt})
                    truth = scene_graph(env)
                    step += 1
                    for view in VIEWS:
                        img, said = vlm.look(view, skip=truth.get("held"))
                        name = f"img_{a.style}_{family}_{seed}_{step}_{view}.png"
                        if saved < a.save_images:
                            img.save(os.path.join(a.out, name))
                            saved += 1
                        record(vlm, truth, img, view, {"family": family, "seed": seed, "step": step, "image": name},
                               said=said or {})
                    if vlm._met(g, truth):
                        break
            env.close()
            print(f"[vlm] {family} {seed}: {step} states, {len(rows)} looks, {time.time()-t0:.0f}s, "
                  f"vlm calls {vlm.calls} ({vlm.seconds:.0f}s)", flush=True)

# ------------------------------------------------------------------ summary
out = [f"model {a.model} style {a.style} | {len(rows)} looks over {len(rows)//3} states | seeds {a.seed_base}+ x {a.episodes} per family"]
by = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])     # unmet, false succ., met, false fail., cubes, right, 'not visible'
for r in rows:
    for key in (r["view"], "any one view"):
        c = by[key]
        for was, now in r["goals"]:
            c[0] += not was
            c[1] += (not was) and now
            c[2] += was
            c[3] += was and not now
        for n, (t, s) in r["cubes"].items():
            if n != r["held"]:
                c[4] += 1
                c[5] += t == s
                c[6] += s == "not_visible"
states = defaultdict(dict)
for r in rows:
    states[(r["family"], r["seed"], r["step"])][r["view"]] = r["goals"]
c = by["majority of 3"]
for views in states.values():
    if len(views) < 3:
        continue
    for i in range(len(views["front"])):
        was = views["front"][i][0]
        now = sum(views[v][i][1] for v in VIEWS) >= 2
        c[0] += not was
        c[1] += (not was) and now
        c[2] += was
        c[3] += was and not now
# do the cameras err together?
both = defaultdict(lambda: [0, 0])
for views in states.values():
    if len(views) < 3:
        continue
    for i in range(len(views["front"])):
        wrong = [views[v][i][0] != views[v][i][1] for v in VIEWS]
        both[sum(wrong)][0] += 1
out.append(f"{'camera':<16}{'unmet goals':>12}{'false succ.':>13}{'met goals':>11}{'false fail.':>13}{'cube right':>11}"
           f"{'not seen':>10}{'wrong':>8}")
for key in list(VIEWS) + ["any one view", "majority of 3"]:
    u, fs, m, ff, n, ok, ns = by[key]
    out.append(f"{key:<16}{u:>12}{(fs/u if u else float('nan')):>13.3f}{m:>11}{(ff/m if m else float('nan')):>13.3f}"
               + (f"{ok/n:>11.3f}{ns/n:>10.3f}{(n-ok-ns)/n:>8.3f}" if n else ""))
tot = sum(v[0] for v in both.values())
out.append("goal readings by number of cameras that got it wrong: "
           + ", ".join(f"{k} wrong: {both[k][0]} ({both[k][0]/tot:.2f})" for k in sorted(both)))
text = "\n".join(out)
print(text)
with open(os.path.join(a.out, f"{tag}_summary.txt"), "w") as f:
    f.write(text + "\n")
