"""The perception-error study.

Grid:  perception error  x  information given to the agent  x  agent  x  family  x  episode

  perception   none | transient | persistent | random(control)
  info         none | label | monitor
  agent        A2 (ReAct) | A5 (ReAct + forced cross-check)
               A6 (+ self-assessment) | A7 (+ belief state) | A8 (+ independent reviewer)
               plus scripted reference policies that need no LLM:
                 naive   one attempt per goal, never looks
                 trust   believe one camera report
                 vote    look from all three cameras, take the majority
                 oracle  read the true scene

Episodes are PAIRED on seed: for a (family, episode) every cell gets the same
block layout and instruction, and the perception=none cell of the same agent and
info level is that episode's fault-free twin.

The run is resumable: rows already in episodes.jsonl are skipped.
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List

from agent_iface.critic import load as load_calib
from agent_iface.executor import FrozenSkillExecutor
from agent_iface.perception import VIEWS, PerceptionModel
from agent_iface.tools import ToolInterface
from envs.scene_graph import _goal_met, scene_graph
from envs.tabletop import MultiObjectTabletopEnv
from envs.task_specs import sample

ap = argparse.ArgumentParser()
ap.add_argument("--agents", default="A2,A5")
ap.add_argument("--baselines", default="naive,trust,vote,oracle")
ap.add_argument("--perception", default="none,transient,persistent")
ap.add_argument("--info", default="none,label,monitor")
ap.add_argument("--families", default="ordered_stack,blocked_rearrange,constrained_place")
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--p-false-success", type=float, default=0.30)
ap.add_argument("--p-false-failure", type=float, default=0.10)
ap.add_argument("--p-drop", type=float, default=0.0, help="executor: chance per skill call of dropping the block")
ap.add_argument("--p-misplace", type=float, default=0.0, help="executor: chance per skill call of a shifted goal")
ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
ap.add_argument("--calib", default="checkpoints/critic_calib.json")
ap.add_argument("--model", default="qwen2.5:14b")
ap.add_argument("--host", default="http://localhost:11434")
ap.add_argument("--max-calls", type=int, default=40)
ap.add_argument("--seed-base", type=int, default=1000)
ap.add_argument("--out", default="results/perception")
a = ap.parse_args()

split = lambda s: [x.strip() for x in s.split(",") if x.strip()]      # noqa: E731
AGENTS, BASES, PERC, INFO, FAMS = map(split, (a.agents, a.baselines, a.perception, a.info, a.families))
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, "episodes.jsonl")
KEY = ("agent", "perception", "info", "family", "episode")
have = set()
if os.path.exists(path):
    with open(path) as f:
        have = {tuple(json.loads(l)[k] for k in KEY) for l in f if l.strip()}
jsonl = open(path, "a")

calib = load_calib(a.calib) if a.calib and os.path.exists(a.calib) else None
# a new simulator for every episode: nothing carries over, so (seed, tool calls) fixes the physics
make_env = lambda: MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=0.19)   # noqa: E731
env = None
llm = None
if AGENTS:
    from agents import graphs
    from agents.llm import LLMBackbone
    from agents.nodes import SYSTEM_V2
    llm = LLMBackbone(model=a.model, host=a.host)


def run_scripted(iface: ToolInterface, spec: Dict[str, Any], how: str) -> None:
    def believes(goal) -> bool:
        if how == "oracle":
            return _goal_met(goal, scene_graph(env))
        votes = []
        for view in (VIEWS if how == "vote" else VIEWS[:1]):
            r = iface.call("observe", {"query": "scene", "view": view})
            votes.append(bool(r.get("ok")) and _goal_met(goal, r["result"]))
        return sum(votes) * 2 > len(votes)

    for g in spec["goals"]:
        obj, tgt = (g["obj"], g["zone"]) if g["type"] == "in_zone" else (g["top"], g["bottom"])
        for _ in range(1 if how == "naive" else 4):
            iface.call("pick", {"obj": obj})
            iface.call("place_on", {"target": tgt})
            if how == "naive" or believes(g):
                break
    iface.call("done")


# info does not reach a scripted policy, so those run once per perception level
cells = [(ag, pm, inf) for ag in AGENTS for pm in PERC for inf in INFO]
cells += [(b, pm, "none") for b in BASES for pm in PERC]
total = len(cells) * len(FAMS) * a.episodes
print(f"[perception] {total} episodes ({len(have)} already done) | agents={AGENTS} baselines={BASES} "
      f"perception={PERC} info={INFO} families={FAMS} model={a.model} "
      f"critic={'yes' if calib else 'NO CALIBRATION'}", flush=True)

t_start, n = time.time(), 0
for agent_name, pmode, info in cells:
    for family in FAMS:
        for ep in range(a.episodes):
            n += 1
            if (agent_name, pmode, info, family, ep) in have:
                continue
            seed = a.seed_base + ep
            env = make_env()
            env.reset(seed=seed)
            spec = sample(family, env.object_names, seed)
            ex = FrozenSkillExecutor(env, policy_path=a.policy, max_steps=100, seed=seed,
                                     p_drop=a.p_drop, p_misplace=a.p_misplace)
            iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=a.max_calls,
                                  perception=PerceptionModel(spec, seed, pmode, a.p_false_success,
                                                             a.p_false_failure),
                                  info=info, critic_calib=calib, hide_claims=True)
            t0, err, usage = time.time(), None, {}
            try:
                if agent_name in BASES:
                    run_scripted(iface, spec, agent_name)
                else:
                    llm.reset_usage()
                    graphs.build(agent_name, iface, llm, None, SYSTEM_V2).invoke(
                        {"instruction": spec["instruction"], "history": [], "reflections": [],
                         "memories": [], "steps": 0, "stop": False},
                        {"recursion_limit": a.max_calls * 4 + 20})
                    usage = llm.usage.as_dict()
            except Exception as e:                      # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            for t in iface.transcript:                   # the per-call scene dump is not needed here
                t.pop("graph", None)
            row = {"agent": agent_name, "perception": pmode, "info": info, "family": family,
                   "episode": ep, "seed": seed, "instruction": spec["instruction"],
                   **iface.score(), **usage, "wall_seconds": round(time.time() - t0, 2),
                   "error": err, "transcript": iface.transcript}
            jsonl.write(json.dumps(row, default=str) + "\n")
            jsonl.flush()
            env.close()
            if n % 20 == 0 or n == total:
                el = time.time() - t_start
                print(f"[perception] {n}/{total}  elapsed {el/60:.1f}m", flush=True)
jsonl.close()

# ------------------------------------------------------------------ summary
rows: List[Dict[str, Any]] = [json.loads(l) for l in open(path) if l.strip()]
agg: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
for r in rows:
    c = agg[(r["agent"], r["perception"], r["info"])]
    for k in ("success", "progress", "calls", "observes_used", "false_completion", "faults_shown",
              "failed_actions", "loop_max", "wall_seconds"):
        c[k].append(float(r[k]))
    c["tokens"].append(float(r.get("total_tokens", 0)))
    c["done"].append(float(r["declared_done"]))
mean = lambda xs: sum(xs) / len(xs) if xs else 0.0                    # noqa: E731
lines = [f"{'agent':<8}{'perception':<12}{'info':<9}{'n':>4}{'succ':>6}{'prog':>6}{'done':>6}"
         f"{'falseD':>8}{'calls':>7}{'looks':>7}{'faults':>8}{'fails':>7}{'loop':>6}{'tokens':>8}{'sec':>6}"]
for (ag, pm, inf), c in sorted(agg.items()):
    lines.append(f"{ag:<8}{pm:<12}{inf:<9}{len(c['success']):>4}{mean(c['success']):>6.2f}"
                 f"{mean(c['progress']):>6.2f}{mean(c['done']):>6.2f}{mean(c['false_completion']):>8.2f}"
                 f"{mean(c['calls']):>7.1f}{mean(c['observes_used']):>7.1f}{mean(c['faults_shown']):>8.2f}"
                 f"{mean(c['failed_actions']):>7.2f}"
                 f"{mean(c['loop_max']):>6.1f}{mean(c['tokens']):>8.0f}{mean(c['wall_seconds']):>6.1f}")
errs = Counter(r["error"] for r in rows if r["error"])
summary = "\n".join(lines) + (f"\nerrors: {dict(errs)}" if errs else "\nerrors: none")
print("\n" + summary, flush=True)
with open(os.path.join(a.out, "summary.txt"), "w") as f:
    f.write(summary + "\n")
print(f"[perception] wrote {path} in {(time.time()-t_start)/60:.1f} min", flush=True)
