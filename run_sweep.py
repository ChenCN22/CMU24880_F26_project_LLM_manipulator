"""Run the architecture sweep.

Episodes are PAIRED: for a given (family, episode index) every agent sees the
same seed, so the same block layout and the same instruction.  Without that the
differences between A1-A4 would be partly differences between scenes.

One process holds one env and one ollama connection and walks the grid
agent x family x episode, writing one JSON line per episode plus a summary.
"""
import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

from agent_iface.executor import FrozenSkillExecutor
from agent_iface.tools import ToolInterface
from agents import graphs
from agents.llm import LLMBackbone
from envs.tabletop import MultiObjectTabletopEnv
from envs.task_specs import FAMILIES, sample

ap = argparse.ArgumentParser()
ap.add_argument("--agents", default="A1,A2,A3,A4")
ap.add_argument("--families", default=",".join(FAMILIES))
ap.add_argument("--episodes", type=int, default=20, help="episodes per (agent, family) cell")
ap.add_argument("--policy", default=None, help="frozen skill checkpoint; omit for the scripted stub")
ap.add_argument("--model", default="llama3.1:8b")
ap.add_argument("--host", default="http://localhost:11434")
ap.add_argument("--n-objects", type=int, default=4)
ap.add_argument("--observe-budget", type=int, default=8)
ap.add_argument("--max-calls", type=int, default=40)
ap.add_argument("--skill-steps", type=int, default=100)
ap.add_argument("--fence", type=float, default=0.19, help="half-width of the workspace rim; 0 disables it")
ap.add_argument("--seed-base", type=int, default=1000)
ap.add_argument("--p-drop", type=float, default=0.0)
ap.add_argument("--p-misplace", type=float, default=0.0)
ap.add_argument("--p-silent", type=float, default=0.0)
ap.add_argument("--memory-scope", default="family", choices=["family", "sweep"],
                help="A4: reset the lesson store per family (default) or let it run all sweep")
ap.add_argument("--out", default="results/sweep")
ap.add_argument("--save-transcripts", action="store_true")
a = ap.parse_args()

AGENTS = [s.strip() for s in a.agents.split(",") if s.strip()]
FAMS = [s.strip() for s in a.families.split(",") if s.strip()]
os.makedirs(a.out, exist_ok=True)
jsonl = open(os.path.join(a.out, "episodes.jsonl"), "a")

env = MultiObjectTabletopEnv(n_objects=a.n_objects, render_mode="rgb_array", fence=a.fence)
llm = LLMBackbone(model=a.model, host=a.host)

memory = None
if "A4" in AGENTS:
    from agents.memory import LessonMemory
    memory = LessonMemory(path=os.path.join(a.out, "memory"))

total = len(AGENTS) * len(FAMS) * a.episodes
agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
t_start = time.time()
done_n = 0

print(f"[sweep] {total} episodes | agents={AGENTS} families={FAMS} "
      f"policy={a.policy or 'scripted stub'} model={a.model}", flush=True)

for agent_name in AGENTS:
    for family in FAMS:
        if agent_name == "A4" and memory is not None and a.memory_scope == "family":
            memory.reset()
        for ep in range(a.episodes):
            seed = a.seed_base + ep                    # PAIRED across agents
            env.reset(seed=seed)
            spec = sample(family, env.object_names, seed)
            ex = FrozenSkillExecutor(env, policy_path=a.policy, max_steps=a.skill_steps,
                                     p_drop=a.p_drop, p_misplace=a.p_misplace,
                                     p_silent=a.p_silent, seed=seed)
            iface = ToolInterface(env, ex, spec, observe_budget=a.observe_budget,
                                  max_calls=a.max_calls)
            llm.reset_usage()
            graph = graphs.build(agent_name, iface, llm, memory)

            t0 = time.time()
            err = None
            try:
                graph.invoke(
                    {"instruction": spec["instruction"], "history": [], "reflections": [],
                     "memories": [], "steps": 0, "stop": False},
                    {"recursion_limit": a.max_calls * 3 + 20},
                )
            except Exception as e:                      # noqa: BLE001
                err = f"{type(e).__name__}: {e}"

            score = iface.score()
            row = {
                "agent": agent_name, "family": family, "episode": ep, "seed": seed,
                "instruction": spec["instruction"],
                **score, **llm.usage.as_dict(),
                "wall_seconds": round(time.time() - t0, 2),
                "error": err,
            }
            if a.save_transcripts:
                row["transcript"] = iface.transcript
            jsonl.write(json.dumps(row, default=str) + "\n")
            jsonl.flush()

            for k in ("success", "progress", "calls", "observes_used", "false_completion",
                      "silent_failures", "total_tokens", "llm_calls", "wall_seconds",
                      "loop_max", "undetected_silent_failures", "retries_after_lie"):
                agg[f"{agent_name}|{family}"][k].append(float(row[k]))
            if row["detection_latency"] is not None:
                agg[f"{agent_name}|{family}"]["detection_latency"].append(float(row["detection_latency"]))
            agg[agent_name][
                "success"].append(float(row["success"]))

            done_n += 1
            if done_n % 5 == 0 or done_n == total:
                el = time.time() - t_start
                eta = el / done_n * (total - done_n)
                print(f"[sweep] {done_n}/{total}  elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)

env.close()
jsonl.close()

# ------------------------------------------------------------------ summary
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

lines = [f"{'cell':<28}{'succ':>6}{'prog':>7}{'calls':>7}{'obs':>6}"
         f"{'falseD':>8}{'silent':>8}{'undet':>7}{'lat':>6}{'loop':>6}{'retry':>7}"
         f"{'tokens':>9}{'sec':>7}"]
for agent_name in AGENTS:
    for family in FAMS:
        c = agg[f"{agent_name}|{family}"]
        lines.append(
            f"{agent_name + ' ' + family:<28}"
            f"{mean(c['success']):>6.2f}{mean(c['progress']):>7.2f}{mean(c['calls']):>7.1f}"
            f"{mean(c['observes_used']):>6.1f}{mean(c['false_completion']):>8.2f}"
            f"{mean(c['silent_failures']):>8.1f}"
            f"{mean(c['undetected_silent_failures']):>7.1f}"
            f"{mean(c['detection_latency']):>6.1f}{mean(c['loop_max']):>6.1f}"
            f"{mean(c['retries_after_lie']):>7.1f}"
            f"{mean(c['total_tokens']):>9.0f}{mean(c['wall_seconds']):>7.1f}")
    lines.append(f"{agent_name + ' OVERALL':<28}{mean(agg[agent_name]['success']):>6.2f}")
summary = "\n".join(lines)
print("\n" + summary, flush=True)
with open(os.path.join(a.out, "summary.txt"), "w") as f:
    f.write(summary + "\n")
print(f"\n[sweep] wrote {a.out}/episodes.jsonl and summary.txt "
      f"in {(time.time()-t_start)/60:.1f} min", flush=True)
