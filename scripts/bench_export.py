"""usage: bench_export.py [--results results] [--holdout <folder of the first hold-out run>] [--out data/benchmark_episodes.csv]

Collects the main benchmark from the run folders into one table, one line per episode:
3 stated + 4 free-choice task families, hold-out seeds 2000-2039, camera right and one bad camera,
every agent and language model that was run.  The transcripts stay in the run folders.

Where each cell comes from (all under --results unless noted):
  scripted rules, qwen2.5:14b A2/A5 on stated tasks   --holdout/{base,A2,A5}     (first hold-out run)
  qwen2.5:14b A9, A10r-none on stated tasks           a10r_holdout/{A9,A10r-none}
  qwen2.5:14b, everything else                        bench/qwen2.5_14b/<agent>
  other models, A2/A5/A9 on stated tasks              models/<model>/<agent>     (not for the Claude models and gpt-6-sol)
  other models, everything else                       bench/<model>/<cell>
"""
import argparse
import csv
import glob
import json
import os

STATED = ["ordered_stack", "blocked_rearrange", "constrained_place"]
FREE = ["free_two_zones", "free_three_zones", "free_stack", "free_two_stacks"]
AGENTS = ["naive", "trust", "vote", "oracle", "A2", "A5", "A9", "A10r-none", "A10-none", "A11-none", "A11-lessons"]
# name in the table: folders, later ones win
MODELS = {
    "qwen2.5:14b": ["@holdout", "a10r_holdout", "bench/qwen2.5_14b"],
    "gemma3:12b": ["models/gemma3_12b", "bench/gemma3_12b"],
    "llama3.1:8b": ["models/llama3.1_8b", "bench/llama3.1_8b"],
    "qwen3.8:27b": ["models/qwen3.8_27b", "bench/qwen3.8_27b"],
    "deepseek-r1:32b": ["models/deepseek-r1_32b", "bench/deepseek-r1_32b"],
    "gpt-6-luna": ["models/openai_gpt-6-luna"],
    "gpt-6.1-terra": ["models/openai_gpt-6.1-terra"],
    "gpt-6-sol": ["bench/openai_gpt-6-sol"],
    "claude-haiku-5-5": ["bench/anthropic_claude-haiku-5-5"],
    "claude-sonnet-5-5": ["bench/anthropic_claude-sonnet-5-5"],
}
SCRIPTS = ("naive", "trust", "vote", "oracle")
COLUMNS = ["model", "agent", "camera", "task_type", "family", "seed", "success", "n_goals", "n_goals_met", "tool_calls",
           "looks", "declared_done", "llm_calls", "input_tokens", "output_tokens", "malformed_answers"]

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--holdout", default=os.path.expanduser("~/24880_term_project/results/holdout"))
ap.add_argument("--out", default="data/benchmark_episodes.csv")
a = ap.parse_args()

rows = {}
for model, folders in MODELS.items():
    for d in folders:
        d = a.holdout if d == "@holdout" else os.path.join(a.results, d)
        for f in sorted(glob.glob(d + "/*/episodes.jsonl")):
            if "/smoke/" in f:
                continue
            for line in open(f):
                r = json.loads(line)
                if (r.get("info", "none") != "none" or r["perception"] not in ("none", "persistent") or r.get("error")
                        or r["family"] not in STATED + FREE or not 2000 <= r["seed"] < 2040 or r["agent"] not in AGENTS
                        or r.get("motion", "low") != "low"):
                    continue
                if r["family"] in FREE and r["perception"] != "none":     # the error model injects nothing there
                    continue
                m = "script" if r["agent"] in SCRIPTS else model
                rows[(m, r["agent"], r["perception"], r["family"], r["seed"])] = [
                    m, r["agent"], "right" if r["perception"] == "none" else "one_bad",
                    "stated" if r["family"] in STATED else "free", r["family"], r["seed"], int(bool(r["success"])),
                    r["n_goals"], r["n_goals_met"], r["calls"], r.get("observes_used", ""), int(bool(r.get("declared_done"))),
                    r.get("llm_calls", 0), r.get("prompt_tokens", 0), r.get("output_tokens", 0), r.get("malformed_json", 0)]
order = {m: i for i, m in enumerate(["script"] + list(MODELS))}
out = sorted(rows.values(), key=lambda x: (order[x[0]], AGENTS.index(x[1]), x[2] != "right", (STATED + FREE).index(x[4]), x[5]))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLUMNS)
    w.writerows(out)
print(f"{len(out)} episodes -> {a.out}")
