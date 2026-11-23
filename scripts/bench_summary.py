"""usage: bench_summary.py [data/benchmark_episodes.csv] [--seeds 25] [--md]

The tables of the main benchmark from the per-episode table (scripts/bench_export.py):
  1. per model: successes of every agent on the stated tasks (camera right, one bad camera) and the
     free-choice tasks, per family, with paired differences;
  2. per model: tool calls, model calls and tokens per episode.
Only the first --seeds seeds of every family are used (25: the set every model has; qwen2.5:14b and
the scripted rules have 40).  A difference is the mean over the episodes both agents have (same
model, camera, family and seed) with a 95% bootstrap interval, in points of task success.
--md prints Markdown tables.
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

STATED = ["ordered_stack", "blocked_rearrange", "constrained_place"]
FREE = ["free_two_zones", "free_three_zones", "free_stack", "free_two_stacks"]
AGENTS = ["naive", "trust", "vote", "oracle", "A2", "A5", "A9", "A10r-none", "A10-none", "A11-none", "A11-lessons"]
PAIRS = [("free", "right", "A10-none", "A9"), ("free", "right", "A10-none", "A2"), ("free", "right", "A9", "A2"),
         ("free", "right", "A11-lessons", "A11-none"), ("free", "right", "A11-lessons", "A10-none"),
         ("stated", "right", "A9", "A2"), ("stated", "right", "A10r-none", "A9"), ("stated", "right", "A10-none", "A9"),
         ("stated", "one_bad", "A9", "A2"), ("stated", "one_bad", "A9", "A5"), ("stated", "one_bad", "A5", "A2"),
         ("stated", "one_bad", "A10r-none", "A9"), ("stated", "one_bad", "A11-lessons", "A9")]

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default="data/benchmark_episodes.csv")
ap.add_argument("--seeds", type=int, default=25)
ap.add_argument("--md", action="store_true")
a = ap.parse_args()

data = defaultdict(dict)        # (model, agent, camera) -> {(family, seed): row}
models = []
for r in csv.DictReader(open(a.csv)):
    if int(r["seed"]) < 2000 + a.seeds:
        data[(r["model"], r["agent"], r["camera"])][(r["family"], int(r["seed"]))] = r
        if r["model"] not in models:
            models.append(r["model"])
rng = np.random.default_rng(0)


def cell(d, fams):
    rs = [r for (f, _), r in d.items() if f in fams]
    return f"{sum(int(r['success']) for r in rs)}/{len(rs)}" if rs else "-"


def diff(d, b, fams):
    keys = sorted(k for k in d if k in b and k[0] in fams)
    if len(keys) < 15:
        return None
    x = np.array([int(d[k]["success"]) - int(b[k]["success"]) for k in keys], dtype=float)
    bt = rng.choice(x, size=(10000, len(x))).mean(axis=1)
    return f"{100 * x.mean():+.1f} [{100 * np.quantile(bt, .025):+.1f}, {100 * np.quantile(bt, .975):+.1f}] (n={len(keys)})"


def table(head, lines):
    if a.md:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for ln in lines:
            print("| " + " | ".join(ln) + " |")
    else:
        w = [max(len(str(x)) for x in col) for col in zip(head, *lines)]
        for ln in [head] + lines:
            print("  ".join(str(x).ljust(n) for x, n in zip(ln, w)))


for m in models:
    print(f"\n{'### ' if a.md else '=== '}{m}: successes / episodes, seeds 2000-{2000 + a.seeds - 1}\n")
    lines = []
    for ag in AGENTS:
        n, p = data.get((m, ag, "right"), {}), data.get((m, ag, "one_bad"), {})
        if n or p:
            lines.append([ag, cell(n, STATED), cell(p, STATED), cell(n, FREE)] + [cell(n, [f]) for f in STATED]
                         + [cell(p, [f]) for f in STATED] + [cell(n, [f]) for f in FREE])
    table(["agent", "stated, camera right", "stated, one bad camera", "free"] + [f + " (right)" for f in STATED]
          + [f + " (one bad)" for f in STATED] + FREE, lines)
    lines = []
    for typ, cam, x, y in PAIRS:
        v = diff(data.get((m, x, cam), {}), data.get((m, y, cam), {}), STATED if typ == "stated" else FREE)
        if v:
            lines.append([f"{typ}, camera {'right' if cam == 'right' else 'one bad'}", f"{x} - {y}", v])
    if lines:
        print()
        table(["tasks", "difference", "points [95% interval]"], lines)
    if m == "script":
        continue
    lines = []
    for ag in AGENTS:
        ln = [ag]
        for cam, fams in (("right", STATED), ("one_bad", STATED), ("right", FREE)):
            rs = [r for (f, _), r in data.get((m, ag, cam), {}).items() if f in fams]
            mean = lambda k: np.mean([float(r[k] or 0) for r in rs])
            ln.append(f"{mean('tool_calls'):.1f} / {mean('llm_calls'):.1f} / {mean('input_tokens'):.0f} / {mean('output_tokens'):.0f}"
                      if rs else "-")
        if ln[1:] != ["-"] * 3:
            lines.append(ln)
    print("\nper episode: tool calls / model calls / input tokens / output tokens\n")
    table(["agent", "stated, camera right", "stated, one bad camera", "free"], lines)
