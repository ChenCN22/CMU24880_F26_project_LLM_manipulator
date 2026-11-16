"""Analysis of the perception-error study.

    python analyze_perception.py results/perception_qwen14b

Reads episodes.jsonl and writes analysis.json + analysis.txt next to it.

Episode level   success, false completion, cost; each faulted cell against its
                same-seed fault-free twin (paired bootstrap on the difference).
Event level     every camera answer that misreported a goal is one event:
                  cross_checked  the agent looked again from another view before acting
                  adopted        its next move followed the wrong report
                                 (false success -> done / moved on;
                                  false failure -> picked the finished block up again)
Fault-free      how often an agent undoes work that was actually correct.
Warning         confusion matrix of the critic warning over all skill calls.
"""
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

LOOKS, ACTS = ("observe", "check_constraints"), ("pick", "place_on", "push")
rng = np.random.default_rng(0)


def goal_met(g, rep) -> Optional[bool]:
    """Goal status according to a camera answer; None if the answer cannot tell."""
    if not isinstance(rep, dict):
        return None
    if "object" in rep:                                    # where:<obj>
        o = rep["object"]
        if g["type"] == "in_zone" and g["obj"] == o:
            return rep.get("in_zone") == g["zone"]
        if g["type"] == "on_top_of" and g["top"] == o:
            return rep.get("on_top_of") == g["bottom"]
        return None
    if g["type"] == "in_zone" and "in_zone" in rep:
        return rep["in_zone"].get(g["obj"]) == g["zone"]
    if g["type"] == "on_top_of" and "on_top_of" in rep:
        return rep["on_top_of"].get(g["top"]) == g["bottom"]
    return None


def goals_of(instr_row) -> List[Dict[str, Any]]:
    return instr_row["_goals"]


def mover(g) -> str:
    return g["obj"] if g["type"] == "in_zone" else g["top"]


def events(row) -> List[Dict[str, Any]]:
    """One event per (faulted look, goal it misreported)."""
    tr, out = row["transcript"], []
    for k, t in enumerate(tr):
        if t["tool"] != "observe" or not t.get("fault"):
            continue
        for g in row["_goals"]:
            shown, real = goal_met(g, t["visible"].get("result")), goal_met(g, t["true_answer"])
            if shown is None or shown == real:
                continue
            later = tr[k + 1:]
            nxt = next((u for u in later if u["tool"] not in LOOKS), None)
            before = later[:later.index(nxt)] if nxt is not None else later
            view = t["visible"].get("view")
            crossed = any(u["tool"] in LOOKS and u["visible"].get("view") not in (None, view)
                          for u in before)
            redo = nxt is not None and nxt["tool"] == "pick" and nxt["args"].get("obj") == mover(g)
            if shown:                                      # false success
                adopted = nxt is not None and not redo     # done, or went on to something else
            else:                                          # false failure
                adopted = redo
            out.append({"type": t["fault"]["type"], "direction": "false_success" if shown else "false_failure",
                        "cross_checked": crossed, "adopted": bool(adopted),
                        "ended_without_move": nxt is None, "success": bool(row["success"])})
    return out


def undone(row) -> int:
    """Times the agent picked up a block that a truthful placement had just put on its goal."""
    target = {mover(g): (g["zone"] if g["type"] == "in_zone" else g["bottom"]) for g in row["_goals"]}
    placed, held, n = set(), None, 0
    for t in row["transcript"]:
        if t["tool"] == "pick":
            o = t["args"].get("obj")
            if o in placed:
                n += 1
                placed.discard(o)
            held = o if t["truth"] else None
        elif t["tool"] == "place_on":
            if held and t["truth"] and target.get(held) == t["args"].get("target"):
                placed.add(held)
            held = None
    return n


def wilson(k: int, n: int):
    if n == 0:
        return (None, None)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(float(c - h), 2), round(float(c + h), 2))


def paired(a: List[float], b: List[float]):
    """mean(b) - mean(a) over paired episodes, with a bootstrap 95% interval."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if len(d) == 0:
        return None
    boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return {"n": len(d), "diff": round(float(d.mean()), 3),
            "ci": [round(float(np.quantile(boot, 0.025)), 3), round(float(np.quantile(boot, 0.975)), 3)]}


def main(out_dir: str) -> None:
    import os
    sys.path.insert(0, ".")
    from envs.task_specs import sample
    NAMES = ["red_block", "green_block", "blue_block", "yellow_block"]
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "episodes.jsonl")) if l.strip()]
    for r in rows:
        r["_goals"] = sample(r["family"], NAMES, r["seed"])["goals"]
        assert sample(r["family"], NAMES, r["seed"])["instruction"] == r["instruction"]

    cell = defaultdict(list)
    for r in rows:
        cell[(r["agent"], r["perception"], r["info"])].append(r)
    by_key = {(r["agent"], r["perception"], r["info"], r["family"], r["episode"]): r for r in rows}

    res: Dict[str, Any] = {"cells": [], "paired_vs_fault_free": [], "paired_info": [], "events": [],
                           "paired_vs_A2": []}
    L: List[str] = []
    L.append(f"{'agent':<8}{'perception':<12}{'info':<9}{'n':>4}{'success':>9}{'95% CI':>16}{'done':>6}"
             f"{'falseD':>8}{'calls':>7}{'looks':>7}{'faults':>8}{'undo':>6}{'tokens':>8}")
    for key in sorted(cell):
        rs = cell[key]
        n, k = len(rs), sum(r["success"] for r in rs)
        m = lambda f: float(np.mean([r.get(f, 0) or 0 for r in rs]))          # noqa: E731
        c = {"agent": key[0], "perception": key[1], "info": key[2], "n": n, "success": round(k / n, 3),
             "ci": wilson(k, n), "declared_done": round(m("declared_done"), 3),
             "false_completion": round(m("false_completion"), 3), "calls": round(m("calls"), 2),
             "looks": round(m("observes_used"), 2), "faults_shown": round(m("faults_shown"), 2),
             "undone_correct_work": round(float(np.mean([undone(r) for r in rs])), 3),
             "tokens": round(m("total_tokens")), "seconds": round(m("wall_seconds"), 1)}
        res["cells"].append(c)
        L.append(f"{key[0]:<8}{key[1]:<12}{key[2]:<9}{n:>4}{c['success']:>9.2f}{str(c['ci']):>16}"
                 f"{c['declared_done']:>6.2f}{c['false_completion']:>8.2f}{c['calls']:>7.1f}{c['looks']:>7.1f}"
                 f"{c['faults_shown']:>8.2f}{c['undone_correct_work']:>6.2f}{c['tokens']:>8}")

    def pairs(ka, kb, field="success"):
        a, b = [], []
        for (ag, pm, inf, fam, ep), r in by_key.items():
            if (ag, pm, inf) == ka and (kb + (fam, ep)) in by_key:
                a.append(float(r[field])); b.append(float(by_key[kb + (fam, ep)][field]))
        return paired(a, b)

    L.append("\nSuccess against the same-seed fault-free twin (faulted minus fault-free)")
    for (ag, pm, inf) in sorted(cell):
        if pm == "none" or (ag, "none", inf) not in cell:
            continue
        p = pairs((ag, "none", inf), (ag, pm, inf))
        res["paired_vs_fault_free"].append({"agent": ag, "perception": pm, "info": inf, **p})
        L.append(f"  {ag:<7}{pm:<12}{inf:<9} n={p['n']:<4} diff {p['diff']:+.3f}  95% CI {p['ci']}")

    L.append("\nEffect of information at a fixed perception level (richer minus 'none')")
    for (ag, pm, inf) in sorted(cell):
        if inf == "none" or (ag, pm, "none") not in cell:
            continue
        for field in ("success", "calls"):
            p = pairs((ag, pm, "none"), (ag, pm, inf), field)
            res["paired_info"].append({"agent": ag, "perception": pm, "info": inf, "field": field, **p})
            L.append(f"  {ag:<7}{pm:<12}{inf:<9}{field:<8} n={p['n']:<4} diff {p['diff']:+.3f}  95% CI {p['ci']}")

    L.append("\nEach other agent against plain ReAct on the same seeds (agent minus A2)")
    for (ag, pm, inf) in sorted(cell):
        if ag == "A2" or not ag.startswith("A") or ("A2", pm, inf) not in cell:
            continue
        for field in ("success", "calls"):
            p = pairs(("A2", pm, inf), (ag, pm, inf), field)
            res["paired_vs_A2"].append({"agent": ag, "perception": pm, "info": inf, "field": field, **p})
            L.append(f"  {ag:<7}{pm:<12}{inf:<9}{field:<8} n={p['n']:<4} diff {p['diff']:+.3f}  95% CI {p['ci']}")

    L.append("\nWhat the agent did with a wrong report (one event per misreported goal)")
    L.append(f"  {'agent':<7}{'perception':<12}{'info':<9}{'direction':<15}{'events':>7}{'cross-chk':>10}"
             f"{'adopted':>9}{'95% CI':>16}{'ep. success':>12}")
    for key in sorted(cell):
        if not key[0].startswith("A"):
            continue
        ev = [e for r in cell[key] for e in events(r)]
        for direction in ("false_success", "false_failure"):
            es = [e for e in ev if e["direction"] == direction]
            if not es:
                continue
            n, ad = len(es), sum(e["adopted"] for e in es)
            rec = {"agent": key[0], "perception": key[1], "info": key[2], "direction": direction, "events": n,
                   "cross_checked": round(float(np.mean([e["cross_checked"] for e in es])), 3),
                   "adopted": round(ad / n, 3), "adopted_ci": wilson(ad, n),
                   "episode_success": round(float(np.mean([e["success"] for e in es])), 3)}
            res["events"].append(rec)
            L.append(f"  {key[0]:<7}{key[1]:<12}{key[2]:<9}{direction:<15}{n:>7}{rec['cross_checked']:>10.2f}"
                     f"{rec['adopted']:>9.2f}{str(rec['adopted_ci']):>16}{rec['episode_success']:>12.2f}")

    w = {k: sum(r.get(k, 0) for r in rows) for k in ("warn_tp", "warn_fp", "warn_fn", "warn_tn")}
    fail, ok = w["warn_tp"] + w["warn_fn"], w["warn_fp"] + w["warn_tn"]
    res["warning"] = {**w, "tpr": round(w["warn_tp"] / fail, 3) if fail else None,
                      "tnr": round(w["warn_tn"] / ok, 3) if ok else None}
    L.append(f"\nCritic warning over all skill calls: {res['warning']}")
    errs = sorted({r["error"] for r in rows if r["error"]})
    L.append(f"episodes {len(rows)} | errors {errs or 'none'}")

    text = "\n".join(L)
    print(text)
    with open(os.path.join(out_dir, "analysis.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/perception_qwen14b")
