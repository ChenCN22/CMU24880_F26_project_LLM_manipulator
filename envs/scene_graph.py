"""Ground-truth symbolic state for the multi-object tabletop.

Two separate jobs, deliberately kept apart:

  * `scene_graph(env)` reads the simulator and returns the TRUE symbolic state.
    Scoring uses this and only this.  The agent never sees it directly.
  * `observe(graph, query)` answers a restricted set of questions about that
    graph.  This is what the agent buys with its perception budget.

Keeping them apart is what lets an episode be scored against reality rather
than against the agent's own report.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from .tabletop import OBJECT_SIZE, ZONE_HALF, ZONES

# table top is at z = 0; table spans x in [-0.85, 0.25], y in [-0.35, 0.35]
TABLE_X = (-0.85, 0.25)
TABLE_Y = (-0.35, 0.35)
STACK_XY_TOL = 0.6 * OBJECT_SIZE      # how aligned two blocks must be to count as stacked
STACK_Z_TOL = 0.4 * OBJECT_SIZE       # tolerance on the one-block z step
GRASP_DIST = 0.055                    # ee-to-object distance that counts as held
GRIPPER_CLOSED = 0.055                # fingers_width below this == closed on something


def _zone_of(pos: np.ndarray) -> Optional[str]:
    for zname, xy in ZONES.items():
        if abs(pos[0] - xy[0]) <= ZONE_HALF and abs(pos[1] - xy[1]) <= ZONE_HALF:
            return zname
    return None


def scene_graph(env) -> Dict[str, Any]:
    """Read the TRUE symbolic state out of the simulator."""
    sim = env.sim
    names: List[str] = list(env.object_names)
    pos = {n: np.asarray(sim.get_base_position(n), dtype=float) for n in names}
    ee = np.asarray(env.robot.get_ee_position(), dtype=float)
    fingers = float(env.robot.get_fingers_width())

    off_table, on_table = [], []
    for n in names:
        p = pos[n]
        if p[2] < -0.05 or not (TABLE_X[0] <= p[0] <= TABLE_X[1]) or not (TABLE_Y[0] <= p[1] <= TABLE_Y[1]):
            off_table.append(n)
        else:
            on_table.append(n)

    in_zone = {n: _zone_of(pos[n]) for n in on_table}

    # stacking: b is directly on a if xy-aligned and exactly one block higher
    on_top_of: Dict[str, Optional[str]] = {n: None for n in names}
    for top in on_table:
        for bottom in on_table:
            if top == bottom:
                continue
            dxy = np.linalg.norm(pos[top][:2] - pos[bottom][:2])
            dz = pos[top][2] - pos[bottom][2]
            if dxy <= STACK_XY_TOL and abs(dz - OBJECT_SIZE) <= STACK_Z_TOL:
                on_top_of[top] = bottom
                break
    supported = {b for b in on_top_of.values() if b is not None}

    held = None
    if fingers < GRIPPER_CLOSED:
        cands = [(np.linalg.norm(pos[n] - ee), n) for n in on_table]
        if cands:
            d, n = min(cands)
            if d <= GRASP_DIST:
                held = n

    return {
        "objects": names,
        "positions": {n: pos[n].round(4).tolist() for n in names},
        "zones": {z: ZONES[z].round(4).tolist() for z in ZONES},
        "in_zone": in_zone,
        "on_top_of": on_top_of,
        "clear": {n: (n not in supported) for n in on_table},
        "off_table": off_table,
        "held": held,
        "ee_position": ee.round(4).tolist(),
        "fingers_width": round(fingers, 4),
    }


# --------------------------------------------------------------- goal checking
def _goal_met(g: Dict[str, Any], graph: Dict[str, Any]) -> bool:
    t = g["type"]
    if t == "in_zone":
        return graph["in_zone"].get(g["obj"]) == g["zone"]
    if t == "on_top_of":
        return graph["on_top_of"].get(g["top"]) == g["bottom"]
    if t == "clear":
        return bool(graph["clear"].get(g["obj"], False))
    if t == "on_table":
        return g["obj"] not in graph["off_table"]
    raise ValueError(f"unknown goal type {t!r}")


def _violated(c: Dict[str, Any], graph: Dict[str, Any]) -> bool:
    t = c["type"]
    if t == "not_on":                       # top must not sit on bottom
        return graph["on_top_of"].get(c["top"]) == c["bottom"]
    if t == "not_in_zone":
        return graph["in_zone"].get(c["obj"]) == c["zone"]
    if t == "keep_on_table":
        return c["obj"] in graph["off_table"]
    raise ValueError(f"unknown constraint type {t!r}")


def evaluate(graph: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Score an episode against ground truth.  Returns task success plus the
    partial-subgoal progress the proposal reports alongside it."""
    goals = spec.get("goals", [])
    met = [_goal_met(g, graph) for g in goals]
    viols = [c for c in spec.get("constraints", []) if _violated(c, graph)]
    return {
        "success": bool(goals) and all(met) and not viols,
        "n_goals": len(goals),
        "n_goals_met": int(sum(met)),
        "progress": (sum(met) / len(goals)) if goals else 0.0,
        "goals_met": met,
        "violations": viols,
        "objects_off_table": list(graph["off_table"]),
    }


def violations(graph: Dict[str, Any], spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [c for c in spec.get("constraints", []) if _violated(c, graph)]


# ------------------------------------------------------------------- observe
OBSERVE_QUERIES = (
    "scene",            # whole symbolic graph (most expensive thing to ask for)
    "zones",            # which object is in which zone
    "stacks",           # on_top_of relations
    "held",             # what the gripper is holding, if anything
    "where:<obj>",      # one object's zone + clear + on_top_of
)


def describe(graph: Dict[str, Any]) -> List[str]:
    """The same scene, one plain sentence per block.  Says nothing a query could
    not already expose and nothing about the task."""
    out = []
    for obj in graph["objects"]:
        if obj in graph.get("hidden", ()):
            out.append(f"{obj}: not detected by this camera")
            continue
        if obj in graph["off_table"]:
            out.append(f"{obj}: off the table")
            continue
        below, zone = graph["on_top_of"].get(obj), graph["in_zone"].get(obj)
        if graph.get("held") == obj:
            where = "held in the gripper"
        elif below:
            where = f"on top of {below}"
        elif zone:
            where = f"on the table inside {zone}"
        else:
            where = "on the table, not inside any zone, not on any block"
        above = [t for t, b in graph["on_top_of"].items() if b == obj]
        out.append(f"{obj}: {where}" + (f"; {above[0]} is on top of it" if above else ""))
    return out


def observe(graph: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Answer one restricted question about the graph.  Anything the agent does
    not ask about, it does not learn."""
    q = query.strip()
    if q == "scene":
        return {k: graph[k] for k in ("in_zone", "on_top_of", "clear", "off_table", "held")}
    if q == "zones":
        return {"in_zone": graph["in_zone"]}
    if q == "stacks":
        return {"on_top_of": graph["on_top_of"], "clear": graph["clear"]}
    if q == "held":
        return {"held": graph["held"]}
    if q.startswith("where:"):
        obj = q.split(":", 1)[1].strip()
        if obj not in graph["objects"]:
            return {"error": f"unknown object {obj!r}", "known": graph["objects"]}
        if obj in graph.get("hidden", ()):          # the camera did not detect it
            return {"object": obj, "visible": False}
        return {
            "object": obj,
            "in_zone": graph["in_zone"].get(obj),
            "on_top_of": graph["on_top_of"].get(obj),
            "clear": graph["clear"].get(obj),
            "off_table": obj in graph["off_table"],
            "held": graph.get("held") == obj,       # in the gripper right now
        }
    return {"error": f"unsupported query {q!r}", "supported": list(OBSERVE_QUERIES)}
