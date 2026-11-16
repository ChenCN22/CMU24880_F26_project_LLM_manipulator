"""A9: an evidence-aware recovery agent.

The earlier agents hand camera reports to the model and let it work out what is
true.  Here the evidence is kept by code, per goal, and the model decides what to
do given a summary of it:

  goals        parsed once from the instruction by the model
  evidence     for every goal, what each camera said since the arm last moved
               (satisfied / not satisfied / block not visible = abstain)
  fusion       log-odds over the cameras on top of a prior: a goal that was
               confirmed before and has not been touched needs more contrary
               evidence to be overturned than a goal that was just attempted
  staleness    any arm movement makes all camera evidence stale -- handling one
               block can knock another -- so every goal is re-read afterwards
  reliability  a camera that was outvoted about a goal is trusted less about
               that goal for the rest of the episode, and is asked last
  cheap first  whether a pick worked is read from the fingers (reliable), so a
               failed pick is retried without placing and photographing
  guards       done() is refused while a goal is unsatisfied and has attempts
               left; picking a block whose goal is confidently satisfied is
               refused; place_on with an empty hand is refused

Nothing here reads the true scene.  The model keeps the decisions that need the
instruction: the order of the goals, what to retry, when to give up, and the
prohibitions.
"""
import json
import math
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph

from agent_iface.perception import VIEWS

from .nodes import ACT_FORMAT, ACT_SCHEMA, TOOL_NAMES, EpisodeState, _args_of

W_TRUSTED = math.log(0.8 / 0.2)     # weight of one camera reading (the camera is right 4 times in 5)
W_DOUBTED = math.log(0.55 / 0.45)   # ... of a camera that has been outvoted about this goal
P_JUST_PLACED, P_WAS_MET, P_WAS_UNMET, P_UNTOUCHED = 0.6, 0.9, 0.05, 0.02
HI, LO = 0.9, 0.2                   # confident "satisfied" / confident "not satisfied"
MAX_PICKS, MAX_REFUSALS, MAX_TURNS = 4, 4, 70

SYSTEM_A9 = """You control a Franka Panda robot on a tabletop. There are coloured blocks (named like \
'red_block') and three zone pads: zone_A, zone_B, zone_C.

TOOLS
  pick(obj)                       close the gripper on a block
  place_on(target)                put the held block on a zone pad, or on top of another block
  push(obj, direction, distance)  direction is "+x", "-x", "+y" or "-y"; distance in metres, max 0.15
  done()                          declare the instruction satisfied and end the episode

HOW THE WORLD WORKS
- The gripper holds at most one block. place_on moves the block that is currently held.
- pick and push both begin by opening the gripper. Calling either one while a block is held drops it.
- Handle one block at a time: pick it, place_on its target, then start on the next block.
- The low-level skill is unreliable. An action that fails can simply be tried again, and a second
  attempt often works.

WHAT YOU ARE GIVEN
You do not read the cameras yourself. After every arm movement a perception system looks from up to
three cameras and keeps an evidence table: for each goal, what each camera saw, and a combined STATUS.
Single cameras can be wrong; the combined status already weighs them against each other.
The GRIPPER line comes from the fingers and is reliable.

RULES
- A block may be picked at most four times.
- Do not pick a block whose goal is SATISFIED.
- If a goal is NOT SATISFIED and its block has attempts left: pick that block, then place_on its target.
- Call done() when every goal is SATISFIED, or when the unsatisfied ones have no attempts left.
- Respect every prohibition in the instruction."""

GOALS_FORMAT = """Reply with exactly one JSON object and nothing else:
{"goals": [{"block": "<block name>", "target": "<zone or block it must end up on>"}, ...]}
List, in the order the instruction gives them, each placement the instruction asks for.
Do not list prohibitions."""


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def reading(goal: Tuple[str, str], rep: Any) -> Optional[bool]:
    """What one whole-scene camera report says about a goal; None = it cannot tell."""
    block, target = goal
    if not isinstance(rep, dict) or "in_zone" not in rep:
        return None
    if block in (rep.get("off_table") or []):
        return False
    iz, ot = rep.get("in_zone") or {}, rep.get("on_top_of") or {}
    if block not in iz:                                   # not detected
        return None
    if target.startswith("zone_"):
        return iz.get(block) == target
    if target not in iz:
        return None
    return ot.get(block) == target


class Evidence:
    """Per-goal camera evidence for one episode."""

    def __init__(self, goals: List[Tuple[str, str]]) -> None:
        self.goals = goals
        self.prior = [P_UNTOUCHED] * len(goals)
        self.reads: List[Dict[str, Optional[bool]]] = [{} for _ in goals]
        self.doubt: List[set] = [set() for _ in goals]    # cameras outvoted about this goal
        self.attempted = [False] * len(goals)
        self.fresh = True                                 # nothing has moved since the last reading
        self.picks: Dict[str, int] = {}
        self.holding: Optional[str] = None

    def p(self, i: int) -> float:
        x = _logit(self.prior[i])
        for view, r in self.reads[i].items():
            if r is not None:
                w = W_DOUBTED if view in self.doubt[i] else W_TRUSTED
                x += w if r else -w
        return _sigmoid(x)

    def moved(self, placed_goal: Optional[int] = None) -> None:
        """The arm moved: what the cameras said before no longer counts as seen."""
        for i in range(len(self.goals)):
            if self.reads[i] or self.fresh:
                was = self.p(i)
                self.prior[i] = (P_WAS_MET if was >= 0.5 else
                                 P_WAS_UNMET if self.attempted[i] else P_UNTOUCHED)
            self.reads[i] = {}
        if placed_goal is not None:
            self.prior[placed_goal] = P_JUST_PLACED
            self.attempted[placed_goal] = True
        self.fresh = False

    def resolved(self) -> bool:
        return all(self.p(i) >= HI or self.p(i) <= LO for i in range(len(self.goals)))

    def order(self) -> List[str]:
        return sorted(VIEWS, key=lambda v: sum(v in d for d in self.doubt))

    def add(self, view: str, rep: Any) -> None:
        for i, g in enumerate(self.goals):
            self.reads[i][view] = reading(g, rep)

    def settle(self) -> None:
        """After a round of looks: remember which camera was outvoted about which goal."""
        for i in range(len(self.goals)):
            p = self.p(i)
            if p >= HI or p <= LO:
                for view, r in self.reads[i].items():
                    if r is not None and r != (p >= 0.5):
                        self.doubt[i].add(view)
        self.fresh = True

    def left(self, i: int) -> int:
        return MAX_PICKS - self.picks.get(self.goals[i][0], 0)

    def pending(self) -> List[int]:
        return [i for i in range(len(self.goals)) if self.p(i) < 0.5 and self.left(i) > 0]

    def table(self) -> str:
        mark = {True: "satisfied", False: "not satisfied", None: "block not visible"}
        rows = []
        for i, (b, t) in enumerate(self.goals):
            p = self.p(i)
            if self.holding == b:
                status = "IN THE GRIPPER"
            elif not self.attempted[i] and p < 0.5:
                status = "NOT SATISFIED (not attempted yet)"
            else:
                status = ("SATISFIED" if p >= 0.5 else "NOT SATISFIED") + ("" if p >= HI or p <= LO
                                                                         else ", low confidence")
            seen = ", ".join(f"{v}: {mark[r]}" for v, r in self.reads[i].items()) or "no camera reading yet"
            rows.append(f"  {i + 1}. {b} on {t}: {status} | cameras: {seen} | "
                        f"picks used {self.picks.get(b, 0)} of {MAX_PICKS}")
        return "\n".join(rows) or "  (no goals could be read from the instruction)"


def build_a9(iface, llm, memory=None, system=None):
    """Evidence-aware recovery: code keeps per-goal evidence, the model decides."""
    names = list(iface.env.object_names)
    zones = ["zone_A", "zone_B", "zone_C"]
    box: Dict[str, Any] = {"ev": Evidence([]), "refused": 0, "streak": 0, "turns": 0}
    goal_schema = {"type": "object", "required": ["goals"], "properties": {"goals": {
        "type": "array", "items": {"type": "object", "required": ["block", "target"], "properties": {
            "block": {"type": "string", "enum": names},
            "target": {"type": "string", "enum": zones + names}}}}}}

    def parse_node(state: EpisodeState) -> EpisodeState:
        out = llm.chat_json([
            {"role": "system", "content": "You read robot task instructions.\n\n" + GOALS_FORMAT},
            {"role": "user", "content": f"INSTRUCTION: {state['instruction']}"}], schema=goal_schema)
        goals = []
        for g in (out or {}).get("goals", []) if isinstance(out, dict) else []:
            pair = (g.get("block"), g.get("target"))
            if pair[0] in names and pair[1] in zones + names and pair[0] != pair[1] and pair not in goals:
                goals.append(pair)
        box["ev"] = Evidence(goals)
        return {"history": [], "steps": 0}

    def ask(ev: Evidence) -> str:
        if ev.holding:
            return f"You are holding {ev.holding}. What is your next tool call?"
        todo = ev.pending()
        if todo:
            b, t = ev.goals[todo[0]]
            return (f"At least one goal is NOT SATISFIED and has picks left (for example {b} on {t}). "
                    "What is your next tool call?")
        return "No goal is left that is both NOT SATISFIED and has picks left. What is your next tool call?"

    def act_node(state: EpisodeState) -> EpisodeState:
        ev: Evidence = box["ev"]
        box["turns"] += 1
        if box["turns"] > MAX_TURNS:
            return {"last": {"tool": "done", "args": {}, "thought": "turn limit"}}
        user = "\n\n".join([
            f"INSTRUCTION: {state['instruction']}",
            "GRIPPER: " + (f"holding {ev.holding}" if ev.holding else "empty"),
            "EVIDENCE TABLE" + ("" if ev.fresh else " (the arm has moved since these readings)") + ":\n" + ev.table(),
            "RECENT CALLS:\n" + ("\n".join(state.get("history", [])[-8:]) or "(nothing yet)"),
            ask(ev)])
        out = llm.chat_json([{"role": "system", "content": SYSTEM_A9 + "\n\n" + ACT_FORMAT},
                             {"role": "user", "content": user}], schema=ACT_SCHEMA)
        if not isinstance(out, dict) or out.get("tool") not in TOOL_NAMES:
            return {"last": {"tool": "observe", "args": {}}}
        return {"last": {"tool": out["tool"], "args": _args_of(out), "thought": out.get("thought", "")}}

    def verify(ev: Evidence) -> None:
        """Look until every goal is confidently read, or the three cameras are used up."""
        for view in ev.order():
            if iface.finished:
                return
            r = iface.call("observe", {"query": "scene", "view": view})
            if r.get("ok"):
                ev.add(view, r.get("result"))
                if ev.resolved():
                    break
        ev.settle()

    def step_node(state: EpisodeState) -> EpisodeState:
        ev: Evidence = box["ev"]
        call = state.get("last") or {"tool": "done", "args": {}}
        tool, args = call.get("tool", "done"), call.get("args") or {}
        hist = list(state.get("history", []))

        def refuse(why: str, hard: bool = False) -> Optional[EpisodeState]:
            """Turn the call down and say why.  A soft refusal is dropped after a few
            times (the model may know better); a hard one never is.  Four refusals in
            a row end the episode: the model and the evidence cannot agree."""
            if not hard and box["refused"] >= MAX_REFUSALS:
                return None
            box["refused"] += 1
            box["streak"] += 1
            line = f"{tool}({json.dumps(args, sort_keys=True)}) -> REFUSED: {why}"
            if box["streak"] >= 4:
                iface.call("done")
                return {"history": hist + [line, "done()  [ended after repeated refusals]"], "stop": True}
            return {"history": hist + [line]}

        if tool in ("observe", "check_constraints", "done") and not ev.fresh and not ev.holding:
            verify(ev)                                      # never decide on stale evidence
            if iface.finished:
                return {"history": hist, "stop": True}
        if tool == "done":
            todo = ev.pending()
            if todo and not ev.holding:
                b, t = ev.goals[todo[0]]
                out = refuse(f"{b} on {t} is NOT SATISFIED and has picks left")
                if out:
                    return out
            iface.call("done")
            return {"history": hist + ["done()"], "stop": True}
        if tool in ("observe", "check_constraints"):
            return {"history": hist + ["(the evidence table is up to date)"]}
        if tool == "pick":
            obj = str(args.get("obj", ""))
            idx = next((i for i, g in enumerate(ev.goals) if g[0] == obj), None)
            out = None
            if ev.holding and ev.holding != obj:
                out = refuse(f"you are holding {ev.holding}; place it first")
            elif idx is not None and ev.fresh and ev.p(idx) >= HI:
                out = refuse(f"the evidence says {obj} is already on {ev.goals[idx][1]}")
            elif obj in names and ev.picks.get(obj, 0) >= MAX_PICKS:
                out = refuse(f"{obj} has no picks left", hard=True)
            if out:
                return out
            box["streak"] = 0
            r = iface.call("pick", args)
            if not r.get("ok"):
                return {"history": hist + [f"pick({json.dumps(args)}) -> {json.dumps(r)}"], "stop": iface.finished}
            ev.picks[obj] = ev.picks.get(obj, 0) + 1
            ev.moved()
            held = None
            if not iface.finished:                          # the fingers say whether it worked
                h = iface.call("observe", {"query": "held"})
                held = (h.get("result") or {}).get("held") if h.get("ok") else None
            ev.holding = held
            if held is not None:                            # it has left wherever it was
                for i, g in enumerate(ev.goals):
                    if g[0] == held:
                        ev.prior[i], ev.attempted[i] = P_WAS_UNMET, True
            note = f"gripper now holds {held}" if held else "gripper is EMPTY, the pick failed"
            return {"history": hist + [f"pick({obj}) -> executed; {note}"], "stop": iface.finished}
        if tool == "place_on":
            if not ev.holding:
                return refuse("the gripper is empty; pick a block first", hard=True)
            box["streak"] = 0
            target, held = str(args.get("target", "")), ev.holding
            r = iface.call("place_on", args)
            if not r.get("ok"):
                return {"history": hist + [f"place_on({json.dumps(args)}) -> {json.dumps(r)}"],
                        "stop": iface.finished}
            idx = next((i for i, g in enumerate(ev.goals) if g == (held, target)), None)
            ev.holding = None
            ev.moved(placed_goal=idx)
            verify(ev)
            return {"history": hist + [f"place_on({target}) -> executed with {held}; cameras were read"],
                    "stop": iface.finished}
        # push
        box["streak"] = 0
        r = iface.call(tool, args)
        ev.holding = None
        ev.moved()
        verify(ev)
        return {"history": hist + [f"{tool}({json.dumps(args, sort_keys=True)}) -> "
                                   + ("executed; cameras were read" if r.get("ok") else json.dumps(r))],
                "stop": iface.finished}

    g = StateGraph(EpisodeState)
    g.add_node("parse", parse_node)
    g.add_node("act", act_node)
    g.add_node("step", step_node)
    g.set_entry_point("parse")
    g.add_edge("parse", "act")
    g.add_edge("act", "step")
    g.add_conditional_edges("step", lambda s: "stop" if s.get("stop") or iface.finished else "act",
                            {"act": "act", "stop": END})
    return g.compile()
