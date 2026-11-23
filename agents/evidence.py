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

Ablations.  build_a9(..., off={...}) switches single parts off; the agents
"A9-<part>" in graphs.BUILDERS are A9 with exactly one part removed:

  single   one camera per verification instead of looking until the goals are resolved
  prior    the same prior (P_FLAT) for every attempted goal, whatever its history
  stale    after a placement only the goal just handled is read again; older
           readings of the other goals keep counting
  doubt    no memory of which camera was outvoted
  fingers  the fingers are not read after a pick; the pick is assumed to have worked
  guards   no call is refused

A10 = A9 for tasks that leave the choice of blocks to the agent (build_a9 with
`choice` set; see agents/skill_memory.py).  The model chooses the blocks from a
layout with positions, and chooses again after every failed attempt; `choice`
names the form in which the memory of earlier episodes reaches it.  Everything
else is A9, and with choice=None nothing below differs from A9.  An instruction that
names its blocks is read as such (NAMED_FORMAT): the plan then has exactly the named
placements and only their order is left to the model.

A10r = A9 with the push skill as a way to recover (build_a9 with `recover` set).  The
goals are read from the instruction as in A9.  After an attempt on a block has failed
the model may push the block toward the middle of the table before it picks it again;
`recover` names what it is shown to decide that: nothing (none), the designer's rule
(told), or the memory of earlier episodes in one of the three forms.  While the memory
is being learned, half of the retries are preceded by a push chosen by a coin, so that
both ways get tried.

A11 = A10 with a memory of what the arm did to the blocks it was not handling (build_a9
with `avoid` set; see agents/disturb_memory.py).  The skill knocks blocks that lie near the
block it picks up or near its path.  Before and after every attempt the agent has the camera
positions of all blocks, so it can record which blocks were moved and how they lay.  When it
chooses which block goes where it is shown the candidate placements with their geometry and,
in the form `avoid` names, what earlier episodes say about it.
"""
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph

from agent_iface.perception import VIEWS

from .nodes import ACT_FORMAT, ACT_SCHEMA, TOOL_NAMES, EpisodeState, _args_of
from .skill_memory import GEOMETRY

W_TRUSTED = math.log(0.8 / 0.2)     # weight of one camera reading (the camera is right 4 times in 5)
W_DOUBTED = math.log(0.55 / 0.45)   # ... of a camera that has been outvoted about this goal
P_JUST_PLACED, P_WAS_MET, P_WAS_UNMET, P_UNTOUCHED = 0.6, 0.9, 0.05, 0.02
P_FLAT = 0.45                       # ablation 'prior': history-free prior of an attempted goal
HI, LO = 0.9, 0.2                   # confident "satisfied" / confident "not satisfied"
MAX_PICKS, MAX_REFUSALS, MAX_TURNS = 4, 4, 70
PUSH_CM, P_EXPLORE = 8.0, 0.5       # A10r: length of a push; share of retries that start with a push while learning

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

PUSH_RULE = {
    "none": "If the last attempt on that block failed, you may push it first and then pick it.",
    "told": "If the last attempt on that block failed, push it first and then pick it.",
    "memory": ("If the last attempt on that block failed, you may push it first and then pick it; what you are "
               "shown from earlier episodes says how picking again and pushing first have gone."),
}


def system_a10r(form: str) -> str:
    """SYSTEM_A9 with the push skill described as what it now is."""
    s = SYSTEM_A9.replace(
        '  push(obj, direction, distance)  direction is "+x", "-x", "+y" or "-y"; distance in metres, max 0.15',
        f"  push(obj)                       slide a block about {PUSH_CM:.0f} cm toward the middle of the table, without lifting it")
    s = s.replace("- Handle one block at a time:",
                  "- A push is one call and does not use up a pick. It can move other blocks that are in its way.\n"
                  "- Handle one block at a time:")
    rule = "- If a goal is NOT SATISFIED and its block has attempts left: pick that block, then place_on its target."
    assert rule in s
    return s.replace(rule, rule + " " + PUSH_RULE.get(form, PUSH_RULE["memory"])
                     + " A block can be pushed once between two picks.")


GOALS_FORMAT = """Reply with exactly one JSON object and nothing else:
{"goals": [{"block": "<block name>", "target": "<zone or block it must end up on>"}, ...]}
List, in the order the instruction gives them, each placement the instruction asks for.
Do not list prohibitions."""


SHAPE_FORMAT = """You read robot task instructions.

Reply with exactly one JSON object and nothing else:
{"zones": [<each zone pad the instruction says a block has to be put in>], "stacks": <how many stacks of two \
blocks the instruction asks to build; 0 if it asks for none>}"""


NAMED_FORMAT = """You read robot task instructions.

Reply with exactly one JSON object and nothing else:
{"named": [{"block": "<block name>", "target": "<zone or block it must end up on>"}, ...]}
List, in the order the instruction gives them, each placement for which the instruction itself says which block \
it is (by its colour) and where it has to end up. If the instruction leaves the blocks open ("one block", \
"another block", "a stack of two blocks"), reply {"named": []}. Do not list prohibitions."""


CHOOSE_SYSTEM = """You plan block placements for a robot arm on a tabletop. There are coloured blocks and three \
zone pads: zone_A, zone_B, zone_C. The arm's pick-and-place skill is unreliable: some picks fail, and a block that \
is hard to pick usually stays hard.

Decide which block goes where. Where the instruction leaves the blocks open, choose so that the robot is likely \
to succeed. A block you do not choose stays where it is and is never picked up. In a stack only the top block is \
picked up; the bottom block stays where it is.

""" + GEOMETRY + """

How many placements a plan has:
- "one block in zone_X and another block in zone_Y": two placements, one into each zone that is named.
- "one stack of two blocks": ONE placement, the top block onto the bottom block. The bottom block gets no placement.
- "two stacks of two blocks": TWO placements, and the four blocks in them are all different.
Never put a block in a zone that the instruction does not name. A block appears in at most one placement, either as \
the block that is moved or as the block it is put on.

Reply with exactly one JSON object and nothing else:
{"reason": "<one sentence>", "goals": [{"block": "<block to pick up>", "target": "<zone or block it must end up on>"}, ...]}
List every placement needed to satisfy the instruction, and nothing else."""


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

    def __init__(self, goals: List[Tuple[str, str]], off=frozenset()) -> None:
        self.goals, self.off = goals, off
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
        flat = "prior" in self.off
        if "stale" in self.off:                           # ablation: only the handled goal is re-read
            if placed_goal is not None:
                self.reads[placed_goal] = {}
                self.prior[placed_goal] = P_FLAT if flat else P_JUST_PLACED
                self.attempted[placed_goal] = True
            return
        for i in range(len(self.goals)):
            if self.reads[i] or self.fresh:
                was = self.p(i)
                if flat:
                    self.prior[i] = P_FLAT if (was >= 0.5 or self.attempted[i]) else P_UNTOUCHED
                else:
                    self.prior[i] = (P_WAS_MET if was >= 0.5 else
                                     P_WAS_UNMET if self.attempted[i] else P_UNTOUCHED)
            self.reads[i] = {}
        if placed_goal is not None:
            self.prior[placed_goal] = P_FLAT if flat else P_JUST_PLACED
            self.attempted[placed_goal] = True
        self.fresh = False

    def resolved(self) -> bool:
        return all(self.p(i) >= HI or self.p(i) <= LO for i in range(len(self.goals)))

    def order(self) -> List[str]:
        return sorted(VIEWS, key=lambda v: sum(v in d for d in self.doubt))

    def add(self, view: str, rep: Any, only: Optional[int] = None) -> None:
        for i, g in enumerate(self.goals):
            if only is None or i == only:
                self.reads[i][view] = reading(g, rep)

    def settle(self) -> None:
        """After a round of looks: remember which camera was outvoted about which goal."""
        for i in range(len(self.goals)):
            p = self.p(i)
            if (p >= HI or p <= LO) and "doubt" not in self.off:
                for view, r in self.reads[i].items():
                    if r is not None and r != (p >= 0.5):
                        self.doubt[i].add(view)
        self.fresh = True

    def retarget(self, goals: List[Tuple[str, str]]) -> None:
        """A10: the plan changed.  Goals that stay keep everything known about them."""
        old = {g: i for i, g in enumerate(self.goals)}
        take = lambda xs, new: [xs[old[g]] if g in old else new() for g in goals]     # noqa: E731
        self.prior, self.reads = take(self.prior, lambda: P_UNTOUCHED), take(self.reads, dict)
        self.doubt, self.attempted = take(self.doubt, set), take(self.attempted, lambda: False)
        self.goals = goals

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


def build_a9(iface, llm, memory=None, system=None, off=frozenset(), choice=None, recover=None, avoid=None):
    """Evidence-aware recovery: code keeps per-goal evidence, the model decides.
    `off` names the parts to leave out (see the module text); empty is A9.
    `choice` (none | episodic | lessons | map) turns on A10: the model chooses the
    blocks, with `memory` (a SkillMemory) shown to it in that form.
    `recover` (none | told | episodic | lessons | map) turns on A10r: push as a way to
    recover from a failed attempt.
    `avoid` (none | told | episodic | lessons | map) turns on A11: A10 whose `memory` is a
    DisturbMemory, shown in that form."""
    off = frozenset(off)
    if avoid:
        from .disturb_memory import DisturbMemory
        choice = choice or "none"
        memory = memory if isinstance(memory, DisturbMemory) else DisturbMemory()
    skill_memory = None if avoid else memory                 # the record of picks and placements (A10, A10r)
    track = bool(choice or recover)                          # keep positions and what last happened to each block
    query = "layout" if track else "scene"
    system_act = system_a10r(recover) if recover else SYSTEM_A9
    learning = bool(recover and memory is not None and memory.learn)
    coin = random.Random(7919 * int(iface.spec.get("seed", 0)) + 17)
    names = list(iface.env.object_names)
    zones = ["zone_A", "zone_B", "zone_C"]
    box: Dict[str, Any] = {"ev": Evidence([], off), "refused": 0, "streak": 0, "turns": 0,
                           "xy": {}, "where": {}, "last": {}, "shape": ([], 0), "fixed": [], "plans": [], "pushed": {}}
    goal_schema = {"type": "object", "required": ["goals"], "properties": {"goals": {
        "type": "array", "items": {"type": "object", "required": ["block", "target"], "properties": {
            "block": {"type": "string", "enum": names},
            "target": {"type": "string", "enum": zones + names}}}}}}

    plan_schema = {"type": "object", "required": ["reason", "goals"], "properties": {
        "reason": {"type": "string"}, "goals": goal_schema["properties"]["goals"]}}
    shape_schema = {"type": "object", "required": ["zones", "stacks"], "properties": {
        "zones": {"type": "array", "items": {"type": "string", "enum": zones}},
        "stacks": {"type": "integer", "minimum": 0, "maximum": 2}}}

    def plan_text(ev: Evidence) -> str:
        return "; ".join(f"{b} on {t}" for b, t in ev.goals)

    def see(view: str, rep: Any, seen: Dict[str, List[List[float]]]) -> None:
        """A10: keep where the cameras put each block, and what it is standing on."""
        if not isinstance(rep, dict):
            return
        for n, xy in (rep.get("positions_cm") or {}).items():
            seen.setdefault(n, []).append(xy)
        for n in names:
            if n in (rep.get("in_zone") or {}):
                below, zone = (rep.get("on_top_of") or {}).get(n), rep["in_zone"].get(n)
                box["where"][n] = f"on top of {below}" if below else f"inside {zone}" if zone else "on the table"

    def settle_positions(seen: Dict[str, List[List[float]]]) -> None:
        for n, pts in seen.items():                          # the mean over the cameras that saw it
            box["xy"][n] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def glance() -> None:
        """A10: one look from the front, for positions only."""
        if iface.finished:
            return
        r = iface.call("observe", {"query": query, "view": box["ev"].order()[0]})
        if r.get("ok"):
            seen: Dict[str, List[List[float]]] = {}
            see("front", r.get("result"), seen)
            settle_positions(seen)

    def choose(instruction: str, why: str = "") -> None:
        """A10: ask the model which block goes where, and adopt the answer if it is a full plan."""
        ev: Evidence = box["ev"]
        rows = []
        for n in names:
            xy = box["xy"].get(n)
            rows.append(f"  {n}: " + (f"at ({xy[0]:+.1f}, {xy[1]:+.1f}), " if xy else "position unknown, ")
                        + box["where"].get(n, "not seen") + f"; picked {ev.picks.get(n, 0)} of {MAX_PICKS} times"
                        + {"failed pick": "; its last pick FAILED", "failed place": "; its last placement FAILED"}
                        .get(box["last"].get(n), ""))
        parts = [f"INSTRUCTION: {instruction}", "LAYOUT:\n" + "\n".join(rows)]
        note = skill_memory.notes(choice, {n: (*box["xy"][n], box["last"].get(n, "first"))
                                           for n in names if n in box["xy"]}) if skill_memory is not None else ""
        if note:
            parts.append(note)
        named, stacks = box["shape"]
        fixed = box["fixed"]
        if avoid:                                            # A11: the placements there are to choose from
            kept = {x for i, g in enumerate(ev.goals) if ev.p(i) >= 0.5 for x in g}
            loose = [n for n in names if n not in kept and n in box["xy"]]
            cand = [(b, z) for b in loose if ev.picks.get(b, 0) < MAX_PICKS for z in named if z not in kept]
            if stacks:
                cand += [(t, b) for t in loose if ev.picks.get(t, 0) < MAX_PICKS for b in loose if b != t]
            if fixed:                                        # the instruction names them: only the order is open
                cand = [g for g in fixed if g[0] in loose and ev.picks.get(g[0], 0) < MAX_PICKS]
            note = memory.notes(avoid, dict(box["xy"]), cand)
            if note:
                parts.append(note)
        shape = " and ".join(x for x in [
            f"one placement into each of {', '.join(named)}" if named else "",
            f"{stacks} placement{'s' if stacks > 1 else ''} of a block onto another block" if stacks else ""] if x)
        shape = f"The plan for this instruction has exactly {shape}, and no other placement."
        if fixed:
            shape = ("The instruction names its placements, so the plan has exactly these and no other: "
                     + "; ".join(f"{b} on {t}" for b, t in fixed) + ". List them in the order in which they should be done.")
        if ev.goals:
            parts.append("CURRENT PLAN:\n" + ev.table())
            parts.append(f"{why}. Give the full plan again. {shape} Keep the placements that are SATISFIED. For the "
                         "others you may keep the same block or change to a different one.")
        else:
            parts.append(f"Give the plan. {shape}")

        def read(out) -> List[Tuple[str, str]]:
            goals: List[Tuple[str, str]] = []
            for g in (out or {}).get("goals", []) if isinstance(out, dict) else []:
                pair = (g.get("block"), g.get("target"))
                if pair[0] in names and pair[1] in zones + names and pair[0] != pair[1] and pair not in goals:
                    goals.append(pair)
            return goals

        def problem(goals: List[Tuple[str, str]]) -> str:
            """Why a plan cannot be one for this instruction; '' if it can.  Checks its shape only
            (how many placements, into which zones), never which blocks were chosen."""
            used = [b for b, _ in goals] + [t for _, t in goals if t in names]
            into = sorted(t for _, t in goals if t in zones)
            if not goals:
                return "it has no placement"
            if fixed and sorted(goals) != sorted(fixed):
                return "the instruction names its placements: " + "; ".join(f"{b} on {t}" for b, t in fixed)
            if len(set(used)) != len(used):
                return "a block appears in more than one placement"
            if into != sorted(named):
                return (f"it needs exactly one placement into each of {', '.join(named)}, and into no other zone"
                        if named else "the instruction names no zone, so no block may be put in a zone")
            if len(goals) - len(into) != stacks:
                return f"it needs exactly {stacks} placement{'' if stacks == 1 else 's'} of a block onto another block"
            for i, g in enumerate(ev.goals):
                if ev.p(i) >= 0.5 and g not in goals:
                    return f"{g[0]} on {g[1]} is SATISFIED and has to stay in the plan"
            for b, t in goals:
                if (b, t) not in ev.goals and ev.picks.get(b, 0) >= MAX_PICKS:
                    return f"{b} has no picks left"
            return ""

        msgs = [{"role": "system", "content": CHOOSE_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
        out = llm.chat_json(msgs, schema=plan_schema)
        goals = read(out)
        wrong = problem(goals)
        tries = 1
        while wrong and tries < 3:                           # told what was wrong, at most twice
            msgs += [{"role": "assistant", "content": json.dumps(out) if isinstance(out, dict) else "{}"},
                     {"role": "user", "content": f"That plan cannot be used: {wrong}. Give the full plan again."}]
            out = llm.chat_json(msgs, schema=plan_schema)
            goals = read(out)
            wrong = problem(goals)
            tries += 1
        box["plans"].append({"why": why or "start", "goals": goals, "refused": wrong, "tries": tries,
                             "reason": (out or {}).get("reason", "") if isinstance(out, dict) else ""})
        if not wrong:
            ev.retarget(goals)

    def parse_node(state: EpisodeState) -> EpisodeState:
        if choice:
            box["ev"], box["fixed"] = Evidence([], off), []
            out = llm.chat_json([{"role": "system", "content": SHAPE_FORMAT},
                                 {"role": "user", "content": f"INSTRUCTION: {state['instruction']}"}], schema=shape_schema)
            out = out if isinstance(out, dict) else {}
            named = [z for z in zones if z in (out.get("zones") or [])]
            stacks = out.get("stacks") if isinstance(out.get("stacks"), int) else 0
            box["shape"] = (named, stacks if named or stacks else 1)
            out = llm.chat_json([{"role": "system", "content": NAMED_FORMAT},
                                 {"role": "user", "content": f"INSTRUCTION: {state['instruction']}"}],
                                schema={"type": "object", "required": ["named"], "properties": {"named": goal_schema["properties"]["goals"]}})
            said = state["instruction"].lower()
            for g in (out or {}).get("named", []) if isinstance(out, dict) else []:
                pair = (g.get("block"), g.get("target"))     # kept only if the instruction has the block's colour in it
                if (pair[0] in names and pair[1] in zones + names and pair[0] != pair[1] and pair not in box["fixed"]
                        and pair[0].split("_")[0] in said and (pair[1] in zones or pair[1].split("_")[0] in said)):
                    box["fixed"].append(pair)
            if box["fixed"]:                                 # a stated task: its shape is what it names
                box["shape"] = ([z for z in zones if z in [t for _, t in box["fixed"]]],
                                sum(t in names for _, t in box["fixed"]))
            glance()
            choose(state["instruction"])
            iface.plans = box["plans"]                       # kept with the episode's log
            return {"history": [], "steps": 0}
        out = llm.chat_json([
            {"role": "system", "content": "You read robot task instructions.\n\n" + GOALS_FORMAT},
            {"role": "user", "content": f"INSTRUCTION: {state['instruction']}"}], schema=goal_schema)
        goals = []
        for g in (out or {}).get("goals", []) if isinstance(out, dict) else []:
            pair = (g.get("block"), g.get("target"))
            if pair[0] in names and pair[1] in zones + names and pair[0] != pair[1] and pair not in goals:
                goals.append(pair)
        box["ev"] = Evidence(goals, off)
        return {"history": [], "steps": 0}

    def ask(ev: Evidence) -> str:
        if not choice:
            return ask_a9(ev)
        if ev.holding:
            t = next((t for b, t in ev.goals if b == ev.holding), None)
            return (f"You are holding {ev.holding}." + (f" The plan puts it on {t}." if t else "")
                    + " What is your next tool call?")
        head = f"The plan for this instruction is: {plan_text(ev)}. "
        if ev.goals and all(ev.p(i) >= 0.5 for i in range(len(ev.goals))):
            return (head + "Every placement of the plan is SATISFIED, so the instruction is fulfilled. "
                    "What is your next tool call?")
        todo = "; ".join(f"pick {ev.goals[i][0]} and place it on {ev.goals[i][1]}" for i in ev.pending())
        if not todo:
            return head + ask_a9(ev)
        return head + f"Still to do: {todo}. What is your next tool call?"

    def ask_a9(ev: Evidence) -> str:
        if ev.holding:
            return f"You are holding {ev.holding}. What is your next tool call?"
        todo = ev.pending()
        if todo and recover:                                 # A10r: put the choice at the moment it exists
            b, t = ev.goals[todo[0]]
            last = box["last"].get(b)
            if last == "pushed":
                return f"{b} has been pushed since its last pick, so the next call for it is pick. What is your next tool call?"
            if last in ("failed pick", "failed place"):
                what = "pick" if last == "failed pick" else "placement"
                how = ("Push it first, then pick it." if recover == "told" else
                       "Either pick it again where it lies, or push it first and pick it after that.")
                return (f"The last {what} of {b} failed, and {b} on {t} is NOT SATISFIED with picks left. {how} "
                        "What is your next tool call?")
        if todo:
            b, t = ev.goals[todo[0]]
            return (f"At least one goal is NOT SATISFIED and has picks left (for example {b} on {t}). "
                    "What is your next tool call?")
        return "No goal is left that is both NOT SATISFIED and has picks left. What is your next tool call?"

    def recovery_note(ev: Evidence) -> str:
        """A10r: for the block that is up next, if its last attempt failed and it has not been pushed since."""
        if not recover or memory is None or ev.holding:
            return ""
        for i in ev.pending():
            b = ev.goals[i][0]
            if box["last"].get(b) in ("failed pick", "failed place"):
                return memory.recover_notes(recover, b, box["xy"].get(b), box["last"][b])
        return ""

    def act_node(state: EpisodeState) -> EpisodeState:
        ev: Evidence = box["ev"]
        box["turns"] += 1
        if box["turns"] > MAX_TURNS:
            return {"last": {"tool": "done", "args": {}, "thought": "turn limit"}}
        user = "\n\n".join(x for x in [
            f"INSTRUCTION: {state['instruction']}",
            "GRIPPER: " + (f"holding {ev.holding}" if ev.holding else "empty"),
            "EVIDENCE TABLE" + ("" if ev.fresh else " (the arm has moved since these readings)") + ":\n" + ev.table(),
            "RECENT CALLS:\n" + ("\n".join(state.get("history", [])[-8:]) or "(nothing yet)"),
            recovery_note(ev),
            ask(ev)] if x)
        out = llm.chat_json([{"role": "system", "content": system_act + "\n\n" + ACT_FORMAT},
                             {"role": "user", "content": user}], schema=ACT_SCHEMA)
        if not isinstance(out, dict) or out.get("tool") not in TOOL_NAMES:
            return {"last": {"tool": "observe", "args": {}}}
        return {"last": {"tool": out["tool"], "args": _args_of(out), "thought": out.get("thought", "")}}

    def verify(ev: Evidence, only: Optional[int] = None) -> None:
        """Look until every goal is confidently read, or the three cameras are used up."""
        seen: Dict[str, List[List[float]]] = {}
        for view in ev.order():
            if iface.finished:
                return
            r = iface.call("observe", {"query": query, "view": view})
            if r.get("ok"):
                ev.add(view, r.get("result"), only)
                if track:
                    see(view, r.get("result"), seen)
                if ev.resolved() or "single" in off:
                    break
        ev.settle()
        settle_positions(seen)

    def step_node(state: EpisodeState) -> EpisodeState:
        ev: Evidence = box["ev"]
        call = state.get("last") or {"tool": "done", "args": {}}
        tool, args = call.get("tool", "done"), call.get("args") or {}
        hist = list(state.get("history", []))

        def refuse(why: str, hard: bool = False) -> Optional[EpisodeState]:
            """Turn the call down and say why.  A soft refusal is dropped after a few
            times (the model may know better); a hard one never is.  Four refusals in
            a row end the episode: the model and the evidence cannot agree."""
            if "guards" in off or (not hard and box["refused"] >= MAX_REFUSALS):
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
            if (recover or choice) and ev.holding:          # not in A9, which is frozen: a held block is not a finished goal
                tgt = next((t for b, t in ev.goals if b == ev.holding), None)
                out = refuse(f"you are holding {ev.holding}; " + (f"place it on {tgt} first" if tgt else "place it first"))
                if out:
                    return out
            if todo and not ev.holding:
                b, t = ev.goals[todo[0]]
                out = refuse(f"{b} on {t} is NOT SATISFIED and has picks left")
                if out:
                    return out
            iface.call("done")
            return {"history": hist + ["done()"], "stop": True}
        if tool in ("observe", "check_constraints"):
            return {"history": hist + ["(the evidence table is up to date)"]}
        def do_push(obj: str) -> bool:
            """A10r: the push skill on `obj`, then the cameras; remembers where the block was."""
            was, why = box["xy"].get(obj), box["last"].get(obj, "first")
            r = iface.call("push", {"obj": obj, "direction": "center", "distance": PUSH_CM / 100})
            ev.holding = None
            ev.moved()
            if "stale" not in off:
                verify(ev)
            if r.get("ok"):
                box["pushed"][obj] = {"from": was, "why": why}
                box["last"][obj] = "pushed"
            return bool(r.get("ok"))

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
            elif choice and ev.goals and idx is None and not ev.holding:
                out = refuse(f"{obj} is not picked up in the plan; the plan picks up "
                             + ", ".join(b for b, _ in ev.goals), hard=True)
            if out:
                return out
            box["streak"] = 0
            if (learning and not ev.holding and box["last"].get(obj) in ("failed pick", "failed place")
                    and coin.random() < P_EXPLORE and not iface.finished):
                if do_push(obj):                            # while learning: half of the retries start with a push
                    hist.append(f"push({obj}) -> executed (tried by the system); cameras were read")
                if iface.finished:
                    return {"history": hist, "stop": True}
            r = iface.call("pick", args)
            if not r.get("ok"):
                return {"history": hist + [f"pick({json.dumps(args)}) -> {json.dumps(r)}"], "stop": iface.finished}
            ev.picks[obj] = ev.picks.get(obj, 0) + 1
            was_at = box["xy"].get(obj)
            before = dict(box["xy"])                         # A11: where the cameras had every block before the attempt
            box["attempt"] = None
            ev.moved()
            held = None
            if "fingers" in off:                            # ablation: assume the pick worked
                ev.holding = obj if obj in names else None
                return {"history": hist + [f"pick({obj}) -> executed"], "stop": iface.finished}
            if not iface.finished:                          # the fingers say whether it worked
                h = iface.call("observe", {"query": "held"})
                held = (h.get("result") or {}).get("held") if h.get("ok") else None
            ev.holding = held
            if held is not None:                            # it has left wherever it was
                for i, g in enumerate(ev.goals):
                    if g[0] == held:
                        ev.prior[i], ev.attempted[i], ev.reads[i] = P_WAS_UNMET, True, {}
            note = f"gripper now holds {held}" if held else "gripper is EMPTY, the pick failed"
            if track and obj in names:
                if skill_memory is not None and was_at:
                    rec = {"tool": "pick", "block": obj, "x": round(was_at[0], 1), "y": round(was_at[1], 1),
                           "prev": box["last"].get(obj, "first"), "ok": held == obj}
                    if rec["prev"] == "pushed":
                        frm = box["pushed"].get(obj) or {}
                        rec["why"] = frm.get("why")
                        if frm.get("from"):
                            rec["fx"], rec["fy"] = round(frm["from"][0], 1), round(frm["from"][1], 1)
                    skill_memory.add(rec)
                box["from"] = was_at
                if avoid and held == obj:
                    box["attempt"] = (obj, before)           # recorded when it has been put down and the cameras were read
                if held != obj:                              # choose again, knowing where things are now
                    box["last"][obj] = "failed pick"
                    if held is None:
                        glance()
                        if avoid:
                            from .disturb_memory import attempt_record
                            memory.add(attempt_record(before, dict(box["xy"]), obj, None))
                        if choice:
                            choose(state["instruction"], f"The pick of {obj} failed")
            return {"history": hist + [f"pick({obj}) -> executed; {note}"], "stop": iface.finished}
        if tool == "place_on":
            if not ev.holding:
                out = refuse("the gripper is empty; pick a block first", hard=True)
                if out:
                    return out
            target, held = str(args.get("target", "")), ev.holding
            want = next((t for b, t in ev.goals if b == held), None) if choice else None
            if want and target != want:
                out = refuse(f"the plan puts {held} on {want}", hard=True)
                if out:
                    return out
            box["streak"] = 0
            r = iface.call("place_on", args)
            if not r.get("ok"):
                if "fingers" in off or "guards" in off:     # nothing was in the hand after all
                    ev.holding = None
                return {"history": hist + [f"place_on({json.dumps(args)}) -> {json.dumps(r)}"],
                        "stop": iface.finished}
            idx = next((i for i, g in enumerate(ev.goals) if g == (held, target)), None)
            ev.holding = None
            ev.moved(placed_goal=idx)
            if "stale" not in off or idx is not None:
                verify(ev, idx if "stale" in off else None)
            if avoid and box.get("attempt") and box["attempt"][0] == held and not iface.finished:
                from .disturb_memory import attempt_record
                memory.add(attempt_record(box["attempt"][1], dict(box["xy"]), held, target))
            box["attempt"] = None
            if track and held in names:
                done = idx is not None and ev.p(idx) >= 0.5
                if skill_memory is not None and box.get("from"):
                    skill_memory.add({"tool": "place_on", "block": held, "x": round(box["from"][0], 1),
                                      "y": round(box["from"][1], 1), "target": target,
                                      "kind": "zone" if target in zones else "stack", "ok": bool(done)})
                box["last"][held] = "placed" if done else "failed place"
                if choice and not done and not iface.finished:
                    choose(state["instruction"], f"{held} did not end up on {target}")
            return {"history": hist + [f"place_on({target}) -> executed with {held}; cameras were read"],
                    "stop": iface.finished}
        if recover:                                         # A10r: push(obj), toward the middle
            obj = str(args.get("obj", ""))
            idx = next((i for i, g in enumerate(ev.goals) if g[0] == obj), None)
            out = None
            if ev.holding:
                out = refuse(f"you are holding {ev.holding}; place it first", hard=True)
            elif obj not in names:
                out = refuse(f"push needs \"obj\", one of {names}", hard=True)
            elif idx is not None and ev.fresh and ev.p(idx) >= HI:
                out = refuse(f"the evidence says {obj} is already on {ev.goals[idx][1]}")
            elif box["last"].get(obj) == "pushed":
                out = refuse(f"{obj} has been pushed since its last pick; pick it now", hard=True)
            if out:
                return out
            box["streak"] = 0
            ok = do_push(obj)
            return {"history": hist + [f"push({obj}) -> " + ("executed; cameras were read" if ok else "not executed")],
                    "stop": iface.finished}
        # push
        box["streak"] = 0
        r = iface.call(tool, args)
        ev.holding = None
        ev.moved()
        if "stale" not in off:
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
