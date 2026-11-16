"""One node library, shared by all four graphs.

Everything that could confound the comparison lives here so that it is
physically the same object in every condition: the system prompt core, the tool
vocabulary, the way results are rendered into context, and the reflection and
memory prompts.  A1-A4 differ only in which of these nodes they wire together
and in what therefore enters the context window.
"""
import json
from typing import Any, Dict, List, Optional, TypedDict

from agent_iface.tools import TOOL_NAMES

# --------------------------------------------------------------------- state
class EpisodeState(TypedDict, total=False):
    instruction: str
    history: List[str]          # rendered call -> result lines, oldest first
    plan: List[Dict[str, Any]]  # A1 only
    cursor: int                 # A1 only
    unverified: bool            # A5: an action has happened since the last look
    belief: str                 # A7: the current goal-status table, rendered
    reviews: List[str]          # A8: notes from the independent reviewer
    reflections: List[str]      # A3, A4
    memories: List[str]         # A4 only
    last: Dict[str, Any]
    steps: int
    stop: bool


# ------------------------------------------------------------------- prompts
SYSTEM_CORE = """You control a Franka Panda robot on a tabletop. There are coloured blocks \
(named like 'red_block') and three zone pads: zone_A, zone_B, zone_C.

TOOLS
  pick(obj)                       close the gripper on a block
  place_on(target)                put the held block on a zone pad, or on top of another block
  push(obj, direction, distance)  direction is "+x", "-x", "+y" or "-y"; distance in metres, max 0.15
  observe(query)                  query is "scene", "zones", "stacks", "held", or "where:<obj>"
  check_constraints()             list the constraints you are currently violating
  done()                          declare the instruction satisfied and end the episode

HOW THE WORLD WORKS
- The gripper holds at most one block. place_on moves the block that is currently held, so it only
  works straight after a pick; with nothing held it returns an error.
- pick and push both begin by opening the gripper. Calling either one while a block is held drops it.
- Handle one block at a time: pick it, place_on its target, then start on the next block.
- The low-level skill is unreliable. An action that fails can simply be tried again, and a second
  attempt often works.
- You have a limited number of calls. Once you believe every part of the instruction is satisfied,
  call done(). Acting on blocks that are already where they belong can undo your own work.

WHAT THE RESULTS MEAN
pick, place_on and push return only what the executor CLAIMS happened. The claim is built from the
gripper's own sensors: pick reports success when the fingers closed on something block-sized, place_on
and push when the gripper reached the commanded pose. It will say success when the wrong thing was
between the fingers, the block slipped out on the way, or it was knocked aside. A claim of success is
evidence, not fact. claimed_success false means the executor itself knows the action did not happen.

observe and check_constraints read the true scene, but they draw on a limited budget for the whole
episode. Spend them where holding a wrong belief would cost you the task, not after every action."""

# The perception study.  Same tools and same account of the executor's claims;
# observe is no longer metered or guaranteed, and takes a camera view.  Nothing
# here says how reliable the cameras are: that is an experimental variable and
# reaches the agent, when it does, through the tool results.
SYSTEM_V2 = """You control a Franka Panda robot on a tabletop. There are coloured blocks \
(named like 'red_block') and three zone pads: zone_A, zone_B, zone_C.

TOOLS
  pick(obj)                       close the gripper on a block
  place_on(target)                put the held block on a zone pad, or on top of another block
  push(obj, direction, distance)  direction is "+x", "-x", "+y" or "-y"; distance in metres, max 0.15
  observe(query, view)            query is "scene", "zones", "stacks", "held", or "where:<obj>";
                                  view is "front", "top" or "side" -- three separate cameras
  check_constraints(view)         list the constraints that camera sees violated
  done()                          declare the instruction satisfied and end the episode

HOW THE WORLD WORKS
- The gripper holds at most one block. place_on moves the block that is currently held, so it only
  works straight after a pick. With nothing held the arm goes through the motion and nothing changes.
- After place_on the gripper is empty, whether or not the block ended up where you wanted. A retry
  therefore always starts with pick; place_on twice in a row does nothing.
- pick and push both begin by opening the gripper. Calling either one while a block is held drops it.
- Handle one block at a time: pick it, place_on its target, then start on the next block.
- The low-level skill is unreliable. An action that fails can simply be tried again, and a second
  attempt often works.
- After placing a block, look at the scene to see whether the block is where the instruction wants it.
  If it is not, that attempt failed: pick the same block and place it again. A part of the instruction
  may need several attempts.
- Before calling done(), compare every part of the instruction with the most recent camera report.
  Call done() only when the report shows every part satisfied, or when you have already tried a part
  four times. Do not move blocks that the report shows are already where they belong.

WHAT THE RESULTS MEAN
pick, place_on and push return only that the motion was executed. They do not say whether it worked:
the block may have slipped out of the gripper, been dropped on the way, or been set down in the wrong
place. The cameras are the only way to find out what the scene looks like after an action.

observe and check_constraints return what one camera's perception system reports about the scene."""

ACT_ASK = ("What is your next tool call? If every part of the instruction has already "
           "been achieved, the next call is done().")
def _ask_v2(state: "EpisodeState") -> str:
    """The question put to the model each turn in the camera study.  It is built
    from the log only -- whether a look has happened since the last action, and
    how often each block has been picked -- never from the true scene."""
    picks: Dict[str, int] = {}
    for line in state.get("history") or []:
        if line.startswith("pick(") or "] pick(" in line:
            try:
                obj = json.loads(line[line.index("(") + 1:line.index(") ->")]).get("obj")
            except (ValueError, AttributeError):
                obj = None
            if obj:
                picks[obj] = picks.get(obj, 0) + 1
    count = ("PICK ATTEMPTS SO FAR: " + (", ".join(f"{o} x{n}" for o, n in picks.items()) or "none")
             + ". A block may be attempted at most four times.\n\n")
    if not state.get("history"):
        return count + "What is your first tool call?"
    if state.get("unverified"):
        return count + ("You have acted since your last look, so you do not yet know the result. "
                        "What is your next tool call?")
    return count + ("Your most recent camera report is up to date; looking again without acting will show "
                    "the same thing. Check each part of the instruction against that report. If a part is "
                    "not shown as satisfied and its block has attempts left, pick that block now and then "
                    "place it again. Only if every part is shown as satisfied, or no attempts are left, is "
                    "the next call done(). What is your next tool call?")


ACT_FORMAT = """Reply with exactly one JSON object and nothing else:
{"thought": "<one sentence>", "tool": "<tool name>", "args": {<arguments>}}
Use {} for args when the tool takes none."""

PLAN_FORMAT = """Reply with exactly one JSON object and nothing else:
{"plan": [{"tool": "<tool name>", "args": {<arguments>}}, ...]}
Give the whole sequence now. You will not be asked again, and you will not see any result until the
sequence has finished, so do not include observe or check_constraints steps that you cannot react to.
End the plan with a done step."""

REFLECT_FORMAT = """Reply with exactly one JSON object and nothing else:
{"assessment": "<what you now believe about the world>", "trust_last_claim": true|false,
 "next_intent": "<what to do about it, one sentence>"}"""

# Enforced by constrained decoding, so the one-sentence thought is really written
# before the tool is chosen (with plain JSON mode the model simply omits it).
ACT_SCHEMA = {"type": "object",
              "properties": {"thought": {"type": "string"},
                             "tool": {"type": "string", "enum": list(TOOL_NAMES)},
                             "args": {"type": "object"}},
              "required": ["thought", "tool", "args"]}

ASSESS_FORMAT = """Reply with exactly one JSON object and nothing else:
{"assessment": "<what you now believe about the world, and on what evidence>",
 "trust_last_result": true|false, "next_intent": "<what to do about it, one sentence>"}"""

BELIEF_FORMAT = """Reply with exactly one JSON object and nothing else:
{"goals": [{"goal": "<one part of the instruction>", "status": "met"|"unmet"|"uncertain",
            "evidence": "<the result that decides it, in at most 12 words; do not quote the log>"}, ...]}
List every part of the instruction, in order, and nothing else."""

REVIEWER_ROLE = """

YOUR ROLE
You are not operating the robot. You are an independent reviewer of another agent's work. You see the
instruction and the raw log of tool calls and results -- not the operator's reasoning, plans or earlier
conclusions. Judge only from the log: decide whether the most recent result should be believed, given
everything else in the log."""

REVIEW_FORMAT = """Reply with exactly one JSON object and nothing else:
{"assessment": "<what the log supports about the world, and on what evidence>",
 "trust_last_result": true|false, "advice": "<what the operator should do about it, one sentence>"}"""

MEMORY_FORMAT = """Reply with exactly one JSON object and nothing else:
{"lesson": "<one transferable sentence about this environment, not about this particular scene>"}"""


def _args_of(step: Dict[str, Any]) -> Dict[str, Any]:
    """Arguments of a proposed call.  Small models often write the arguments
    next to "tool" instead of inside "args"; that is a formatting slip, not a
    decision, so both layouts are accepted -- identically in every condition."""
    args = step.get("args")
    if isinstance(args, dict) and args:
        return args
    return {k: v for k, v in step.items() if k not in ("tool", "args", "thought")}


def render_result(tool: str, args: Dict[str, Any], visible: Dict[str, Any]) -> str:
    """How a tool result enters the context window.  Identical in every
    condition -- if this differed, nothing else would be comparable."""
    call = f"{tool}({json.dumps(args, sort_keys=True)})" if args else f"{tool}()"
    return f"{call} -> {json.dumps(visible, sort_keys=True)}"


def _user_block(state: EpisodeState, extra: str = "") -> str:
    parts = [f"INSTRUCTION: {state['instruction']}"]
    if state.get("memories"):
        parts.append("LESSONS FROM EARLIER EPISODES:\n" + "\n".join(f"- {m}" for m in state["memories"]))
    if state.get("history"):
        parts.append("WHAT HAS HAPPENED SO FAR:\n" + "\n".join(state["history"]))
    else:
        parts.append("WHAT HAS HAPPENED SO FAR:\n(nothing yet)")
    if state.get("belief"):
        parts.append("YOUR CURRENT BELIEF ABOUT EACH GOAL:\n" + state["belief"])
    if state.get("reviews"):
        parts.append("NOTES FROM AN INDEPENDENT REVIEWER OF YOUR LOG:\n"
                     + "\n".join(f"- {r}" for r in state["reviews"][-3:]))
    if state.get("reflections"):
        parts.append("YOUR EARLIER ASSESSMENTS:\n" + "\n".join(f"- {r}" for r in state["reflections"][-3:]))
    parts.append(f"CALLS USED SO FAR: {state.get('steps', 0)}")
    rep = _repeats(state.get("history") or [])
    if rep >= 2:
        # Loop guard, shared by every closed-loop condition: a small model at
        # temperature 0 will otherwise re-issue one call until the budget is gone.
        parts.append(f"NOTE: your last {rep} calls were identical and returned the same result. "
                     "Repeating it again will change nothing. Choose a different call, or done() "
                     "if the instruction is satisfied.")
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def _repeats(history: List[str]) -> int:
    """Length of the run of identical lines at the end of the history."""
    n = 0
    for line in reversed(history):
        if line == history[-1]:
            n += 1
        else:
            break
    return n if history else 0


# --------------------------------------------------------------------- nodes
def make_plan_node(llm, system=None):
    """A1: produce the whole sequence up front, with no feedback."""
    def plan_node(state: EpisodeState) -> EpisodeState:
        out = llm.chat_json([
            {"role": "system", "content": (system or SYSTEM_CORE) + "\n\n" + PLAN_FORMAT},
            {"role": "user", "content": _user_block(state)},
        ])
        plan = []
        if isinstance(out, dict):
            raw = out.get("plan") or out.get("steps") or []
            if isinstance(raw, list):
                for s in raw:
                    if isinstance(s, dict) and s.get("tool") in TOOL_NAMES:
                        plan.append({"tool": s["tool"], "args": _args_of(s)})
        if not plan:                                  # unusable plan: still an episode
            plan = [{"tool": "done", "args": {}}]
        return {"plan": plan, "cursor": 0}
    return plan_node


def make_step_plan_node():
    """A1: hand the next pre-planned call to the executor, blind to results."""
    def step_plan_node(state: EpisodeState) -> EpisodeState:
        plan, cur = state.get("plan", []), state.get("cursor", 0)
        if cur >= len(plan):
            return {"last": {"tool": "done", "args": {}}, "stop": True}
        return {"last": dict(plan[cur]), "cursor": cur + 1}
    return step_plan_node


def make_act_node(llm, system=None):
    """A2-A4: choose one call, having seen everything so far."""
    def act_node(state: EpisodeState) -> EpisodeState:
        out = llm.chat_json([
            {"role": "system", "content": (system or SYSTEM_CORE) + "\n\n" + ACT_FORMAT},
            {"role": "user", "content": _user_block(
                state, _ask_v2(state) if system is SYSTEM_V2 else ACT_ASK)},
        ], schema=ACT_SCHEMA)
        if not isinstance(out, dict) or out.get("tool") not in TOOL_NAMES:
            # an unparseable turn is a wasted turn, not a crash
            return {"last": {"tool": "observe", "args": {"query": "scene"}},
                    "history": state.get("history", []) + ["(no valid tool call produced)"]}
        return {"last": {"tool": out["tool"], "args": _args_of(out),
                         "thought": out.get("thought", "")}}
    return act_node


def make_execute_node(iface):
    """Shared by every condition: the only way to touch the world."""
    def execute_node(state: EpisodeState) -> EpisodeState:
        call = state.get("last") or {"tool": "done", "args": {}}
        tool, args = call.get("tool", "done"), call.get("args") or {}
        visible = iface.call(tool, args)
        line = render_result(tool, args, visible)
        if call.get("forced"):
            line = "[required check before done] " + line
        out = {
            "history": state.get("history", []) + [line],
            "steps": state.get("steps", 0) + 1,
            "last": {**call, "visible": visible},
            "stop": iface.finished,
        }
        if tool in ("pick", "place_on", "push"):
            out["unverified"] = True
        elif tool in ("observe", "check_constraints") and visible.get("ok"):
            out["unverified"] = False
        return out
    return execute_node


def make_gate_node():
    """A5: done() is not accepted while an action is still unchecked.  The
    call is replaced by a look at the scene, and the agent is asked again."""
    def gate_node(state: EpisodeState) -> EpisodeState:
        last = state.get("last") or {}
        if last.get("tool") == "done" and state.get("unverified", True):
            return {"last": {"tool": "observe", "args": {"query": "scene", "view": "front"},
                             "forced": True}}
        return {}
    return gate_node


def make_crosscheck_node(iface):
    """A5: every look is repeated from a second camera, and from the third if
    the first two disagree.  Control flow does this, not the model."""
    from agent_iface.perception import VIEWS

    def answer(v: Dict[str, Any]) -> str:
        return json.dumps(v.get("result", v.get("violations")), sort_keys=True)

    def crosscheck_node(state: EpisodeState) -> EpisodeState:
        last = state.get("last") or {}
        tool, args, first = last.get("tool"), dict(last.get("args") or {}), last.get("visible") or {}
        if not first.get("ok") or iface.finished:
            return {}
        others = [v for v in VIEWS if v != first.get("view", "front")]
        hist, steps = list(state.get("history", [])), state.get("steps", 0)
        a2 = {**args, "view": others[0]}
        second = iface.call(tool, a2)
        hist.append("[automatic second view] " + render_result(tool, a2, second))
        steps += 1
        seen = [first] + ([second] if second.get("ok") else [])
        if second.get("ok") and answer(second) != answer(first) and not iface.finished:
            a3 = {**args, "view": others[1]}
            third = iface.call(tool, a3)
            hist.append("[automatic third view, the first two disagreed] " + render_result(tool, a3, third))
            steps += 1
            if third.get("ok"):
                seen.append(third)
        # Fuse the views in control flow: the answer most of them give becomes the
        # report the model is asked to act on.  No truth is involved.
        if len(seen) >= 2:
            counts: Dict[str, int] = {}
            for v in seen:
                counts[answer(v)] = counts.get(answer(v), 0) + 1
            best, votes = max(counts.items(), key=lambda kv: kv[1])
            bare = {k: v for k, v in args.items() if k != "view"}
            if votes >= 2:
                agreed = next(v for v in seen if answer(v) == best)
                fused = {k: v for k, v in agreed.items() if k != "view"}
                hist.append(f"[combined report: {votes} of {len(seen)} views agree] " + render_result(tool, bare, fused))
            else:
                hist.append(f"[combined report: the {len(seen)} views all disagree; the state of this is unknown] "
                            + render_result(tool, bare, {"ok": False, "error": "views disagree"}))
        return {"history": hist, "steps": steps, "unverified": False, "stop": iface.finished}
    return crosscheck_node


def should_crosscheck(state: EpisodeState) -> str:
    if state.get("stop"):
        return "stop"
    tool = (state.get("last") or {}).get("tool")
    return "crosscheck" if tool in ("observe", "check_constraints") else "act"


def make_reflect_node(llm, system=None):
    """A3, A4: after an action whose claim may be wrong, say what you believe."""
    def reflect_node(state: EpisodeState) -> EpisodeState:
        out = llm.chat_json([
            {"role": "system", "content": (system or SYSTEM_CORE) + "\n\n" + REFLECT_FORMAT},
            {"role": "user", "content": _user_block(
                state, "Assess the most recent result. Do you believe the claim?")},
        ])
        if not isinstance(out, dict):
            return {}
        note = (f"{out.get('assessment', '')} "
                f"(trusted_claim={out.get('trust_last_claim')}) -> {out.get('next_intent', '')}").strip()
        return {"reflections": state.get("reflections", []) + [note]}
    return reflect_node


def make_assess_node(llm, system=None):
    """A6: after EVERY result -- an action's claim or a camera report -- say
    whether you believe it.  The step is mandatory; what to conclude is not."""
    def assess_node(state: EpisodeState) -> EpisodeState:
        out = llm.chat_json([
            {"role": "system", "content": (system or SYSTEM_CORE) + "\n\n" + ASSESS_FORMAT},
            {"role": "user", "content": _user_block(
                state, "Assess the most recent result. Do you believe it?")},
        ])
        if not isinstance(out, dict):
            return {}
        note = (f"{out.get('assessment', '')} "
                f"(trusted={out.get('trust_last_result')}) -> {out.get('next_intent', '')}").strip()
        return {"reflections": state.get("reflections", []) + [note]}
    return assess_node


def make_review_node(llm, system=None):
    """A8: a second agent audits every result.  Same model, same cadence and the
    same question as A6's self-assessment; the difference is isolation.  The
    reviewer is given the instruction and the raw log only -- no thoughts, no
    earlier notes (its own or the operator's) -- so it cannot be led by them."""
    def review_node(state: EpisodeState) -> EpisodeState:
        log = "\n".join(state.get("history") or []) or "(nothing yet)"
        out = llm.chat_json([
            {"role": "system", "content": (system or SYSTEM_CORE) + REVIEWER_ROLE + "\n\n" + REVIEW_FORMAT},
            {"role": "user", "content": f"INSTRUCTION: {state['instruction']}\n\nLOG:\n{log}\n\n"
                                        "Review the most recent result. Should it be believed?"},
        ])
        if not isinstance(out, dict):
            return {}
        note = (f"{out.get('assessment', '')} "
                f"(trusted={out.get('trust_last_result')}) -> {out.get('advice', '')}").strip()
        return {"reviews": state.get("reviews", []) + [note]}
    return review_node


def make_belief_node(llm, system=None):
    """A7: keep an explicit status for every goal, rewritten after each result.
    The act node then chooses from this table as well as from the raw history."""
    def belief_node(state: EpisodeState) -> EpisodeState:
        out = llm.chat_json([
            {"role": "system", "content": (system or SYSTEM_CORE) + "\n\n" + BELIEF_FORMAT},
            {"role": "user", "content": _user_block(
                state, "Update your belief about each goal in the light of the most recent result.")},
        ])
        goals = out.get("goals") if isinstance(out, dict) else None
        if not isinstance(goals, list) or not goals:
            return {}
        lines = [f"- {g.get('goal', '?')}: {str(g.get('status', '?')).upper()} ({g.get('evidence', '')})"
                 for g in goals if isinstance(g, dict)]
        return {"belief": "\n".join(lines)} if lines else {}
    return belief_node


def make_store_node(llm, memory, system=None):
    """A4: distil one transferable lesson at the end of the episode."""
    def store_node(state: EpisodeState) -> EpisodeState:
        if memory is None:
            return {}
        out = llm.chat_json([
            {"role": "system", "content": (system or SYSTEM_CORE) + "\n\n" + MEMORY_FORMAT},
            {"role": "user", "content": _user_block(
                state, "The episode has ended. State one lesson that would transfer to a different scene.")},
        ])
        lesson = (out or {}).get("lesson") if isinstance(out, dict) else None
        if isinstance(lesson, str) and lesson.strip():
            memory.add(lesson.strip(), state["instruction"])
        return {}
    return store_node


def make_recall_node(memory, k: int = 3):
    """A4: retrieve lessons before acting."""
    def recall_node(state: EpisodeState) -> EpisodeState:
        if memory is None:
            return {"memories": []}
        return {"memories": memory.search(state["instruction"], k=k)}
    return recall_node


# ------------------------------------------------------------------- routing
def should_reflect(state: EpisodeState) -> str:
    """Reflect after an action whose claim could be false.  Deterministic, so
    the A2/A3 difference is control flow and not a sampling artefact."""
    if state.get("stop"):
        return "stop"
    last = state.get("last") or {}
    if last.get("tool") in ("pick", "place_on", "push"):
        return "reflect"
    return "act"


def not_stopped(state: EpisodeState) -> str:
    return "stop" if state.get("stop") else "act"
