"""The four agent architectures, as four wirings of one node library.

  A1  plan-and-execute   plan once, execute blind, never look back
  A2  ReAct              one call at a time, seeing every result
  A3  A2 + reflection    after every world-changing action, say what you believe
  A4  A3 + memory        plus lessons retrieved from, and written to, earlier episodes

Nothing below changes the tools, the backbone, the prompt core or the way results
are rendered.  The only variables are the edges and, through them, what is in the
context window when the model is asked to choose.
"""
import functools
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, StateGraph

from .nodes import (
    EpisodeState,
    make_act_node,
    make_assess_node,
    make_belief_node,
    make_crosscheck_node,
    make_execute_node,
    make_gate_node,
    make_plan_node,
    make_review_node,
    make_recall_node,
    make_reflect_node,
    make_step_plan_node,
    not_stopped,
    should_crosscheck,
    should_reflect,
)


def _plan_exhausted(state: EpisodeState) -> str:
    if state.get("stop"):
        return "stop"
    return "step" if state.get("cursor", 0) < len(state.get("plan", [])) else "stop"


def build_a1(iface, llm, memory=None, system=None):
    g = StateGraph(EpisodeState)
    g.add_node("plan", make_plan_node(llm, system))
    g.add_node("step", make_step_plan_node())
    g.add_node("execute", make_execute_node(iface))
    g.set_entry_point("plan")
    g.add_edge("plan", "step")
    g.add_edge("step", "execute")
    g.add_conditional_edges("execute", _plan_exhausted, {"step": "step", "stop": END})
    return g.compile()


def build_a2(iface, llm, memory=None, system=None):
    g = StateGraph(EpisodeState)
    g.add_node("act", make_act_node(llm, system))
    g.add_node("execute", make_execute_node(iface))
    g.set_entry_point("act")
    g.add_edge("act", "execute")
    g.add_conditional_edges("execute", not_stopped, {"act": "act", "stop": END})
    return g.compile()


def build_a3(iface, llm, memory=None, system=None):
    g = StateGraph(EpisodeState)
    g.add_node("act", make_act_node(llm, system))
    g.add_node("execute", make_execute_node(iface))
    g.add_node("reflect", make_reflect_node(llm, system))
    g.set_entry_point("act")
    g.add_edge("act", "execute")
    g.add_conditional_edges("execute", should_reflect,
                            {"reflect": "reflect", "act": "act", "stop": END})
    g.add_edge("reflect", "act")
    return g.compile()


def build_a4(iface, llm, memory=None, system=None):
    from .nodes import make_store_node
    g = StateGraph(EpisodeState)
    g.add_node("recall", make_recall_node(memory))
    g.add_node("act", make_act_node(llm, system))
    g.add_node("execute", make_execute_node(iface))
    g.add_node("reflect", make_reflect_node(llm, system))
    g.add_node("store", make_store_node(llm, memory, system))
    g.set_entry_point("recall")
    g.add_edge("recall", "act")
    g.add_edge("act", "execute")
    g.add_conditional_edges("execute", should_reflect,
                            {"reflect": "reflect", "act": "act", "stop": "store"})
    g.add_edge("reflect", "act")
    g.add_edge("store", END)
    return g.compile()


def build_a5(iface, llm, memory=None, system=None):
    """ReAct with a forced cross-check: every look is repeated from a second
    camera, and done() is refused until the last action has been looked at."""
    g = StateGraph(EpisodeState)
    g.add_node("act", make_act_node(llm, system))
    g.add_node("gate", make_gate_node())
    g.add_node("execute", make_execute_node(iface))
    g.add_node("crosscheck", make_crosscheck_node(iface))
    g.set_entry_point("act")
    g.add_edge("act", "gate")
    g.add_edge("gate", "execute")
    g.add_conditional_edges("execute", should_crosscheck,
                            {"crosscheck": "crosscheck", "act": "act", "stop": END})
    g.add_conditional_edges("crosscheck", not_stopped, {"act": "act", "stop": END})
    return g.compile()


def _after_every_result(node_name, make_node):
    def builder(iface, llm, memory=None, system=None):
        g = StateGraph(EpisodeState)
        g.add_node("act", make_act_node(llm, system))
        g.add_node("execute", make_execute_node(iface))
        g.add_node(node_name, make_node(llm, system))
        g.set_entry_point("act")
        g.add_edge("act", "execute")
        g.add_conditional_edges("execute", not_stopped, {"act": node_name, "stop": END})
        g.add_edge(node_name, "act")
        return g.compile()
    return builder


build_a6 = _after_every_result("assess", make_assess_node)     # ReAct + mandatory assessment
build_a8 = _after_every_result("review", make_review_node)     # ReAct + independent reviewer agent
build_a7 = _after_every_result("belief", make_belief_node)     # ReAct + explicit goal-status table


from .evidence import build_a9  # noqa: E402

BUILDERS: Dict[str, Callable[..., Any]] = {
    "A1": build_a1, "A2": build_a2, "A3": build_a3, "A4": build_a4, "A5": build_a5, "A6": build_a6, "A7": build_a7, "A8": build_a8,
    "A9": build_a9,
}
# A10: A9 for tasks that leave the blocks open, with a cross-episode memory of the skill in one of four forms
for _form in ("none", "episodic", "lessons", "map"):
    BUILDERS[f"A10-{_form}"] = functools.partial(build_a9, choice=_form)
# A10r: A9 with the push skill as a way to recover; the form is what the model is shown to decide when to push
for _form in ("none", "told", "episodic", "lessons", "map"):
    BUILDERS[f"A10r-{_form}"] = functools.partial(build_a9, recover=_form)
A9_PARTS = ("single", "prior", "stale", "doubt", "fingers", "guards")
for _part in A9_PARTS:                                    # A9 with exactly one part removed
    BUILDERS[f"A9-{_part}"] = functools.partial(build_a9, off=frozenset({_part}))
DESCRIPTIONS = {
    "A1": "plan-and-execute (no feedback)",
    "A2": "ReAct",
    "A3": "ReAct + reflection",
    "A4": "ReAct + reflection + cross-episode memory",
    "A5": "ReAct + forced cross-check of perception",
    "A6": "ReAct + mandatory assessment of every result",
    "A7": "ReAct + explicit belief state over the goals",
    "A8": "ReAct + independent reviewer agent (sees the log, not the reasoning)",
    "A9": "evidence-aware recovery: per-goal evidence kept by code, the model decides on its summary",
}


def build(name: str, iface, llm, memory=None, system=None):
    if name not in BUILDERS:
        raise KeyError(f"unknown agent {name!r}; choose from {list(BUILDERS)}")
    return BUILDERS[name](iface, llm, memory, system)
