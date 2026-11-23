"""The six-tool interface, the same for every agent and every condition.

Agents differ in control flow and in what enters the prompt, never in what they can do.
Enforced here rather than trusted to the agents:

  1. An agent sees only what a tool returns.  Every `_`-prefixed field from the executor
     (ground truth, step counts, forced drops) is stripped before the result is returned
     and kept in the transcript.
  2. With hide_claims (every run of the camera study) an action returns only that the
     motion was executed.  The cameras are then the only way to learn its outcome.
  3. An episode ends at done() or after max_calls tool calls.  observe_budget can meter
     looking separately (the first study, run_sweep.py); the camera study runs with
     observe_budget=None.

The descriptions in TOOL_SCHEMAS are documentation.  The agents' prompts spell the tools
out themselves (SYSTEM_V2 in agents/nodes.py, SYSTEM_A9 in agents/evidence.py).
"""
import json
from typing import Any, Dict, List, Optional

from envs.scene_graph import describe, evaluate, observe, scene_graph, violations
from envs.tabletop import ZONES

from .critic import warning
from .perception import VIEWS

INFO_LEVELS = ("none", "label", "monitor")
RELIABILITY_LABEL = ("camera reports are sometimes wrong. Reports of the same thing from two different "
                     "cameras are independent checks; asking the same camera again may just repeat its error.")

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "pick",
        "description": "Close the gripper on an object. Returns that the motion was executed (and, without hide_claims, "
                       "the executor's claimed outcome).",
        "parameters": {"type": "object", "properties": {
            "obj": {"type": "string", "description": "e.g. 'red_block'"}}, "required": ["obj"]},
    },
    {
        "name": "place_on",
        "description": "Place the held object on a zone pad or on top of another block. "
                       "Returns as pick does.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "'zone_A'|'zone_B'|'zone_C' or a block name"}},
            "required": ["target"]},
    },
    {
        "name": "push",
        "description": "Slide an object over the table, along an axis or toward the middle of the "
                       "workspace ('center'). Returns as pick does.",
        "parameters": {"type": "object", "properties": {
            "obj": {"type": "string"},
            "direction": {"type": "string", "enum": ["+x", "-x", "+y", "-y", "center"]},
            "distance": {"type": "number", "description": "metres, 0.0-0.15"}},
            "required": ["obj", "direction", "distance"]},
    },
    {
        "name": "observe",
        "description": "Ask one camera (view: front, top or side) about part of the scene. "
                       "Queries: 'scene', 'zones', 'stacks', 'held', 'where:<obj>'.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "check_constraints",
        "description": "Ask one camera which constraints it sees violated.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "done",
        "description": "Declare the instruction satisfied and end the episode.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

TOOL_NAMES = [t["name"] for t in TOOL_SCHEMAS]
# Every argument a tool call can carry, for back ends whose schemas cannot leave "args" open.
ARG_PROPERTIES: Dict[str, Any] = {k: {x: y for x, y in v.items() if x != "description"}
                                  for t in TOOL_SCHEMAS for k, v in t["parameters"]["properties"].items()}
ARG_PROPERTIES["view"] = {"type": "string"}            # observe: the camera to ask


class BudgetExhausted(Exception):
    pass


class ToolInterface:
    """Mediates every interaction between an agent and the world."""

    def __init__(
        self,
        env,
        executor,
        spec: Dict[str, Any],
        observe_budget: Optional[int] = 8,
        max_calls: int = 40,
        perception=None,
        info: str = "none",
        critic_calib: Optional[Dict[str, Any]] = None,
        hide_claims: bool = False,
    ) -> None:
        """perception=None keeps the original study's interface (a truthful,
        metered observe).  With a PerceptionModel, observe takes a `view`, can be
        wrong, and `info` sets what else the agent is told:
          none     the camera report only
          label    plus a fixed statement of how reliable the camera is
          monitor  plus a receipt that cross-checks the report against what the
                   gripper did, and an execution warning from the skill's critic
        observe_budget=None removes the perception meter."""
        if info not in INFO_LEVELS:
            raise ValueError(f"info must be one of {INFO_LEVELS}")
        self.env, self.executor, self.spec = env, executor, spec
        self.perception, self.info, self.critic_calib = perception, info, critic_calib
        # hide_claims: an action returns only that it was executed.  The executor's
        # own success claim is kept for the monitor and the log, not shown.
        self.hide_claims = hide_claims
        # what the gripper side believes, built only from claims: never from truth
        self.believed_held: Optional[str] = None
        self.handled: Dict[str, Dict[str, Any]] = {}
        self.observe_budget = observe_budget
        self.max_calls = max_calls
        self.observes_used = 0
        self.calls = 0
        self.done_called = False
        self.transcript: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------- helpers
    @property
    def finished(self) -> bool:
        return self.done_called or self.calls >= self.max_calls

    def _strip(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove everything the agent is not allowed to see."""
        return {k: v for k, v in result.items() if not k.startswith("_")}

    def _log(self, name: str, args: Dict[str, Any], full: Dict[str, Any],
             visible: Dict[str, Any]) -> None:
        self.transcript.append({
            "i": self.calls, "tool": name, "args": args,
            "visible": visible,
            "truth": full.get("_truth"),
            "claimed": full.get("claimed_success"),
            "info": full.get("_info"),
            "fault": full.get("_fault"),            # perception error shown in this answer
            "true_answer": full.get("_true"),       # what a perfect camera would have said
            "warned": full.get("_warned"),          # critic warning (computed in every condition)
            "observes_used": self.observes_used,
            "graph": scene_graph(self.env),
        })

    # -------------------------------------------------------------- telemetry
    def _note_action(self, name: str, args: Dict[str, Any], full: Dict[str, Any]) -> None:
        """Update the gripper-side record and attach the critic's warning.  The
        warning is computed in every condition (so it can be scored) and shown
        to the agent only under info='monitor'."""
        if not full.get("ok") or "claimed_success" not in full:
            return
        claimed = bool(full["claimed_success"])
        target = str(args.get("target", ""))
        kind = "pick" if name == "pick" else "push" if name == "push" else (
            "place_zone" if target in ZONES else "place_stack")
        fired, at = warning(self.critic_calib, kind, (full.get("_info") or {}).get("q_trace"))
        full["_warned"] = {"fired": fired, "step": at, "kind": kind}
        if self.info == "monitor" and self.critic_calib and kind in self.critic_calib:
            full["execution_warning"] = fired
        if name == "pick":
            obj = str(args.get("obj", ""))
            self.believed_held = obj if claimed else None
            self.handled[obj] = {"tool": "pick", "claimed": claimed, "warned": fired, "i": self.calls}
        elif name == "place_on":
            obj, self.believed_held = self.believed_held, None
            if obj is not None:
                self.handled[obj] = {"tool": "place_on", "target": target, "claimed": claimed,
                                     "warned": fired, "i": self.calls}
        else:
            self.believed_held = None

    def _receipt(self, shown: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-check a camera report against what the gripper did.  Uses the
        executor's claims and the critic warning only -- it has no access to the
        truth, so it can itself be wrong; it says where two sources disagree."""
        if "object" in shown:                                   # a where:<obj> answer
            o = shown["object"]
            zones = {o: shown.get("in_zone")} if "in_zone" in shown else None
            stacks = {o: shown.get("on_top_of")} if "on_top_of" in shown else None
            unseen = [o] if shown.get("visible") is False else []
        else:
            zones, stacks = shown.get("in_zone"), shown.get("on_top_of")
            unseen = [o for o in self.handled if zones is not None and o not in zones]
        conflicts: List[str] = []
        for obj, rec in self.handled.items():
            if obj in unseen:
                conflicts.append(f"{obj}: the camera does not report it at all, but it was handled at call {rec['i']}")
                continue
            if rec["tool"] != "place_on":
                continue
            tgt = rec["target"]
            table = zones if tgt in ZONES else stacks
            if table is None or obj not in table:
                continue
            there = table.get(obj) == tgt
            if rec["claimed"] and not rec["warned"] and not there:
                conflicts.append(f"{obj}: the camera does not show it at {tgt}, but place_on({tgt}) at call "
                                 f"{rec['i']} reached its target with no execution warning")
            elif there and not rec["claimed"]:
                conflicts.append(f"{obj}: the camera shows it at {tgt}, but place_on({tgt}) at call "
                                 f"{rec['i']} did not reach its target")
            elif there and rec["warned"]:
                conflicts.append(f"{obj}: the camera shows it at {tgt}, but place_on({tgt}) at call "
                                 f"{rec['i']} raised an execution warning")
        out: Dict[str, Any] = {"conflicts": conflicts}
        if conflicts:
            out["options"] = ["observe the same query from a different view",
                              "pick the block and place_on its target again",
                              "accept the camera report"]
        return out

    # ------------------------------------------------------------------- call
    def call(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute one tool call and return ONLY what the agent may see."""
        args = dict(args) if isinstance(args, dict) else {}
        if self.done_called:
            return {"ok": False, "error": "episode already ended"}
        if self.calls >= self.max_calls:
            return {"ok": False, "error": "call budget exhausted"}
        if name not in TOOL_NAMES:
            self.calls += 1
            return {"ok": False, "error": f"unknown tool {name!r}", "available": TOOL_NAMES}

        self.calls += 1
        full: Dict[str, Any]

        if name in ("pick", "place_on", "push"):
            # A model can put anything in an argument slot (a list for obj, the
            # string "far" for distance).  That is a wasted call, never a crash.
            try:
                if name == "pick":
                    full = self.executor.pick(str(args.get("obj", "")))
                elif name == "place_on":
                    full = self.executor.place_on(str(args.get("target", "")))
                else:
                    full = self.executor.push(str(args.get("obj", "")), str(args.get("direction", "")),
                                              float(args.get("distance", 0.0)))
            except (TypeError, ValueError) as e:
                full = {"ok": False, "error": f"bad arguments for {name}: {e}"}
            if self.hide_claims and str(full.get("error", "")).startswith("nothing is held"):
                # With claims hidden, an empty-handed place_on must not reveal that the
                # pick failed: the arm goes through the motion and nothing changes.
                full = {"ok": True, "claimed_success": False, "_truth": False, "_info": {"empty_hand": True}}
            self._note_action(name, args, full)
        elif name in ("observe", "check_constraints"):
            view = str(args.get("view") or "front")
            if self.observe_budget is not None and self.observes_used >= self.observe_budget:
                full = {"ok": False, "error": "perception budget exhausted",
                        "budget": self.observe_budget}
            elif self.perception is not None and view not in VIEWS:
                full = {"ok": False, "error": f"unknown view {view!r}", "views": list(VIEWS)}
            else:
                self.observes_used += 1
                true_graph = scene_graph(self.env)
                seen, fault = (self.perception.perceive(true_graph, view)
                               if self.perception is not None else (true_graph, None))
                if name == "observe":
                    q = str(args.get("query", "scene"))
                    key, shown, real = "result", observe(seen, q), observe(true_graph, q)
                else:
                    key, shown, real = "violations", violations(seen, self.spec), violations(true_graph, self.spec)
                full = {"ok": True, key: shown, "_true": real,
                        # a fault only counts if this particular answer shows it
                        "_fault": fault if shown != real else None}
                if self.perception is not None:
                    full["view"] = view
                if self.observe_budget is not None:
                    full["budget_left"] = self.observe_budget - self.observes_used
                if self.info in ("label", "monitor"):
                    full["reliability"] = RELIABILITY_LABEL
                if self.info == "monitor" and name == "observe":
                    full["monitor"] = self._receipt(shown)
                if self.perception is not None and name == "observe" and "held" in shown and "in_zone" in shown:
                    # the whole-scene report also comes as one sentence per block
                    full[key] = {"description": describe(seen), **shown}
                    full["_true"] = {"description": describe(true_graph), **real}
        else:  # done
            self.done_called = True
            full = {"ok": True, "ended": True}

        visible = self._strip(full)
        if self.hide_claims and "claimed_success" in visible:
            del visible["claimed_success"]
            visible["executed"] = True
        self._log(name, args, full, visible)
        return visible

    # ----------------------------------------------------------------- scoring
    def score(self) -> Dict[str, Any]:
        """Ground-truth episode score plus the cost side of the comparison."""
        graph = scene_graph(self.env)
        result = evaluate(graph, self.spec)
        claimed = [t for t in self.transcript if t["truth"] is not None]
        silent = [t for t in claimed if t.get("claimed") and t["truth"] is False]
        failed = [t for t in claimed if t["truth"] is False]
        wasted = [t for t in self.transcript
                  if t["tool"] in ("observe", "check_constraints") and t["visible"].get("ok")]
        # Pathologies worth counting rather than only reading in a transcript.
        sig = [(t["tool"], json.dumps(t["args"], sort_keys=True)) for t in self.transcript]
        immediate_repeats = sum(1 for i in range(1, len(sig)) if sig[i] == sig[i - 1])
        # longest run of a repeating 1- or 2-cycle, e.g. pick/push/pick/push
        loop_max = 0
        for period in (1, 2):
            run = 0
            for i in range(period, len(sig)):
                run = run + 1 if sig[i] == sig[i - period] else 0
                loop_max = max(loop_max, run + period if run else 0)
        # how long the agent carried a false belief: calls from each silent
        # failure to the next time it spent perception. None == never checked.
        verify_at = [t["i"] for t in self.transcript
                     if t["tool"] in ("observe", "check_constraints") and t["visible"].get("ok")]
        latencies = []
        for t in silent:
            later = [v for v in verify_at if v > t["i"]]
            latencies.append(later[0] - t["i"] if later else None)
        measured = [x for x in latencies if x is not None]
        # did the agent ever go back to an object a claim had lied about?
        lied_objs = {t["args"].get("obj") or t["args"].get("target") for t in silent}
        lied_objs.discard(None)
        retried = sum(1 for t in self.transcript
                      if t["tool"] in ("pick", "place_on", "push")
                      and (t["args"].get("obj") or t["args"].get("target")) in lied_objs
                      and any(s["i"] < t["i"] for s in silent))
        seen = [t for t in self.transcript
                if t["tool"] in ("observe", "check_constraints") and t["visible"].get("ok")]
        warned = [t for t in claimed if (t.get("warned") or {}).get("kind") in (self.critic_calib or {})]
        result.update({
            "failed_actions": len(failed),
            "faults_shown": sum(1 for t in seen if t.get("fault")),
            "warnings": sum(1 for t in warned if t["warned"]["fired"]),
            "warn_tp": sum(1 for t in warned if t["warned"]["fired"] and t["truth"] is False),
            "warn_fp": sum(1 for t in warned if t["warned"]["fired"] and t["truth"] is True),
            "warn_fn": sum(1 for t in warned if not t["warned"]["fired"] and t["truth"] is False),
            "warn_tn": sum(1 for t in warned if not t["warned"]["fired"] and t["truth"] is True),
            "calls": self.calls,
            "observes_used": self.observes_used,
            "declared_done": self.done_called,
            # the agent said it was finished but ground truth disagrees
            "false_completion": bool(self.done_called and not result["success"]),
            "action_calls": len(claimed),
            "silent_failures": len(silent),
            "silent_failure_rate": (len(silent) / len(claimed)) if claimed else 0.0,
            "verification_calls": len(wasted),
            "immediate_repeats": immediate_repeats,
            "loop_max": loop_max,
            "detection_latency": (sum(measured) / len(measured)) if measured else None,
            "undetected_silent_failures": sum(1 for x in latencies if x is None),
            "retries_after_lie": retried,
        })
        return result

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"spec": self.spec, "score": self.score(),
                       "transcript": self.transcript}, f, indent=2, default=str)
