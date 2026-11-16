"""Imperfect perception, between the true scene graph and what `observe` returns.

The simulator still knows the truth and scoring still uses only the truth.  What
changes is that the agent's camera can be wrong about the goals of the task.

Errors are defined against the truth, so their rates mean what they say:
  false success   a goal that is not met is reported as met        (p_fs, default 0.30)
  false failure   a goal that is met is reported as unmet          (p_ff, default 0.10)
                  -- as a wrong relation, or by not detecting the block at all
False success is the commoner error on purpose: it is the direction real success
detectors err in, and the one that makes an agent stop early.

Modes
  none        the camera is always right
  transient   every look draws its errors independently; looking again usually helps
  persistent  errors are a function of (camera view, true scene): the same view
              repeats them exactly, another view draws afresh
  random      control: an arbitrary block's zone is misreported with probability p,
              independently per look -- noise unrelated to the task
"""
import copy
import json
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VIEWS = ("front", "top", "side")
MODES = ("none", "transient", "persistent", "random")


def _reclear(g: Dict[str, Any]) -> None:
    supported = {b for b in g["on_top_of"].values() if b is not None}
    g["clear"] = {n: (n not in supported) for n in g["clear"]}


def _report(g: Dict[str, Any]) -> str:
    """Everything a query could expose, as one comparable string."""
    return json.dumps({k: g[k] for k in ("in_zone", "on_top_of", "clear", "off_table", "held")}
                      | {"hidden": g.get("hidden", [])}, sort_keys=True)


class PerceptionModel:
    def __init__(self, spec: Dict[str, Any], seed: int, mode: str = "none",
                 p_fs: float = 0.30, p_ff: float = 0.10, p: float = 0.25) -> None:
        if mode not in MODES:
            raise ValueError(f"perception mode must be one of {MODES}")
        self.spec, self.seed, self.mode, self.p = spec, seed, mode, p
        self.p_fs, self.p_ff = p_fs, p_ff
        self.rng = np.random.default_rng(seed + 7919)
        self.goals = [g for g in spec.get("goals", []) if g["type"] in ("in_zone", "on_top_of")]
        objs: List[str] = []
        for g in self.goals:
            objs += [g["obj"]] if g["type"] == "in_zone" else [g["top"], g["bottom"]]
        self.goal_objs = list(dict.fromkeys(objs))

    # ------------------------------------------------------------- candidates
    @staticmethod
    def _met(g, graph) -> bool:
        if g["type"] == "in_zone":
            return graph["in_zone"].get(g["obj"]) == g["zone"]
        return graph["on_top_of"].get(g["top"]) == g["bottom"]

    def _random_flip(self, graph, rng) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Control: one arbitrary block's zone is reported wrongly, task-blind."""
        on = list(graph["in_zone"])
        if not on:
            return None
        obj = on[int(rng.integers(len(on)))]
        zones = [z for z in list(graph["zones"]) + [None] if z != graph["in_zone"][obj]]
        out = copy.deepcopy(graph)
        out["in_zone"][obj] = zones[int(rng.integers(len(zones)))]
        return out, {"type": "random", "object": obj}

    # ---------------------------------------------------------------- perceive
    def _hide(self, out, obj) -> None:
        for k in ("in_zone", "clear", "on_top_of"):
            out[k].pop(obj, None)
        out["on_top_of"] = {k: (None if v == obj else v) for k, v in out["on_top_of"].items()}
        if out["held"] == obj:
            out["held"] = None
        out.setdefault("hidden", []).append(obj)

    def _set(self, out, g, met: bool) -> None:
        if g["type"] == "in_zone":
            out["in_zone"][g["obj"]] = g["zone"] if met else None
            if met:
                out["on_top_of"][g["obj"]] = None
        else:
            out["on_top_of"][g["top"]] = g["bottom"] if met else None
            if met:
                out["in_zone"][g["top"]] = out["in_zone"].get(g["bottom"])
        if met and out["held"] == (g.get("obj") or g.get("top")):
            out["held"] = None

    def perceive(self, graph: Dict[str, Any], view: str = "front"
                 ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Return (what this camera reports, the fault that was injected or None).

        Errors are conditioned on the truth, goal by goal:
          a goal that is NOT met is shown as met with probability p_fs
          a goal that IS met is shown as unmet with probability p_ff -- half the
          time as a wrong relation, half the time by not detecting the block
        """
        if self.mode == "none":
            return graph, None
        if self.mode == "persistent":
            key = f"{self.seed}|{view}|{_report(graph)}"
            rng = np.random.default_rng(zlib.crc32(key.encode()))
        else:
            rng = self.rng
        if self.mode == "random":
            if rng.random() >= self.p:
                return graph, None
            made = self._random_flip(graph, rng)
            if made is None:
                return graph, None
            made[1].update(view=view, mode=self.mode)
            return made
        out, errors = copy.deepcopy(graph), []
        for g in self.goals:
            u, how = rng.random(), rng.random()            # always drawn, so the stream is stable
            objs = [g["obj"]] if g["type"] == "in_zone" else [g["top"], g["bottom"]]
            if any(o not in out["in_zone"] for o in objs):  # off the table or already hidden
                continue
            met = self._met(g, graph)
            if not met and u < self.p_fs:
                self._set(out, g, True)
                errors.append({"goal": g, "direction": "false_success", "type": "flip"})
            elif met and u < self.p_ff:
                if how < 0.5:
                    self._set(out, g, False)
                    errors.append({"goal": g, "direction": "false_failure", "type": "flip"})
                else:
                    self._hide(out, objs[0])
                    errors.append({"goal": g, "direction": "false_failure", "type": "miss"})
        _reclear(out)
        if not errors or _report(out) == _report(graph):
            return graph, None
        return out, {"type": errors[0]["type"], "direction": errors[0]["direction"],
                     "errors": errors, "view": view, "mode": self.mode}
