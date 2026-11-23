"""Cross-episode memory of what the arm did to the blocks it was not handling.

The skill is frozen and it is not careful: while it picks a block up and carries it, the hand
or the carried block can hit other blocks.  The agent can see that for itself.  It has the
camera positions of all blocks before an attempt and after it, so for every attempt it can
note, for each block it was not handling, how that block lay (how far from the block that was
picked up, how far from the straight line to the target) and whether it was moved.  The true
scene is never read.

`mode` is the form in which this reaches the model when it chooses which block goes where:

  none      nothing
  told      the designer's rule, no memory (a reference: what a memory could at best teach)
  episodic  the most recent attempts, as they are
  lessons   sentences the model wrote itself, updated after every episode it learned from
  map       for each candidate placement, how often earlier carries with other blocks lying
            like this moved one of them: counted by code, not by the model

In every form the model is shown the same table of candidate placements with their geometry
(length of the carry, the other block nearest to the path, the other block nearest to the
block picked up), so the forms differ only in what is known about what that geometry means.
"""
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

MODES = ("none", "told", "episodic", "lessons", "map")
ZONE_CM = {"zone_A": (11.0, -11.0), "zone_B": (11.0, 0.0), "zone_C": (11.0, 11.0)}
MOVED_CM = 2.0          # a block counts as moved from this displacement on
SIGMA_CM = 2.0          # width of "lying like this" in both distances
FAR_CM = 20.0           # distances beyond this are all the same: far
RECENT = 16             # how many attempts the episodic form shows
MAX_LESSONS = 6
XY = Tuple[float, float]

TOLD = ("RULE (from the designer of the skill): the arm carries a block low and in a straight line from where it lies "
        "to its target. Another block within about 6 cm of that line is usually pushed away; beyond 10 cm it is safe. "
        "A block within about 6 cm of the block that is picked up is often hit as well. A block that is pushed away can "
        "leave the pad or the stack it was on, or end at the rim. Choose placements whose path is clear of other blocks.")

LESSON_SYSTEM = """You keep a short list of lessons about a robot arm's pick-and-place skill. While the arm picks a \
block up and carries it to its target, it sometimes moves other blocks that it was not asked to move; a block that is \
moved can leave the pad or the stack it was on. After each episode you see what was tried and what happened to the \
other blocks, and you update the list so that a later episode can choose which block to carry to which target in a \
way that leaves the other blocks alone.

Reply with exactly one JSON object and nothing else:
{"lessons": ["<lesson>", ...]}
At most """ + str(MAX_LESSONS) + """ lessons, each one sentence of at most 30 words. A lesson must be general: about \
how the other blocks have to lie, relative to the block that is picked up and to its path, for them to be moved or to \
stay, and what to choose because of it. Keep earlier lessons that still hold, correct the ones the new episode \
contradicts, and merge repeats."""


def _dist(a: XY, b: XY) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_dist(p: XY, a: XY, b: XY) -> float:
    """Distance of p from the straight line segment a -> b."""
    ab = (b[0] - a[0], b[1] - a[1])
    L = ab[0] ** 2 + ab[1] ** 2
    t = 0.0 if L < 1e-9 else min(1.0, max(0.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / L))
    return _dist(p, (a[0] + t * ab[0], a[1] + t * ab[1]))


def end_of(target: Optional[str], xy: Dict[str, XY]) -> Optional[XY]:
    return ZONE_CM.get(target) or xy.get(target) if target else None


def lay(xy: Dict[str, XY], block: str, target: Optional[str]) -> List[Tuple[str, float, Optional[float]]]:
    """For every other block: (name, distance from `block`, distance from the path block -> target)."""
    start, end = xy[block], end_of(target, xy)
    return [(n, _dist(p, start), path_dist(p, start, end) if end else None)
            for n, p in xy.items() if n not in (block, target)]


def attempt_record(before: Dict[str, XY], after: Dict[str, XY], block: str, target: Optional[str]) -> Optional[Dict[str, Any]]:
    """One attempt on `block` as the agent saw it: target None = the pick failed, nothing was carried."""
    if block not in before or (target and end_of(target, before) is None):
        return None
    others = [{"d_pick": round(dp, 1), "d_path": None if dl is None else round(dl, 1), "moved": round(_dist(after[n], before[n]), 1)}
              for n, dp, dl in lay(before, block, target) if n in after]
    end = end_of(target, before)
    return {"carried": target is not None, "to": None if not target else "zone" if target in ZONE_CM else "block",
            "carry": round(_dist(before[block], end), 1) if end else None, "others": others}


def describe(rec: Dict[str, Any]) -> str:
    def other(o: Dict[str, Any]) -> str:
        lies = (f"a block {o['d_path']:.1f} cm from the straight path and {o['d_pick']:.1f} cm from the picked block"
                if rec["carried"] else f"a block {o['d_pick']:.1f} cm from it")
        return lies + (f" was MOVED {o['moved']:.0f} cm" if o["moved"] >= MOVED_CM else " stayed")
    head = (f"carried a block {rec['carry']:.0f} cm to {'a zone pad' if rec['to'] == 'zone' else 'the top of another block'}"
            if rec["carried"] else "tried to pick a block up and failed")
    return head + ": " + ("; ".join(other(o) for o in rec["others"]) or "no other block was seen")


class DisturbMemory:
    def __init__(self, path: Optional[str] = None, learn: bool = False) -> None:
        self.path, self.learn = path, learn
        self.records: List[Dict[str, Any]] = []
        self.lessons: List[str] = []
        self.episodes = 0
        self._episode: List[Dict[str, Any]] = []
        if path and os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            self.records, self.lessons, self.episodes = d["records"], d["lessons"], d.get("episodes", 0)

    # ------------------------------------------------------------------ writing
    def add(self, rec: Optional[Dict[str, Any]]) -> None:
        if self.learn and rec:
            self._episode.append(rec)

    def end_episode(self, llm=None, lessons: bool = True) -> None:
        """Commit this episode's attempts; with an LLM, let it revise its lessons."""
        if not self.learn:
            return
        new, self._episode = self._episode, []
        self.records += new
        self.episodes += 1
        if llm is not None and lessons and new:
            out = llm.chat_json([
                {"role": "system", "content": LESSON_SYSTEM},
                {"role": "user", "content": "LESSONS SO FAR:\n" + ("\n".join(f"- {x}" for x in self.lessons) or "(none)")
                 + "\n\nTHIS EPISODE:\n" + "\n".join(f"- {describe(r)}" for r in new)
                 + "\n\nWrite the updated list."}],
                schema={"type": "object", "required": ["lessons"], "properties": {
                    "lessons": {"type": "array", "maxItems": MAX_LESSONS, "items": {"type": "string"}}}})
            got = (out or {}).get("lessons") if isinstance(out, dict) else None
            if isinstance(got, list) and got:
                self.lessons = [str(x).strip() for x in got if str(x).strip()][:MAX_LESSONS]
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w") as f:
                json.dump({"episodes": self.episodes, "lessons": self.lessons, "records": self.records}, f)

    # ------------------------------------------------------------------ reading
    def expect_moved(self, d_pick: float, d_path: float) -> Optional[float]:
        """How often a block lying like this was moved in earlier carries (kernel average); None = no such case."""
        n = k = 0.0
        a, b = min(d_pick, FAR_CM), min(d_path, FAR_CM)
        for r in self.records:
            if not r["carried"]:
                continue
            for o in r["others"]:
                w = math.exp(-((min(o["d_pick"], FAR_CM) - a) ** 2 + (min(o["d_path"], FAR_CM) - b) ** 2) / (2 * SIGMA_CM ** 2))
                n += w
                k += w * (o["moved"] >= MOVED_CM)
        return None if n < 1.0 else (k + 0.5) / (n + 1.0)

    def notes(self, mode: str, xy: Dict[str, XY], candidates: List[Tuple[str, str]]) -> str:
        """The candidate placements with their geometry, and what is known about what it means."""
        rows = []
        for block, target in candidates:
            if block not in xy or end_of(target, xy) is None:
                continue
            others = lay(xy, block, target)
            row = f"  {block} -> {target}: carry {_dist(xy[block], end_of(target, xy)):.0f} cm"
            if others:
                n1, _, d1 = min(others, key=lambda o: o[2])
                n2, d2, _ = min(others, key=lambda o: o[1])
                row += (f"; other block nearest to the straight path: {n1}, {d1:.1f} cm from it"
                        f"; other block nearest to {block}: {n2}, {d2:.1f} cm away")
                if mode == "map" and self.records:
                    ps = [(self.expect_moved(dp, dl), n) for n, dp, dl in others]
                    if any(p is None for p, _ in ps):
                        row += " | no earlier carry with blocks lying like this"
                    else:
                        any_moved = 1 - math.prod(1 - p for p, _ in ps)
                        p, n = max(ps)
                        row += f" | moved another block in about {100 * any_moved:.0f}% of earlier carries like this (most at risk: {n})"
            rows.append(row)
        text = ("CANDIDATE PLACEMENTS (lengths in cm, worked out from the camera positions; the path is the straight line "
                "from the block to its target):\n" + "\n".join(rows)) if rows else ""
        know = ""
        if mode == "told":
            know = TOLD
        elif mode == "episodic" and self.records:
            know = ("EARLIER ATTEMPTS (the most recent ones from earlier episodes; what happened to the blocks that were "
                    "not being handled):\n" + "\n".join(f"  {describe(r)}" for r in self.records[-RECENT:]))
        elif mode == "lessons" and self.lessons:
            know = "LESSONS (you wrote these after earlier episodes):\n" + "\n".join(f"  - {x}" for x in self.lessons)
        elif mode == "map" and self.records:
            n = sum(r["carried"] for r in self.records)
            know = (f"SKILL RECORD: the percentages in the table were counted by the system from {n} carries in earlier "
                    "episodes. A block that is moved can leave the pad or the stack it was on.")
        return "\n\n".join(x for x in (text, know) if x)
