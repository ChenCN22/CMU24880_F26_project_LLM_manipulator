"""Cross-episode memory of how the skill's calls turned out.

A9 keeps, inside one episode, what the cameras said.  This keeps, across
episodes, what the skill did: every pick and placement the agent made, with
what the agent itself could tell at the time -- where its camera put the block,
what had last happened to that block, and whether the fingers closed on it
(pick) or the fused camera evidence said the goal was met (placement).  The
true scene is never read.

`mode` decides in what form the same records reach the model when it chooses
which blocks to use:

  none      nothing
  episodic  the most recent picks, as they are
  lessons   sentences the model wrote itself, updated after every episode it
            learned from (it saw that episode's records)
  map       for each candidate block, how often earlier picks near that spot
            worked: counted by code, not by the model
"""
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

MODES = ("none", "episodic", "lessons", "map")
SIGMA_CM = 2.5          # width of "near that spot"
OTHER_HISTORY = 0.25    # weight of a record whose block had a different history (fresh / just failed)
RECENT = 24             # how many records the episodic form shows
MAX_LESSONS = 6

GEOMETRY = ("Positions are in cm. x points away from the robot and y to its left. The workspace is a square "
            "from -19 to 19 in both directions with a low rim around it. The zone pads are at x = 11: "
            "zone_A at y = -11, zone_B at y = 0, zone_C at y = 11.")

LESSON_SYSTEM = """You keep a short list of lessons about a robot arm's pick-and-place skill. The skill is \
unreliable: some picks fail. After each episode you see what was tried and how it went, and you update the list \
so that a later episode can choose blocks that the robot handles well.

""" + GEOMETRY + """

Reply with exactly one JSON object and nothing else:
{"lessons": ["<lesson>", ...]}
At most """ + str(MAX_LESSONS) + """ lessons, each one sentence of at most 30 words. A lesson must be general: about \
which blocks, positions or situations make a pick or a placement likely to work or fail, and what to do about it. \
Keep earlier lessons that still hold, correct the ones the new episode contradicts, and merge repeats."""


def describe(rec: Dict[str, Any]) -> str:
    if rec["tool"] == "pick":
        after = {"failed pick": " right after a failed pick of the same block",
                 "failed place": " after its placement had failed", "placed": " (it had been placed before)"}
        return (f"pick of a block at ({rec['x']:+.1f}, {rec['y']:+.1f}){after.get(rec.get('prev'), '')}: "
                + ("worked" if rec["ok"] else "FAILED"))
    return (f"placement on {'a zone pad' if rec.get('kind') == 'zone' else 'another block'} of a block picked up at "
            f"({rec['x']:+.1f}, {rec['y']:+.1f}): " + ("worked" if rec["ok"] else "FAILED"))


class SkillMemory:
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
    def add(self, rec: Dict[str, Any]) -> None:
        if self.learn:
            self._episode.append(rec)

    def end_episode(self, llm=None, lessons: bool = True) -> None:
        """Commit this episode's records; with an LLM, let it revise its lessons."""
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
    def expect_pick(self, x: float, y: float, prev: str = "first") -> Tuple[Optional[float], float, float]:
        """(expected success, weighted picks near this spot, weighted successes among them).
        A kernel average over earlier picks; picks whose block had the other kind of
        history (fresh, or just failed) count for less.  None when nothing is near."""
        failed = prev == "failed pick"
        n = k = 0.0
        for r in self.records:
            if r["tool"] != "pick":
                continue
            w = math.exp(-((r["x"] - x) ** 2 + (r["y"] - y) ** 2) / (2 * SIGMA_CM ** 2))
            if (r.get("prev") == "failed pick") != failed:
                w *= OTHER_HISTORY
            n += w
            k += w * bool(r["ok"])
        if n < 1.0:
            return None, n, k
        return (k + 0.5) / (n + 1.0), n, k

    def notes(self, mode: str, blocks: Dict[str, Tuple[float, float, str]]) -> str:
        """What the model is shown about the skill when it chooses blocks.
        blocks: name -> (x cm, y cm, what last happened to it)."""
        if mode == "none" or (not self.records and not self.lessons):
            return ""
        if mode == "episodic":
            picks = [r for r in self.records if r["tool"] == "pick"][-RECENT:]
            return ("EARLIER PICKS (the most recent ones from earlier episodes, as they happened):\n"
                    + "\n".join(f"  {describe(r)}" for r in picks))
        if mode == "lessons":
            return ("LESSONS (you wrote these after earlier episodes):\n"
                    + ("\n".join(f"  - {x}" for x in self.lessons) or "  (none yet)"))
        rows = []
        for name, (x, y, prev) in blocks.items():
            p, n, k = self.expect_pick(x, y, prev)
            rows.append(f"  {name}: " + ("no earlier picks near this spot" if p is None else
                        f"expected pick success {100 * p:.0f}% "
                        f"(about {k:.0f} of {n:.0f} earlier picks near this spot worked)"))
        total = sum(r["tool"] == "pick" for r in self.records)
        return (f"SKILL RECORD (counted by the system from {total} picks in earlier episodes; it covers picking a "
                "block up, not putting it down):\n" + "\n".join(rows))
