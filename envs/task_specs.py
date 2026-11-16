"""The four task families, as episode specs.

Each generator returns {instruction, goals, constraints} over the objects that
actually exist in the env, so the same spec format scores every family.  Horizon
grows across the families, which is the axis the architecture comparison is
supposed to bite on.
"""
from typing import Any, Dict, List

import numpy as np

ZONE_LIST = ["zone_A", "zone_B", "zone_C"]


def color_sort(objects: List[str], rng: np.random.Generator) -> Dict[str, Any]:
    """Shortest horizon: each named block goes to its own zone."""
    k = min(3, len(objects))
    chosen = list(rng.choice(objects, size=k, replace=False))
    zones = list(rng.permutation(ZONE_LIST))[:k]
    goals = [{"type": "in_zone", "obj": o, "zone": z} for o, z in zip(chosen, zones)]
    pairs = ", ".join(f"the {o.split('_')[0]} block in {z}" for o, z in zip(chosen, zones))
    return {"family": "color_sort",
            "instruction": f"Sort the blocks: put {pairs}.",
            "goals": goals, "constraints": [{"type": "keep_on_table", "obj": o} for o in chosen]}


def ordered_stack(objects: List[str], rng: np.random.Generator) -> Dict[str, Any]:
    """Medium horizon: a two-block stack in a stated order.  (Three blocks was
    the original design; the frozen skill's placement precision makes a third
    level unreachable even for a ground-truth-retry script, so it was cut.)"""
    k = min(2, len(objects))
    order = list(rng.choice(objects, size=k, replace=False))
    goals = [{"type": "on_top_of", "top": order[i + 1], "bottom": order[i]} for i in range(k - 1)]
    top, bottom = order[1].split("_")[0], order[0].split("_")[0]
    return {"family": "ordered_stack",
            "instruction": f"Put the {top} block on top of the {bottom} block.",
            "goals": goals, "constraints": [{"type": "keep_on_table", "obj": o} for o in order]}


def blocked_rearrange(objects: List[str], rng: np.random.Generator) -> Dict[str, Any]:
    """Long horizon: the wanted zone is already taken, so it must be cleared
    before the goal block can go in.  The blocker's own destination is stated,
    which is what makes the clearing step checkable rather than arbitrary."""
    assert len(objects) >= 2
    goal_obj, blocker = list(rng.choice(objects, size=2, replace=False))
    target, elsewhere = list(rng.permutation(ZONE_LIST))[:2]
    return {"family": "blocked_rearrange",
            "instruction": (f"First put the {blocker.split('_')[0]} block in {elsewhere}, "
                            f"then put the {goal_obj.split('_')[0]} block in {target}."),
            "goals": [{"type": "in_zone", "obj": blocker, "zone": elsewhere},
                      {"type": "in_zone", "obj": goal_obj, "zone": target}],
            "constraints": [{"type": "keep_on_table", "obj": goal_obj},
                            {"type": "keep_on_table", "obj": blocker}]}


def constrained_place(objects: List[str], rng: np.random.Generator) -> Dict[str, Any]:
    """Long horizon with a forbidden relation -- the condition where an agent
    that never verifies can satisfy every goal and still fail the episode."""
    assert len(objects) >= 3
    a, b, c = list(rng.choice(objects, size=3, replace=False))
    z1, z2 = list(rng.permutation(ZONE_LIST))[:2]
    return {"family": "constrained_place",
            "instruction": (f"Put the {a.split('_')[0]} block in {z1} and the {b.split('_')[0]} block in {z2}. "
                            f"The {a.split('_')[0]} block must never end up on top of the {c.split('_')[0]} block."),
            "goals": [{"type": "in_zone", "obj": a, "zone": z1},
                      {"type": "in_zone", "obj": b, "zone": z2}],
            "constraints": [{"type": "not_on", "top": a, "bottom": c},
                            {"type": "keep_on_table", "obj": a},
                            {"type": "keep_on_table", "obj": b}]}


# ---- tasks that leave the blocks open -------------------------------------------------
# The families above name every block, so an agent that knows where the skill is
# weak can only stop retrying sooner.  These leave the choice of blocks to the
# agent: which ones to move, and (for stacks) which one stays where it is.
def free_two_zones(objects: List[str], rng: np.random.Generator) -> Dict[str, Any]:
    """Two of four blocks are spare."""
    z1, z2 = list(rng.permutation(ZONE_LIST))[:2]
    return {"family": "free_two_zones",
            "instruction": f"Put one block in {z1} and another block in {z2}. You may choose which blocks to use.",
            "goals": [{"type": "zone_filled", "zone": z1}, {"type": "zone_filled", "zone": z2}],
            "constraints": []}


def free_three_zones(objects: List[str], rng: np.random.Generator) -> Dict[str, Any]:
    """Longer, and only one block is spare."""
    return {"family": "free_three_zones",
            "instruction": "Put one block in each of zone_A, zone_B and zone_C. You may choose which blocks to use.",
            "goals": [{"type": "zone_filled", "zone": z} for z in ZONE_LIST], "constraints": []}


def free_stack(objects: List[str], rng: np.random.Generator) -> Dict[str, Any]:
    return {"family": "free_stack",
            "instruction": ("Build one stack of two blocks. You may choose which two blocks to use and which of "
                            "them goes on top."),
            "goals": [{"type": "stacks", "n": 1}], "constraints": []}


def free_two_stacks(objects: List[str], rng: np.random.Generator) -> Dict[str, Any]:
    """All four blocks are needed: a hard block cannot be left out, only given the job it can do."""
    return {"family": "free_two_stacks",
            "instruction": ("Build two separate stacks of two blocks each. You may choose how to pair the blocks "
                            "and which block of each pair goes on top."),
            "goals": [{"type": "stacks", "n": 1}, {"type": "stacks", "n": 2}], "constraints": []}


FAMILIES = {
    "free_two_zones": free_two_zones,
    "free_three_zones": free_three_zones,
    "free_stack": free_stack,
    "free_two_stacks": free_two_stacks,
    "color_sort": color_sort,
    "ordered_stack": ordered_stack,
    "blocked_rearrange": blocked_rearrange,
    "constrained_place": constrained_place,
}


def sample(family: str, objects: List[str], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    spec = FAMILIES[family](list(objects), rng)
    spec["seed"] = seed
    return spec
