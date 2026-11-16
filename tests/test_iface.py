"""Smoke test for the frozen tool interface.

Checks the two properties the whole study rests on: the agent cannot see ground
truth, and perception is actually metered.  Runs with the scripted stub
executor so it does not need a trained checkpoint.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_iface.executor import FrozenSkillExecutor          # noqa: E402
from agent_iface.tools import TOOL_NAMES, TOOL_SCHEMAS, ToolInterface  # noqa: E402
from envs.scene_graph import scene_graph                       # noqa: E402
from envs.tabletop import MultiObjectTabletopEnv               # noqa: E402
from envs.task_specs import FAMILIES, sample                   # noqa: E402


class TestInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _fresh(self, family="color_sort", seed=0, **kw):
        self.env.reset(seed=seed)
        spec = sample(family, self.env.object_names, seed)
        ex = FrozenSkillExecutor(self.env, policy_path=None, max_steps=40, seed=seed)
        return ToolInterface(self.env, ex, spec, **kw), spec

    def test_schemas_match_dispatch(self):
        self.assertEqual(len(TOOL_SCHEMAS), 6)
        self.assertEqual(set(TOOL_NAMES),
                         {"pick", "place_on", "push", "observe", "check_constraints", "done"})

    def test_scene_graph_shape(self):
        self.env.reset(seed=1)
        g = scene_graph(self.env)
        self.assertEqual(len(g["objects"]), 4)
        self.assertEqual(set(g["zones"]), {"zone_A", "zone_B", "zone_C"})
        self.assertIsNone(g["held"])
        for n in g["objects"]:                   # blocks must start well separated
            self.assertIsNone(g["on_top_of"][n])

    def test_all_four_families_generate(self):
        for fam in FAMILIES:
            _, spec = self._fresh(fam, seed=3)
            self.assertTrue(spec["instruction"])
            self.assertTrue(spec["goals"], f"{fam} produced no goals")
            for g in spec["goals"]:
                self.assertIn(g["type"], ("in_zone", "on_top_of", "clear", "on_table"))

    def test_ground_truth_never_leaks(self):
        """The single most important invariant in the codebase."""
        iface, spec = self._fresh()
        obj = self.env.object_names[0]
        results = [
            iface.call("pick", {"obj": obj}),
            iface.call("place_on", {"target": "zone_A"}),
            iface.call("push", {"obj": obj, "direction": "+y", "distance": 0.05}),
            iface.call("observe", {"query": "zones"}),
            iface.call("check_constraints"),
        ]
        for r in results:
            for k in r:
                self.assertFalse(k.startswith("_"), f"leaked private field {k!r} to the agent")
            self.assertNotIn("_truth", r)
        # ...and the transcript must have kept ground truth for every call that
        # actually reached the executor.  A rejected call (place_on with nothing
        # held, say) carries no truth because no action was attempted.
        acted = [t for t in iface.transcript
                 if t["tool"] in ("pick", "place_on", "push") and t["visible"].get("ok")]
        self.assertGreater(len(acted), 0, "no action call reached the executor")
        for t in acted:
            self.assertIn(t["truth"], (True, False),
                          f"{t['tool']} returned ok but logged no ground truth")
        # every action the agent saw must report a claim it could act on
        for t in acted:
            self.assertIn("claimed_success", t["visible"])

    def test_perception_budget_is_metered(self):
        iface, _ = self._fresh(observe_budget=3)
        oks = [iface.call("observe", {"query": "held"}).get("ok") for _ in range(5)]
        self.assertEqual(oks, [True, True, True, False, False])
        self.assertEqual(iface.observes_used, 3)

    def test_call_cap_and_done(self):
        iface, _ = self._fresh(max_calls=4)
        for _ in range(4):
            iface.call("observe", {"query": "held"})
        self.assertTrue(iface.finished)
        self.assertFalse(iface.call("observe", {"query": "held"})["ok"])

        iface2, _ = self._fresh()
        self.assertTrue(iface2.call("done")["ended"])
        self.assertTrue(iface2.finished)
        self.assertFalse(iface2.call("observe", {"query": "held"})["ok"])

    def test_bad_arguments_are_rejected_not_crashed(self):
        iface, _ = self._fresh()
        self.assertFalse(iface.call("pick", {"obj": "nonexistent_block"})["ok"])
        self.assertFalse(iface.call("push", {"obj": self.env.object_names[0],
                                             "direction": "sideways", "distance": 0.05})["ok"])
        self.assertFalse(iface.call("place_on", {"target": "zone_Z"})["ok"])
        self.assertFalse(iface.call("teleport", {})["ok"])

    def test_score_reports_cost_and_truth(self):
        iface, _ = self._fresh()
        iface.call("pick", {"obj": self.env.object_names[0]})
        iface.call("observe", {"query": "scene"})
        iface.call("done")
        s = iface.score()
        for k in ("success", "progress", "calls", "observes_used", "false_completion",
                  "silent_failures", "silent_failure_rate", "verification_calls"):
            self.assertIn(k, s)
        self.assertEqual(s["observes_used"], 1)
        self.assertTrue(s["declared_done"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
