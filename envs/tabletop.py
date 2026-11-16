"""Multi-object tabletop for the 24-880 term project.

This is NOT an RL training environment.  It is the physics substrate the agent
acts in: N coloured blocks and three goal zones on the panda-gym table.  The
frozen skill was trained on single-object PandaPickAndPlace-v3 and has never
seen the other blocks -- the executor hands it a single-object *view* of this
scene (see agent_iface/executor.py), which is where collisions and silent
failures come from.  Task success is scored symbolically against ground-truth
simulator state by envs/scene_graph.py, never by this class.
"""
from typing import Any, Dict, List, Optional

import numpy as np
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

# Block palette.  Order matters: the first n_objects colours are the ones used.
COLORS: Dict[str, np.ndarray] = {
    "red":    np.array([0.90, 0.10, 0.10, 1.0]),
    "green":  np.array([0.10, 0.80, 0.10, 1.0]),
    "blue":   np.array([0.10, 0.25, 0.90, 1.0]),
    "yellow": np.array([0.95, 0.85, 0.10, 1.0]),
    "orange": np.array([0.95, 0.50, 0.10, 1.0]),
    "purple": np.array([0.55, 0.15, 0.80, 1.0]),
    "cyan":   np.array([0.10, 0.80, 0.85, 1.0]),
    "pink":   np.array([0.95, 0.45, 0.70, 1.0]),
}

# Three zones on the far side of the workspace, so reaching one is a transport
# problem rather than a no-op.  Every position here must stay inside the skill's
# TRAINING distribution: PandaPickAndPlace-v3 samples objects and goals in
# [-0.15, 0.15]^2, so anything outside that box is out of distribution for the
# frozen policy and it simply fails -- which would be our bug, not its failure.
ZONES: Dict[str, np.ndarray] = {
    "zone_A": np.array([0.11, -0.11]),
    "zone_B": np.array([0.11,  0.00]),
    "zone_C": np.array([0.11,  0.11]),
}
ZONE_HALF = 0.045          # 9 cm pads, 2 cm apart at 0.11 spacing
OBJECT_SIZE = 0.04


class TabletopTask(Task):
    """N blocks + three zone pads.  Blocks spawn on the near side, well clear
    of the zones, with rejection sampling so they never start interpenetrating."""

    def __init__(
        self,
        sim: PyBullet,
        n_objects: int = 4,
        obj_x_range: tuple = (-0.14, -0.02),
        obj_y_range: tuple = (-0.13, 0.13),
        min_separation: float = 1.6 * OBJECT_SIZE,
        fence: float = 0.0,
    ) -> None:
        super().__init__(sim)
        assert 1 <= n_objects <= len(COLORS), f"n_objects must be in 1..{len(COLORS)}"
        self.n_objects = n_objects
        self.object_size = OBJECT_SIZE
        self.obj_x_range = obj_x_range
        self.obj_y_range = obj_y_range
        self.min_separation = min_separation
        self.fence = fence
        self.object_names: List[str] = [f"{c}_block" for c in list(COLORS)[:n_objects]]
        self.zone_names: List[str] = list(ZONES)
        with self.sim.no_rendering():
            self._create_scene()

    # ------------------------------------------------------------------ scene
    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        for i, name in enumerate(self.object_names):
            color = COLORS[name.rsplit("_", 1)[0]]
            self.sim.create_box(
                body_name=name,
                half_extents=np.ones(3) * self.object_size / 2,
                mass=1.0,
                position=np.array([0.0, 0.1 * i, self.object_size / 2]),
                rgba_color=color,
            )
        if self.fence:
            # A low rim around the skill's training workspace.  A block knocked
            # outside [-0.15, 0.15]^2 is out of distribution for the frozen
            # policy and can never be picked up again, by any agent; the rim
            # keeps a fumble recoverable instead of terminal.
            f, t, h = self.fence, 0.01, OBJECT_SIZE / 2
            for i, (cx, cy, hx, hy) in enumerate([(f + t, 0, t, f + 2 * t), (-f - t, 0, t, f + 2 * t),
                                                  (0, f + t, f + 2 * t, t), (0, -f - t, f + 2 * t, t)]):
                self.sim.create_box(body_name=f"fence_{i}", half_extents=np.array([hx, hy, h]), mass=0.0,
                                    position=np.array([cx, cy, h]), rgba_color=np.array([0.3, 0.3, 0.3, 1.0]))
        for zname, xy in ZONES.items():
            self.sim.create_box(
                body_name=zname,
                half_extents=np.array([ZONE_HALF, ZONE_HALF, 0.001]),
                mass=0.0,
                ghost=True,
                position=np.array([xy[0], xy[1], 0.001]),
                rgba_color=np.array([0.6, 0.6, 0.6, 0.35]),
            )

    # -------------------------------------------------------------- Task API
    def get_obs(self) -> np.ndarray:
        """Full multi-object state.  The *agent* never reads this (it gets a
        symbolic scene graph); it exists so the env is a well-formed gym env."""
        parts = []
        for name in self.object_names:
            parts += [
                self.sim.get_base_position(name),
                self.sim.get_base_rotation(name),
                self.sim.get_base_velocity(name),
                self.sim.get_base_angular_velocity(name),
            ]
        return np.concatenate(parts)

    def get_achieved_goal(self) -> np.ndarray:
        return np.concatenate([self.sim.get_base_position(n) for n in self.object_names])

    def get_goal(self) -> np.ndarray:
        return self.goal

    def reset(self) -> None:
        self.goal = np.zeros(3 * self.n_objects, dtype=np.float32)
        for name, pos in zip(self.object_names, self._sample_layout()):
            self.sim.set_base_pose(name, pos, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_layout(self) -> List[np.ndarray]:
        placed: List[np.ndarray] = []
        for _ in range(self.n_objects):
            for _attempt in range(200):
                xy = np.array([
                    self.np_random.uniform(*self.obj_x_range),
                    self.np_random.uniform(*self.obj_y_range),
                ])
                if all(np.linalg.norm(xy - p[:2]) >= self.min_separation for p in placed):
                    placed.append(np.array([xy[0], xy[1], self.object_size / 2]))
                    break
            else:  # could not find a free spot; fall back to a grid slot
                k = len(placed)
                placed.append(np.array([
                    self.obj_x_range[0] + 0.06 * (k % 3),
                    self.obj_y_range[0] + 0.09 * (k // 3),
                    self.object_size / 2,
                ]))
        return placed

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray,
                   info: Dict[str, Any] = {}) -> np.ndarray:
        """Not meaningful here -- episode success is judged symbolically by
        scene_graph.evaluate() against the episode's task spec."""
        return np.array(False)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray,
                       info: Dict[str, Any] = {}) -> np.ndarray:
        return np.array(0.0, dtype=np.float32)


class MultiObjectTabletopEnv(RobotTaskEnv):
    """Panda + TabletopTask.  Physics substrate for the agent experiments."""

    def __init__(
        self,
        n_objects: int = 4,
        render_mode: str = "rgb_array",
        control_type: str = "ee",
        renderer: str = "Tiny",
        fence: float = 0.0,
        **kwargs: Any,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False,
                      base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = TabletopTask(sim, n_objects=n_objects, fence=fence)
        super().__init__(robot, task, **kwargs)
        # Bullet orders contact pairs by memory address unless told otherwise, so the
        # same seed and the same actions could end in different places from one run to
        # the next (measured: 10-14 of 60 scripted episodes changed outcome).
        sim.physics_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

    # convenience passthroughs used by the tool layer
    @property
    def object_names(self) -> List[str]:
        return self.task.object_names

    @property
    def zone_names(self) -> List[str]:
        return self.task.zone_names
