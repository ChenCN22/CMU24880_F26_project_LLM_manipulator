"""The frozen executor: compiles a tool call into a goal pose, hands it to the
frozen skill policy, and reports a *claimed* outcome.

The claim is gripper-centric on purpose.  The policy was trained on
single-object PandaPickAndPlace, where reaching the commanded goal with the
object in hand is the same event as succeeding; it reports success when *its
gripper* arrives.  On a cluttered table those two events come apart -- the
block can be knocked loose, blocked by a neighbour, or dropped en route -- so
silent failure is emergent from the layering rather than hand-authored.

Ground truth is computed at the same instant and returned separately under
`_truth`.  The tool layer strips that before anything reaches the agent; the
logger keeps it.  Nothing here is ever shown to the agent directly.
"""
from typing import Any, Dict, Optional, Tuple

import functools
import numpy as np

from envs.scene_graph import GRIPPER_CLOSED, scene_graph

from .critic import q_value
from envs.tabletop import OBJECT_SIZE, ZONES

REACH_TOL = 0.05          # same threshold the skill was trained against
PICK_LIFT = 0.06          # how far up a pick commands the block
ACTION_GAIN = 0.5         # the skill's xyz action is halved: at full gain it overshoots and
                          # oscillates through the goal (measured on table goals: stays within
                          # 5 cm 74% of the time at gain 1.0, 100% at 0.5)
ALIGN_TOL = 0.008         # scripted final alignment before release

# A pick cannot be compiled as "goal = where the block already is": the skill is
# a HER goal-conditioned policy, so achieved_goal == desired_goal reads as an
# already-solved task and it outputs ~zero actions. A pick is therefore a short
# upward displacement, which is also what makes the grasp necessary.
DIRECTIONS = {
    "+x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
}


@functools.lru_cache(maxsize=4)
def _load_policy(path: str):
    """Load an SB3 checkpoint without needing to be told which algorithm it is.

    Cached: the policy is only ever asked for deterministic predictions, so one
    loaded copy can serve every episode of a sweep."""
    import torch
    torch.set_num_threads(1)       # a 3-layer MLP: more threads only add contention
    from stable_baselines3 import SAC
    try:
        from sb3_contrib import TQC
    except ImportError:
        TQC = None
    # The checkpoints were trained with HerReplayBuffer, whose saved kwargs make
    # load() demand an env.  We only ever call predict(), so the buffer is
    # overridden away rather than reconstructed.
    from stable_baselines3.common.buffers import DictReplayBuffer
    custom = {"replay_buffer_class": DictReplayBuffer, "replay_buffer_kwargs": {}}
    errors = []
    for cls in ([TQC, SAC] if "tqc" in path.lower() else [SAC, TQC]):
        if cls is None:
            continue
        try:
            return cls.load(path, device="cpu", custom_objects=custom)
        except Exception as e:                      # noqa: BLE001
            errors.append(f"{cls.__name__}: {type(e).__name__} {e}")
    raise RuntimeError("could not load policy from " + path + "\n  " + "\n  ".join(errors))


class FrozenSkillExecutor:
    """Turns the six-tool vocabulary into skill invocations.

    policy_path=None selects a scripted proportional controller instead of the
    learned skill.  That mode exists so the tool interface can be exercised
    before a checkpoint is picked -- it is NOT an experimental condition.
    """

    def __init__(
        self,
        env,
        policy_path: Optional[str] = None,
        max_steps: int = 100,  # twice the training horizon, because the action gain is halved
        p_drop: float = 0.0,
        p_misplace: float = 0.0,
        p_silent: float = 0.0,
        seed: int = 0,
        retract: bool = True,
        gain: float = ACTION_GAIN,
        pick_gain: float = 0.7,
        pick_hold: int = 3,
        home_jitter: float = 0.02,
        stack_standoff: float = 0.08,
        align: bool = True,
        push_policy_path: Optional[str] = None,
    ) -> None:
        self.env = env
        # A second frozen skill, trained on single-object PandaPush: it slides a block over the
        # table with the fingers closed.  None keeps the executor as it was (push() below then
        # still drives the pick-and-place policy).
        self.push_policy = _load_policy(push_policy_path) if push_policy_path else None
        # The skill only ever started an episode from the neutral pose.  Chaining
        # calls starts it from wherever the last one ended (low, over a zone),
        # which is out of distribution.  retract=True returns the open gripper to
        # the pose it had at construction -- straight up first -- before a pick.
        self.retract = retract
        self.gain, self.align = gain, align
        self.pick_gain, self.pick_hold = pick_gain, pick_hold
        self.home_jitter = home_jitter
        self.stack_standoff = stack_standoff
        self.home = np.asarray(env.robot.get_ee_position(), dtype=float).copy()
        self.policy = _load_policy(policy_path) if policy_path else None
        self.scripted = self.policy is None
        self.max_steps = max_steps
        self.p_drop, self.p_misplace, self.p_silent = p_drop, p_misplace, p_silent
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------- single-object view
    def _obs_for(self, obj: str, goal: np.ndarray) -> Dict[str, np.ndarray]:
        """Build exactly the 19-dim observation the skill was trained on, with
        `obj` standing in for the one object that existed during training.

        layout: [ee_pos(3), ee_vel(3), fingers_width(1),
                 obj_pos(3), obj_rot(3), obj_vel(3), obj_angvel(3)]
        """
        sim, robot = self.env.sim, self.env.robot
        observation = np.concatenate([
            robot.get_obs(),
            sim.get_base_position(obj),
            self._canonical_rotation(obj),
            sim.get_base_velocity(obj),
            sim.get_base_angular_velocity(obj),
        ]).astype(np.float32)
        return {
            "observation": observation,
            "achieved_goal": np.asarray(sim.get_base_position(obj), dtype=np.float32),
            "desired_goal": np.asarray(goal, dtype=np.float32),
        }

    def _canonical_rotation(self, obj: str) -> np.ndarray:
        """Euler angles of the block, reduced by the cube's symmetry group.

        In training the block only ever sat flat and nearly unrotated, so the
        skill saw roll = pitch = 0 and small yaw.  Here a block that has been
        dropped or knocked can land on another face: physically the same cube,
        but its Euler angles read 1.57 and the policy -- which has never seen
        that input -- stops being able to pick it up at all (measured: 0 of 42
        picks on such blocks).  Of the 24 equivalent orientations, report the
        one closest to upright."""
        from scipy.spatial.transform import Rotation as R
        if not hasattr(self, "_cube_group"):
            self._cube_group = R.create_group("O")
        rot = R.from_quat(np.asarray(self.env.sim.get_base_orientation(obj), dtype=float))
        cands = rot * self._cube_group
        best = int(np.argmin(cands.magnitude()))
        return cands[best].as_euler("xyz").astype(np.float32)

    def _scripted_action(self, obj: str, goal: np.ndarray) -> np.ndarray:
        """Proportional controller used only in stub mode."""
        sim, robot = self.env.sim, self.env.robot
        ee = np.asarray(robot.get_ee_position(), dtype=float)
        op = np.asarray(sim.get_base_position(obj), dtype=float)
        if np.linalg.norm(ee - op) > 0.03:
            target, grip = op, 1.0            # approach with fingers open
        else:
            target, grip = goal, -1.0         # transport with fingers closed
        d = np.clip((target - ee) * 5.0, -1.0, 1.0)
        return np.array([d[0], d[1], d[2], grip], dtype=np.float32)

    # -------------------------------------------------------------- skill call
    def _run(self, obj: str, goal: np.ndarray, claim_fn, max_steps: Optional[int] = None,
             gain: float = 1.0) -> Tuple[bool, Dict[str, Any]]:
        """Drive the skill toward `goal` with `obj` as its single object.

        `claim_fn()` is the executor's own notion of having finished -- built
        only from proprioception (where the gripper is, whether it is closed),
        never from the object's true state.  The loop stops when it fires, and
        its value at exit is what the executor claims.  Ground truth is read
        separately by each tool.
        """
        goal = np.asarray(goal, dtype=np.float32)
        if self.p_misplace and self.rng.random() < self.p_misplace:
            goal = goal + np.concatenate([self.rng.uniform(-0.08, 0.08, 2), [0.0]]).astype(np.float32)

        steps = max_steps or self.max_steps
        dropped_at, t, fired = None, 0, False
        q_trace = []                 # the critic's value at each step; hidden from the agent
        for t in range(steps):
            if self.scripted:
                action = self._scripted_action(obj, goal)
            else:
                obs = self._obs_for(obj, goal)
                action, _ = self.policy.predict(obs, deterministic=True)
                q_trace.append(round(q_value(self.policy, obs, action), 2))
            action = np.asarray(action, dtype=np.float32).copy()
            if not self.scripted:
                action[:3] *= gain
            self.env.robot.set_action(action)
            self.env.sim.step()

            if self.p_drop and dropped_at is None and self.rng.random() < self.p_drop / steps:
                # force the gripper open for one step: whatever was held falls
                self.env.robot.set_action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
                self.env.sim.step()
                dropped_at = t

            if claim_fn():
                fired = True
                break

        claimed = fired
        if self.p_silent and self.rng.random() < self.p_silent:
            claimed = True
        ee = np.asarray(self.env.robot.get_ee_position(), dtype=float)
        op = np.asarray(self.env.sim.get_base_position(obj), dtype=float)
        return claimed, {
            "steps": t + 1,
            "ee_goal_dist": round(float(np.linalg.norm(ee - goal)), 4),
            "obj_goal_dist": round(float(np.linalg.norm(op - goal)), 4),
            "fingers_width": round(float(self.env.robot.get_fingers_width()), 4),
            "forced_drop_step": dropped_at,
            "q_trace": q_trace,
        }

    def _go_home(self, max_steps: int = 40, grip: float = 1.0) -> None:
        """Scripted, open-fingered return to the start pose: release, rise, then
        translate.  Not a skill call and makes no claim.  grip=-1 closes the fingers
        on arrival, which is how the push skill starts."""
        robot, sim = self.env.robot, self.env.sim
        for _ in range(3):
            robot.set_action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)); sim.step()
        # a slightly different start each time, so that retrying a failed call
        # is not a replay of the same deterministic rollout
        home = self.home.copy()
        home[:2] += self.rng.uniform(-self.home_jitter, self.home_jitter, 2)
        for _ in range(max_steps):
            ee = np.asarray(robot.get_ee_position(), dtype=float)
            target = home.copy()
            if ee[2] < home[2] - 0.03:        # rise before moving sideways
                target[:2] = ee[:2]
            if np.linalg.norm(ee - home) < 0.01:
                break
            d = np.clip((target - ee) / 0.05, -1.0, 1.0)
            robot.set_action(np.array([d[0], d[1], d[2], 1.0], dtype=np.float32)); sim.step()
        for _ in range(6 if grip < 0 else 0):               # close only once the hand is up and away
            robot.set_action(np.array([0.0, 0.0, 0.0, grip], dtype=np.float32)); sim.step()

    def _align(self, goal: np.ndarray, max_steps: int = 30) -> None:
        """Scripted final approach: with the fingers left closed, centre the
        hand over the target just above release height.  The skill is only
        trained to within REACH_TOL; a 9 cm pad or a 4 cm block top needs
        better.  Moves the hand, not the block -- if the block is not in the
        hand, this aligns nothing and the claim is still made."""
        robot, sim = self.env.robot, self.env.sim
        target = np.array([goal[0], goal[1], goal[2] + 0.015], dtype=float)
        cruise = target[2] + 0.02                                  # cross above release height
        for _ in range(max_steps):
            ee = np.asarray(robot.get_ee_position(), dtype=float)
            t = target.copy()
            off = np.linalg.norm(ee[:2] - target[:2]) > ALIGN_TOL
            if off and ee[2] < cruise - 0.01:                      # too low to cross: rise first
                t[:2] = ee[:2]; t[2] = cruise
            elif off:                                              # then centre, without descending
                t[2] = ee[2]
            if np.linalg.norm(ee - target) < ALIGN_TOL:
                break
            d = np.clip((t - ee) / 0.05, -1.0, 1.0)
            robot.set_action(np.array([d[0], d[1], d[2], -1.0], dtype=np.float32)); sim.step()

    # ------------------------------------------------- proprioceptive predicates
    def _ee_at(self, goal: np.ndarray, tol: float = REACH_TOL):
        def f() -> bool:
            ee = np.asarray(self.env.robot.get_ee_position(), dtype=float)
            return bool(np.linalg.norm(ee - np.asarray(goal, dtype=float)) < tol)
        return f

    def _closed_on_something(self, hold: int = 3):
        """"My fingers have stopped on something block-sized."  Finger width
        only (about 0.04 around a block, 0.0 on nothing), held for a few
        consecutive steps so an opening/closing transient does not count.  It
        does not know WHICH block is between the fingers, or whether it stays."""
        streak = [0]
        def f() -> bool:
            ok = 0.02 < self.env.robot.get_fingers_width() < GRIPPER_CLOSED
            streak[0] = streak[0] + 1 if ok else 0
            return streak[0] >= hold
        return f

    # --------------------------------------------------------- goal compilation
    def _target_position(self, target: str) -> np.ndarray:
        """Resolve a place_on target -- a zone pad or the top of another block."""
        if target in ZONES:
            xy = ZONES[target]
            return np.array([xy[0], xy[1], OBJECT_SIZE / 2], dtype=np.float32)
        if target in self.env.object_names:
            p = np.asarray(self.env.sim.get_base_position(target), dtype=float)
            return np.array([p[0], p[1], p[2] + OBJECT_SIZE], dtype=np.float32)
        raise KeyError(f"unknown place_on target {target!r}")

    # ------------------------------------------------------------------- tools
    def pick(self, obj: str) -> Dict[str, Any]:
        if obj not in self.env.object_names:
            return {"ok": False, "error": f"unknown object {obj!r}; args need \"obj\", one of {list(self.env.object_names)}"}
        if self.retract:
            self._go_home()
        here = np.asarray(self.env.sim.get_base_position(obj), dtype=np.float32)
        goal = here + np.array([0.0, 0.0, PICK_LIFT], dtype=np.float32)
        claimed, info = self._run(obj, goal, self._closed_on_something(self.pick_hold),
                                  max_steps=self.max_steps // 2, gain=self.pick_gain)
        truth = scene_graph(self.env)["held"] == obj
        return {"ok": True, "claimed_success": claimed, "_truth": bool(truth), "_info": info}

    def place_on(self, target: str, obj: Optional[str] = None) -> Dict[str, Any]:
        held = scene_graph(self.env)["held"]
        obj = obj or held
        if obj is None:
            return {"ok": False, "error": "nothing is held: pick a block first, then place_on"}
        try:
            goal = self._target_position(target)
        except KeyError:
            return {"ok": False, "error": f"unknown target {target!r}; args need \"target\", a zone "
                                          f"{list(ZONES)} or a block {list(self.env.object_names)}"}
        approach = goal
        if target in self.env.object_names and self.stack_standoff and not self.scripted:
            # Driving the skill straight at a stack target shoves the block
            # underneath (it arrives low and from the side).  Send it to a
            # table-level staging point short of the target instead; the
            # scripted alignment then rises, crosses over the top and descends.
            here = np.asarray(self.env.sim.get_base_position(obj), dtype=float)
            away = here[:2] - goal[:2]
            n = float(np.linalg.norm(away))
            away = away / n if n > 1e-6 else np.array([-1.0, 0.0])
            xy = np.clip(goal[:2] + away * self.stack_standoff, -0.15, 0.15)
            approach = np.array([xy[0], xy[1], OBJECT_SIZE / 2], dtype=np.float32)
        claimed, info = self._run(obj, approach, self._ee_at(approach), gain=self.gain)
        if claimed and self.align and not self.scripted:
            self._align(goal)
        for _ in range(8):                      # release and let physics settle
            self.env.robot.set_action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            self.env.sim.step()
        op = np.asarray(self.env.sim.get_base_position(obj), dtype=float)
        info["obj_goal_dist_after_release"] = round(float(np.linalg.norm(op[:2] - goal[:2])), 4)
        # Ground truth has to match what the target meant.  For a zone, landing
        # inside the pad is the whole claim.  For a stack, xy proximity is not
        # enough -- a block resting *beside* the tower is within tolerance of
        # the tower's xy, so the relation itself is the only honest test.
        graph = scene_graph(self.env)
        if target in self.env.object_names:
            truth = graph["on_top_of"].get(obj) == target
        else:
            truth = graph["in_zone"].get(obj) == target
        return {"ok": True, "claimed_success": claimed, "_truth": bool(truth), "_info": info}

    def push(self, obj: str, direction: str, distance: float) -> Dict[str, Any]:
        """Move a block over the table by `distance` along an axis, or toward the middle of
        the workspace ("center").  With a push skill loaded this is that skill (push_to);
        without one it is the pick-and-place skill sent to a table goal, as before."""
        if obj not in self.env.object_names:
            return {"ok": False, "error": f"unknown object {obj!r}; args need \"obj\", one of {list(self.env.object_names)}"}
        if direction not in DIRECTIONS and direction != "center":
            return {"ok": False, "error": f"direction must be one of {list(DIRECTIONS) + ['center']}"}
        distance = float(np.clip(distance, 0.0, 0.15))
        start = np.asarray(self.env.sim.get_base_position(obj), dtype=float)
        if direction == "center":
            r = float(np.linalg.norm(start[:2]))
            unit = np.array([-start[0] / r, -start[1] / r, 0.0]) if r > 1e-6 else np.zeros(3)
            distance = min(distance, r)                      # never past the middle
        else:
            unit = DIRECTIONS[direction]
        goal = (start + unit * distance).astype(np.float32)
        goal[2] = OBJECT_SIZE / 2
        if self.push_policy is not None:
            fence = float(getattr(self.env.task, "fence", 0.0) or 0.0)
            if fence:                                        # a goal the rim allows the block to reach
                lim = fence - OBJECT_SIZE / 2 - 0.01
                goal[:2] = np.clip(goal[:2], -lim, lim)
            return self.push_to(obj, goal[:2])
        claimed, info = self._run(obj, goal, self._ee_at(goal))
        now = np.asarray(self.env.sim.get_base_position(obj), dtype=float)
        moved = float(np.linalg.norm(now[:2] - start[:2]))
        info["moved"] = round(moved, 4)
        # the push counts only if the block ended up near where it was sent
        truth = bool(np.linalg.norm(now[:2] - goal[:2]) < REACH_TOL)
        return {"ok": True, "claimed_success": claimed, "_truth": truth, "_info": info}

    def push_to(self, obj: str, xy, steps: int = 50) -> Dict[str, Any]:
        """The push skill: slide `obj` over the table to the point `xy`, fingers closed.

        Runs the push policy for its training horizon and then lifts the hand clear.  The
        claim is only that the motion was executed; `_truth` says whether the block ended
        within the skill's training tolerance of the point."""
        if obj not in self.env.object_names:
            return {"ok": False, "error": f"unknown object {obj!r}; args need \"obj\", one of {list(self.env.object_names)}"}
        if self.push_policy is None:
            return {"ok": False, "error": "no push skill is loaded"}
        sim, robot = self.env.sim, self.env.robot
        if self.retract:
            self._go_home(grip=-1.0)
        goal = np.array([float(xy[0]), float(xy[1]), OBJECT_SIZE / 2], dtype=np.float32)
        start = np.asarray(sim.get_base_position(obj), dtype=float)
        for _ in range(steps):
            pos = np.asarray(sim.get_base_position(obj), dtype=np.float32)
            obs = {"observation": np.concatenate([
                       robot.get_obs()[:6], pos, self._canonical_rotation(obj),
                       sim.get_base_velocity(obj), sim.get_base_angular_velocity(obj)]).astype(np.float32),
                   "achieved_goal": pos, "desired_goal": goal}
            action, _ = self.push_policy.predict(obs, deterministic=True)
            robot.set_action(np.array([action[0], action[1], action[2], -1.0], dtype=np.float32)); sim.step()
        for _ in range(8):                                   # straight up, off the block
            robot.set_action(np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)); sim.step()
        now = np.asarray(sim.get_base_position(obj), dtype=float)
        info = {"moved": round(float(np.linalg.norm(now[:2] - start[:2])), 4),
                "obj_goal_dist": round(float(np.linalg.norm(now[:2] - goal[:2])), 4)}
        return {"ok": True, "claimed_success": True, "_truth": bool(info["obj_goal_dist"] < REACH_TOL), "_info": info}
