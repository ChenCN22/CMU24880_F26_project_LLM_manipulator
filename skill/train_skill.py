"""Train the frozen low-level skill for the 24-880 term project.

Per the proposal, the executor's skill is a SAC+HER policy trained ONLY on
isolated pick-and-place.  We deliberately want a mid-training checkpoint whose
success rate sits in the 70-90% band, so checkpoints + eval are dense and the
run is never 'finished' in the usual sense -- we pick from the curve afterwards.
"""
import argparse, os
import gymnasium as gym
import panda_gym  # noqa: F401  (registers Panda* envs)
from stable_baselines3 import SAC, HerReplayBuffer
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

p = argparse.ArgumentParser()
p.add_argument("--env", default="PandaPickAndPlace-v3")
p.add_argument("--algo", default="sac", choices=["sac", "tqc"])
p.add_argument("--steps", type=int, default=1_000_000)
p.add_argument("--ckpt-every", type=int, default=25_000)
p.add_argument("--eval-every", type=int, default=25_000)
p.add_argument("--eval-episodes", type=int, default=50)
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None)
p.add_argument("--goal-z", type=float, default=None,
               help="override goal_z_range. Stock is 0.2, which puts most goals in mid-air -- far "
                    "outside this project's deployment distribution and the main reason stock "
                    "converges slowly. The skill here is asked for table-level placement and a "
                    "6 cm lift when picking, so 0.08 covers exactly what it will see. Recorded "
                    "in the run name so a tuned run is never confused with stock.")
a = p.parse_args()

out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "runs", f"skill_{a.algo}{'' if a.goal_z is None else f'_gz{a.goal_z:g}'}_{a.env}_seed{a.seed}")
os.makedirs(out, exist_ok=True)

def make(tag):
    e = gym.make(a.env)
    if a.goal_z is not None:
        # _sample_goal() reads goal_range_high at every reset, so setting the z
        # component here fixes the goal-height distribution for good.
        e.unwrapped.task.goal_range_high[2] = a.goal_z
    return Monitor(e, filename=os.path.join(out, tag))

env, eval_env = make("train"), make("eval")

# rl-baselines3-zoo panda-gym hyperparameters
Algo = {"sac": SAC, "tqc": TQC}[a.algo]
model = Algo(
    "MultiInputPolicy", env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    buffer_size=1_000_000, batch_size=2048, gamma=0.95, tau=0.05,
    learning_rate=1e-3, learning_starts=1000,
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    device=a.device, seed=a.seed, verbose=1,
    tensorboard_log=os.path.join(out, "tb"),
)

callbacks = [
    CheckpointCallback(save_freq=a.ckpt_every, save_path=os.path.join(out, "ckpt"),
                       name_prefix=f"{a.algo}_her", save_replay_buffer=False),
    EvalCallback(eval_env, n_eval_episodes=a.eval_episodes, eval_freq=a.eval_every,
                 log_path=os.path.join(out, "evallog"),
                 best_model_save_path=os.path.join(out, "best"),
                 deterministic=True, render=False, verbose=1),
]

print(f"[train_skill] algo={a.algo} env={a.env} steps={a.steps} device={a.device} -> {out}", flush=True)
model.learn(total_timesteps=a.steps, callback=callbacks, log_interval=20)
model.save(os.path.join(out, "final"))
print("[train_skill] done", flush=True)
