"""Save what the agent is given at a look, next to pictures of the same moment.

    python tests/dump_agent_view.py --family ordered_stack --seed 1000 --out results/agent_view

The LLM receives TEXT only.  For every whole-scene look this writes
  <tag>_call<k>_{front,top,side}.png   renders for a human to compare against
  <tag>_call<k>_prompt.txt             the complete prompt of the model's next turn
                                       (system prompt, log so far, this camera report)
                                       followed by what the model answered
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from agent_iface.executor import FrozenSkillExecutor             # noqa: E402
from agent_iface.perception import PerceptionModel               # noqa: E402
from agent_iface.tools import ToolInterface                      # noqa: E402
from agents import graphs                                        # noqa: E402
from agents.llm import LLMBackbone                               # noqa: E402
from agents.nodes import SYSTEM_V2                               # noqa: E402
from envs.tabletop import MultiObjectTabletopEnv                 # noqa: E402
from envs.task_specs import sample                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--family", default="ordered_stack")
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--agent", default="A2")
ap.add_argument("--perception", default="none")
ap.add_argument("--policy", default="checkpoints/tqc_her_600000_steps.zip")
ap.add_argument("--model", default="qwen2.5:14b")
ap.add_argument("--out", default="results/agent_view")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = f"{a.family}_{a.seed}_{a.agent}_{a.perception}"

CAMS = {"front": dict(yaw=90, pitch=-25, distance=0.75),     # the three viewpoints, for the human reader
        "top": dict(yaw=90, pitch=-89, distance=0.75),
        "side": dict(yaw=0, pitch=-25, distance=0.75)}

env = MultiObjectTabletopEnv(n_objects=4, render_mode="rgb_array", fence=0.19)
env.reset(seed=a.seed)
spec = sample(a.family, env.object_names, a.seed)
ex = FrozenSkillExecutor(env, policy_path=a.policy, max_steps=100, seed=a.seed)
iface = ToolInterface(env, ex, spec, observe_budget=None, max_calls=40,
                      perception=PerceptionModel(spec, a.seed, a.perception), info="none", hide_claims=True)
llm = LLMBackbone(model=a.model, keep_messages=True)

raw, looks = iface.call, []
def call(name, args=None):                                       # noqa: E302
    out = raw(name, args)
    if name == "observe" and (args or {}).get("query", "scene") == "scene":
        k = iface.calls
        for view, cam in CAMS.items():
            img = env.sim.render(width=640, height=480, target_position=np.zeros(3), roll=0, **cam)
            Image.fromarray(np.asarray(img, dtype=np.uint8)[..., :3]).save(
                os.path.join(a.out, f"{tag}_call{k:02d}_{view}.png"))
        looks.append((k, len(llm.trace)))                        # the model's next turn is trace[len]
    return out
iface.call = call                                                # noqa: E305

graphs.build(a.agent, iface, llm, None, SYSTEM_V2).invoke(
    {"instruction": spec["instruction"], "history": [], "reflections": [], "memories": [],
     "steps": 0, "stop": False}, {"recursion_limit": 200})

for k, i in looks:
    if i >= len(llm.trace):
        continue
    t = llm.trace[i]
    with open(os.path.join(a.out, f"{tag}_call{k:02d}_prompt.txt"), "w") as f:
        for m in t["messages"]:
            f.write(f"===== {m['role'].upper()} =====\n{m['content']}\n\n")
        f.write(f"===== MODEL ANSWER =====\n{t['out']}\n")
score = iface.score()
print(f"{tag}: instruction={spec['instruction']!r} success={score['success']} calls={score['calls']} "
      f"scene looks saved={len(looks)}")
for t in iface.transcript:
    print("  ", t["i"], t["tool"], t["args"], "truth=", t["truth"])
env.close()
