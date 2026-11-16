# LLM agents on top of a frozen manipulation skill

Term project for CMU 24-880 *AI Agents for Engineers*, Fall 2026.

A Franka Panda in PyBullet has one low-level skill: a pick-and-place policy
(TQC + HER) trained on a single block and then frozen. An LLM agent sits above it
and solves multi-block tabletop tasks by calling six tools. The study asks what
happens when the agent's **camera is wrong**: does it notice, does it recover,
and does extra information or a different agent design help?

Everything runs locally: the simulator on CPU, the LLM through
[Ollama](https://ollama.com) (`qwen2.5:14b`, about 9 GB, temperature 0).

## Setup

```bash
conda create -n proj880 python=3.10 -y
conda activate proj880
pip install -r requirements.txt

ollama pull qwen2.5:14b        # needs the Ollama server running on localhost:11434
```

The trained skill and the critic calibration are in `checkpoints/`, so nothing
has to be trained to run the experiments.

Check the installation (no LLM needed, about two minutes):

```bash
python tests/test_iface.py
python tests/test_perception.py
python tests/test_determinism.py      # must end with DETERMINISTIC
```

## Running an agent

```bash
scripts/run_agents.sh A6               # one agent over the whole grid
scripts/run_agents.sh A6,A7,A8         # several, one after another
EPISODES=5 scripts/run_agents.sh A8    # quick look: 5 seeds per task family
OUT=results/mine scripts/run_agents.sh A7
```

The script runs `run_perception_sweep.py` and then `analyze_perception.py`.
A run can be interrupted and started again with the same command; episodes that
are already in `episodes.jsonl` are skipped. To combine agents run on different
machines, concatenate their `episodes.jsonl` files into one folder and run
`python analyze_perception.py <folder>`.

For full control call the sweep directly:

```bash
python run_perception_sweep.py --agents A2,A5 --baselines naive,trust,vote,oracle \
    --perception none,transient,persistent --info none,label,monitor \
    --episodes 20 --out results/perception
```

Rough cost per episode with `qwen2.5:14b` on an RTX 4090: A2 and A5 about 5-10 s,
A6 and A8 about 10-30 s, A7 slower. The default grid for one agent with
`--info none,monitor` is 3 x 2 x 3 families x 20 seeds = 360 episodes.

To watch an episode in the PyBullet window:

```bash
python demo_gui.py --driver A8 --perception persistent --info monitor
```

## What is being compared

**Agents** (`agents/graphs.py`, nodes in `agents/nodes.py`)

| name | design |
|---|---|
| A2 | ReAct: one tool call at a time, sees every result |
| A5 | A2 + forced cross-check: every action must be looked at, every look is repeated from a second camera (a third if the two disagree) before the agent may finish |
| A6 | A2 + mandatory self-assessment after every result |
| A7 | A2 + an explicit belief table over the goals, updated after every result |
| A8 | A2 + an independent reviewer: a second LLM call that sees only the instruction and the raw tool log, not the operator's reasoning |

A1, A3 and A4 (plan-and-execute, reflection, cross-episode memory) belong to the
earlier, perfect-camera study and are run with `run_sweep.py`.

Scripted reference policies need no LLM: `naive` (one attempt per goal, never
looks), `trust` (believes one camera report, retries up to 4 times), `vote`
(three cameras, majority), `oracle` (reads the true scene).

**Tools** (`agent_iface/tools.py`): `pick`, `place_on`, `push`, `observe`,
`check_constraints`, `done`. An action returns only that the motion was executed;
the cameras (`front`, `top`, `side`) are the only way to learn its outcome.

**Perception error** (`agent_iface/perception.py`), defined against the truth:
a goal that is *not* met is reported as met with probability 0.30 (false
success); a goal that *is* met is reported as unmet with probability 0.10 (false
failure).

| mode | meaning |
|---|---|
| `none` | the camera is always right |
| `transient` | each look errs independently; looking again usually helps |
| `persistent` | the same camera repeats the same error on the same scene; another camera draws afresh |
| `random` | control: an arbitrary block is misplaced in the report, unrelated to the task |

**Information given to the agent**

| level | meaning |
|---|---|
| `none` | raw camera reports |
| `label` | each report carries a fixed note on how reliable the cameras are |
| `monitor` | label + a receipt that cross-checks the report against what the gripper did, + an early warning from the skill's own critic |

**Tasks** (`envs/task_specs.py`): `ordered_stack`, `blocked_rearrange`,
`constrained_place`, four blocks on the table.

## Reproducibility

Episodes are paired on seed: for a given task family and episode index every
agent and every condition starts from the same block layout and instruction, and
the `perception=none` cell is the fault-free twin of each faulted cell.

The physics is deterministic: the same seed and the same tool calls give
bit-identical block positions, in any order and across processes. This needs
both PyBullet's `deterministicOverlappingPairs` (set in `envs/tabletop.py`) and a
new simulator per episode (done in `run_perception_sweep.py`);
`tests/test_determinism.py` checks it. The LLM is called at temperature 0.

## Output

`results/<out>/episodes.jsonl` has one line per episode with the score and the
full tool transcript (what the agent saw, what was true, any injected fault).
`analyze_perception.py` writes `analysis.txt` and `analysis.json`: success with
Wilson intervals, paired-bootstrap differences against the fault-free twin and
against A2, what the agent did after each wrong report (cross-checked it or not,
acted on it or not), and the hit rate of the critic warning.

## Layout

```
envs/          tabletop scene, symbolic scene graph, task families
agent_iface/   frozen-skill executor, tool interface, perception model, critic warning
agents/        LLM backbone, LangGraph nodes and graphs (A1-A8)
skill/         training script for the pick-and-place policy
checkpoints/   the frozen skill (TQC+HER, 600k steps) and the critic calibration
tests/         interface, perception and determinism tests; calibration scripts
scripts/       run_agents.sh
```

## Adding an agent

Write a `build_xx(iface, llm, memory=None, system=None)` function in
`agents/graphs.py` that returns a compiled LangGraph, register it in the
`BUILDERS` table there, and run it with `scripts/run_agents.sh Axx`.

## License

MIT, see `LICENSE`.
