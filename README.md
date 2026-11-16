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
| A5 | A2 + forced cross-check: every action must be looked at before the agent may finish, every look is repeated from a second camera (a third if the two disagree), and the majority answer is handed to the model as one combined report |
| A6 | A2 + mandatory self-assessment after every result |
| A7 | A2 + an explicit belief table over the goals, updated after every result |
| A8 | A2 + an independent reviewer: a second LLM call that sees only the instruction and the raw tool log, not the operator's reasoning |
| A9 | evidence-aware recovery (`agents/evidence.py`): code keeps, for every goal, what each camera said since the arm last moved, fuses it in log-odds on top of a prior, treats "block not visible" as no evidence, re-reads every goal after any arm movement, reads pick success from the fingers, and refuses `done` or a re-pick that the evidence contradicts. The model parses the goals and chooses the actions |

A1, A3 and A4 (plan-and-execute, reflection, cross-episode memory) belong to the
earlier, perfect-camera study and are run with `run_sweep.py`.

Scripted reference policies need no LLM: `naive` (one attempt per goal, never
looks), `trust` (believes one camera report, retries up to 4 times), `vote`
(three cameras, majority), `oracle` (reads the true scene).

**Tools** (`agent_iface/tools.py`): `pick`, `place_on`, `push`, `observe`,
`check_constraints`, `done`. An action returns only that the motion was executed;
the cameras (`front`, `top`, `side`) are the only way to learn its outcome.

**Perception error** (`agent_iface/perception.py`), defined against the truth:
*false success* reports a goal that is not met as met; *false failure* reports a
goal that is met as unmet. A block in the gripper is never misreported.

| mode | meaning |
|---|---|
| `none` | the camera is always right |
| `transient` | every look errs independently, the same for all cameras (false success 0.30, false failure 0.10); looking again usually helps |
| `persistent` | each goal has one badly placed camera that is wrong about it most of the time (0.90 / 0.30) while the other two are nearly always right (0.05 / 0.02). The same camera repeats its error on an unchanged scene, so only another view helps. One look from a random camera is wrong about as often as in `transient` (0.33), a majority of three is not (0.09 against 0.22) |
| `random` | control: an arbitrary block is misplaced in the report, unrelated to the task |
| `vlm` | nothing is injected: a vision-language model reads a rendered camera image (see below) |

**Perception from images** (`agent_iface/vlm_perception.py`). With `--perception vlm`
every look renders the asked camera (front, top or side) and a vision-language
model (`qwen2.5vl:7b`) finds the four cubes in the image. Calibrated camera
geometry then turns each box into a place: which pad the cube is on, or which
cube it is stacked on. Before a picture is taken the empty hand moves up and
back out of the cameras' way. The errors are the perception system's own and
depend on the viewpoint, for example a cube standing behind another one looks
stacked from the side camera.

```bash
# how well does the reader do, per camera, against the true scene?
python tests/eval_vlm.py --style detect --episodes 8 --out results/vlm_eval
# closed loop
python run_perception_sweep.py --agents A2,A5 --perception vlm,none --info none --out results/vlm
```

`--style scene` and `--style cube` ask the model directly where each cube is; on
this scene that was far less accurate than detection plus geometry.
`qwen2.5vl` needs a recent Ollama (0.7 or later); `--vlm-host` and `--host` say
which server serves the VLM and the agent LLM.

**Information given to the agent**

| level | meaning |
|---|---|
| `none` | raw camera reports |
| `label` | each report carries a fixed note that cameras can be wrong and that different cameras are independent checks |
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
`tests/test_determinism.py` checks it.

The LLM is called at temperature 0, but Ollama is still not bit-reproducible:
the same prompt can return a different answer on another run (measured: 2 of 8
ReAct episodes changed outcome between two identical runs). Each run folder
therefore keeps an answer cache, `llm_cache.jsonl`: every distinct prompt is put
to the model once and its answer is reused. Two conditions of the same agent
then behave differently only from the point where their prompts differ, and
rerunning a folder reproduces it exactly. Run one sweep per output folder at a
time; several folders can run in parallel.

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
