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
| A9-`part` | A9 with one part removed, for the ablation: `single` (one camera per check), `prior` (flat prior), `stale` (only the handled goal is re-read after a movement), `doubt` (no memory of outvoted cameras), `fingers` (a pick is assumed to work), `guards` (nothing is refused) |
| A10-`form` | experimental: A9 on the free-choice tasks. The model chooses which blocks to use from a layout with positions and chooses again after every failed pick or placement. `form` is what it is told about the skill from earlier episodes: `none`, `episodic` (the last raw pick records), `lessons` (short rules the model wrote itself), `map` (the success rate of earlier picks near each block). `--memory <file>` names the store, `--learn` lets the run add to it. The model decides the plan; the code holds it: the shape of a plan (which zones, how many stacks) is read from the instruction once and every plan is checked against it, a block outside the plan is not picked up, a held block goes only to its target in the plan, and `done` is refused while a block is held |
| A10r-`form` | A9 with the push skill as a way to recover (needs `--push-policy`). The goals are read from the instruction as in A9. When the last attempt on a block has failed, the model is asked whether to pick it again where it lies or to push it toward the middle of the table first. `form` is what it is shown to decide: `none`, `told` (the designer's rule: push first), or the memory of earlier episodes as `episodic`, `lessons` or `map` (how often picking again and pushing first worked, counted by code). With `--learn`, half of the retries start with a push chosen by a coin, so both ways get tried. One more guard than A9: `done` is refused while a block is held |

A1, A3 and A4 (plan-and-execute, reflection, cross-episode memory) belong to the
earlier, perfect-camera study and are run with `run_sweep.py`.

Scripted reference policies need no LLM: `naive` (one attempt per goal, never
looks), `trust` (believes one camera report, retries up to 4 times), `vote`
(three cameras, majority), `oracle` (reads the true scene).

**Tools** (`agent_iface/tools.py`): `pick`, `place_on`, `push`, `observe`,
`check_constraints`, `done`. An action returns only that the motion was executed;
the cameras (`front`, `top`, `side`) are the only way to learn its outcome.

`push(obj, direction, distance)` slides a block along an axis or toward the middle
of the workspace (`"center"`). Without `--push-policy` it is carried out by the
pick-and-place skill, as in every run up to the comparison of language models.
With `--push-policy checkpoints/tqc_her_push_350000_steps.zip` it is a second
frozen skill (TQC + HER on PandaPush).

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
`constrained_place`, four blocks on the table. In the free-choice families
(`free_two_zones`, `free_three_zones`, `free_stack`, `free_two_stacks`) the
instruction says what to build but not with which blocks.

## Where the skill fails

```bash
# scripted episodes on seeds 6000+: every pick and placement with what was known before it
python tests/skill_limits.py --episodes 200 --out results/skill_limits
python analyze_skill_limits.py results/skill_limits/episodes.jsonl
# what knowing these limits is worth when the task leaves a choice of blocks (seeds 6200+)
python tests/skill_choice.py --fit results/skill_limits/episodes.jsonl --out results/skill_choice
```

The first script logs, for each call, where the block was, how close the rim
and the other blocks were, and what happened to the block last time. The second
compares scripted ways of choosing blocks: the ones the task names, a fitted
model of pick success, switching after a failure blindly, by a one-sentence
rule, or by the model.

What a push is worth after a failure (seeds 6400+, scripted, reads the true scene):

```bash
# one block from one failed state: pick again, or push first (directly, and through the tool)
python tests/push_recover.py --arms repick,push,center,axis --out results/push_recover_tool
# whole episodes: never push, push at the rim only, push after every failure
python tests/push_episode.py --out results/push_episode
```

The agents built on these measurements, A10r (push after a failure) and A10 (choice of
blocks), with what they are told about the skill from earlier episodes. The memory is
learned once and then frozen; the file used in the evaluation is `checkpoints/a10r_memory.json`.

```bash
# the memory: 200 episodes per fixed family (seeds 8000+) with A10r-none; half of the retries start with a push chosen by a coin
python run_perception_sweep.py --baselines= --agents A10r-none --perception none --info none --seed-base 8000 --episodes 200 \
    --push-policy checkpoints/tqc_her_push_350000_steps.zip --memory results/a10r_learn/memory.json --learn --out results/a10r_learn
# A9 and the A10r forms on the hold-out seeds, then paired differences to A9
scripts/a10r_eval.sh results/a10r_holdout 2000 40 none,persistent
python scripts/a10r_pairs.py results/a10r_holdout
# the A10 forms on the free-choice tasks: hold-out seeds and a second set, then paired differences to A10-none
scripts/a10_eval.sh results/a10_holdout 2000 40 none
scripts/a10_eval.sh results/a10_ext 9000 120 none
python scripts/a10_pooled.py results/a10_holdout results/a10_ext
```

What came out with `qwen2.5:14b` (paired bootstrap, 95% intervals, in points of task success):

- **Push as a recovery helps.** On the hold-out seeds (120 episodes per cell) every A10r form is
  8 to 11 points above A9 with the camera right (A9 85%) and 7 to 9 points above it with one
  bad camera (A9 79%); every interval excludes 0. The forms do not differ: the model pushes in
  nearly every retry whatever it is shown, so the gain comes from having the push, not from the memory.
- **Memory of the skill does not measurably help the choice of blocks.** Over 640 episodes per form
  (160 hold-out + 480 on the second set) A10 without memory completes 85.5%; against it
  `episodic` is -1.1 [-3.9, +1.7], `lessons` +0.8 [-2.5, +3.9], `map` +2.0 [-1.1, +5.2].
  With the map more picks work (79% against 75%) and the first pick of an episode works more
  often (95% against 91%), but task success does not follow. Three of the four families are
  near their ceiling without memory; in `free_two_stacks` the agent completes 53 to 63%, where
  a scripted policy that switches blocks by a fitted model of the skill completes 81%
  (with four picks per goal as its budget instead of the 40 calls).
- `persistent` injects no fault on the free-choice tasks (the error model acts on
  goals that name their blocks), so A10 is evaluated with the camera right only.

One episode as an MP4 from the front camera, with every call and what it really did written
under the image (needs ffmpeg). An agent episode is replayed without new model calls when
`--llm-cache` points at a copy of that run's answer cache:

```bash
python scripts/record_episode.py --driver always --family constrained_place --seed 6557 --out push.mp4
python scripts/record_episode.py --driver A10r-none --perception persistent --family constrained_place --seed 2031 \
    --llm-cache <copy of results/a10r_holdout/A10r-none/llm_cache.jsonl> --memory results/a10r_learn/memory.json --out agent.mp4
```

## Seeds

| seeds | used for |
|---|---|
| 1000-1019 | test runs up to A9 |
| 2000-2039 | hold-out for A9 and its ablation, and the comparison of language models. No agent was run or tuned on them before; `tests/calibrate_critic.py` drew scripted episodes from 2000+ and 3000+. Later also the hold-out of A10r and of A10 (developed on 5000+, 7000+ and 8000+) |
| 4000+, 5000+, 7000+ | development |
| 6000-6199 | measuring the skill (`tests/skill_limits.py`) |
| 6200-6399 | choice experiment (`tests/skill_choice.py`) |
| 6400-6599 | push as a recovery (`tests/push_recover.py`, `tests/push_episode.py`) |
| 8000-8199 | learning episodes of the A10r memory |
| 9000-9119 | A10 block choice, second evaluation set on the free-choice tasks (run once, code unchanged after the hold-out run) |

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
agents/        LLM backbone, LangGraph nodes and graphs (A1-A10), evidence fusion, skill memory
skill/         training script for the pick-and-place policy
checkpoints/   the frozen pick-and-place skill (TQC+HER, 600k steps), the frozen push skill (350k steps),
               the critic calibration and the skill memory used for A10 and A10r
tests/         interface, perception and determinism tests; calibration scripts; skill-limit, choice and push measurements
scripts/       run_agents.sh; holdout.sh, ablation.sh, multi_llm.sh and holdout_pairs.py for the hold-out runs; a10r_eval.sh, a10_eval.sh,
               a10r_pairs.py and a10_pooled.py for A10r and A10; record_episode.py for videos
```

## Adding an agent

Write a `build_xx(iface, llm, memory=None, system=None)` function in
`agents/graphs.py` that returns a compiled LangGraph, register it in the
`BUILDERS` table there, and run it with `scripts/run_agents.sh Axx`.

## License

MIT, see `LICENSE`.
