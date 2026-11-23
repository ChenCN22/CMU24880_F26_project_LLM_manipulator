# LLM agents on top of a frozen manipulation skill

Term project for CMU 24-880 *AI Agents for Engineers*, Fall 2026.

A Franka Panda in PyBullet has one low-level skill: a pick-and-place policy
(TQC + HER) trained on a single block and then frozen. An LLM agent sits above it
and solves multi-block tabletop tasks by calling six tools. The skill fails without
saying so and the camera that reports the outcome can be wrong. The study asks how
the agent layer should be built to cope with both, and whether the answer holds
across language models.

The simulator runs on CPU. The language model is a local one through
[Ollama](https://ollama.com) (developed with `qwen2.5:14b`, temperature 0) or an
API model (`openai:<id>`, `anthropic:<id>`).

**Main results** (hold-out seeds, ten language models; tables in [Main benchmark](#main-benchmark)):

- With one bad camera, an agent that keeps the camera evidence in code (A9) completes 60 to 64 of 75
  stated tasks on eight models; ReAct (A2) completes 27 to 58. A9 is ahead on all ten models and the
  gap is largest for the strongest models.
- On tasks that leave the choice of blocks open, an assignment that is held in code and can be revised
  by the model (A10) beats a fixed one (A9) by 17 to 34 points on all eight models. Against ReAct it is
  22 to 32 points ahead on the weaker models and about 5 (not established) on the stronger ones.
- The code-held designs use a fifth to a half of ReAct's input tokens.
- A second skill for recovery (push) adds 3 to 12 points. Memory of the skill did not raise success.

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

## Main benchmark

Three stated task families and four free-choice families, hold-out seeds 2000-2039 (the tables use
2000-2024, the set almost every model has), camera right and one bad camera, the same block layout and
instruction for every agent and model.

```bash
# qwen2.5:14b: fills the cells that the earlier hold-out runs do not have (40 seeds per family)
scripts/bench_fill.sh results/bench/qwen2.5_14b
# further local models, one after the other: "<model> <Ollama host> <think: off|on|->"; HOURS limits the GPU time
EPISODES=25 HOURS=12 scripts/bench_models.sh "gemma3:12b http://localhost:11434 -" "qwen3.8:27b http://localhost:11434 off"
# an API model with a spending cap: <provider:model> <cap in dollars> <$ per 1M input tokens> <$ per 1M output tokens>
scripts/bench_api.sh anthropic:claude-haiku-5-5 6 0.10 0.50
STATED_CAMS=persistent CELLS="A9__free A10-none__free A2__free A9__stated A2__stated" \
    scripts/bench_api.sh anthropic:claude-sonnet-5-5 16.5 2 10
# one table for everything, then the summary
python scripts/bench_export.py --results results --holdout <folder of the first hold-out run> --out data/benchmark_episodes.csv
python scripts/bench_summary.py data/benchmark_episodes.csv            # --seeds 40, --md
python scripts/bench_table.py results/bench/<model>/*/episodes.jsonl   # one model, straight from its run folders
```

`data/benchmark_episodes.csv` has one line per episode (15,034 episodes: model, agent, camera, family, seed,
success, tool calls, model calls, tokens). `data/benchmark_summary.txt` is the output of `bench_summary.py`:
per model, successes per family, paired differences with 95% bootstrap intervals, and calls and tokens per episode.
The launchers assume the conda environment `proj880`; `bench_models.sh` also knows a second, user-level Ollama
on port 11435. Runs stop cleanly on a kill and continue when started again; an API run that reaches its cap
stops and can be continued with a higher cap.

Successes on seeds 2000-2024. Scripted rules on the stated tasks (75 episodes): naive 46, oracle 67 with the
camera right; naive 46, trust 56, vote 61, oracle 67 with one bad camera.

Stated tasks with one bad camera, of 75:

| model | A2 | A5 | A9 | A10r-none | A9 - A2 |
|---|---|---|---|---|---|
| qwen2.5:14b | 58 | 57 | 60 | 66 | +2.7 [-5.3, +10.7] |
| gemma3:12b | 47 | 56 | 61 | 68 | +18.7 [+9.3, +28.0] |
| llama3.1:8b | 34 | 34 | 43 | 46 | +12.0 [-2.7, +26.7] |
| qwen3.8:27b | 42 | 51 | 64 | 66 | +29.3 [+18.7, +40.0] |
| deepseek-r1:32b | 45 | 42 | 63 | 63 | +24.0 [+13.3, +34.7] |
| claude-haiku-5-5 | 34 | 52 | 64 | 65 | +40.0 [+28.0, +52.0] |
| claude-sonnet-5-5 | 35 | - | 63 | - | +37.3 [+25.3, +49.3] |
| gpt-6-sol (of 48) | 22 | - | 41 | - | +39.6 [+25.0, +54.2] |
| gpt-6-luna | 42 | 49 | 64 | - | +29.3 [+18.7, +40.0] |
| gpt-6.1-terra | 27 | 54 | 64 | - | +49.3 [+36.0, +61.3] |

Free-choice tasks with the camera right, of 100:

| model | A2 | A9 | A10-none | A11-lessons | A10 - A9 | A10 - A2 |
|---|---|---|---|---|---|---|
| qwen2.5:14b | 55 | 55 | 81 | 87 | +26.0 [+16.0, +36.0] | +26.0 [+16.0, +36.0] |
| gemma3:12b | 40 | 53 | 72 | 71 | +19.0 [+12.0, +27.0] | +32.0 [+23.0, +41.0] |
| llama3.1:8b | 36 | 28 | 62 | 47 | +34.0 [+24.0, +44.0] | +26.0 [+15.0, +37.0] |
| deepseek-r1:32b | 60 | 49 | 82 | 84 | +33.0 [+22.0, +44.0] | +22.0 [+12.0, +32.0] |
| qwen3.8:27b | 79 | 67 | 85 | 82 | +18.0 [+8.0, +28.0] | +6.0 [-3.0, +15.0] |
| claude-haiku-5-5 | 82 | 54 | 87 | 94 | +33.0 [+22.0, +44.0] | +5.0 [-4.0, +14.0] |
| claude-sonnet-5-5 | 86 | 74 | 91 | - | +17.0 [+7.0, +27.0] | +5.0 [-2.0, +13.0] |
| gpt-6-sol (of 64) | 55 | 47 | 59 | - | +18.8 [+7.8, +29.7] | +6.2 [-1.6, +15.6] |

With the camera right, every design completes 61 to 69 of 75 stated tasks on the capable models (the oracle
rule: 67); only the push skill passes that (A10r-none 69 to 74). The error model injects nothing on the
free-choice tasks, so they are run with the camera right only. gpt-6-luna and gpt-6.1-terra ran A2, A5 and A9
on the stated tasks; claude-sonnet-5-5 and gpt-6-sol ran the cells shown.

Why the designs differ on the free-choice tasks (which block failed, which was tried again, what was left
on the table) is read from the logs by the scripts in `scripts/free_task_analysis/`.

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
| A10-`form` | A9 for the free-choice tasks. The model chooses which blocks to use from a layout with positions and chooses again after every failed pick or placement. `form` is what it is told about the skill from earlier episodes: `none`, `episodic` (the last raw pick records), `lessons` (short rules the model wrote itself), `map` (the success rate of earlier picks near each block). `--memory <file>` names the store, `--learn` lets the run add to it. The model decides the plan; the code holds it: the shape of a plan (which zones, how many stacks) is read from the instruction once and every plan is checked against it, a block outside the plan is not picked up, a held block goes only to its target in the plan, and `done` is refused while a block is held |
| A10r-`form` | A9 with the push skill as a way to recover (needs `--push-policy`). The goals are read from the instruction as in A9. When the last attempt on a block has failed, the model is asked whether to pick it again where it lies or to push it toward the middle of the table first. `form` is what it is shown to decide: `none`, `told` (the designer's rule: push first), or the memory of earlier episodes as `episodic`, `lessons` or `map` (how often picking again and pushing first worked, counted by code). With `--learn`, half of the retries start with a push chosen by a coin, so both ways get tried. One more guard than A9: `done` is refused while a block is held |
| A11-`form` | A10 with a memory of what the arm did to blocks it was not handling (`agents/disturb_memory.py`). Before and after every attempt the agent has the camera positions of all blocks, so it records which blocks were moved and how they lay; no ground truth is used. When it chooses which block goes where it is shown the candidate placements with their geometry (in every form) and what earlier episodes say: `none` (the table only), `told` (the designer's rule), `episodic`, `lessons`, `map`. `--memory <file>` names the store, `--learn` adds to it |

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

**Language models** (`agents/llm.py`). `--model <name>` with `--host` asks an Ollama server; `--think off`
switches a thinking model's reasoning off. `--model openai:<id>` and `--model anthropic:<id>` go through the
providers' HTTP APIs (keys in `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`); `--effort none`
asks for answers without reasoning. Every model answers with one JSON object under a schema. The Claude API
takes no temperature and no seed, and its schemas must be closed, so the open `args` object of a tool call is
sent with every tool argument as an optional field. Models that cannot answer without reasoning
(gpt-6.1-sol, gpt-6.2-sol, claude-opus-5-5) do not fit this setup.

**How the hand moves** (`agent_iface/executor.py`, `--motion` or `EXECUTOR_MOTION`). `low`, the default and the
motion of every run in the benchmark: the learned policy approaches, grasps and carries the block at table
level. A quarter of all picks and placements then move a block they were not about. `clear` scripts the
transport around the policy (lift straight up, carry above the other blocks, lower straight down). With it
scripted success on development seeds rises from 73% to 95% and every agent reaches 93 to 100% with the camera
right (`results/clear_holdout` in our runs). It is a reference only: the study keeps the flawed skill and asks
what the agent can do about it.

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

What the arm does to blocks it is not handling, from true positions:

```bash
# replays a finished run from its answer cache and logs the true positions around every pick, placement and push
python tests/disturbance_audit.py --runs results/a10_holdout/A10-none results/a10_holdout/A10-map --out results/disturbance_audit_a10
# scripted episodes under one motion: which calls knock which blocks, and how they lay
python tests/motion_audit.py --motions low --seed-base 4000 --episodes 200 --out results/motion_audit_low_pos
# what scripted rules that avoid risky moves gain (order of the goals, choice of blocks)
python tests/avoid_audit.py --out results/avoid_audit
python tests/avoid_choice_audit.py --out results/avoid_choice_audit
```

In 1,280 audited A10 episodes, 171 of 173 failed episodes contain a block that the arm knocked, toppled or
threw. A pick works 86% of the time for a block inside the workspace, 42% at one rim and 10% in a corner.
A carried block knocks a block within 4 cm of its straight path 71 to 85% of the time and one beyond 10 cm
under 3%. Scripted rules that use this gain about 4 points on the stated tasks and 7 on the free-choice tasks.

A11 learns the same regularity from its own camera reports (`checkpoints/a11_memory.json`, 400 learning
episodes on seeds 11000+):

```bash
scripts/a11_eval.sh results/a11_eval 10000 120 none           # the A11 forms on the free-choice tasks
scripts/a10_eval.sh results/a11_eval 10000 120 none A10-none  # the reference, into the same folder
python scripts/a11_pairs.py results/a11_eval                  # BASE=A11-none changes the reference
```

With `qwen2.5:14b` (480 episodes per agent) success does not differ between the forms (80.0 to 83.3%). In the
two zone families the share of carries that knock another block falls from 32.7% (A10-none) to 25.6% (table
only) to 20.2% (lessons); a scripted rule that takes the nearest block reaches 5.2%.

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
| 2000-2039 | hold-out for A9 and its ablation, and the comparison of language models. No agent was run or tuned on them before; `tests/calibrate_critic.py` drew scripted episodes from 2000+ and 3000+. Later also the hold-out of A10r and of A10 (developed on 4000+, 5000+ and 8000+) and the main benchmark |
| 4000+, 5000+, 7000+ | development |
| 6000-6199 | measuring the skill (`tests/skill_limits.py`) |
| 6200-6399 | choice experiment (`tests/skill_choice.py`) |
| 6400-6599 | push as a recovery (`tests/push_recover.py`, `tests/push_episode.py`) |
| 8000-8199 | learning episodes of the A10r memory |
| 9000-9119 | A10 block choice, second evaluation set on the free-choice tasks (run once, code unchanged after the hold-out run) |
| 10000-10199 | scripted avoidance rules (`tests/avoid_audit.py`, `tests/avoid_choice_audit.py`); 10000-10119 also the A11 evaluation |
| 11000-11099 | learning episodes of the A11 memory |

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
agents/        LLM back ends, LangGraph nodes and graphs (A1-A11), evidence fusion (A9-A11), skill memory, disturbance memory
skill/         training script for the pick-and-place policy
checkpoints/   the frozen pick-and-place skill (TQC+HER, 600k steps), the frozen push skill (350k steps),
               the critic calibration, and the two frozen memories (a10r_memory.json, a11_memory.json)
data/          the main benchmark as one table (benchmark_episodes.csv) and its summaries
tests/         interface, perception and determinism tests; calibration scripts; skill-limit, choice, push,
               disturbance, motion and avoidance measurements
scripts/       bench_*.sh / bench_*.py for the main benchmark; run_agents.sh; holdout.sh, ablation.sh, multi_llm.sh and
               holdout_pairs.py for the first hold-out runs; a10r_eval.sh, a10_eval.sh, a11_eval.sh with their *_pairs.py /
               a10_pooled.py; free_task_analysis/ for the failure analysis; record_episode.py for videos
```

## Adding an agent

Write a `build_xx(iface, llm, memory=None, system=None)` function in
`agents/graphs.py` that returns a compiled LangGraph, register it in the
`BUILDERS` table there, and run it with `scripts/run_agents.sh Axx`.

## License

MIT, see `LICENSE`.
