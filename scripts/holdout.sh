#!/bin/bash
# Hold-out test of A9.  Seeds 2000-2039 (40 per family, 120 episodes per cell): no agent was run or tuned on them before.
# (tests/calibrate_critic.py drew scripted pick-and-place episodes from 2000+ to calibrate the skill critic; no agent, no prompt.)
# Code frozen at commit adc01ca; no parameter of any agent was changed after the test-seed run (results/v7).
# Scripts, A2 and A5 run on the same seeds as the comparison.
cd ~/24880_term_project
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate proj880; unset PYTHONPATH
O=results/holdout; mkdir -p $O
C="--seed-base 2000 --episodes 40 --info none"
R="python -u run_perception_sweep.py"
$R --agents= --baselines naive,trust,vote,oracle $C --out $O/base > $O/base.log 2>&1
for ag in A9 A2 A5; do
  $R --baselines= --agents $ag --perception persistent,transient,none $C --out $O/$ag > $O/$ag.log 2>&1 &
done
wait
echo TEXT_DONE $(date)
ollama stop qwen2.5:14b; sleep 5
H="--host http://127.0.0.1:11435"
$R --agents= --baselines naive,trust,vote,oracle --perception vlm $C --out $O/base >> $O/base.log 2>&1
$R --baselines= --agents A9 --perception vlm $C $H --out $O/A9 >> $O/A9.log 2>&1
for ag in A2 A5; do
  $R --baselines= --agents $ag --perception vlm $C $H --out $O/$ag >> $O/$ag.log 2>&1 &
done
wait
echo HOLDOUT_DONE $(date)
