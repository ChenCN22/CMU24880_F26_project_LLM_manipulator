#!/bin/bash
# usage: ablation.sh <part> <part> ...   A9 with one part removed, on the hold-out seeds (2000-2039), text conditions.
# Runs from ~/24880_dev (code after the hold-out freeze; the default A9 path replays the v7 run call for call).
cd ~/24880_dev
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate proj880; unset PYTHONPATH
for part in "$@"; do
  for pm in persistent transient none; do
    # the hold-out run moves the text model to the user-level server when its text cells are done
    if grep -q TEXT_DONE ~/24880_term_project/results/holdout.log; then H=http://127.0.0.1:11435; else H=http://localhost:11434; fi
    python -u run_perception_sweep.py --baselines= --agents A9-$part --perception $pm --info none \
      --seed-base 2000 --episodes 40 --host $H --out results/ablation/$part >> results/ablation/$part.log 2>&1
  done
done
echo ABLATION_DONE "$@" $(date)
