#!/bin/bash
# usage: multi_llm.sh <model> ...   A9, A2, A5 with other language models, hold-out seeds 2000-2039, text conditions.
# Starts only after the hold-out run and both ablation streams have finished (gives up after 6 hours).
cd ~/24880_dev
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate proj880; unset PYTHONPATH
for i in $(seq 1 2160); do
  if grep -q HOLDOUT_DONE ~/24880_term_project/results/holdout.log && grep -q ABLATION_DONE results/ablation_1.log && grep -q ABLATION_DONE results/ablation_2.log; then break; fi
  sleep 10
done
grep -q HOLDOUT_DONE ~/24880_term_project/results/holdout.log || { echo GAVE_UP; exit 1; }
OLLAMA_HOST=127.0.0.1:11435 ~/ollama_user/bin/ollama stop qwen2.5:14b; OLLAMA_HOST=127.0.0.1:11435 ~/ollama_user/bin/ollama stop qwen2.5vl:7b
ollama stop qwen2.5:14b; sleep 5
for model in "$@"; do
  tag=$(echo $model | tr ':/' '__'); O=results/models/$tag; mkdir -p $O
  for ag in A9 A2 A5; do
    python -u run_perception_sweep.py --baselines= --agents $ag --perception persistent,none,transient --info none \
      --seed-base 2000 --episodes 40 --model $model --out $O/$ag > $O/$ag.log 2>&1 &
  done
  wait
  ollama stop $model; sleep 5
  echo MODEL_DONE $model $(date)
done
echo MULTI_LLM_DONE
