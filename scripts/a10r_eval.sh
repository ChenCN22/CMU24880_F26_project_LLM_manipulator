#!/bin/bash
# usage: scripts/a10r_eval.sh <out folder> <seed base> <episodes per family> <perception list> [agents]
# A9 and the five A10r forms on the fixed task families, with the skill memory frozen.
# The memory was learned once with A10r-none --learn on seeds 8000-8199; MEMORY=<file> names another one.
cd "$(dirname "$0")/.."
M=${MEMORY:-checkpoints/a10r_memory.json}
O=$1; mkdir -p $O
for ag in ${5:-A9 A10r-none A10r-told A10r-episodic A10r-lessons A10r-map}; do
  python -u run_perception_sweep.py --baselines= --agents $ag --perception $4 --info none --seed-base $2 --episodes $3 \
    --push-policy checkpoints/tqc_her_push_350000_steps.zip --memory $M --out $O/$ag >> $O/$ag.log 2>&1
done
echo "A10R_EVAL_DONE $O $(date)"
