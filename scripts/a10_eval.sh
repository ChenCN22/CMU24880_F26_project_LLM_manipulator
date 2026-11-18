#!/bin/bash
# usage: scripts/a10_eval.sh <out folder> <seed base> <episodes per family> <perception list> [agents]
# The block-choosing agent A10 on the four free-choice task families, with the skill memory frozen
# (the same file as for A10r; MEMORY=<file> names another one).
cd "$(dirname "$0")/.."
M=${MEMORY:-checkpoints/a10r_memory.json}
O=$1; mkdir -p $O
for ag in ${5:-A10-none A10-episodic A10-lessons A10-map}; do
  python -u run_perception_sweep.py --baselines= --agents $ag --perception $4 --info none --seed-base $2 --episodes $3 \
    --families free_two_zones,free_three_zones,free_stack,free_two_stacks \
    --memory $M --out $O/$ag >> $O/$ag.log 2>&1
done
echo "A10_EVAL_DONE $O $(date)"
