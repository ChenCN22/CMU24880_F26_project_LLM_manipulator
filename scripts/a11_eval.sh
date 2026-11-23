#!/bin/bash
# usage: scripts/a11_eval.sh <out folder> <seed base> <episodes per family> <perception list> [agents]
# A11 (A10 + memory of knocked blocks) on the four free-choice task families, memory frozen.
# The memory was learned once with A11-none --learn on seeds 11000-11099; MEMORY=<file> names another one.
# Compare with:  scripts/a10_eval.sh <out folder> <seed base> <episodes> <perception> A10-none   (same folder)
# and then:      python scripts/a11_pairs.py <out folder>
cd "$(dirname "$0")/.."
M=${MEMORY:-checkpoints/a11_memory.json}
O=$1; mkdir -p $O
for ag in ${5:-A11-none A11-episodic A11-lessons A11-map A11-told}; do
  python -u run_perception_sweep.py --baselines= --agents $ag --perception $4 --info none --seed-base $2 --episodes $3 \
    --families free_two_zones,free_three_zones,free_stack,free_two_stacks \
    --memory $M --out $O/$ag >> $O/$ag.log 2>&1
done
echo "A11_EVAL_DONE $O $(date)"
