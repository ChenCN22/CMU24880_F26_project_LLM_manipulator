#!/bin/bash
# usage: scripts/bench_fill.sh <out folder> [model] [host] [extra sweep arguments]
# Fills the main benchmark: every main agent on the 3 stated + 4 free-choice families, hold-out seeds
# 2000-2039, camera right and one bad camera.  A2/A5/A9/A10r on the stated families exist already
# (results/models/<model>, a10r_holdout), so those agents run on the free families only here.
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh; conda activate proj880; unset PYTHONPATH
O=$1; MODEL=${2:-qwen2.5:14b}; HOST=${3:-http://localhost:11434}; mkdir -p $O
FREE=free_two_zones,free_three_zones,free_stack,free_two_stacks
STATED=ordered_stack,blocked_rearrange,constrained_place
run() { python -u run_perception_sweep.py --baselines= --agents $1 --perception none,persistent --info none --seed-base 2000 --episodes 40 \
          --families $2 --model $MODEL --host $HOST "${@:3}" --out $O/$1 >> $O/$1.log 2>&1; }
echo "RUNNING $(date)" > $O/STATUS
for ag in ${AGENTS_FREE:-A2 A5 A9}; do run $ag $FREE ${EXTRA} & done
for ag in ${AGENTS_PUSH:-A10r-none}; do run $ag $FREE --push-policy checkpoints/tqc_her_push_350000_steps.zip --memory checkpoints/a10r_memory.json ${EXTRA} & done
for ag in ${AGENTS_A10:-A10-none}; do run $ag $STATED,$FREE --memory checkpoints/a10r_memory.json ${EXTRA} & done
for ag in ${AGENTS_A11:-A11-none A11-lessons}; do run $ag $STATED,$FREE --memory checkpoints/a11_memory.json ${EXTRA} & done
wait
echo "DONE $(date)" > $O/STATUS
