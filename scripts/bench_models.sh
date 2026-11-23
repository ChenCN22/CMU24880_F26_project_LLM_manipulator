#!/bin/bash
# usage: scripts/bench_models.sh "<model> <host> <think: off|on|->" ...
# The main benchmark's later agents with further language models, one model after the other
# (hold-out seeds 2000-2024: 25 episodes per family, EPISODES=40 for all of them).
# A2/A5/A9 on the stated families exist in results/models/<tag>.
#   A10r-none                          stated families, camera right + one bad camera
#   A10-none, A11-none, A11-lessons    the same, and the free-choice families with the camera right
#                                      (the camera-error model has no errors for free-choice goals)
#   A2, A5, A9                         the free-choice families with the camera right
# The memories are frozen and were learned with qwen2.5:14b.
#
# The GPU is ours for HOURS (default 12) only.  At that time the running sweeps are stopped, the
# models unloaded and the remaining models skipped.  Each model runs in two parts (1. camera right,
# 2. one bad camera), so a stop in part 2 leaves part 1 whole.  A sweep that was stopped continues
# where it stopped when this script is started again with the same arguments.
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh; conda activate proj880; unset PYTHONPATH
U="env OLLAMA_HOST=127.0.0.1:11435 $HOME/ollama_user/bin/ollama"
FREE=free_two_zones,free_three_zones,free_stack,free_two_stacks
STATED=ordered_stack,blocked_rearrange,constrained_place
N=${EPISODES:-25}
ROOT=${ROOT:-results/bench}
S=$ROOT/models_STATUS
END=$(( $(date +%s) + $(python -c "print(int(${HOURS:-12} * 3600))") ))
left() { [ "$(date +%s)" -lt "$END" ]; }
unload() { [ -n "$DRY" ] && return; ollama stop $1 2>/dev/null; $U stop $1 2>/dev/null; sleep 5; }
alive() { for p in $PIDS; do kill -0 $p 2>/dev/null && return 0; done; return 1; }
part() {   # waits for the sweeps of one part; at the time limit stops them and returns 1
  while alive; do
    if ! left; then kill $PIDS 2>/dev/null; wait; PIDS=; return 1; fi
    sleep ${POLL:-20}
  done
  wait; PIDS=; return 0
}
echo "RUNNING $(date), $N episodes per family, until $(date -d @$END)" > $S
unload qwen2.5:14b
for spec in "$@"; do
  set -- $spec; model=$1; host=$2; think=$3
  if ! left; then echo "SKIPPED $model (time limit) $(date)" >> $S; continue; fi
  T=""; [ "$think" != "-" ] && T="--think $think"
  tag=$(echo $model | tr ':/' '__'); O=$ROOT/$tag; mkdir -p $O
  # run <agent> <camera conditions> <families> <folder> [more arguments]: one sweep in the background
  run() {
    if [ -n "$DRY" ]; then (echo "$model $*" >> $O/dry.txt; exec sleep $DRY) &
    else (exec python -u run_perception_sweep.py --baselines= --agents $1 --perception $2 --info none --seed-base 2000 \
            --episodes $N --families $3 --model $model --host $host $T "${@:5}" --out $O/$4 >> $O/$4.log 2>&1) &
    fi
    PIDS="$PIDS $!"
  }
  PUSH="--push-policy checkpoints/tqc_her_push_350000_steps.zip --memory checkpoints/a10r_memory.json"
  stated() {   # the stated families under the camera condition $1
    run A10r-none $1 $STATED A10r-none $PUSH
    run A10-none $1 $STATED A10-none__stated --memory checkpoints/a10r_memory.json
    for ag in A11-none A11-lessons; do run $ag $1 $STATED ${ag}__stated --memory checkpoints/a11_memory.json; done
  }
  stated none
  run A10-none none $FREE A10-none__free --memory checkpoints/a10r_memory.json
  for ag in A11-none A11-lessons; do run $ag none $FREE ${ag}__free --memory checkpoints/a11_memory.json; done
  for ag in A2 A5 A9; do run $ag none $FREE ${ag}__free; done   # how the older agents fail on open instructions
  if part; then
    echo "PART_DONE $model camera right $(date)" >> $S
    stated persistent
    if part; then echo "MODEL_DONE $model $(date)" >> $S
    else echo "STOPPED $model in part 2 (one bad camera) at the time limit $(date)" >> $S; fi
  else
    echo "STOPPED $model in part 1 (camera right) at the time limit $(date)" >> $S
  fi
  unload $model
done
echo "DONE $(date)" >> $S
