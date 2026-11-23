#!/bin/bash
# usage: scripts/bench_api.sh <provider:model id> <cap in dollars> <$ per 1M input tokens> <$ per 1M output tokens>
#   e.g. scripts/bench_api.sh anthropic:claude-haiku-5-5 6 0.10 0.50
#        STATED_CAMS=persistent CELLS="A9__free A2__stated" scripts/bench_api.sh openai:gpt-6.2-sol 5 2 10
# The benchmark table for one model asked through an API (anthropic: or openai:) without thinking, no GPU:
# hold-out seeds 2000-2024 (25 episodes per family, EPISODES), the cells of scripts/bench_models.sh
# plus A2/A5/A9 on the stated families.  The key is read from the environment (ANTHROPIC_API_KEY or
# CLAUDE_API_KEY; OPENAI_API_KEY).
#
# The money spent is added up from the answers written to the llm_cache.jsonl files (every answered
# call is written there with its token counts; a provider's discount for cached input is not counted,
# so the sum is an upper bound).  At the cap the sweeps are stopped; started again, they continue
# where they stopped and answered calls are not paid twice.
# JOBS (default 6) sweeps run at a time; CELLS limits the run to the folders named (space-separated);
# STATED_CAMS (default none,persistent) is the camera conditions of the stated families.
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh; conda activate proj880; unset PYTHONPATH
model=$1; CAP=$2; PIN=$3; POUT=$4
[ -z "$POUT" ] && { echo "usage: $0 <provider:model id> <cap in dollars> <\$/1M in> <\$/1M out>"; exit 1; }
FREE=free_two_zones,free_three_zones,free_stack,free_two_stacks
STATED=ordered_stack,blocked_rearrange,constrained_place
N=${EPISODES:-25}
O=${ROOT:-results/bench}/$(echo $model | tr ':/' '__'); mkdir -p $O
SC=${STATED_CAMS:-none,persistent}
S=$O/STATUS
spent() { python - "$O" "$PIN" "$POUT" <<'EOF'
import glob, json, sys
i = o = 0
for f in glob.glob(sys.argv[1] + "/*/llm_cache.jsonl"):
    for l in open(f):
        try:
            e = json.loads(l)
        except ValueError:                      # a line being written
            continue
        i += e.get("pt", 0); o += e.get("ot", 0)
print(f"{(i * float(sys.argv[2]) + o * float(sys.argv[3])) / 1e6:.3f}")
EOF
}
over() { python -c "import sys; sys.exit(0 if float('$(spent)') >= float('$CAP') else 1)"; }
alive() { local n=0; for p in $PIDS; do kill -0 $p 2>/dev/null && n=$((n + 1)); done; echo $n; }
PUSH="--push-policy checkpoints/tqc_her_push_350000_steps.zip --memory checkpoints/a10r_memory.json"
M10="--memory checkpoints/a10r_memory.json"; M11="--memory checkpoints/a11_memory.json"
# folder | agent | camera conditions | families | more arguments
CELLS_ALL="A9__free|A9|none|$FREE|
A10-none__free|A10-none|none|$FREE|$M10
A11-none__free|A11-none|none|$FREE|$M11
A11-lessons__free|A11-lessons|none|$FREE|$M11
A2__free|A2|none|$FREE|
A5__free|A5|none|$FREE|
A9__stated|A9|$SC|$STATED|
A10r-none|A10r-none|$SC|$STATED|$PUSH
A10-none__stated|A10-none|$SC|$STATED|$M10
A11-none__stated|A11-none|$SC|$STATED|$M11
A11-lessons__stated|A11-lessons|$SC|$STATED|$M11
A2__stated|A2|$SC|$STATED|
A5__stated|A5|$SC|$STATED|"
echo "RUNNING $model, $N episodes per family, cap \$$CAP, since $(date)" > $S
STOP=""
while IFS='|' read -r folder agent cams fams more; do
  [ -n "$CELLS" ] && [[ " $CELLS " != *" $folder "* ]] && continue
  while [ "$(alive)" -ge "${JOBS:-6}" ]; do
    if over; then STOP=1; break; fi
    sleep ${POLL:-15}
  done
  [ -n "$STOP" ] && break
  (exec python -u run_perception_sweep.py --baselines= --agents $agent --perception $cams --info none --seed-base 2000 \
      --episodes $N --families $fams --model $model --effort none $more --out $O/$folder >> $O/$folder.log 2>&1) &
  PIDS="$PIDS $!"
done <<< "$CELLS_ALL"
while [ -z "$STOP" ] && [ "$(alive)" -gt 0 ]; do
  if over; then STOP=1; break; fi
  sleep ${POLL:-15}
done
if [ -n "$STOP" ]; then
  kill $PIDS 2>/dev/null; wait
  echo "STOPPED at the cap: \$$(spent) spent, $(date)" >> $S
else
  wait
  echo "DONE \$$(spent) spent, $(date)" >> $S
fi
