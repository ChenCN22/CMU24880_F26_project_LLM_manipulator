#!/bin/bash
# Run one or more LLM agents over the perception-error grid.
#
#   scripts/run_agents.sh A6            # one agent
#   scripts/run_agents.sh A6,A7,A8      # several, one after another
#   EPISODES=5 scripts/run_agents.sh A8 # a quick look (5 seeds per task family)
#
# Results go to results/<OUT>/episodes.jsonl. The run is resumable: start the same
# command again and episodes that are already in the file are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
AGENTS="${1:?usage: scripts/run_agents.sh A6[,A7,A8]}"
OUT="${OUT:-results/perception}"
python -u run_perception_sweep.py \
    --agents "$AGENTS" --baselines= \
    --perception "${PERCEPTION:-none,transient,persistent}" \
    --info "${INFO:-none,monitor}" \
    --episodes "${EPISODES:-20}" \
    --model "${MODEL:-qwen2.5:14b}" \
    --out "$OUT"
python analyze_perception.py "$OUT"
