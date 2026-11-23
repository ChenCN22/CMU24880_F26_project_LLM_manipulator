#!/bin/bash
# usage: scripts/bench_claude.sh <claude model id> <cap in dollars> <$ per 1M input tokens> <$ per 1M output tokens>
# The same as scripts/bench_api.sh anthropic:<claude model id> ...
exec bash "$(dirname "$0")/bench_api.sh" "anthropic:$1" "${@:2}"
