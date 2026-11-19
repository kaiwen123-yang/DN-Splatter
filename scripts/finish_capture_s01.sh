#!/usr/bin/env bash
# Run inside a separate project tmux pane when handing a live run to the instance.
set -eo pipefail
cd "$(dirname "$0")/.."
export DN_RUN_OVERRIDE="$PWD/.local/runs/capture-s01-dn-30k-20261119-v1"
export DN_SCENE_OVERRIDE="$DN_RUN_OVERRIDE/inputs"
source scripts/autodl_env.sh
while tmux has-session -t '=dn-capture-s01-full' 2>/dev/null; do sleep 5; done
if [[ ! -s "$DN_RUN_ROOT/state/publish.done.json" && -s "$DN_RUN_ROOT/training_complete.json" ]]; then
  # Replay unfinished post-processing once, keeping the bound final checkpoint.
  bash scripts/launch_capture_s01.sh || true
fi
if [[ -s "$DN_RUN_ROOT/state/publish.done.json" ]]; then
  python scripts/capture_s01_delivery.py complete
else
  python scripts/capture_s01_delivery.py blocked
fi
