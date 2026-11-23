#!/usr/bin/env bash
# Own the continuation, adopting any active frozen validation chain before proceeding.
set -eo pipefail
cd "$(dirname "$0")/.."
export DN_CAPTURE_PROTOCOL=capture_full_test_protocol
export DN_RUN_OVERRIDE="$PWD/.local/runs/capture-full-quality-v3-20261122-v1"
export DN_SCENE_OVERRIDE="$DN_RUN_OVERRIDE/inputs"
source scripts/autodl_env.sh
export LD_LIBRARY_PATH="$PWD/.local/capture-geometry/20261119-s01-5hz-3982cb26/runtime/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
r="$DN_RUN_ROOT"
exec 7>"$r/state/continuation.lock"
flock -n 7 || exit 75
exec >>"$r/logs/continuation.log" 2>&1
trap 'rc=$?; echo "$rc" > "$r/state/continuation.exit"' EXIT
# This train-only CPU selection freezes crops and camera names before A4 scores are read.
"$DN_ENV_PREFIX/bin/python" -u scripts/capture_surface_texture.py --freeze
# The earlier launcher holds this lock until the last A4 output is closed.
flock -s "$r/state/frozen_validation_chain.lock" true
if [[ ! -s "$r/state/frozen_pair_val.json" ]]; then
    bash scripts/launch_capture_surface_validation.sh
fi
"$DN_ENV_PREFIX/bin/python" -u scripts/capture_surface_continue.py
