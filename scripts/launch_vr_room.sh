#!/usr/bin/env bash
# Invoke from the checkout, e.g. tmux new -d -s dn-vr-room-30k 'bash scripts/launch_vr_room.sh'
set -euo pipefail
source "$(dirname "$0")/autodl_env.sh"
cd "$DN_PROJECT_ROOT"
export PYOPENGL_PLATFORM=egl
exec python scripts/vr_room_run.py >>"$DN_RUN_ROOT/logs/workflow.log" 2>&1
