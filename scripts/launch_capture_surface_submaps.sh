#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
export DN_CAPTURE_PROTOCOL=capture_full_test_protocol
export DN_RUN_OVERRIDE="$PWD/.local/runs/capture-full-quality-v3-20261122-v1"
export DN_SCENE_OVERRIDE="$DN_RUN_OVERRIDE/inputs"
source scripts/autodl_env.sh
export LD_LIBRARY_PATH="$PWD/.local/capture-geometry/20261119-s01-5hz-3982cb26/runtime/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
label=${1:-B2}
case "$label" in B2|B3) ;; *) exit 2 ;; esac
exec 9>"$DN_RUN_ROOT/state/${label}_submap_launcher.lock"
flock -n 9 || exit 75
printf '%s\n' "$$" > "$DN_RUN_ROOT/state/${label}_submap_launcher.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$DN_RUN_ROOT/state/${label}_submap_exit.tmp"; mv "$DN_RUN_ROOT/state/${label}_submap_exit.tmp" "$DN_RUN_ROOT/state/${label}_submap_exit"' EXIT
export OMP_NUM_THREADS="$DN_CPU_BUDGET"
echo "Submap launch $(date -u +%FT%TZ) budget=$DN_CPU_BUDGET pid=$$" >> "$DN_RUN_ROOT/logs/${label}_submaps.log"
"$DN_ENV_PREFIX/bin/python" -u scripts/capture_surface_submaps.py "$label" >> "$DN_RUN_ROOT/logs/${label}_submaps.log" 2>&1
