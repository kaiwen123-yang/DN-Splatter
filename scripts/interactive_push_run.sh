#!/usr/bin/env bash
# Server-only D and PushT branch. Public input recovery is independent of the hall experiments.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
run="$root/.local/runs/interactive-world-20261128-v1"
e="$run/e_interaction"
py="$root/.local/iwenv/rigid/bin/python"
torchpy="$root/.local/conda/lingbot-depth-bw/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$e/receipts/server_pipeline" "$run/d_world/memory"
exec 9>"$e/receipts/server_pipeline/lock"
flock -n 9 || { echo "D/E pipeline already active"; exit 0; }
echo "$$" > "$e/receipts/server_pipeline/pid"
date -Iseconds > "$e/receipts/server_pipeline/start"
trap 'echo $? > "$e/receipts/server_pipeline/status"' EXIT
stage() {
    name=$1
    shift
    folder="$e/receipts/$name"
    mkdir -p "$folder"
    (
        exec 8>"$folder/lock"
        flock 8
        if [ -s "$folder/status" ] && [ "$(cat "$folder/status")" = 0 ]; then return 0; fi
        echo "$BASHPID" > "$folder/pid"
        date -Iseconds > "$folder/start"
        set +e
        "$@" >> "$folder/run.log" 2>&1
        rc=$?
        set -e
        date -Iseconds > "$folder/end"; echo "$rc" > "$folder/status"
        return "$rc"
    )
}
stage evidence_memory "$py" scripts/interactive_physics_memory.py "$run/d_world/memory/results.json"
test -s "$e/assets/pusht/source.json" && test -s "$e/assets/tblock/source.json"
if [ ! -s "$e/prepare/split.json" ]; then
    stage prepare "$py" scripts/interactive_push_prepare.py "$e/assets/pusht" "$e/assets/tblock" "$e/prepare"
fi
track_v2() {
    if [ ! -s "$e/track_v2/tracker_freeze.json" ]; then
        "$py" scripts/interactive_push_track.py calibrate "$e/assets/pusht" "$e/prepare" "$e/track_v2" || return $?
    fi
    "$py" -u scripts/interactive_push_track.py track "$e/assets/pusht" "$e/prepare" "$e/track_v2" --workers 2
}
stage track_v2 track_v2
test -s "$e/track_v2/tracking_all.json"
stage fit "$py" -u scripts/interactive_push_replay.py fit "$e"
test -s "$e/calibration/freeze.json"
stage replay "$py" -u scripts/interactive_push_replay.py evaluate "$e" &
replay_pid=$!
stage generate "$py" -u scripts/interactive_generate.py "$e" --workers 2 &
gen_pid=$!
rc=0
wait "$gen_pid" || rc=$?
if [ -s "$e/generated/manifest.json" ]; then
    stage probe "$torchpy" -u scripts/interactive_probe.py "$e" --device cpu --threads 2 || rc=$?
fi
wait "$replay_pid" || rc=$?
if [ -s "$e/calibration/causal_online.json" ] && [ -f scripts/interactive_push_media.py ]; then
    stage media "$py" -u scripts/interactive_push_media.py "$e" || rc=$?
fi
exit "$rc"
