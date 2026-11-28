#!/usr/bin/env bash
# Persistent, idempotent entry for a single already-admitted training treatment.
set -eu -o pipefail
project="$(cd "$(dirname "$0")/.." && pwd -P)"
variant="${1:?supply R0C, R1-R7, VR0C, VR1, or VR2}"
phase="${2:-all}"
case "$variant" in R0C|R[1-7]) lane=c_dn ;; VR0C|VR1|VR2) lane=f_vr ;; *) exit 2 ;; esac
case "$phase" in prepare|train|evaluate|all) ;; *) exit 2 ;; esac
root="$project/.local/runs/interactive-world-20261128-v1"
out="$root/$lane/$variant"
mkdir -p "$out/logs" "$out/state"
if [ "${3:-}" != "--worker" ]; then
    exec 9>"$out/launch.lock"
    flock 9
    if [ -s "$out/state/worker.pid" ]; then
        oldpid="$(cat "$out/state/worker.pid")"
        if [ -r "/proc/$oldpid/cmdline" ] && "$project/.local/conda/dn-splatter-bw/bin/python" - "$oldpid" "$project/scripts/interactive_dn_launch.sh" "$variant" <<'PY'
import os,sys
from pathlib import Path
p=Path("/proc")/sys.argv[1]/"cmdline"
args=p.read_bytes().rstrip(b"\0").split(b"\0") if p.exists() else []
sys.exit(0 if len(args)>=5 and args[-4:-2]==list(map(os.fsencode,sys.argv[2:])) and args[-1]==b"--worker" else 1)
PY
        then
            echo "active $variant pid=$oldpid"
            exit 0
        fi
    fi
    log="$out/logs/${phase}-$(date -u +%Y%m%dT%H%M%S)-$$.log"
    if [ -s "$out/state/worker.exit" ]; then
        cp "$out/state/worker.exit" "$log.previous_exit"
    fi
    : > "$out/state/worker.exit"
    : > "$out/state/worker.ended"
    nohup setsid bash "$project/scripts/interactive_dn_launch.sh" "$variant" "$phase" --worker \
        9>&- </dev/null >"$log" 2>&1 &
    echo "$!" >"$out/state/worker.pid"
    printf '%s\n' "$log" >"$out/state/current_log"
    echo "started $variant pid=$! log=$log"
    exit 0
fi
threads="${IW_DN_THREADS:-8}"
export DN_RUN_OVERRIDE="$out"
if [ "$lane" = f_vr ]; then
    export DN_SCENE_OVERRIDE="$root/restore/tree/.local/data/room_datasets/vr_room"
else
    export DN_SCENE_OVERRIDE="$root/restore/tree/.local/runs/capture-full-quality-v3-20261122-v1/inputs"
fi
trap 'rc=$?; echo "$rc" >"$out/state/worker.exit"; date -u +%FT%TZ >"$out/state/worker.ended"' EXIT
export DN_CAPTURE_PROTOCOL=capture_full_test_protocol
# Some historical packages have prefix-bound editable source paths.
source "$project/scripts/autodl_env.sh"
export DN_CPU_BUDGET="$threads" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_HOME="$root/shared/torch" XDG_CACHE_HOME="$root/shared/xdg" MPLCONFIGDIR="$root/shared/xdg/matplotlib"
mkdir -p "$TORCH_HOME" "$MPLCONFIGDIR"
alexnet=alexnet-owt-7be5be79.pth
restored="$root/restore/tree/.local/cache/torch/hub/checkpoints/$alexnet"
if [ -f "$root/restore/metric.verified.json" ] && [ -f "$restored" ]; then
    mkdir -p "$TORCH_HOME/hub/checkpoints"
    if [ ! -e "$TORCH_HOME/hub/checkpoints/$alexnet" ]; then
        ln "$restored" "$TORCH_HOME/hub/checkpoints/$alexnet" || test -f "$TORCH_HOME/hub/checkpoints/$alexnet"
    fi
fi
cd "$project"
printf '%s %s threads=%s\n' "$(date -u +%FT%TZ)" "$variant" "$threads"
"$DN_ENV_PREFIX/bin/python" -u scripts/interactive_dn_train.py "$variant" --phase "$phase"
