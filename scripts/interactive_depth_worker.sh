#!/usr/bin/env bash
# Durable A-line worker. Each stage's actual exit status and immutable artifacts are retained.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
a="$root/.local/runs/interactive-world-20261128-v1/a_depth"
p="$root/.local/conda/lingbot-depth-bw/bin/python"
mode="${1:-all}"
mkdir -p "$a/logs" "$a/locks" "$a/status"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

event() {
    "$p" - "$a" "$@" <<'PY'
import json,sys,time,os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
base=Path(sys.argv[1])
stage,phase,code=sys.argv[2:]
row=dict(stage=stage,status=phase,exit_code=int(code) if code else None,
         time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),pid=os.getppid())
path=base/"status"/(stage+".json")
tmp=path.with_suffix(".tmp."+str(os.getpid()))
tmp.write_text(json.dumps(row)+"\n"); os.replace(tmp,path)
with (base/"logs"/(stage+".events.jsonl")).open("a") as f:
    f.write(json.dumps(row)+"\n")
PY
}

stage() (
    local name="$1" lock="$2" marker="$3"
    shift 3
    exec 8>"$lock"
    flock 8
    if [[ -s "$marker" ]]; then
        event "$name" completed 0
        return 0
    fi
    local log="$a/logs/$name.$(date +%Y%m%dT%H%M%S).log"
    event "$name" running ""
    set +e
    "$@" >> "$log" 2>&1
    local rc=$?
    set -e
    if [[ "$rc" -eq 0 && ! -s "$marker" ]]; then rc=65; fi
    if [[ "$rc" -ne 0 ]]; then event "$name" failed "$rc"; return "$rc"; fi
    event "$name" completed 0
)

hall() {
    local packet=()
    if [[ -s "$a/native/manifest.json" ]]; then packet=(--packet "$a/native"); fi
    stage A_freeze "$a/locks/freeze" "$a/freeze.json" "$p" scripts/interactive_depth_inputs.py --out "$a" "${packet[@]}"
    stage A_R1 "$a/sensor.lock" "$a/supervision/R1/manifest.json" "$p" scripts/interactive_depth_candidates.py --workers "${IW_A_CPU_WORKERS:-3}"
    stage A_natural "$a/mdm.lock" "$a/natural_manifest.json" "$p" scripts/interactive_depth_mdm.py natural
    stage A_masked "$a/mdm.lock" "$a/masked_manifest.json" "$p" scripts/interactive_depth_mdm.py masked
    for k in 0 1 2 3 4; do
        stage "A_fold_$k" "$a/locks/quality" "$a/quality/fold_$k/R3_frames.json" \
            "$p" scripts/interactive_depth_metrics.py --fold "$k"
    done
    # Idempotent aggregation also recovers an interruption between the last fold and the final two manifests.
    stage A_R3 "$a/locks/quality" "$a/supervision/R3/manifest.json" "$p" scripts/interactive_depth_metrics.py --fold 4
}

vr() {
    local v="$a/vr_transfer"
    stage A_VR_masks "$a/locks/vr_prepare" "$v/masked_manifest.json" "$p" scripts/interactive_depth_vr.py make
    event A_VR_zero_shot pending ""
    while [[ ! -s "$a/quality/fold_0/calibration.json" ]]; do
        # A failed/dependency-blocked hall fit needs diagnosis; preserve the independent VR inputs.
        if "$p" -c 'import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); sys.exit(0 if any(json.loads(x.read_text())["status"]=="failed" for x in p.glob("A_*.json") if not x.name.startswith("A_VR")) else 1)' "$a/status"; then
            event A_VR_zero_shot blocked 66
            return 66
        fi
        sleep 45
    done
    stage A_VR_zero_shot "$a/locks/vr_score" "$v/hall_fold0_zero_shot.json" "$p" scripts/interactive_depth_vr.py evaluate
    stage A_VR_adapt "$a/locks/vr_adapt" "$v/vr_train_only_adaptation.json" "$p" scripts/interactive_depth_vr.py adapt
}

case "$mode" in
    hall) hall ;;
    vr) vr ;;
    all)
        hall & hall_pid=$!
        vr & vr_pid=$!
        rc_h=0; wait "$hall_pid" || rc_h=$?
        rc_v=0; wait "$vr_pid" || rc_v=$?
        if [[ "$rc_h" -ne 0 ]]; then exit "$rc_h"; fi
        exit "$rc_v" ;;
    *) echo "Usage: bash scripts/interactive_depth_worker.sh [all|hall|vr]" >&2; exit 2 ;;
esac
