#!/usr/bin/env bash
# B3 native coordinator. Never treats missing implementations as completed evidence.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export DN_CAPTURE_PROTOCOL=capture_full_test_protocol
export DN_RUN_OVERRIDE="$PWD/.local/runs/capture-full-quality-v3-20261122-v1"
export DN_SCENE_OVERRIDE="$DN_RUN_OVERRIDE/inputs"
source scripts/autodl_env.sh
mkdir -p "$DN_RUN_ROOT/state" "$DN_RUN_ROOT/logs"
exec 9>"$DN_RUN_ROOT/state/native_pipeline.lock"
flock -n 9 || { echo 'B3 native pipeline already running' >&2; exit 75; }
exec >>"$DN_RUN_ROOT/logs/native_pipeline.log" 2>&1
printf '%s\n' "$$" > "$DN_RUN_ROOT/state/native_pipeline.pid"
status() {
    "$DN_ENV_PREFIX/bin/python" - "$DN_RUN_ROOT/state/native_pipeline.json" "$1" "$2" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]); tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps(dict(run_id='capture-full-quality-v3-20261122-v1',phase=sys.argv[2],status=sys.argv[3],pid=os.getppid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
os.replace(tmp,p)
PY
}
phase=initializing
trap 'rc=$?; if (( rc != 0 && rc != 78 )); then status "$phase" "failed_exit_$rc"; fi; printf "%s\n" "$rc" > "$DN_RUN_ROOT/state/native_pipeline.exit"' EXIT
phase=A2_repeatability
status "$phase" running
# This existing launcher waits for A1's lock and validates both frozen identities.
bash scripts/launch_capture_surface_evidence.sh B2
phase=A3_fusion
status "$phase" implementation_incomplete
printf '%s\n' 'A3 spatial ownership, clipping and assembly are not implemented; refusing to claim completion or launch downstream stages.' >&2
exit 78
