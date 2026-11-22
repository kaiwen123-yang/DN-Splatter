#!/usr/bin/env bash
# Resume A3 only; frozen submaps and tolerances are immutable inputs.
set -eo pipefail
cd "$(dirname "$0")/.."
export DN_RUN_OVERRIDE="$PWD/.local/runs/capture-full-quality-v3-20261122-v1"
export DN_CAPTURE_PROTOCOL=capture_full_test_protocol
source scripts/autodl_env.sh
export LD_LIBRARY_PATH="$PWD/.local/capture-geometry/20261119-s01-5hz-3982cb26/runtime/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
r="$DN_RUN_ROOT"
exec 9>"$r/state/native_pipeline.lock"
flock -n 9 || exit 75
exec >>"$r/logs/A3_fusion.log" 2>&1
status(){
  "$DN_ENV_PREFIX/bin/python" - "$r/state/current.json" "$1" "$2" "$$" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
v=dict(run_id=p.parent.parent.name,phase=sys.argv[2],status=sys.argv[3],pid=int(sys.argv[4]),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
t=p.with_suffix('.tmp');t.write_text(json.dumps(v)+'\n');os.replace(t,p)
print(v,flush=True)
PY
}
phase=A3_index
trap 'rc=$?; if ((rc)); then status "$phase" "failed_exit_$rc"; fi; echo "$rc" > "$r/state/A3_fusion.exit"' EXIT
status "$phase" adopting
flock -s "$r/state/A3_index.lock" true
if [[ ! -s "$r/export/B2/tiles/complete.json" ]]; then
    "$DN_ENV_PREFIX/bin/python" -u scripts/capture_surface_tiles.py B2
fi
for branch in model sensor; do
    phase=A3_${branch}_votes
    status "$phase" running
    if [[ ! -s "$r/export/B2/layers_$branch/complete.json" ]]; then
        "$DN_ENV_PREFIX/bin/python" -u scripts/capture_surface_layers.py B2 "$branch"
    fi
    phase=A3_${branch}_assembly
    status "$phase" running
    "$DN_ENV_PREFIX/bin/python" -u scripts/capture_surface_assemble.py B2 "$branch"
done
phase=A3_global_meshes
status "$phase" completed
