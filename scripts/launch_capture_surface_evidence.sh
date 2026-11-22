#!/usr/bin/env bash
# Run the native-Z repeatability pass after the matching full local TSDF export.
set -eo pipefail
cd "$(dirname "$0")/.."
export DN_CAPTURE_PROTOCOL=capture_full_test_protocol
export DN_RUN_OVERRIDE="$PWD/.local/runs/capture-full-quality-v3-20261122-v1"
export DN_SCENE_OVERRIDE="$DN_RUN_OVERRIDE/inputs"
source scripts/autodl_env.sh
export LD_LIBRARY_PATH="$PWD/.local/capture-geometry/20261119-s01-5hz-3982cb26/runtime/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
label=${1:-B2}
case "$label" in B2|B3) ;; *) exit 2 ;; esac
state="$DN_RUN_ROOT/state"
out="$DN_RUN_ROOT/export/$label"
log="$DN_RUN_ROOT/logs/${label}_repeatability.log"
exec 8>"$state/${label}_repeatability.lock"
flock -n 8 || exit 75
exec >>"$log" 2>&1
echo "Waiting for $label local surfaces at $(date -u +%FT%TZ), pid=$$"
flock -s "$state/${label}_submap_launcher.lock" true
if [[ ! -s "$out/submaps.json" ]]; then
    bash scripts/launch_capture_surface_submaps.sh "$label"
fi
test -s "$out/submaps.json"
trap 'rc=$?; echo "$rc" > "$state/${label}_repeatability_exit.tmp"; mv "$state/${label}_repeatability_exit.tmp" "$state/${label}_repeatability_exit"' EXIT
echo "Calculating reciprocal native Z residuals at $(date -u +%FT%TZ)"
if [[ ! -s "$out/tolerances.json" ]]; then
    "$DN_ENV_PREFIX/bin/python" -u scripts/capture_surface_repeatability.py "$label"
fi
"$DN_ENV_PREFIX/bin/python" - "$out" <<'PY'
import hashlib, json, pathlib, sys
d = pathlib.Path(sys.argv[1])
load = lambda name: json.loads((d / name).read_text())
tol = load("tolerances.json")
assert tol["camera_sha256"] == load("identity.json")["cameras"]["sha256"]
assert len(tol["bins"]) == 4 and tol["train_pairs"] >= 1477
assert load("submaps.json")["identity"] == hashlib.sha256((d / "identity.json").read_bytes()).hexdigest()
print(tol["bins"])
PY
echo "Completed at $(date -u +%FT%TZ)"
