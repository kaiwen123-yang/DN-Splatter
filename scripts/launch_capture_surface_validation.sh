#!/usr/bin/env bash
# Adopt the active A3 lock, validate its two exports, and evaluate each against the frozen incumbent.
set -eo pipefail
cd "$(dirname "$0")/.."
export DN_CAPTURE_PROTOCOL=capture_full_test_protocol
export DN_RUN_OVERRIDE="$PWD/.local/runs/capture-full-quality-v3-20261122-v1"
source scripts/autodl_env.sh
export LD_LIBRARY_PATH="$PWD/.local/capture-geometry/20261119-s01-5hz-3982cb26/runtime/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
r="$DN_RUN_ROOT"; b="$PWD/.local/runs/capture-full-quality-v2-20261121-v1"
exec 8>"$r/state/frozen_validation_chain.lock"
flock -n 8 || exit 75
exec >>"$r/logs/frozen_validation_chain.log" 2>&1
trap 'echo $? > "$r/state/frozen_validation_chain.exit"' EXIT
echo "Adopting existing B3 fusion at $(date -u +%FT%TZ)"
flock -s "$r/state/native_pipeline.lock" true
if [[ ! -s "$r/export/B2/fusion_sensor/complete.json" || ! -s "$r/export/B2/fusion_model/complete.json" ]]; then
    bash scripts/launch_capture_surface_fusion.sh
fi
for branch in model sensor; do
    d="$r/export/B2/fusion_$branch"; dest="$r/evaluation/A4_$branch"
    "$DN_ENV_PREFIX/bin/python" - "$r" "$b" "$d" "$dest" "$branch" <<'PY'
import sys,time,os
from pathlib import Path
sys.path.insert(0,"scripts")
from capture_surface_tiles import read,write,sha,ROOT
run,baseline,d,dest=map(Path,sys.argv[1:5]); branch=sys.argv[5]
product=read(d/"complete.json"); mesh=d/"global.ply"
if product["branch"]!=branch or product["mesh"]["sha256"]!=sha(mesh):raise ValueError("Global identity mismatch")
write(dest/"manifest.json",dict(scene=str(baseline/"inputs"),output=str(dest),
      meshes=dict(supported_poisson=str(baseline/"meshes/supported/supported.ply"),**{"new_"+branch+"_surface":str(mesh)})))
write(run/"state/current.json",dict(run_id=run.name,phase="A4_frozen_validation",branch=branch,status="running",pid=os.getppid(),utc_epoch=time.time()))
PY
    if [[ ! -s "$dest/complete.json" ]]; then
        "$DN_ENV_PREFIX/bin/python" -u scripts/capture_surface_evaluate.py "$dest/manifest.json" val
        "$DN_ENV_PREFIX/bin/python" - "$dest" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,"scripts")
from capture_surface_tiles import read,write,sha
d=Path(sys.argv[1]);result=read(d/"surfaces_val.json")
assert len(read(d/"surface_frames_val.json"))==184
assert all(v["reference"]==29222080 for v in result["groups"]["all"].values())
write(d/"complete.json",dict(split="val",views=184,reference_pixels=29222080,metrics_sha256=sha(d/"surfaces_val.json")))
PY
    fi
done
"$DN_ENV_PREFIX/bin/python" - "$r" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,"scripts")
from capture_surface_tiles import read,write,sha
r=Path(sys.argv[1]);a={b:read(r/"evaluation"/("A4_"+b)/"complete.json") for b in ("model","sensor")}
write(r/"state/frozen_pair_val.json",a)
write(r/"state/current.json",dict(run_id=r.name,phase="A4_frozen_validation",status="completed",branches=list(a)))
PY
