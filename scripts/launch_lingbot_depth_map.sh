#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
RUN="$ROOT/.local/runs/lingbot-depth-map-20261128-v1"
TMUX_BIN=${LINGBOT_TMUX:-$(command -v tmux || true)}
if [[ -z "$TMUX_BIN" ]]; then TMUX_BIN=/root/autodl-tmp/lingbot-20261128-bootstrap/bin/tmux; fi
mkdir -p "$RUN"/{logs,state,cache}
stage=${1:?choose prepare, depth, map, evaluate-depth, evaluate-map, media-depth, media-map or report}
case "$stage" in
  prepare|depth|evaluate-depth|media-depth|report) env_name=lingbot-depth-bw ;;
  map|evaluate-map|media-map) env_name=lingbot-map2-bw ;;
  *) echo "Unknown stage $stage" >&2; exit 2 ;;
esac
if [[ ${2:-} != worker ]]; then
  name="hall-$stage"
  if "$TMUX_BIN" -L lingbot has-session -t "=$name" 2>/dev/null; then
    echo "Already running in $name"; exit 0
  fi
  "$TMUX_BIN" -L lingbot new-session -d -s "$name" bash "$ROOT/scripts/launch_lingbot_depth_map.sh" "$stage" worker
  exit
fi
exec 9>"$RUN/state/$stage.lock"
flock -n 9 || exit 75
attempt=$(date -u +%Y%m%dT%H%M%SZ)
log="$RUN/logs/$stage-$attempt.log"
exec >>"$log" 2>&1
ln -sfn "$(basename "$log")" "$RUN/logs/$stage-latest.log"
echo "$$" > "$RUN/state/$stage.pid"
date -Is > "$RUN/state/$stage.started"
trap 'rc=$?; echo "$rc" > "$RUN/logs/$stage-$attempt.exit"; echo "$rc" > "$RUN/state/$stage.exit"; date -Is > "$RUN/state/$stage.ended"' EXIT
export PATH="$ROOT/.local/conda/$env_name/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$RUN/cache/$stage" TORCH_HOME="$RUN/cache/torch" HF_HOME="$RUN/cache/hf"
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor" TRITON_CACHE_DIR="$RUN/cache/triton"
export FLASHINFER_WORKSPACE_BASE="$RUN/cache/flashinfer" MPLCONFIGDIR="$RUN/cache/matplotlib"
export TMPDIR="$RUN/cache/tmp" PIP_CACHE_DIR="$RUN/cache/pip" TORCH_EXTENSIONS_DIR="$RUN/cache/extensions"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 MAX_JOBS=5 TORCHINDUCTOR_COMPILE_THREADS=4
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$MPLCONFIGDIR"
case "$stage" in
  prepare) python scripts/lingbot_prepare.py ;;
  depth)
    test -f "$RUN/prepare/freeze.json"
    export PYTHONPATH="$ROOT/.local/upstream/lingbot-depth"
    if [[ ! -f "$RUN/state/example0.done" ]]; then
      (cd "$PYTHONPATH" && python "$ROOT/scripts/lingbot_depth_backend.py" "$PYTHONPATH/example.py" "$RUN/state/depth_kernel_check.json" --example 0 --device cuda --model "$RUN/weights/depth/model.pt" --output "$RUN/depth-example0")
      test -s "$RUN/depth-example0/depth.npy" || test -s "$RUN/depth-example0/point_cloud.ply"
      sha256sum "$RUN/weights/depth/model.pt" > "$RUN/state/example0.done"
    fi
    python scripts/lingbot_depth_run.py "$RUN/weights/depth/model.pt" ;;
  map)
    test -f "$RUN/prepare/freeze.json"
    python scripts/lingbot_map_run.py "$ROOT/.local/upstream/lingbot-map-v2" "$RUN/weights/map2" ;;
  evaluate-depth) python scripts/lingbot_depth_evaluate.py ;;
  evaluate-map) python scripts/lingbot_map_evaluate.py ;;
  media-depth) python scripts/lingbot_media.py depth ;;
  media-map) python scripts/lingbot_media.py map ;;
  report) python scripts/lingbot_report.py ;;
esac
