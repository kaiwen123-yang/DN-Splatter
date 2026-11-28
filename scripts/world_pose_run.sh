#!/usr/bin/env bash
set -eu
project="$(cd "$(dirname "$0")/.." && pwd -P)"
run="$project/.local/runs/interactive-world-20261128-v1"
base="$run/b_map/matching"
cd "$project"
export PATH="$project/.local/conda/dn-splatter-bw/bin:/usr/local/cuda-12.8/bin:/root/miniconda3/bin:$PATH"
export PYTHONPATH="$base/python:$base/lib/mast3r"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_HOME="$base/hf" XDG_CACHE_HOME="$base/cache" TORCH_HOME="$base/torch"
mkdir -p "$base/cache" "$run/logs" "$run/state"
exec 9>"$run/state/pose-launch.lock"
flock -n 9 || exit 75
stamp="$(date -u +%Y%m%dT%H%M%S)"
exec >>"$run/logs/pose-$stamp.log" 2>&1
echo "$$" >"$run/state/pose.pid"
date -Is >"$run/state/pose.started"
trap 'rc=$?; echo "$rc" >"$run/state/pose.exit"; date -Is >"$run/state/pose.ended"' EXIT
test -s "$base/weights/model.safetensors"
python scripts/world_pose_refine.py all
