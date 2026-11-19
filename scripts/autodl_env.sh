#!/usr/bin/env bash
# Source this file from the project runner. No shell startup files are changed.
export DN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export DN_ENV_PREFIX="$DN_PROJECT_ROOT/.local/conda/dn-splatter-bw"
export DN_SOURCE_DIR="$DN_PROJECT_ROOT/.local/upstream/dn-splatter"
export DN_DATA_ROOT="$DN_PROJECT_ROOT/.local/data"
export DN_SCENE_DIR="${DN_SCENE_OVERRIDE:-$DN_DATA_ROOT/room_datasets/vr_room}"
export DN_RUN_ROOT="${DN_RUN_OVERRIDE:-$DN_PROJECT_ROOT/.local/runs/vr-room-iphone-30k}"
export CONDA_PKGS_DIRS="$DN_PROJECT_ROOT/.local/cache/conda_pkgs"
export PIP_CACHE_DIR="$DN_PROJECT_ROOT/.local/cache/pip"
export HF_HOME="$DN_PROJECT_ROOT/.local/cache/hf"
export TORCH_HOME="$DN_PROJECT_ROOT/.local/cache/torch"
export XDG_CACHE_HOME="$DN_PROJECT_ROOT/.local/cache/xdg"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TORCH_EXTENSIONS_DIR="$DN_PROJECT_ROOT/.local/cache/cuda_extensions/torch271-cu128-sm120"
export TMPDIR="$DN_PROJECT_ROOT/.local/tmp"
export CUDA_HOME="${DN_CUDA_TOOLKIT:-/usr/local/cuda-12.8}"
export CUDA_VISIBLE_DEVICES="${DN_GPU_INDEX:-0}"
export TORCH_CUDA_ARCH_LIST=12.0
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONPATH PYTHONHOME
export PATH="$DN_ENV_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$DN_ENV_PREFIX/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_MAX_ACTIVE_LEVELS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR" "$DN_RUN_ROOT/logs" "$DN_RUN_ROOT/state" "$DN_DATA_ROOT/downloads"
read -r DN_CPU_BUDGET MAX_JOBS < <("$DN_ENV_PREFIX/bin/python" - <<'PY'
import os
from pathlib import Path
cg = Path("/sys/fs/cgroup")
cpus = len(os.sched_getaffinity(0))
if (cg / "cpu.max").exists():
    quota, period = (cg / "cpu.max").read_text().split()
    if quota != "max":
        cpus = min(cpus, max(1, int(quota) // int(period)))
else:
    quota = int((cg / "cpu/cpu.cfs_quota_us").read_text())
    if quota > 0:
        cpus = min(cpus, max(1, quota // int((cg / "cpu/cpu.cfs_period_us").read_text())))
mem = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:"))) * 1024
for limit, used in [("memory.max", "memory.current"),
                    ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes")]:
    if (cg / limit).is_file() and (cg / limit).read_text().strip() != "max":
        mem = min(mem, int((cg / limit).read_text()) - int((cg / used).read_text()))
print(cpus, min(cpus, max(1, mem // (5 * 1024**3))))
PY
)
export DN_CPU_BUDGET MAX_JOBS
