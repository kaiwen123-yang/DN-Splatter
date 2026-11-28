#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
seq=${1:-full_train_1478}
case "$seq" in short_train_256|full_train_1478|full_all_1846_exploratory) ;; *) exit 2 ;; esac
run="$root/.local/runs/lingbot-depth-map-20261128-v1"
python="$root/.local/conda/lingbot-map2-bw/bin/python"
directory=$("$python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["audit"]["directory"])' "$run/map/$seq/complete.json")
export XDG_CACHE_HOME="$run/cache/viewer" MPLCONFIGDIR="$run/cache/matplotlib" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
cd "$root/.local/upstream/lingbot-map-v2"
exec "$python" demo.py --predictions "$directory" --host 127.0.0.1 --port "${2:-18788}"
