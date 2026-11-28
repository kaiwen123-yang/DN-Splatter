#!/usr/bin/env bash
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
run="$root/.local/runs/interactive-world-20261128-v1"
e="$run/e_interaction"; d="$e/phystwin"
py="$root/.local/iwenv/physwarp/bin/python"
depth_py="$root/.local/conda/lingbot-depth-bw/bin/python"
weight="$root/.local/runs/lingbot-depth-map-20261128-v1/weights/depth/model.pt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$root/.local/upstream/lingbot-depth:$root/scripts"
export TORCH_HOME="$root/.local/iwenv/torch-cache" XDG_CACHE_HOME="$root/.local/iwenv/cache"
mkdir -p "$d"
exec 9>"$d/pipeline.lock";flock -n 9 || exit 0
echo "$$" > "$d/pid"
trap 'echo $? > "$d/exit.code"' EXIT
phase() {
    local case=$1 name=$2;shift 2
    local folder="$d/$case/receipts/$name"
    mkdir -p "$folder"
    if [ "$(cat "$folder/status" 2>/dev/null)" = 0 ]; then return 0; fi
    date -Iseconds > "$folder/start"
    "$@" >> "$folder/run.log" 2>&1
    local code=$?
    echo "$code" > "$folder/status";date -Iseconds > "$folder/end";echo "$case $name exit=$code"
    return "$code"
}
rc=0
for case in single_push_rope_1 single_push_rope_4; do
    phase "$case" controls "$py" scripts/interactive_phystwin.py controls "$e" "$case" || { rc=1;continue; }
    phase "$case" prepare "$py" scripts/interactive_phystwin.py prepare "$e" "$case" || { rc=1;continue; }
    phase "$case" infer "$depth_py" scripts/interactive_phystwin.py infer "$e" "$case" --weights "$weight" || rc=1
    phase "$case" G-NN "$py" scripts/interactive_phystwin.py predict "$e" "$case" --variant G-NN || rc=1
    if [ -s "$d/$case/G-LingBot.npy" ]; then
        phase "$case" G-LingBot "$py" scripts/interactive_phystwin.py predict "$e" "$case" --variant G-LingBot || rc=1
    fi
done
if ! [ -s "$d/single_push_rope_1/camera0_input.npz" ] && ! [ -s "$d/single_push_rope_4/camera0_input.npz" ]; then exit 1; fi
fold="$run/a_depth/quality/fold_0"
if ! [ -s "$fold/calibration.json" ]; then
    echo "Waiting for frozen hall fold_0/calibration.json for G-Q1"
    while ! [ -s "$fold/calibration.json" ]; do sleep 60; done
fi
for case in single_push_rope_1 single_push_rope_4; do
    input="$d/$case/camera0_input.npz";test -s "$input" || continue
    phase "$case" transfer "$depth_py" scripts/interactive_depth_transfer.py "$input" "$d/$case/transfer.npz" --fold-dir "$fold" || { rc=1;continue; }
    phase "$case" G-Q1 "$py" scripts/interactive_phystwin.py predict "$e" "$case" --variant G-Q1 || rc=1
done
exit "$rc"
