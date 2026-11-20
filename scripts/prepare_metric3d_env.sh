#!/usr/bin/env bash
# Project-scoped setup for Metric3D on the existing Blackwell PyTorch ABI.
set -eo pipefail
cd "$(dirname "$0")/.."
export DN_RUN_OVERRIDE="$PWD/.local/runs/capture-full-metric3d-30k-20261120-v1"
source scripts/autodl_env.sh
dest="$PWD/.local/conda/metric3d-bw"
src="$PWD/.local/upstream/Metric3D"
pin=eb5b6fac0dc155e4e52f576e304fbf11655ff339
if [ ! -e "$src" ]; then
    git init -q "$src"
    git -C "$src" remote add origin https://github.com/YvanYin/Metric3D.git
fi
if ! git -C "$src" cat-file -e "$pin^{commit}" 2>/dev/null; then
    (source /etc/network_turbo >/dev/null 2>&1 && git -C "$src" fetch -q --depth=1 origin "$pin") ||
        git -C "$src" fetch -q --depth=1 origin "$pin"
fi
if ! git -C "$src" rev-parse --verify HEAD >/dev/null 2>&1; then git -C "$src" checkout -q --detach "$pin"; fi
test "$(git -C "$src" rev-parse HEAD)" = "$pin"
for compat in "$PWD"/patches/metric3d-eb5b6fac-*.patch; do
    if ! git -C "$src" apply --reverse --check "$compat" 2>/dev/null; then
        git -C "$src" apply --check "$compat"
        git -C "$src" apply "$compat"
    fi
done
if [ ! -e "$dest" ]; then
    /root/miniconda3/bin/conda create -y --offline --prefix "$dest" --clone "$DN_ENV_PREFIX"
fi
test -f "$dest/conda-meta/history"
if ! "$dest/bin/python" -c 'import mmengine; assert mmengine.__version__ == "0.10.7"' 2>/dev/null; then
    "$dest/bin/python" -m pip install --no-input 'mmengine==0.10.7'
fi
(cd "$src" && "$dest/bin/python" -c 'import hubconf, torch; assert torch.__version__ == "2.7.1+cu128"')
