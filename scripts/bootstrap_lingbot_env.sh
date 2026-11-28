#!/usr/bin/env bash
# Rebuild an independent prefix on the verified AutoDL CUDA 12.8 image.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
branch=${1:?depth or map2}
case "$branch" in
  depth) repo=lingbot-depth; commit=f3a237e434ae987bc38281476d6cfb5df3e4d739 ;;
  map2) repo=lingbot-map-v2; commit=078b54485830c74ab1d3797667073e3e8353f3b9 ;;
  *) exit 2 ;;
esac
run="$root/.local/runs/lingbot-depth-map-20261128-v1"
prefix="$root/.local/conda/lingbot-$branch-bw"
source_dir="$root/.local/upstream/$repo"
base=${LINGBOT_BASE_PREFIX:-/root/miniconda3}
export PIP_CACHE_DIR="$run/cache/pip" CONDA_PKGS_DIRS="$run/cache/conda" HF_HOME="$run/cache/hf"
export MODELSCOPE_CACHE="$run/cache/modelscope" TMPDIR="$run/cache/tmp" XDG_CACHE_HOME="$run/cache/bootstrap"
export CUDA_HOME=/usr/local/cuda-12.8 PYTHONNOUSERSITE=1
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$source_dir" "$run/weights/$branch"
if ! test -d "$source_dir/.git"; then
  git -C "$source_dir" init -q
  git -C "$source_dir" remote add origin "https://github.com/Robbyant/$repo.git"
fi
test "$(git -C "$source_dir" remote get-url origin)" = "https://github.com/Robbyant/$repo.git"
if ! git -C "$source_dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
  (source /etc/network_turbo >/dev/null 2>&1; git -C "$source_dir" fetch --depth 1 origin "$commit") ||
    git -C "$source_dir" fetch --depth 1 origin "$commit"
fi
if [[ "$(git -C "$source_dir" rev-parse --verify HEAD 2>/dev/null || true)" != "$commit" ]]; then
  test -z "$(git -C "$source_dir" status --porcelain)"
  git -C "$source_dir" checkout -q --detach "$commit"
fi
if ! test -x "$prefix/bin/python"; then
  "$base/bin/python" -c 'import torch; assert torch.__version__=="2.8.0+cu128" and "sm_120" in torch.cuda.get_arch_list()'
  cp -an "$base/pkgs/." "$CONDA_PKGS_DIRS/"
  CONDA_OFFLINE=true "$base/bin/conda" create -y --clone "$base" --offline --copy -p "$prefix"
fi
export PATH="$prefix/bin:$CUDA_HOME/bin:$PATH"
python -c 'import pathlib,sys,torch; assert pathlib.Path(torch.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()) and torch.__version__=="2.8.0+cu128"'
printf '%s\n' 'torch==2.8.0+cu128' > "$run/cache/torch-$branch.txt"
export PIP_CONSTRAINT="$run/cache/torch-$branch.txt"
core=(opencv-python-headless==5.1.0.95 scipy==1.18.1 trimesh==5.2.2 imageio-ffmpeg==0.8.2 modelscope==1.40.3)
if [[ "$branch" == depth ]]; then
  python -m pip install "${core[@]}" xformers==0.0.32.post2 huggingface_hub==0.35.3 click
  python - "$run/weights/depth" <<'PY'
from modelscope import snapshot_download
import sys
snapshot_download("Robbyant/lingbot-depth-pretrain-vitl-14-v0.5", revision="8d8942390f40f5c4e2e0c3b3e0964d94d9fe5afc",
                  allow_patterns=["model.pt"], local_dir=sys.argv[1])
PY
  (cd "$run/weights/depth"; echo 'b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23  model.pt' | sha256sum -c -)
else
  python -m pip install "${core[@]}" flashinfer-python==0.7.6 einops==0.8.2 viser==1.1.1 onnxruntime
  (cd "$source_dir"; python download_model.py --source modelscope --output "$run/weights/map2")
  (cd "$run/weights/map2"; echo 'd3ae62a09d3caaeb0947c4372f6fb23a4c0e85304867fa9702948ba030310e27  model_weights.pt' | sha256sum -c -;
    echo 'e42d91271d05bf28e87f224713f2848af6d6b23ce70f94e06117c216b6e8f491  config.json' | sha256sum -c -)
fi
