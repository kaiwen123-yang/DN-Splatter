#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/autodl_env.sh"
cd "$DN_PROJECT_ROOT"
exec 8>"$DN_RUN_ROOT/state/install.lock"
flock -n 8 || { echo "Installation already running"; exit 75; }
exec >>"$DN_RUN_ROOT/logs/install.log" 2>&1
trap 'echo $? > "$DN_RUN_ROOT/state/install.exit"' EXIT
echo "Begin installation $(date -Is) CPU=$DN_CPU_BUDGET compile_jobs=$MAX_JOBS"
python -m pip install 'setuptools<81' wheel ninja 'numpy==1.26.4' 'packaging<26'
torch_url='https://download-r2.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=d6c3cba198dc93f93422a8545f48a6697890366e4b9701f54351fc27e2304bd3'
vision_url='https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=538f4db667286d939b4eee0a66d31ed21b51186668006b0e0ffe20338ecc7e00'
if ! python -c 'from importlib.metadata import version; assert version("torch")=="2.7.1+cu128" and version("torchvision")=="0.22.1+cu128"' 2>/dev/null; then
  python -m pip install --prefer-binary "$torch_url" "$vision_url"
fi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability()); assert torch.cuda.get_device_capability() == (12, 0)'
src="$DN_PROJECT_ROOT/.local/upstream/gsplat"
test "$(git -C "$DN_SOURCE_DIR" rev-parse HEAD)" = 811f5cdf342d92d12a9d9a4be185d374c430c184
test "$(git -C "$src" rev-parse HEAD)" = c7b0a383657307a13dff56cb2f832e3ab7f029fd
test "$(git -C "$src/gsplat/cuda/csrc/third_party/glm" rev-parse HEAD)" = 45008b225e28eb700fa0f7d3ff69b7c1db94fadf
for patch in "$DN_PROJECT_ROOT"/patches/*.patch; do
  case "$(basename "$patch")" in
    gsplat-1.0.0-*) tree="$src" ;;
    dn-811f5cd-*) tree="$DN_SOURCE_DIR" ;;
    omnidata-tools-0.0.23-backbone.patch) continue ;;
    *) echo "Unknown patch target: $(basename "$patch")"; exit 1 ;;
  esac
  if ! git -C "$tree" apply --reverse --check "$patch" 2>/dev/null; then git -C "$tree" apply "$patch"; fi
done
if ! python - <<'PY'
from importlib.metadata import distribution
from importlib.util import module_from_spec, spec_from_file_location
import torch
try:
    dist = distribution("gsplat")
    assert dist.version == "1.0.0"
    for name in ("csrc", "csrc_legacy"):
        spec = spec_from_file_location(name, dist.locate_file(f"gsplat/{name}.so"))
        spec.loader.exec_module(module_from_spec(spec))
except (ImportError, AssertionError, FileNotFoundError):
    raise SystemExit(1)
PY
then
  python -m pip install --no-build-isolation --no-deps -v "$src"
fi
python -m pip install --prefer-binary -c scripts/blackwell_constraints.txt -e "$DN_SOURCE_DIR"
site_dir="$(python -c 'from importlib.metadata import distribution; d=distribution("omnidata-tools"); assert d.version=="0.0.23"; print(d.locate_file(""))')"
dep_patch="$DN_PROJECT_ROOT/patches/omnidata-tools-0.0.23-backbone.patch"
if ! patch --batch --dry-run -R -p1 -d "$site_dir" -i "$dep_patch" >/dev/null; then
  patch --batch --forward -p1 -d "$site_dir" -i "$dep_patch"
fi
python -m pip check
python -m pip list --format=freeze > "$DN_RUN_ROOT/environment.freeze"
