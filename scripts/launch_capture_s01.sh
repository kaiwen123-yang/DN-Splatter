#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
export DN_RUN_OVERRIDE="$PWD/.local/runs/capture-s01-dn-30k-20261119-v1"
export DN_SCENE_OVERRIDE="$DN_RUN_OVERRIDE/inputs"
source scripts/autodl_env.sh
exec python scripts/capture_s01_run.py "$@"
