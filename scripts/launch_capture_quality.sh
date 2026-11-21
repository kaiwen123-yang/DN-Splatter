#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
export DN_CAPTURE_PROTOCOL=capture_full_test_protocol
export DN_RUN_OVERRIDE="$PWD/.local/runs/capture-full-quality-v2-20261121-v1"
export DN_SCENE_OVERRIDE="$DN_RUN_OVERRIDE/inputs"
source scripts/autodl_env.sh
exec "$DN_ENV_PREFIX/bin/python" -u scripts/capture_quality_run.py
