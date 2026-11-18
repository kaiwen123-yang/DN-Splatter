#!/usr/bin/env bash
# Run a GitHub operation with project-scoped network settings.
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd -P)"
export GIT_TERMINAL_PROMPT=0
if [[ -r /etc/network_turbo ]] && (set +u; source /etc/network_turbo >/dev/null; "$@"); then exit 0; fi
if [[ -r "$root/.local/network/proxy.url" ]] && (
  https_proxy="$(cat "$root/.local/network/proxy.url")"
  export https_proxy
  "$@"
); then exit 0; fi
exec "$@"
