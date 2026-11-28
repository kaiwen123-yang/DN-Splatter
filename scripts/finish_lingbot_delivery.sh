#!/usr/bin/env bash
# Run inside the experiment's tmux server. A restarted coordinator reuses verified stages.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
r="$root/.local/runs/lingbot-depth-map-20261128-v1"
tm=${LINGBOT_TMUX:-$(command -v tmux || true)}
[[ -n "$tm" ]] || tm=/root/autodl-tmp/lingbot-20261128-bootstrap/bin/tmux
if [[ ${1:-} != worker ]]; then
  "$tm" -L lingbot has-session -t '=hall-delivery' 2>/dev/null ||
    "$tm" -L lingbot new-session -d -s hall-delivery bash "$root/scripts/finish_lingbot_delivery.sh" worker
  exit
fi
exec 9>"$r/state/delivery.lock"; flock -n 9 || exit 75
exec >>"$r/logs/delivery.log" 2>&1
trap 'rc=$?; echo "$rc" > "$r/state/delivery.exit"; date -Is >> "$r/logs/delivery.log"' EXIT
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3
export XDG_CACHE_HOME="$r/cache/delivery" MPLCONFIGDIR="$r/cache/matplotlib" TMPDIR="$r/cache/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
d="$root/.local/conda/lingbot-depth-bw/bin/python"
m="$root/.local/conda/lingbot-map2-bw/bin/python"
await_file() {
  local file=$1 session=$2
  until [[ -s "$file" ]]; do
    if ! "$tm" -L lingbot has-session -t "=$session" 2>/dev/null; then
      [[ -s "$file" ]] || { echo "Missing $file after $session ended"; return 1; }
    fi
    sleep 3
  done
}
echo "Final delivery coordinator started $(date -Is), PID $$"
await_file "$r/depth/manifest.json" hall-depth
if [[ ! -s "$r/depth/evaluation/metrics.json" ]]; then
  if "$tm" -L lingbot has-session -t '=hall-evaluate-depth' 2>/dev/null; then
    await_file "$r/depth/evaluation/metrics.json" hall-evaluate-depth
  else "$d" scripts/lingbot_depth_evaluate.py; fi
fi
if "$tm" -L lingbot has-session -t '=hall-media-depth' 2>/dev/null; then
  await_file "$root/reports/lingbot-depth-map/20261128-v1/previews/depth_recovery.png" hall-media-depth
elif [[ ! -s "$root/reports/lingbot-depth-map/20261128-v1/previews/depth_recovery.png" ]]; then "$d" scripts/lingbot_media.py depth; fi
for seq in short_train_256 full_train_1478 full_all_1846_exploratory; do
  await_file "$r/map/$seq/complete.json" hall-map
  if [[ "$seq" == full_train_1478 ]] && "$tm" -L lingbot has-session -t '=lb-full-train-delivery' 2>/dev/null; then
    await_file "$r/media/map/$seq/display.json" lb-full-train-delivery
  fi
  if [[ ! -s "$r/map/$seq/evaluation/metrics.json" ]]; then "$m" scripts/lingbot_map_evaluate.py "$seq"; fi
  if [[ ! -s "$r/media/map/$seq/display.json" ]]; then "$m" scripts/lingbot_media.py "$seq"; fi
  echo "Verified predictions, evaluation and replay for $seq"
done
"$d" scripts/lingbot_report.py
echo "Report and SHA-256 catalogs ready $(date -Is)"
