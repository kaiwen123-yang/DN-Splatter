#!/usr/bin/env bash
set -eu
project="$(cd "$(dirname "$0")/.." && pwd -P)"
run="$project/.local/runs/interactive-world-20261128-v1"
tmux_bin="${IW_TMUX:-$(command -v tmux || true)}"
if [ -z "$tmux_bin" ]; then tmux_bin=/root/autodl-tmp/lingbot-20261128-bootstrap/bin/tmux; fi
session=interactive-world
sock=dn-interactive
case "${1:-start}" in
    start)
        mkdir -p "$run/logs" "$run/state"
        exec 9>"$run/state/tmux-start.lock"
        flock 9
        if ! "$tmux_bin" -L "$sock" has-session -t "=$session" 2>/dev/null; then
            "$tmux_bin" -L "$sock" new-session -d -s "$session" -n coordinator -c "$project" \
                "exec /root/miniconda3/bin/python -u scripts/run_interactive_world.py serve >> '$run/logs/coordinator.log' 2>&1" 9>&-
        fi
        "$tmux_bin" -L "$sock" list-panes -t "$session" -F '#{session_name}/#{window_name} pid=#{pane_pid} dead=#{pane_dead}'
        ;;
    attach) exec "$tmux_bin" -L "$sock" attach-session -t "=$session" ;;
    status) "$tmux_bin" -L "$sock" list-windows -t "=$session"; exec /root/miniconda3/bin/python "$project/scripts/run_interactive_world.py" status ;;
    *) echo "Use start, status, or attach" >&2; exit 2 ;;
esac
