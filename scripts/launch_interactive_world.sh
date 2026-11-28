#!/usr/bin/env bash
set -eu
project="$(cd "$(dirname "$0")/.." && pwd -P)"
run="$project/.local/runs/interactive-world-20261128-v1"
tmux_bin="${IW_TMUX:-$(command -v tmux || true)}"
if [ -z "$tmux_bin" ]; then tmux_bin=/root/autodl-tmp/lingbot-20261128-bootstrap/bin/tmux; fi
session=interactive-world
sock=dn-interactive
window() {
    local name="$1" command="$2" existing
    existing=$("$tmux_bin" -L "$sock" list-panes -s -t "=$session" -F '#{window_name} #{pane_id} #{pane_dead}' | awk -v n="$name" '$1==n {print $2,$3;exit}')
    if [[ -z "$existing" ]]; then
        "$tmux_bin" -L "$sock" new-window -d -t "$session" -n "$name" -c "$project" "$command" 9>&-
    elif [[ "${existing##* }" == 1 ]]; then
        "$tmux_bin" -L "$sock" respawn-pane -t "${existing%% *}" -c "$project" "$command" 9>&-
    fi
}
case "${1:-start}" in
    start)
        mkdir -p "$run/logs" "$run/state"
        exec 9>"$run/state/tmux-start.lock"
        flock 9
        if ! "$tmux_bin" -L "$sock" has-session -t "=$session" 2>/dev/null; then
            "$tmux_bin" -L "$sock" new-session -d -s "$session" -n coordinator -c "$project" \
                "exec /root/miniconda3/bin/python -u scripts/run_interactive_world.py serve >> '$run/logs/coordinator.log' 2>&1" 9>&-
        fi
        if [[ -s "$run/a_depth/native/manifest.json" && -x "$project/.local/conda/lingbot-depth-bw/bin/python" &&
              ( ! -s "$run/a_depth/supervision/R3/manifest.json" || ! -s "$run/a_depth/vr_transfer/vr_train_only_adaptation.json" ) ]]; then
            if ! "$tmux_bin" -L iw-a has-session -t '=iw-a-coordinator' 2>/dev/null; then
                "$tmux_bin" -L iw-a new-session -d -s iw-a-coordinator -c "$project" 'exec bash scripts/interactive_depth_worker.sh all' 9>&-
            fi
        fi
        if [[ -s "$run/b_map/matching/weights/model.safetensors" && ! -s "$run/b_map/pose/accepted.json" ]]; then
            window pose 'exec bash scripts/world_pose_run.sh'
            if [[ -s "$run/b_map/pose/edges.json" && ! -s "$run/b_map/pose/matches_complete.json" ]]; then
                window pose-cpu "exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '$project/.local/conda/lingbot-map2-bw/bin/python' -u scripts/world_pose_cpu_pipeline.py >> '$run/logs/pose-cpu.log' 2>&1"
            fi
        fi
        if [[ -s "$run/e_interaction/assets/pusht/source.json" &&
             ( ! -s "$run/e_interaction/probe/manifest.json" || ! -s "$run/e_interaction/media/manifest.json" ) ]]; then
            window interaction 'exec bash scripts/interactive_push_run.sh'
        fi
        if [[ -d "$run/e_interaction/assets/rope/data" &&
             ( ! -s "$run/e_interaction/phystwin/single_push_rope_1/G-Q1_metrics.json" || ! -s "$run/e_interaction/phystwin/single_push_rope_4/G-Q1_metrics.json" ) ]]; then
            window rope 'exec bash scripts/interactive_phystwin_run.sh'
        fi
        window journal "exec /root/miniconda3/bin/python -u scripts/world_daily_log.py watch >> '$run/logs/journal.log' 2>&1"
        "$tmux_bin" -L "$sock" list-panes -s -t "$session" -F '#{session_name}/#{window_name} pid=#{pane_pid} dead=#{pane_dead}'
        ;;
    attach) exec "$tmux_bin" -L "$sock" attach-session -t "=$session" ;;
    status) "$tmux_bin" -L "$sock" list-windows -t "=$session"; exec /root/miniconda3/bin/python "$project/scripts/run_interactive_world.py" status ;;
    *) echo "Use start, status, or attach" >&2; exit 2 ;;
esac
