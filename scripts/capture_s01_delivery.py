"""Publish only this capture's source and small evidence, then verify the remote ref."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from capture_rectify import read, save
from capture_s01_inputs import P, REPORT, ensure
from vr_room_run import ROOT, RUN, STATE, latest_checkpoint, now

TARGET = "https://github.com/kaiwen123-yang/DN-Splatter.git"


def git(*args, **kwargs):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, **kwargs).strip()


def network(*args):
    command = [
        "git",
        "-c",
        "credential.https://github.com.helper=",
        "-c",
        "credential.https://github.com.helper=!gh auth git-credential",
        *args,
    ]
    shell = 'if [ -r /etc/network_turbo ]; then (source /etc/network_turbo >/dev/null 2>&1 && "$@") && exit 0; fi; exec "$@"'
    return subprocess.check_output(
        ["bash", "-c", shell, "git-transport", *command],
        cwd=ROOT,
        env=dict(os.environ, GIT_TERMINAL_PROMPT="0"),
        text=True,
    ).strip()


def update_progress(mode):
    status = read(STATE / "current.json")
    if mode == "complete":
        ensure(
            status["stage"] == "complete"
            and read(REPORT.parent / "latest.json")["status"] == "completed",
            "completion gate",
        )
        for name in P["deliverable"]["required_files"]:
            ensure((REPORT / name).is_file(), name)
        return
    ckpt, step, gaussians = latest_checkpoint()
    pointer = read(REPORT.parent / "latest.json")
    pointer.update(
        status=mode,
        normal_priors_generated=(REPORT / "normal_prior.json").is_file(),
        training_started=(STATE / "training_segments.jsonl").is_file(),
        cumulative_updates=step + 1,
        observed_checkpoint=str(ckpt) if ckpt else None,
        observed_utc=now(),
        run_directory=str(RUN),
        note="Progress snapshot at the latest fully readable checkpoint.",
    )
    save(REPORT.parent / "latest.json", pointer)
    info = dict(
        observed_utc=now(),
        stage=status["stage"],
        saved_step=step,
        updates_in_saved_checkpoint=step + 1,
        gaussians_in_saved_checkpoint=gaussians,
        checkpoint=str(ckpt) if ckpt else None,
        tmux="dn-capture-s01-full",
        continuation="bash scripts/launch_capture_s01.sh",
        logs=str(RUN / "logs"),
    )
    if mode == "blocked":
        info["reason"] = status.get("reason", "See stage logs")
        info["stage_records"] = [
            {k: r[k] for k in ("stage", "exit_code", "duration_seconds")}
            for r in map(json.loads, (STATE / "stages.jsonl").read_text().splitlines())
        ]
        bad = next((row for row in reversed(info["stage_records"]) if row["exit_code"]), None)
        if bad:
            lines = (RUN / "logs" / (bad["stage"] + ".log")).read_text(errors="replace").splitlines()
            info["last_error_lines"] = [line[:500] for line in lines[-6:]]
    save(REPORT / ("handoff_" + mode + ".json"), info)


def deliver(mode):
    ensure(
        git("branch", "--show-current") == "main"
        and git("remote", "get-url", "--push", "origin") == TARGET,
        "git target",
    )
    update_progress(mode)
    scope = [
        ROOT / "scripts/autodl_env.sh",
        ROOT / "scripts/launch_capture_s01.sh",
        ROOT / "scripts/finish_capture_s01.sh",
        *sorted((ROOT / "scripts").glob("capture_s01_*.py")),
        REPORT.parent / "latest.json",
        *sorted(p for p in REPORT.rglob("*") if p.is_file()),
    ]
    names = [str(p.relative_to(ROOT)) for p in scope]
    ensure(
        set(git("diff", "--cached", "--name-only").splitlines()) <= set(names),
        "unrelated staged changes",
    )
    ensure(sum(p.stat().st_size for p in scope) < 50 * 1024**2, "public size limit")
    for path in scope:
        ensure(
            not path.is_symlink()
            and path.suffix in {".py", ".sh", ".json", ".md", ".yml", ".jpg"}
            and path.stat().st_size < 8 * 1024**2,
            "invalid public file " + path.name,
        )
    git("add", "--", *names)
    git("diff", "--cached", "--check")
    diff = git("diff", "--cached", "--no-ext-diff")
    sensitive = r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|https?://[^\s/]+:[^\s/]+@|connect\.[\w.-]*seetacloud"
    ensure(not re.search(sensitive, diff), "staged secret scan")
    if diff:
        git(
            "commit",
            "-m",
            {
                "running": "Run capture-s01 baseline from verified Omnidata inputs",
                "complete": "Publish final capture-s01 30k evaluation and raw Poisson evidence",
                "blocked": "Record capture-s01 runtime failure and recovery evidence",
            }[mode],
            "--",
            *names
        )
    for attempt in range(3):
        try:
            network("fetch", "origin", "main")
            git("merge", "--ff-only", "origin/main")
            network("push", "origin", "HEAD:refs/heads/main")
            commit = git("rev-parse", "HEAD")
            ensure(
                network("ls-remote", "origin", "refs/heads/main").split()[0] == commit,
                "remote verification",
            )
            save(
                STATE / ("github_" + mode + ".json"), dict(mode=mode, commit=commit, verified_utc=now())
            )
            print(commit, flush=True)
            return
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(6 * (attempt + 1))


if __name__ == "__main__":
    deliver(sys.argv[1])
