"""Publish the scoped B1 evidence and seal the remote commit for the Windows receiver."""
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import time

from capture_metric3d_common import P, REPORT, ROOT, RUN, STATE, demand, member, read, write
from capture_s01_delivery import TARGET, git, network
from capture_full_delivery import SENSITIVE


def push():
    for retry in range(3):
        try:
            network("fetch", "origin", "main")
            if subprocess.run(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=ROOT).returncode:
                if not subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"], cwd=ROOT).returncode:
                    git("merge", "--ff-only", "origin/main")
                else:
                    demand(not git("status", "--porcelain"), "concurrent local edits during publication")
                    try:
                        git("rebase", "origin/main")
                    except subprocess.CalledProcessError:
                        git("rebase", "--abort")
                        raise
            network("push", "origin", "HEAD:refs/heads/main")
            commit = git("rev-parse", "HEAD")
            demand(network("ls-remote", "origin", "refs/heads/main").split()[0] == commit, "published remote main")
            return commit
        except subprocess.CalledProcessError:
            if retry == 2:
                raise
            time.sleep(5 * (retry + 1))


def deliver(mode):
    demand(git("branch", "--show-current") == "main" and git("remote", "get-url", "--push", "origin") == TARGET,
           "authorized GitHub destination")
    pointer = REPORT.parent / "latest.json"
    latest = read(pointer)
    utc = datetime.now(timezone.utc).isoformat()
    current = read(STATE / "current.json")
    latest.update(working_interpretation="User confirmed Metric3D v2 ViT-Large", server_accessed_this_run=True,
                  observed_utc=utc, run_id=RUN.name, recovery_command="bash scripts/launch_capture_metric3d.sh")
    marker = dict(run_id=RUN.name, report_dir=REPORT.relative_to(ROOT).as_posix())
    if mode == "complete":
        video = read(REPORT / "video_manifest.json")
        demand(latest["status"] == "completed" and latest["actual_updates"] == 30000 and
               video["frames"] == 1846 and (REPORT/"surface_summary.json").is_file(), "final B1 evidence")
        marker["video"] = member(Path(video["path"]))
        demand(marker["video"]["sha256"] == video["sha256"], "published video identity")
    elif mode == "blocked":
        reason = current.get("reason", "Stage failed")
        stage = current.get("failed_stage", current["stage"])
        detail = []
        if re.fullmatch(r"[a-zA-Z0-9_]+", stage):
            path = RUN/"logs"/(stage+".log")
            if path.exists():
                with path.open("rb") as f:
                    f.seek(max(0, path.stat().st_size-3500))
                    detail = [s.strip()[:300] for s in f.read().decode(errors="replace").splitlines()[-5:]
                              if not SENSITIVE.search(s) and "proxy" not in s.lower() and "http" not in s.lower()]
        write(REPORT/"interruption.json", dict(stage=stage, reason=reason, tail=detail, utc=utc,
              recovery="bash scripts/launch_capture_metric3d.sh"))
        latest.update(status="blocked", observed_stage=stage, note=reason)
        marker["reason"] = reason
    else:
        demand(mode == "running", mode)
        latest.update(status="running", observed_stage=current["stage"], note="实测运行中，终点指标待正式产物。")
    for key in ("prior_progress.json",):
        if (STATE/key).is_file():
            latest["actual_inferred_frames"] = read(STATE/key)["completed"]
    if (REPORT/"normal_prior.json").exists():
        prior = read(REPORT/"normal_prior.json")
        latest.update(actual_inferred_frames=prior["completed"], actual_weights_sha256=prior["weight"]["sha256"])
    write(pointer, latest)
    files = [p for p in REPORT.rglob("*") if p.is_file()] + [pointer]
    for p in files:
        demand(not p.is_symlink() and p.suffix.lower() in (".json", ".md", ".csv", ".jpg", ".png", ".gz", ".yml", ".patch") and
               p.stat().st_size < 25*1024**2, "publication member " + p.name)
    names = [str(p.relative_to(ROOT)) for p in files]
    demand(set(git("diff", "--cached", "--name-only").splitlines()) <= set(names), "unrelated staged changes")
    git("add", "--", *names)
    git("diff", "--cached", "--check")
    diff = git("diff", "--cached", "--no-ext-diff")
    demand(not SENSITIVE.search(diff), "sensitive staged content")
    if diff:
        git("commit", "-m", {"running":"Record active ViT-Large full-capture comparison",
            "blocked":"Record Metric3D comparison interruption and recovery",
            "complete":"Publish measured full-capture Metric3D comparison and surface coverage"}[mode], "--", *names)
    commit = push()
    receipt = dict(**marker, commit=commit, status=latest["status"], verified_utc=utc)
    write(STATE/("github_running.json" if mode=="running" else "delivery_ready.json"), receipt)


if __name__ == "__main__":
    import sys
    deliver(sys.argv[1])
