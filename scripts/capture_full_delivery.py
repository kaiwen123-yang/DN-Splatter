"""Publish the isolated full-run evidence and verify the pushed ref."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import re
import subprocess
import sys
import time

from capture_rectify import read, save
from capture_s01_delivery import TARGET, git, network
from capture_s01_inputs import P, REPORT, ensure
from vr_room_run import ROOT, STATE, now

SENSITIVE=re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|https?://[^\s/]+:[^\s/]+@|connect\.[\w.-]*seetacloud")


def deliver(mode):
    pointer=REPORT.parent/"latest.json"
    ensure(git("branch","--show-current")=="main" and git("remote","get-url","--push","origin")==TARGET,"repository target")
    if mode=="complete":
        ensure(read(pointer)["status"] in ("completed","completed_with_coverage_gaps"),"final report")
        for name in P["deliverable"]["required_files"]:
            ensure((REPORT/name).is_file(),name)
    elif mode=="blocked":
        REPORT.mkdir(parents=True,exist_ok=True)
        current=read(STATE/"current.json")
        safe=dict(stage=current.get("failed_stage"),reason=current.get("reason"),utc=now(),
                  resume="bash scripts/launch_capture_full.sh")
        if safe["stage"] and re.fullmatch(r"[a-z_]+",safe["stage"]):
            log=STATE.parent/"logs"/(safe["stage"]+".log")
            if log.is_file():
                with log.open("rb") as f:
                    f.seek(max(0,log.stat().st_size-4096))
                    lines=f.read().decode(errors="replace").splitlines()[-6:]
                safe["error_tail"]=[line[:400] for line in lines if not SENSITIVE.search(line) and "proxy" not in line.lower()]
        save(REPORT/"interruption.json",safe)
        data=read(pointer)
        if data["status"] in ("completed","completed_with_coverage_gaps") and safe["stage"]=="delivery":
            data.update(delivery_status="pending",observed_utc=safe["utc"])
        else:
            data.update(status="blocked",observed_utc=safe["utc"],note="See interruption.json for stage, error and same-run recovery.")
        save(pointer,data)
    public=[pointer,*sorted(p for p in REPORT.rglob("*") if p.is_file())]
    ensure(sum(p.stat().st_size for p in public)<50*1024**2,"report size")
    for p in public:
        ensure(not p.is_symlink() and p.suffix in {".md",".yml",".jpg",".json",".sh",".patch"} and p.stat().st_size<10*1024**2,"report member "+p.name)
    names=[str(p.relative_to(ROOT)) for p in public]
    ensure(set(git("diff","--cached","--name-only").splitlines())<=set(names),"unrelated staged changes")
    git("add","--",*names)
    git("diff","--cached","--check")
    diff=git("diff","--cached","--no-ext-diff")
    ensure(not SENSITIVE.search(diff),"unexpected sensitive text in evidence")
    if diff:
        git("commit","-m",("Publish measured capture-full results" if mode=="complete" else "Record capture-full interruption and recovery"),"--",*names)
    for attempt in range(3):
        try:
            network("fetch","origin","main")
            if subprocess.run(["git","merge-base","--is-ancestor","origin/main","HEAD"],cwd=ROOT).returncode:
                if subprocess.run(["git","merge-base","--is-ancestor","HEAD","origin/main"],cwd=ROOT).returncode==0:
                    git("merge","--ff-only","origin/main")
                else:
                    ensure(not git("status","--porcelain"),"new worktree changes during delivery")
                    try:
                        git("rebase","origin/main")
                    except subprocess.CalledProcessError:
                        git("rebase","--abort")
                        raise
            network("push","origin","HEAD:refs/heads/main")
            head=git("rev-parse","HEAD")
            ensure(network("ls-remote","origin","refs/heads/main").split()[0]==head,"remote ref did not match")
            save(STATE/("github_"+mode+".json"),dict(commit=head,verified_utc=now(),
                 verified_at_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat()))
            return
        except subprocess.CalledProcessError:
            if attempt==2:
                raise
            time.sleep(6*(attempt+1))


if __name__=="__main__":
    deliver(sys.argv[1])
