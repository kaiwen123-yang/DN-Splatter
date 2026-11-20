"""Windows-side, read-only, resumable retrieval of the verified B1 video and reports."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TRIAL = json.loads((ROOT/"configs/capture_metric3d_protocol.json").read_text(encoding="utf-8"))["trial"]
PATHS = json.loads((ROOT/"configs/capture_full_test_protocol.json").read_text(encoding="utf-8"))["paths"]
DEST = ROOT / Path(PATHS["local_transfer"]).parent / "metric3d-20261120-v1"
RUN = PurePosixPath(TRIAL["run_directory"])
REPORT = PurePosixPath(TRIAL["report_directory"])


def check(ok, message):
    if not ok:
        raise ValueError(message)


def write_state(**fields):
    path=DEST/"receiver.json"
    temp=path.with_suffix(".pending")
    temp.write_text(json.dumps(dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**fields),
                               indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    temp.replace(path)


@contextmanager
def locked():
    with (DEST/"receiver.lock").open("a+b") as lock:
        if os.name=="nt":
            import msvcrt
            if not lock.seek(0,2):
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def git(*args, binary=False):
    raw=subprocess.check_output(["git",*map(str,args)],cwd=ROOT,stderr=subprocess.DEVNULL)
    return raw if binary else raw.decode("utf-8").strip()


def file_hash(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024**2),b""):
            h.update(block)
    return h.hexdigest()


def fetch_once(ssh):
    server=shlex.quote(PATHS["server_project_root"])
    marker=shlex.quote(str(RUN/"state/delivery_ready.json"))
    cmd=f"cd {server} && if test -s {marker}; then cat {marker}; fi"
    result=subprocess.check_output([*ssh,cmd],timeout=35,stderr=subprocess.DEVNULL)
    if not result.strip():
        write_state(status="waiting_for_verified_delivery",run_id=TRIAL["run_id"])
        return False
    receipt=json.loads(result)
    commit=receipt["commit"]
    check(receipt["run_id"]==TRIAL["run_id"] and receipt["report_dir"]==str(REPORT) and
          re.fullmatch("[a-f0-9]{40}",commit), "Invalid server delivery identity")
    check(re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)kaiwen123-yang/dn-splatter(?:\.git)?/?",
                       git("remote","get-url","origin"), re.I), "Unexpected Git destination")
    git("fetch","--quiet","origin","main")
    github=git("rev-parse","refs/remotes/origin/main")
    git("merge-base","--is-ancestor",commit,github)
    subprocess.run([*ssh,f"cd {server} && git merge-base --is-ancestor {commit} HEAD"],
                   check=True,timeout=35,stderr=subprocess.DEVNULL)
    paths=[str(REPORT),str(REPORT.parent/"latest.json")]
    archive=git("archive","--format=zip",commit,"--",*paths,binary=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for entry in z.infolist():
            name=PurePosixPath(entry.filename)
            if entry.is_dir():
                continue
            check(not name.is_absolute() and ".." not in name.parts and
                  (name.is_relative_to(REPORT) or name==REPORT.parent/"latest.json"), "Unexpected Git archive member")
            target=DEST/"reports"/name.relative_to(REPORT.parent)
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_suffix(target.suffix+".partial")
            tmp.write_bytes(z.read(entry))
            tmp.replace(target)
    if git("branch","--show-current")=="main":
        # Git rejects conflicting local edits and divergent branches. The archived report remains usable.
        subprocess.run(["git","merge","--ff-only",github],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    local=git("rev-parse","HEAD")
    identity=dict(commit=commit,github_main=github,local_head=local,verified_server_ancestry=True)
    if receipt["status"]=="blocked":
        write_state(status="server_reported_blocked",reason=receipt.get("reason"),**identity)
        return True
    check(receipt["status"].startswith("completed"), "Unexpected terminal state")
    info=receipt["video"]
    remote=PurePosixPath(info["path"])
    check(not remote.is_absolute() and ".." not in remote.parts and remote.is_relative_to(RUN) and
          remote.suffix.lower()==".mp4" and re.fullmatch("[0-9a-f]{64}",info["sha256"]) and
          isinstance(info["bytes"],int) and info["bytes"]>0,"Invalid video manifest")
    public=json.loads((DEST/"reports"/REPORT.name/"video_manifest.json").read_text(encoding="utf-8"))
    check(public["sha256"]==info["sha256"] and public["bytes"]==info["bytes"] and public["frames"]==1846,
          "Video differs from the published 1846-position manifest")
    movie=DEST/remote.name
    part=movie.with_suffix(".mp4.part")
    def valid(p):
        return p.is_file() and p.stat().st_size==info["bytes"] and file_hash(p)==info["sha256"]
    if not valid(movie):
        if movie.exists():
            movie.rename(movie.with_name(movie.name+f".previous.{time.time_ns()}"))
        if part.exists() and part.stat().st_size>=info["bytes"] and not valid(part):
            part.rename(part.with_name(part.name+f".rejected.{time.time_ns()}"))
        first=part.stat().st_size if part.exists() else 0
        if first<info["bytes"]:
            write_state(status="receiving_video",resumed_bytes=first,expected_bytes=info["bytes"],**identity)
            with part.open("ab") as sink:
                subprocess.run([*ssh,f"cd {server} && tail -c +{first+1} -- {shlex.quote(str(remote))}"],
                               stdout=sink,stderr=subprocess.DEVNULL,check=True)
        check(valid(part),"Downloaded video digest mismatch")
        part.replace(movie)
    write_state(status="received",path=str(movie),sha256=info["sha256"],bytes=info["bytes"],**identity)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    DEST.mkdir(exist_ok=True,parents=True)
    alias=(ROOT/".local/transfer/full-host").read_text(encoding="utf-8-sig").strip()
    check(re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]*",alias),"Invalid saved project connection")
    ssh=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=12","-o","ServerAliveInterval=20",alias]
    with locked():
        while True:
            try:
                if fetch_once(ssh) or args.once:
                    return
            except (subprocess.SubprocessError,OSError,ValueError,KeyError,zipfile.BadZipFile) as exc:
                # Do not record command arguments, connection details or authentication output.
                write_state(status="retrying",error_type=type(exc).__name__)
                if args.once:
                    raise SystemExit(1) from None
            time.sleep(60)


if __name__=="__main__":
    main()
