"""Receive the B2 delivery into a sibling of the existing B1 capture-full results."""
import argparse
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

from capture_archive_transport import ROOT, SERVER, atomic, check, digest, safe_rel, ssh_args, utc
from capture_archive_windows import extended_path


def document(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


PLAN = document(ROOT / "configs/capture_quality_protocol.json")
ID = PLAN["run_id"]
RUN = PurePosixPath(".local/runs") / ID
REPORT = PurePosixPath("reports/capture-full-quality-v2")
OLD = document(ROOT / "configs/capture_full_test_protocol.json")["paths"]["local_transfer"]
DEST = ROOT / Path(OLD).parent / ID
STATE = DEST / "receiver.json"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)


def lock_receiver(handle):
    if os.name == "nt":
        import msvcrt
        if not handle.seek(0, 2):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def fetch_file(row, ssh):
    remote = PurePosixPath(safe_rel(row["path"]))
    check(remote.is_relative_to(RUN) and remote != RUN, "Artifact outside B2 run")
    size, key = row["bytes"], row["sha256"]
    check(isinstance(size, int) and size >= 0 and re.fullmatch("[0-9a-f]{64}", key), "Artifact identity")
    dest = DEST / "artifacts" / remote.relative_to(RUN)
    check(dest.resolve().is_relative_to((DEST / "artifacts").resolve()), "Local artifact path")
    dest = extended_path(dest)
    if dest.exists():
        check(dest.stat().st_size == size and digest(dest) == key, "Previously received artifact differs")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + "." + key[:12] + ".partial")
    if part.exists() and part.stat().st_size > size:
        raise ValueError("Partial artifact exceeds declared size")
    offset = part.stat().st_size if part.exists() else 0
    atomic(STATE, dict(status="receiving", run_id=ID, utc=utc(), path=str(remote), offset=offset, bytes=size))
    with part.open("ab") as out:
        if offset < size:
            src = shlex.quote(SERVER + "/" + str(remote))
            subprocess.run([*ssh, f"tail -c +{offset + 1} -- {src}"], check=True,
                           stdout=out, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    if part.stat().st_size == size and digest(part) != key:
        # Only this receiver's invalid temporary file is discarded.
        part.unlink()
        raise OSError("SHA256 mismatch during receipt")
    check(part.stat().st_size == size, "Incomplete receipt")
    part.replace(dest)


def receive():
    ssh = ssh_args()
    marker = shlex.quote(SERVER + "/" + str(RUN / "state/delivery_ready.json"))
    data = subprocess.check_output([*ssh, f"if test -s {marker}; then cat {marker}; fi"],
                                   timeout=40, stderr=subprocess.DEVNULL)
    if not data.strip():
        pending=shlex.quote(SERVER+"/"+str(RUN/"state/delivery_candidate.json"))
        raw=subprocess.check_output([*ssh,f"if test -s {pending}; then cat {pending}; fi"],
                                    timeout=40,stderr=subprocess.DEVNULL)
        if raw.strip():
            publish_candidate(json.loads(raw))
            return False
        current=shlex.quote(SERVER+"/"+str(RUN/"state/current.json"))
        progress=shlex.quote(SERVER+"/"+str(RUN/"state/training_progress.json"))
        items=subprocess.check_output([*ssh,f"for f in {current} {progress}; do if test -s \"$f\"; then cat \"$f\"; else echo '{{}}'; fi; echo '---'; done"],
                                      timeout=40,stderr=subprocess.DEVNULL).split(b"---")
        stage,training_next=[json.loads(i) for i in items[:2]]
        atomic(STATE, dict(status="server_stage_failed" if stage.get("stage")=="failed" else "waiting_for_server_delivery",
                           run_id=ID,pid=os.getpid(),utc=utc(),server=stage,training=training_next))
        return False
    receipt = json.loads(data)
    commit = receipt["commit"]
    check(receipt["run_id"] == ID and re.fullmatch("[0-9a-f]{40}", commit), "B2 delivery identity")
    check(re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)kaiwen123-yang/DN-Splatter(?:\.git)?/?",
                       git("remote", "get-url", "origin").decode().strip(), re.I), "Unexpected remote")
    git("fetch", "-q", "origin", "main")
    git("merge-base", "--is-ancestor", commit, "origin/main")
    git("merge-base", "--is-ancestor", "7359479b407be1b3b8bacd20f9635a3710544a90", commit)
    subprocess.run([*ssh, f"cd {shlex.quote(SERVER)} && git merge-base --is-ancestor {commit} HEAD"],
                   check=True, timeout=35, stderr=subprocess.DEVNULL)
    status = receipt["status"]
    check(status in ("blocked", "completed"), "Unknown B2 delivery state")
    rows = receipt.get("artifacts", [])
    check(len({r["path"] for r in rows}) == len(rows), "Duplicate artifact paths")
    if status == "completed":
        extensions = [PurePosixPath(r["path"]).suffix.lower() for r in rows]
        check(".mp4" in extensions and extensions.count(".ply") >= 3 and ".ckpt" in extensions,
              "Expected video, three PLY surfaces and checkpoint")
    with zipfile.ZipFile(io.BytesIO(git("archive", "--format=zip", commit, str(REPORT)))) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = PurePosixPath(safe_rel(entry.filename)).relative_to(REPORT)
            target = DEST / "reports" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = archive.read(entry)
            if target.exists():
                check(hashlib.sha256(content).hexdigest() == digest(target), "Report already exists with different content")
            else:
                tmp = target.with_name(target.name + ".part")
                tmp.write_bytes(content)
                tmp.replace(target)
    for row in rows:
        fetch_file(row, ssh)
    if git("branch", "--show-current").strip() == b"main":
        subprocess.run(["git", "merge", "--ff-only", "origin/main"], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atomic(DEST / "server_receipt.json", receipt)
    atomic(STATE, dict(status="received" if status == "completed" else "server_reported_blocked",
                       run_id=ID, commit=commit, utc=utc(), pid=os.getpid(), reason=receipt.get("reason"),
                       verified_files=len(rows), verified_bytes=sum(r["bytes"] for r in rows),
                       local_head=git("rev-parse", "HEAD").decode().strip(),
                       github_main=git("rev-parse", "origin/main").decode().strip()))
    return True


def publish_candidate(receipt):
    """Use the existing local GitHub credential only for an audited, fast-forward report commit."""
    commit=receipt["commit"]
    check(receipt["run_id"]==ID and re.fullmatch("[0-9a-f]{40}",commit),"candidate identity")
    host=(ROOT/".local/transfer/full-host").read_text(encoding="utf-8-sig").strip()
    check(re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]*",host),"invalid project alias")
    check(git("remote","get-url","origin").decode().strip().removesuffix(".git").lower() in
          ("https://github.com/kaiwen123-yang/dn-splatter","git@github.com:kaiwen123-yang/dn-splatter"),
          "unexpected publication destination")
    git("fetch","-q","origin","main")
    git("fetch","-q",host+":"+SERVER,"refs/heads/main")
    git("merge-base","--is-ancestor",commit,"FETCH_HEAD")
    if subprocess.run(["git","merge-base","--is-ancestor",commit,"origin/main"],cwd=ROOT).returncode==0: return
    git("merge-base","--is-ancestor","origin/main",commit)
    paths=git("diff","--name-only","-z","origin/main",commit).split(b"\0")
    for raw in filter(None,paths):
        p=PurePosixPath(safe_rel(raw.decode()))
        check(p.is_relative_to(REPORT),"candidate contains changes outside B2 report")
        check(p.suffix.lower() in (".md",".json",".gz",".jsonl",".jpg",".yml",".diff"),"unexpected report format")
        obj=git("show",commit+":"+p.as_posix())
        check(len(obj)<25*1024**2 and not re.search(rb"BEGIN [A-Z ]*PRIVATE KEY|Bearer [A-Za-z0-9._-]{20,}|sk-[A-Za-z0-9_-]{30,}",obj),
              "candidate file review")
    latest=json.loads(git("show",commit+":"+str(REPORT/"latest.json")))
    check(latest["run_id"]==ID and latest["status"]=="completed" and latest["updates"]==240000,"completed report identity")
    git("push","origin",commit+":refs/heads/main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / "receiver.lock").open("a+b") as handle:
        lock_receiver(handle)
        while True:
            try:
                if receive() or args.once:
                    break
            except (OSError, ValueError, KeyError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
                atomic(STATE, dict(status="retrying", run_id=ID, utc=utc(), pid=os.getpid(),
                                   error_type=type(exc).__name__,
                                   reason=str(exc) if isinstance(exc, ValueError) else None))
                if args.once:
                    raise SystemExit(1) from None
            time.sleep(60)
