"""Receive B3 and verify referenced B2 bytes in place. Run on the existing Windows workstation."""
import argparse
import base64
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
from capture_quality_receive import document, fetch_file, git, lock_receiver

PLAN = document(ROOT / "configs/capture_surface_quality_protocol.json")
ID = PLAN["run_id"]
OLD_ID = PLAN["baseline"]["run_id"]
RUN = PurePosixPath(".local/runs", ID)
OLD_RUN = RUN.parent / OLD_ID
REPORT = "reports/capture-full-quality-v3/20261122-v1"
PARENT = (ROOT / document(ROOT / "configs/capture_full_test_protocol.json")["paths"]["local_transfer"]).parent
DEST = PARENT / ID
STATE = DEST / "receiver.json"


def metadata():
    # One SSH read. A single copy of the bytes serves both the digest and payload.
    script = """import base64, hashlib, json, pathlib, sys
r = pathlib.Path(sys.argv[1])
out = {}
for name in ('state/delivery_ready.json','state/B2_local_surfaces.json','state/native_pipeline.json','state/current.json',
             'state/A3_progress.json','state/A3_fusion_progress.json','state/training_progress.json','state/training_decision.json',
             'export/B2/fusion_model/complete.json','export/B2/fusion_sensor/complete.json',
             'export/B2/tolerances.json','export/B2/tiles/complete.json','export/B2/identity.json'):
 p = r / name
 if p.is_file():
  data = p.read_bytes()
  out[name] = [hashlib.sha256(data).hexdigest(), base64.b64encode(data).decode()]
print(json.dumps(out))
"""
    args = [SERVER + "/.local/conda/dn-splatter-bw/bin/python", "-c", script, SERVER + "/" + str(RUN)]
    raw = subprocess.check_output([*ssh_args(), shlex.join(args)], timeout=55, stderr=subprocess.DEVNULL)
    out, hashes = {}, {}
    for name, (key, encoded) in json.loads(raw).items():
        b = base64.b64decode(encoded, validate=True)
        check(hashlib.sha256(b).hexdigest() == key, "Remote evidence checksum")
        local = DEST / "server_state" / safe_rel(name)
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name+f".{os.getpid()}.part")
        tmp.write_bytes(b)
        check(digest(tmp) == key, "Local evidence checksum")
        os.replace(tmp, local)
        out[name], hashes[name] = json.loads(b), key
    return out, hashes


def receive():
    meta, hashes = metadata()
    ready = meta.get("state/delivery_ready.json")
    if ready is None:
        atomic(STATE, dict(run_id=ID, pid=os.getpid(), utc=utc(), status="waiting_for_delivery",
              metadata_sha256=hashes, submaps=meta.get("state/B2_local_surfaces.json"),
              server=meta.get("state/current.json"), fusion=meta.get("state/A3_fusion_progress.json"),
              global_meshes={k:meta.get(f"export/B2/fusion_{k}/complete.json") for k in ("model","sensor")},
              training_decision=meta.get("state/training_decision.json"), training=meta.get("state/training_progress.json")))
        return False
    commit = ready["commit"]
    check(ready["run_id"] == ID and ready["schema_version"] == 1 and re.fullmatch("[a-f0-9]{40}", commit),
          "B3 delivery identity")
    remote = git("remote", "get-url", "origin").decode().strip()
    check(re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)kaiwen123-yang/DN-Splatter(?:\.git)?/?",
                       remote, re.I), "Remote repository identity")
    git("fetch", "-q", "origin", "main")
    git("merge-base", "--is-ancestor", "11086c1", commit)
    git("merge-base", "--is-ancestor", commit, "origin/main")
    subprocess.run([*ssh_args(), f"cd {shlex.quote(SERVER)} && git merge-base --is-ancestor {commit} HEAD"],
                   check=True, timeout=40, stderr=subprocess.DEVNULL)
    check(ready["status"] in ("completed", "blocked"), "B3 server state")
    rows, refs = ready["artifacts"], ready.get("references", [])
    paths = [r["path"] for r in rows+refs]
    check(len(set(paths)) == len(paths), "Duplicate B3 receipt path")
    if ready["status"] == "completed":
        published = json.loads(git("show", f"{commit}:{REPORT}/artifacts.json"))
        for key, actual in (("artifacts", rows), ("references", refs)):
            check(published[key] == actual, "Git report/source disagreement: " + key)
        updates = ready["training_updates"]
        check(updates in (0, 240000), "Unexpected training budget")
        suffix = [PurePosixPath(r["path"]).suffix for r in rows]
        check(suffix.count(".ply") >= 2 and ".mp4" in suffix, "Missing B3 surface/video deliverables")
        if updates:
            steps = {str(ready["selected_step"]).zfill(9), "000239999"}
            names = {PurePosixPath(r["path"]).name for r in rows}
            check(all(f"step-{s}.ckpt" in names for s in steps), "Missing B3 checkpoint")
        else:
            check(any(r["path"].endswith(".mp4") for r in refs), "Missing reused B2 RGB timeline")
        check(any(r["sha256"] == PLAN["baseline"]["selected_checkpoint_sha256"] for r in refs),
              "Missing frozen 210k checkpoint reference")
    old = {r["path"]: r for r in document(PARENT / OLD_ID / "server_receipt.json")["artifacts"]}
    for row in refs:
        path = PurePosixPath(safe_rel(row["path"]))
        check(path.is_relative_to(OLD_RUN) and all(old[str(path)][k] == row[k] for k in ("bytes", "sha256")),
              "Unrecognized B2 reference")
        local = extended_path(PARENT / OLD_ID / "artifacts" / path.relative_to(OLD_RUN))
        check(local.stat().st_size == row["bytes"] and digest(local) == row["sha256"], "B2 reference readback")
    with zipfile.ZipFile(io.BytesIO(git("archive", "--format=zip", commit, REPORT))) as reports:
        for entry in reports.infolist():
            if entry.is_dir():
                continue
            rel = PurePosixPath(safe_rel(entry.filename)).relative_to(REPORT)
            target = DEST / "reports" / commit / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            content = reports.read(entry)
            target.write_bytes(content)
            check(digest(target) == hashlib.sha256(content).hexdigest(), "Report readback")
    for row in rows:
        fetch_file(row, ssh_args(), run=RUN, destination=DEST, state=STATE, run_id=ID)
    atomic(DEST / "server_receipt.json", ready)
    if git("branch", "--show-current").strip() == b"main":
        git("merge", "--ff-only", "origin/main")
    atomic(STATE, dict(run_id=ID, utc=utc(), status="received" if ready["status"]=="completed" else "server_blocked",
          commit=commit, artifacts=len(rows), verified_new_bytes=sum(r["bytes"] for r in rows),
          references=len(refs), verified_existing_bytes=sum(r["bytes"] for r in refs), metadata_sha256=hashes))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    opts = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / "receiver.lock").open("a+b") as lock:
        lock_receiver(lock)
        while True:
            try:
                if receive() or opts.once:
                    break
            except (OSError, ValueError, KeyError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
                atomic(STATE, dict(run_id=ID, utc=utc(), pid=os.getpid(), status="retrying",
                                   error_type=type(exc).__name__, reason=str(exc) if isinstance(exc, ValueError) else None))
                if opts.once:
                    raise
            time.sleep(60)
