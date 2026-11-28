"""Restore only protocol assets from the sealed 2026-11-25 archive to the new run.

The sender reads verified source chunks in place. The Linux receiver publishes a
file only after its complete SHA-256 matches. Its chunk store is restartable.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import zlib


ID = "interactive-world-20261128-v1"
ROOT = Path(__file__).resolve().parents[1]
RESTORE = ROOT / ".local/runs" / ID / "restore"
B3 = ".local/runs/capture-full-quality-v3-20261122-v1/"
B2 = ".local/runs/capture-full-quality-v2-20261121-v1/"
VR = ".local/data/room_datasets/vr_room/"
ENV = ".local/conda/dn-splatter-bw/"
UP = tuple(".local/upstream/" + n + "/" for n in ("dn-splatter", "gsplat", "pytorch3d"))
GROUPS = ("env", "hall", "vr", "frozen", "metric")


def selected(group, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Non-relative archive member")
    if group == "metric":
        return name == ".local/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
    if group == "env":
        if any(p in {".git-credentials", ".netrc", ".ssh"} for p in path.parts):
            return False
        if name.endswith("/.git/config") or ("/.git/" in name and path.name == "config"):
            return False
        return (name + "/").startswith((ENV, *UP))
    if group == "hall":
        return (name.startswith(B3 + "inputs/") or
                (name.startswith(B3 + "weights/B2/frame_") and name.endswith(".npy")) or
                name in {B3+"state/trainer/refinement.jsonl", B3+"geometry/tracks.npz", B3+"geometry/added_tracks.npz"})
    if group == "vr":
        if not name.startswith(VR):
            return False
        rel = name[len(VR):]
        if rel in {"gt_mesh.ply", "gt_pd.ply", "icp_iphone.json", "iphone/long_capture/iphone_pointcloud.ply",
                   "iphone/long_capture/test.txt"}:
            return True
        if not rel.startswith(("iphone/long_capture/", "iphone/short_capture/")):
            return False
        return ("/images/" in rel or "/depth/" in rel or "/reference_depth/" in rel or
                ("/normals_from_pretrain/" in rel and rel.endswith(".png")) or
                path.name in {"transformations.json", "transformations_colmap.json", "test.txt"})
    checkpoints = (B2+"training/capture-full-b2/dn-splatter/quality-v2/",
                   B3+"training/capture-full-b3/dn-splatter/surface-v3/")
    return any(name in {p+"config.yml", p+"nerfstudio_models/step-000209999.ckpt"} for p in checkpoints)


def hash_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(data)
    os.replace(f.name, path)


def read_manifest(group):
    path = RESTORE / (group+".manifest.json.gz")
    with gzip.open(path, "rt", encoding="utf8") as f:
        data = json.load(f)
    if data["group"] != group or any(not selected(group, row["p"]) for row in data["rows"]):
        raise ValueError("Scope mismatch")
    return data


def dest_root(group):
    return ROOT if group == "env" else RESTORE / "tree"


def blob(key):
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("Invalid SHA-256")
    return RESTORE / "blobs" / key[:2] / key


def recovered_hall_source(row):
    if not row["p"].startswith(B3+"inputs/"):
        return None
    rel = row["p"][len(B3+"inputs/"):]
    path = ROOT / ".local/runs/lingbot-depth-map-20261128-v1/source/b2" / rel
    if path.is_file() and path.stat().st_size == row["s"] and hash_file(path) == row["h"]:
        return path
    return None


def receive_plan(group):
    RESTORE.mkdir(parents=True, exist_ok=True)
    manifest = RESTORE / (group+".manifest.json.gz")
    encoded = sys.stdin.buffer.read()
    if manifest.exists() and manifest.read_bytes() != encoded:
        raise ValueError("A different restore manifest is already present")
    atomic_bytes(manifest, encoded)
    data = read_manifest(group)
    known = {}
    needed = set()
    for row in data["rows"]:
        if "h" not in row:
            continue
        target = dest_root(group) / row["p"]
        if target.exists():
            if target.is_symlink() or target.stat().st_size != row["s"] or hash_file(target) != row["h"]:
                raise FileExistsError(target)
            known[row["h"]] = str(target)
        elif group == "hall":
            other = recovered_hall_source(row)
            if other:
                known[row["h"]] = str(other)
    for row in data["rows"]:
        if "h" in row and row["s"] and row["h"] not in known:
            needed.update(row.get("parts") or [row["h"]])
    missing = sorted(h for h in needed if not blob(h).is_file() or hash_file(blob(h)) != h)
    atomic_bytes(RESTORE / (group+".reuse.json"), json.dumps(known).encode())
    print(json.dumps(missing), flush=True)


def receive_chunks():
    inp = sys.stdin.buffer
    for line in inp:
        key, size, wire = json.loads(line)
        target = blob(key)
        if not 0 <= size <= 32*1024**2 or not 0 < wire <= size+131072:
            raise ValueError("Invalid transfer dimensions")
        data = zlib.decompress(inp.read(wire))
        if len(data) != size or hashlib.sha256(data).hexdigest() != key:
            raise IOError("Chunk verification failed: " + key)
        if not target.exists() or hash_file(target) != key:
            atomic_bytes(target, data)


def finish(group):
    data = read_manifest(group)
    known = json.loads((RESTORE/(group+".reuse.json")).read_text())
    dest = dest_root(group)
    count = size = links = 0
    for row in data["rows"]:
        target = dest / row["p"]
        if not target.parent.resolve().is_relative_to(dest.resolve()):
            raise ValueError("Symlink in destination path: "+row["p"])
        if "h" not in row:
            continue
        if target.exists():
            if target.is_symlink() or hash_file(target) != row["h"]:
                raise FileExistsError(target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
                inputs = [Path(known[row["h"]])] if row["h"] in known else [blob(h) for h in row.get("parts") or ([row["h"]] if row["s"] else [])]
                for source in inputs:
                    with source.open("rb") as part:
                        shutil.copyfileobj(part, f, 1024*1024)
            temp = Path(f.name)
            if temp.stat().st_size != row["s"] or hash_file(temp) != row["h"]:
                raise IOError("Final file verification failed: "+row["p"])
            os.chmod(temp, stat.S_IMODE(row["m"]))
            os.replace(temp, target)
        count += 1
        size += row["s"]
    for row in data["rows"]:
        if "l" in row:
            target = dest / row["p"]
            if not target.parent.resolve().is_relative_to(dest.resolve()) or not (target.parent/row["l"]).resolve().is_relative_to(dest.resolve()):
                raise ValueError("External archive symlink: "+row["p"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not os.path.lexists(target):
                target.symlink_to(row["l"])
            if not target.is_symlink() or os.readlink(target) != row["l"]:
                raise FileExistsError(target)
            links += 1
        elif stat.S_ISDIR(row["m"]):
            (dest/row["p"]).mkdir(parents=True,exist_ok=True)
    receipt = dict(group=group, verified_regular_files=count, logical_bytes=size, links=links,
                   archive_source_sha256=data["archive"], manifest_sha256=hash_file(RESTORE/(group+".manifest.json.gz")),
                   destination=str(dest))
    atomic_bytes(RESTORE/(group+".verified.json"), (json.dumps(receipt,indent=2)+"\n").encode())
    print(json.dumps(receipt),flush=True)


def send(args):
    sys.path.insert(0, str(args.archive.resolve() / "tools"))
    import offline
    archive = offline.Archive()
    rows = [{k:v for k,v in r.items() if k in {"p","m","h","s","parts","l"}} for r in archive.entries
            if selected(args.group,r["p"])]
    if not rows:
        raise ValueError("No archive members selected")
    payload = dict(group=args.group, rows=rows, archive=archive.layout["source_sha256"])
    wire = gzip.compress(json.dumps(payload,separators=(",",":")).encode(),mtime=0)
    def command(op, stream=False):
        remote = shlex.join([args.remote_python, args.remote_root+"/scripts/world_restore.py", op, args.group])
        return [*args.transport, "stream" if stream else "run", remote]
    result = subprocess.run(command("receive-plan",True), input=wire, stdout=subprocess.PIPE, check=True)
    missing = set(json.loads(result.stdout))
    print(f"{args.group}: {len(rows)} selected members; {len(missing)} missing chunks",flush=True)
    local = defaultdict(list)
    for h in missing:
        if h in archive.local_chunks:
            path,offset,n = archive.local_chunks[h]
            local[path].append((h,offset,n))
    packs = sorted({archive.pack_index[h][0] for h in missing if h not in archive.local_chunks})
    receipts = archive.receipts()
    work = [("disk", p) for p in sorted(local)] + [("pack", i) for i in packs]
    def worker(jobs):
        with subprocess.Popen(command("receive-chunks",True), stdin=subprocess.PIPE) as proc:
            def emit(h,content):
                if h not in missing:
                    return
                if hashlib.sha256(content).hexdigest() != h:
                    raise IOError("Source hash mismatch: "+h)
                compressed = zlib.compress(content,1)
                proc.stdin.write(json.dumps([h,len(content),len(compressed)]).encode()+b"\n")
                proc.stdin.write(compressed)
            for index,(kind,key) in enumerate(jobs):
                if kind == "disk":
                    with offline.disk_path(key).open("rb") as src:
                        for h,off,n in local[key]:
                            src.seek(off)
                            emit(h,src.read(n))
                else:
                    offline.readpack(archive.zstd,key,archive.layout["packs"][key],receipts[str(key)],emit)
                if (index+1)%100==0:
                    print(f"{args.group}: stream {index+1}/{len(jobs)} verified source groups",flush=True)
            proc.stdin.close()
            if proc.wait() != 0:
                raise IOError("Receiver failed")
    with ThreadPoolExecutor(max_workers=args.streams) as executor:
        list(executor.map(worker, [work[i::args.streams] for i in range(args.streams) if work[i::args.streams]]))
    subprocess.run(command("finish"),check=True)


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=["send","receive-plan","receive-chunks","finish"])
    cli.add_argument("group",choices=GROUPS)
    cli.add_argument("--archive",type=Path)
    cli.add_argument("--remote-root",default="/root/autodl-tmp/DN-Splatter")
    cli.add_argument("--remote-python",default="/root/miniconda3/bin/python")
    cli.add_argument("--streams",type=int,default=2)
    cli.add_argument("--transport",nargs=argparse.REMAINDER,help="Authorized local adapter accepting run/stream and one remote command")
    a=cli.parse_args()
    if a.action=="send":
        send(a)
    elif a.action=="receive-plan":
        receive_plan(a.group)
    elif a.action=="receive-chunks":
        receive_chunks()
    else:
        finish(a.group)
