"""Stream just the frozen hall inputs from the November archive to a fresh instance."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import zlib


B2 = ".local/runs/capture-full-quality-v2-20261121-v1/inputs"
GEO = ".local/capture-geometry/20261119-full-5hz-3982cb26"


def needed(archive):
    result = {}
    for name, row in archive.by_path.items():
        if not row.get("h"):
            continue
        if name.startswith(B2 + "/"):
            rel = name[len(B2) + 1:]
            if rel.startswith(("images/", "depth/")) or rel in {
                "transforms.json", "splits.json", "training_manifest.json", "geometry.json",
            }:
                result["b2/" + rel] = row
        elif name.startswith(GEO + "/"):
            rel = name[len(GEO) + 1:]
            if rel.startswith("rect/1280/depth/") or rel in {
                "transfer/selection.json", "transfer/calibration.json", "rect/grids.json",
            }:
                result["geometry/" + rel] = row
    for prefix in ("b2/images/", "b2/depth/", "geometry/rect/1280/depth/"):
        if sum(k.startswith(prefix) for k in result) != 1846:
            raise ValueError("Incomplete source inventory for " + prefix)
    return result


def receive(directory):
    source = sys.stdin.buffer
    directory.mkdir(exist_ok=True, parents=True)
    for line in source:
        name, digest, size, wire = json.loads(line)
        if not (name.startswith(("geometry/", "b2/")) or name == "archive_inventory.json"):
            raise ValueError("Unknown input role")
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or wire > 50_000_000:
            raise ValueError("Invalid archive member")
        payload = source.read(wire)
        if len(payload) != wire:
            raise EOFError(name)
        data = zlib.decompress(payload)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise IOError("Transfer integrity failure: " + name)
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise FileExistsError(path)
            continue
        path.parent.mkdir(exist_ok=True, parents=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
            out.write(data)
        os.replace(out.name, path)


def send(args):
    sys.path.insert(0, str(args.archive.resolve() / "tools"))
    import offline

    archive = offline.Archive()
    rows = needed(archive)
    inventory = {k: dict(path=v["p"], bytes=v["s"], sha256=v["h"]) for k, v in rows.items()}
    full_index = json.dumps(inventory, separators=(",", ":"), sort_keys=True).encode()
    entries = {k: v for k, v in rows.items() if not k.startswith("b2/depth/") or args.send_b2_depth}
    command = shlex.join([args.remote_python, args.remote_script, "receive", str(args.destination)])
    ssh = args.ssh
    if ssh and ssh[0] == "--":
        ssh = ssh[1:]
    if not ssh:
        raise ValueError("Pass the SSH client and destination after --")
    listing = shlex.join(["find", str(args.destination), "-type", "f", "-printf", "%P\\n"])
    result = subprocess.run(ssh + [listing], capture_output=True)
    existing = set(result.stdout.decode().splitlines()) if result.returncode == 0 else set()
    paths = sorted((p for p in entries if p not in existing), key=lambda p: ("/frame_" in p, p))
    paths.insert(0, "archive_inventory.json")

    def upload(part):
        with subprocess.Popen(ssh + [command], stdin=subprocess.PIPE) as process:
            for i, name in enumerate(part):
                if name == "archive_inventory.json":
                    data = full_index
                    digest = hashlib.sha256(data).hexdigest()
                else:
                    digest = rows[name]["h"]
                    if digest in archive.reused:
                        data = offline.disk_path(archive.reused[digest]).read_bytes()
                    else:
                        with tempfile.TemporaryDirectory() as work:
                            temp = Path(work) / "member"
                            archive.materialize({digest: temp})
                            data = temp.read_bytes()
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise IOError("Archive file integrity: " + name)
                enc = zlib.compress(data, 1)
                header = json.dumps([name, digest, len(data), len(enc)]).encode() + b"\n"
                process.stdin.write(header)
                process.stdin.write(enc)
                if (i + 1) % 100 == 0:
                    print(f"{name} ({i+1}/{len(part)} on stream)", flush=True)
            process.stdin.close()
            if process.wait():
                raise RuntimeError("SSH receiver exited with an error")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.streams) as pool:
        list(pool.map(upload, [paths[i::args.streams] for i in range(args.streams)]))
    print(f"Transferred {len(paths)} members. The prepare stage verifies the complete inventory.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    rx = commands.add_parser("receive")
    rx.add_argument("destination", type=Path)
    tx = commands.add_parser("send")
    tx.add_argument("--archive", type=Path, required=True)
    tx.add_argument("--destination", required=True)
    tx.add_argument("--remote-script", required=True)
    tx.add_argument("--remote-python", default="/root/miniconda3/bin/python")
    tx.add_argument("--streams", type=int, default=4)
    tx.add_argument("--send-b2-depth", action="store_true", help="Also transfer archived range-masked NPYs")
    tx.add_argument("ssh", nargs=argparse.REMAINDER)
    args = cli.parse_args()
    if args.action == "receive":
        receive(args.destination)
    else:
        send(args)
