"""Fetch the four pinned inputs, or relay them from a client with Zenodo access.

Relay mode streams directly into the same resumable AutoDL file. It never
writes a dataset or a weight file on the client.
"""

import argparse
import concurrent.futures
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request


FILES = {
    "vr_room_iphone.tar.gz": (10230733, 244279042, "6c411b7bad82041e2fb5eb8d6a46fbf9"),
    "omnidata_dpt_normal_v2.ckpt": (10447888, 1947430960, "f2e0962ea6e248a1648f20d74b1b034d"),
    "vr_room_mesh_pd.tar.gz": (10222321, 218765772, "7a56daf91c3c610cb126c54ba9216f79"),
    "reference_depth.tar.gz": (10438963, 3849632019, "e19690ad110f623ae958d19afb627059"),
}
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".local/data/downloads"
CHUNK = 64 * 1024**2


def size_on_target(name):
    final = DEST / name
    part = DEST / (name + ".part")
    if final.is_file():
        if final.stat().st_size != FILES[name][1]:
            raise ValueError("Existing finished file has the wrong size")
        return final.stat().st_size
    return part.stat().st_size if part.exists() else 0


def finish(name):
    import fcntl

    with (DEST / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        final = DEST / name
        src = final if final.exists() else DEST / (name + ".part")
        digest = hashlib.md5()
        with src.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024**2), b""):
                digest.update(block)
        if src.stat().st_size != FILES[name][1] or digest.hexdigest() != FILES[name][2]:
            raise ValueError(f"Zenodo size or MD5 mismatch for {name}")
        if src != final:
            src.rename(final)


def receive(name, offset, length, stream):
    import fcntl

    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if size_on_target(name) != offset or offset + length > FILES[name][1]:
            raise ValueError("Transfer offset changed")
        with (DEST / (name + ".part")).open("ab") as out:
            left = length
            while left:
                block = stream.read(min(1024**2, left))
                if not block:
                    raise EOFError("Incomplete input range")
                out.write(block)
                left -= len(block)


def transfer(name, host, remote_root):
    record, total, _ = FILES[name]
    api = f"https://zenodo.org/api/records/{record}/files/{name}/content"
    web = f"https://zenodo.org/records/{record}/files/{name}?download=1"

    def ssh_command(action, *args):
        py = remote_root + "/.local/conda/dn-splatter-bw/bin/python"
        cmd = shlex.join([py, remote_root + "/scripts/zenodo_inputs.py", action, name, *map(str, args)])
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=20", host, cmd]

    failures = 0
    previous = -1
    while True:
        try:
            start = (int(subprocess.check_output(ssh_command("size"), timeout=40))
                     if host else size_on_target(name))
            if start > previous:
                failures = 0
            previous = start
            if start == total:
                if host:
                    subprocess.run(ssh_command("finish"), check=True)
                else:
                    finish(name)
                print(f"{name} verified {total} bytes", flush=True)
                return
            if not 0 <= start < total:
                raise ValueError("Invalid saved offset")
            size = min(CHUNK, total - start)
            end = start + size - 1
            url = (web, api)[failures % 2]
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                                       "User-Agent": "DN-Splatter-reproduction/1.0"})
            with urllib.request.urlopen(req, timeout=75) as response:
                cr = response.headers.get("Content-Range")
                if not (response.status == 206 and cr == f"bytes {start}-{end}/{total}"
                        or response.status == 200 and start == 0 and int(response.headers.get("Content-Length", -1)) == total):
                    raise ValueError(f"Unexpected range response ({response.status}, {cr})")
                if host:
                    with subprocess.Popen(ssh_command("receive", start, size), stdin=subprocess.PIPE) as sink:
                        try:
                            left = size
                            while left:
                                block = response.read(min(1024**2, left))
                                if not block:
                                    raise EOFError("Short Zenodo response")
                                sink.stdin.write(block)
                                left -= len(block)
                        finally:
                            sink.stdin.close()
                        if sink.wait() != 0:
                            raise OSError("Remote receiver exited")
                else:
                    receive(name, start, size, response)
            print(f"{name} {end+1}/{total}", flush=True)
        except (OSError, EOFError, ValueError, subprocess.SubprocessError) as exc:
            failures += 1
            print(f"{name} at {previous} bytes: {type(exc).__name__}: {exc}", flush=True)
            if failures >= 8:
                raise
            time.sleep(min(60, failures * 5))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["download", "size", "finish", "receive"])
    ap.add_argument("name", choices=list(FILES) + ["all"])
    ap.add_argument("numbers", type=int, nargs="*")
    ap.add_argument("--ssh-host")
    ap.add_argument("--remote-root", default="/root/autodl-tmp/DN-Splatter")
    a = ap.parse_args()
    if a.action == "download":
        names = list(FILES) if a.name == "all" else [a.name]
        with concurrent.futures.ThreadPoolExecutor(len(names)) as pool:
            futures = [pool.submit(transfer, n, a.ssh_host, a.remote_root) for n in names]
            for f in concurrent.futures.as_completed(futures):
                f.result()
    elif a.action == "size":
        print(size_on_target(a.name))
    elif a.action == "finish":
        finish(a.name)
    else:
        receive(a.name, *a.numbers, sys.stdin.buffer)
