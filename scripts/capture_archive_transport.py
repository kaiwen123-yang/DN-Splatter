"""Verified private SSH transfers for the capture-full comparison. No project credentials are copied."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
ID = "capture-full-metric3d-30k-20261120-v1"
RESULT = "7d4a8a1d7d47379102cc5a17d2bbb7e07231ac42"
SERVER = "/root/autodl-tmp/DN-Splatter"
STAGE = ".local/archives/" + ID + "-" + RESULT[:8]
B1 = ".local/runs/" + ID
B0 = ".local/runs/capture-full-dn-30k-20261119-v1"
GEOID = "20261119-full-5hz-3982cb26"
GEO = ".local/data/custom/capture-full/" + GEOID
GEORUN = ".local/capture-geometry/" + GEOID
REPORT = "reports/capture-full-metric3d-30k/20261120-v1"
BASE_REPORT = "reports/capture-full-dn-30k/20261119-v1"
VIDEO = B1 + "/video/reference_B0_B1.mp4"
BLOCK = 8 * 1024**2


def utc():
    return datetime.now(timezone.utc).isoformat()


def check(ok, message):
    if not ok:
        raise ValueError(str(message))


def read(path):
    return Path(path).read_text(encoding="utf-8-sig")


def document(path):
    return json.loads(read(path))


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else stream_digest(f)


def stream_digest(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(4 * 1024**2), b""):
        h.update(block)
    return h.hexdigest()


def checksum_record(path):
    p = Path(path)
    return dict(bytes=p.stat().st_size, sha256=digest(p))


def safe_rel(name):
    p = PurePosixPath(name)
    check(name and not p.is_absolute() and str(p) == name and "\\" not in name and
          not any(v in ("", ".", "..") or ":" in v for v in p.parts), "Invalid archive member")
    return name


def ssh_args():
    host = read(ROOT / ".local/transfer/full-host").strip()
    check(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host), "Invalid saved project connection")
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20",
            "-o", "ServerAliveCountMax=4", "-o", "IPQoS=none", host]


def remote_bytes(rel, optional=False):
    path = shlex.quote(SERVER + "/" + safe_rel(rel))
    cmd = f"if test -f {path}; then cat -- {path}; fi" if optional else f"cat -- {path}"
    for attempt in range(5):
        try:
            r = subprocess.run(ssh_args() + [cmd], capture_output=True, timeout=120)
            if r.returncode == 0:
                return r.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(1+attempt)
    raise OSError("SSH metadata unavailable: " + rel)


def range_hashes(path, block=BLOCK):
    chunks = []
    with Path(path).open("rb") as f:
        while True:
            at = f.tell()
            b = f.read(block)
            if not b:
                return chunks
            chunks.append([at, len(b), hashlib.sha256(b).hexdigest()])


def download(ssh, remote_rel, target, info, workers, partial=None, progress=None):
    """Write independent byte ranges at fixed offsets and promote only after full SHA verification."""
    target = Path(target)
    partial = Path(partial) if partial is not None else target.with_suffix(target.suffix + ".ranges")
    size, key = info["bytes"], info["sha256"]
    chunks = info["chunks"]
    pos = 0
    for at, n, h in chunks:
        check(at == pos and n > 0 and re.fullmatch("[a-f0-9]{64}", h), "Invalid range descriptor")
        pos += n
    check(pos == size and re.fullmatch("[a-f0-9]{64}", key), "Invalid source descriptor")
    if target.exists():
        check(target.stat().st_size == size and digest(target) == key, "Existing target differs: " + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = partial.exists()
    have = [0] * len(chunks)
    valid = set()
    with partial.open("r+b" if existed else "w+b") as f:
        previous_size = f.seek(0, 2)
        check(previous_size <= size, "Unexpected partial length")
        if existed:
            for i, (a,n,h) in enumerate(chunks):
                if a + n <= previous_size:
                    f.seek(a)
                    if hashlib.sha256(f.read(n)).hexdigest() == h:
                        have[i] = n
                        valid.add(i)
        f.truncate(size)
    remote = shlex.quote(SERVER + "/" + safe_rel(remote_rel))
    lock = threading.Lock()
    def update():
        if progress:
            progress(sum(have), len(valid), len(chunks))
    update()
    def fetch(i):
        a,n,h = chunks[i]
        filled = 0
        for attempt in range(7):
            with partial.open("r+b", buffering=0) as f:
                f.seek(a + filled)
                command = f"dd if={remote} iflag=skip_bytes,count_bytes bs=262144 skip={a+filled} count={n-filled} status=none"
                with subprocess.Popen(ssh + [command], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
                    while filled < n:
                        b = proc.stdout.read(min(262144, n-filled))
                        if not b:
                            break
                        f.write(b)
                        filled += len(b)
                        with lock:
                            have[i] = filled
                            update()
                    proc.stdout.close()
                    proc.wait()
                if filled == n:
                    f.seek(a)
                    if hashlib.sha256(f.read(n)).hexdigest() == h:
                        with lock:
                            valid.add(i)
                            update()
                        return
                    filled = 0
                    with lock:
                        have[i] = 0
            time.sleep(min(10, attempt + 1))
        raise OSError(f"Unverified byte range {i} in {remote_rel}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(fetch, [i for i in range(len(chunks)) if i not in valid]))
    check(len(valid) == len(chunks) and digest(partial) == key, "Final SHA256 mismatch: " + target.name)
    os.replace(partial, target)


def finish_video():
    from capture_metric3d_receive import DEST, locked
    movie = DEST / "reference_B0_B1.mp4"
    with locked():
        pub = document(ROOT / REPORT / "video_manifest.json")
        old = DEST / "transfer_ranges.json"
        info = document(old) if old.is_file() else dict(bytes=pub["bytes"], sha256=pub["sha256"], chunks=[])
        check((pub["sha256"],pub["bytes"]) == (info["sha256"],info["bytes"]), "Public video manifest mismatch")
        if not info["chunks"] and not movie.exists():
            raise ValueError("Original byte-range manifest required to resume incomplete video")
        if not movie.exists():
            last = [0.]
            def progress(b,n,t):
                if time.monotonic() - last[0] >= 4 or n == t:
                    atomic(DEST/"transfer_progress.json", dict(utc=utc(),bytes=b,total=info["bytes"],complete_ranges=n,ranges=t))
                    last[0] = time.monotonic()
            download(ssh_args(), VIDEO, movie, info, 12, DEST/"reference_B0_B1.mp4.ranges", progress)
        check(checksum_record(movie) == dict(bytes=pub["bytes"],sha256=pub["sha256"]), "Video identity")
        atomic(DEST/"receiver.json", dict(status="received",utc=utc(),path=str(movie),bytes=pub["bytes"],
                                         sha256=pub["sha256"],commit=RESULT))
        print("Full video SHA256 verified", pub["sha256"], flush=True)
    return movie


if __name__ == "__main__":
    finish_video()
