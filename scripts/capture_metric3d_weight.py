"""Fetch the pinned official HF object in verified, resumable HTTP byte ranges."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
NAME = "metric_depth_vit_large_800k.pth"
REVISION = "80d2d1410afb4b23cd9d18c6be9144483d4b70b6"
URL = f"https://huggingface.co/JUGGHM/Metric3D/resolve/{REVISION}/{NAME}"
SIZE = 1647972663
DIGEST = "15328ffc42b528b95f188687418f6f03b3f123eb34ccdbd686c112abbea6d972"
CHUNK = 8 * 1024**2


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(CHUNK), b""):
            h.update(part)
    return h.hexdigest()


def fetch():
    out = ROOT / ".local/data/downloads" / NAME
    out.parent.mkdir(exist_ok=True, parents=True)
    partial = out.with_suffix(".pth.part")
    journal = out.with_suffix(".pth.ranges.json")
    with out.with_suffix(".pth.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out.is_file():
            if out.stat().st_size != SIZE or digest(out) != DIGEST:
                raise ValueError("Existing checkpoint does not match the published LFS object")
            return
        finished = set(json.loads(journal.read_text())) if journal.exists() and partial.exists() else set()
        if not journal.exists() and partial.exists() and partial.stat().st_size < SIZE:
            finished.update(range(partial.stat().st_size // CHUNK))
        def record():
            tmp = journal.with_suffix(".tmp")
            tmp.write_text(json.dumps(sorted(finished)))
            tmp.replace(journal)
        record()
        proxy = ROOT / ".local/network/proxy.url"
        proxies = {"https": proxy.read_text().strip()} if proxy.is_file() else {}
        with partial.open("r+b" if partial.exists() else "wb") as f:
            def get(i):
                start = i * CHUNK
                end = min(start + CHUNK, SIZE) - 1
                for attempt in range(6):
                    try:
                        with requests.get(URL + f"?range_start={start}&try={attempt}",
                                          headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                                          proxies=proxies, timeout=(15, 70)) as response:
                            if (response.status_code != 206 or
                                    response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}" or
                                    len(response.content) != end - start + 1):
                                raise ValueError(f"Invalid range at byte {start} (HTTP {response.status_code})")
                            return i, response.content
                    except (requests.RequestException, ValueError) as exc:
                        if attempt == 5:
                            raise RuntimeError(f"Range at byte {start} failed ({type(exc).__name__})") from None
                        time.sleep(2 ** attempt)
            count = (SIZE + CHUNK - 1) // CHUNK
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                pending = {pool.submit(get, i) for i in range(count) if i not in finished}
                while pending:
                    ready, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in ready:
                        i, data = future.result()
                        f.seek(i * CHUNK)
                        f.write(data)
                        f.flush()
                        os.fsync(f.fileno())
                        finished.add(i)
                        record()
                    print(f"Verified HTTP ranges {len(finished)}/{count}", flush=True)
        if partial.stat().st_size != SIZE or digest(partial) != DIGEST:
            # Preserve the bytes. A subsequent run must obtain every range again.
            partial.rename(partial.with_name(partial.name + f".rejected.{time.time_ns()}"))
            journal.unlink()
            raise ValueError("Assembled weight SHA256 mismatch")
        partial.replace(out)
        print(f"Verified {NAME}, {SIZE} bytes, SHA256 {DIGEST}", flush=True)


if __name__ == "__main__":
    fetch()
