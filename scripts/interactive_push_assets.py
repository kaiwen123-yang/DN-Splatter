"""Recover the versioned public inputs for the interaction experiments on the server."""

import argparse
import collections
import hashlib
import io
import json
import os
import re
import shutil
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath


SOURCES = {
    "pusht": ("datasets", "shashuo0104/xarm7_pusht", "9d9e4e31ff82580ca8841664ba4a891b92dbdc35"),
    "tblock": ("models", "shashuo0104/phystwin-T-block", "25cea998cbde8957e766d7a49b28d72d4fb14e97"),
    "rope": ("datasets", "Jianghanxiao/PhysTwin", "05665bcd0edad8f069c7def9cbb58d2eb75d68d0"),
}
ZIP_SHA = "586a237a0f870b2a0a144ee69c3f04b636cac4fe251df969c1fd1feaf67eeae2"
ZIP_BYTES = 8075884740
CASES = {"single_push_rope_1": (92, 64), "single_push_rope_4": (83, 58)}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + f".{os.getpid()}.tmp")
    part.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    os.replace(part, path)


def file_hash(path):
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {path.stat().st_size}\x00".encode())
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
            git.update(block)
            crc = zlib.crc32(block, crc)
    return sha.hexdigest(), git.hexdigest(), crc & 0xffffffff


def fetch(url, byte_range=None):
    if os.environ.get("DN_PUBLIC_INPUT_PIPE") == "1":
        sys.stdout.buffer.write(("HF_FETCH " + json.dumps([url, byte_range]) + "\n").encode())
        sys.stdout.buffer.flush()
        n_bytes = struct.unpack("!Q", sys.stdin.buffer.read(8))[0]
        block = sys.stdin.buffer.read(n_bytes)
        if len(block) != n_bytes:
            raise OSError("Public input relay disconnected")
        return block
    for attempt in range(5):
        try:
            headers = {"Accept-Encoding": "identity", "User-Agent": "DN-Splatter-data/1"}
            target = url
            if byte_range is not None:
                start, end = byte_range
                headers["Range"] = f"bytes={start}-{end}"
                target += ("&" if "?" in target else "?") + f"range={start}-{end}"
            with urllib.request.urlopen(urllib.request.Request(target, headers=headers), timeout=60) as response:
                if byte_range is None:
                    return response.read()
                parsed = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if response.status != 206 or not parsed or tuple(map(int, parsed.groups())) != (start, end, ZIP_BYTES):
                    raise ValueError("Host did not serve the requested slice of the pinned PhysTwin archive")
                block = response.read(end - start + 2)
                if len(block) != end - start + 1:
                    raise OSError("Truncated range response")
                return block
        except (OSError, urllib.error.URLError) as error:
            if attempt == 4:
                raise
            print(f"Network retry {attempt+1}/4 ({type(error).__name__})", flush=True)
            time.sleep(2 ** attempt)


class RangeArchive(io.RawIOBase):
    """Bounded shared block cache for ZIP directory and sequential member reads."""

    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.blocks = collections.OrderedDict()
        self.chunk = 4 << 20
        self.downloaded = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        origin = (0, self.pos, ZIP_BYTES)[whence]
        if origin + offset < 0:
            raise ValueError("negative offset")
        self.pos = origin + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        size = max(0, ZIP_BYTES - self.pos) if size < 0 else min(size, max(0, ZIP_BYTES-self.pos))
        parts = []
        while size:
            key, start = divmod(self.pos, self.chunk)
            if key not in self.blocks:
                base = key * self.chunk
                chunk = fetch(self.url, (base, min(base + self.chunk, ZIP_BYTES) - 1))
                self.downloaded += len(chunk)
                self.blocks[key] = chunk
                if len(self.blocks) > 4:
                    self.blocks.popitem(last=False)
            self.blocks.move_to_end(key)
            chunk = self.blocks[key]
            count = min(len(chunk)-start, size)
            parts.append(chunk[start:start+count])
            self.pos += count
            size -= count
        return b"".join(parts)


def safe_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unexpected member name {name!r}")
    target = (root / path).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"Outside recovery directory: {name!r}")
    return target


def recover(root, source, endpoint, cases, local_zip=None):
    root.mkdir(parents=True, exist_ok=True)
    kind, repo, revision = SOURCES[source]
    identity = json.loads(fetch(f"{endpoint}/api/{kind}/{repo}/revision/{revision}?blobs=true"))
    if identity["sha"] != revision:
        raise ValueError("Unexpected HF revision")
    prefix = "datasets/" if kind == "datasets" else ""
    base = f"{endpoint}/{prefix}{repo}/resolve/{revision}/"
    manifest = {"repo": repo, "revision": revision, "license": identity.get("cardData", {}).get("license"), "files": []}
    if source != "rope":
        for info in identity["siblings"]:
            name = info["rfilename"]
            target = safe_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            candidate = target if target.is_file() else target.with_suffix(target.suffix + f".{os.getpid()}.part")
            if candidate != target:
                candidate.write_bytes(fetch(base + urllib.parse.quote(name)))
            sha, git, _ = file_hash(candidate)
            valid = sha == info["lfs"]["sha256"] if info.get("lfs") else git == info["blobId"]
            if not valid or candidate.stat().st_size != info["size"]:
                raise ValueError(f"Pinned identity mismatch for {name}")
            if candidate != target:
                os.replace(candidate, target)
            manifest["files"].append({"path": name, "bytes": info["size"], "sha256": sha})
    else:
        info = next(r for r in identity["siblings"] if r["rfilename"] == "data.zip")
        if (info["size"], info["lfs"]["sha256"]) != (ZIP_BYTES, ZIP_SHA):
            raise ValueError("PhysTwin revision does not point to the frozen archive")
        if local_zip and (local_zip.stat().st_size != ZIP_BYTES or file_hash(local_zip)[0] != ZIP_SHA):
            raise ValueError("Local full archive hash mismatch")
        archive = local_zip if local_zip else RangeArchive(base + "data.zip")
        manifest.update(archive_lfs_sha256=ZIP_SHA, archive_bytes=ZIP_BYTES, whole_file_hash_computed=bool(local_zip))
        prefixes = tuple(f"data/different_types/{case}/" for case in cases)
        with zipfile.ZipFile(archive) as zf:
            entries = sorted((i for i in zf.infolist() if i.filename.startswith(prefixes) and not i.is_dir()), key=lambda i: i.header_offset)
            if not all(any(i.filename == prefix + "split.json" for i in entries) for prefix in prefixes):
                raise ValueError("Case split absent in frozen ZIP")
            for entry in entries:
                target = safe_path(root, entry.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                candidate = target if target.is_file() else target.with_suffix(target.suffix + f".{os.getpid()}.part")
                if candidate != target:
                    with zf.open(entry) as inp, candidate.open("wb") as out:
                        shutil.copyfileobj(inp, out, 1 << 20)
                sha, _, crc = file_hash(candidate)
                if crc != entry.CRC or candidate.stat().st_size != entry.file_size:
                    raise ValueError(f"ZIP member CRC or length mismatch for {entry.filename}")
                if candidate != target:
                    os.replace(candidate, target)
                manifest["files"].append({"path": entry.filename, "bytes": entry.file_size, "sha256": sha, "zip_crc32": f"{crc:08x}"})
        if not local_zip:
            manifest["transferred_range_bytes_this_attempt"] = archive.downloaded
        for case in cases:
            directory = root / "data/different_types" / case
            n, cut = CASES[case]
            split = json.loads((directory / "split.json").read_text())
            counts = {f"{dtype}/{cam}": len(list((directory/dtype/str(cam)).glob(pattern)))
                      for dtype, pattern in (("color", "*.png"), ("depth", "*.npy")) for cam in range(3)}
            if split.get("train") != [0, cut] or split.get("test") != [cut, n]:
                raise ValueError(f"{case} split changed: {split}")
            if any(count != n for count in counts.values()):
                raise ValueError(f"{case} frame count mismatch: {counts}")
            manifest.setdefault("cases", {})[case] = {"split": split, "expected_prefix_end": cut, "frame_counts": counts}
    outfile = root / ("source-" + "-".join(cases) + ".json" if source == "rope" else "source.json")
    write_json(outfile, manifest)
    print(f"Verified {len(manifest['files'])} files in {outfile}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=SOURCES)
    parser.add_argument("root", type=Path)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--local-zip", type=Path)
    args = parser.parse_args()
    recover(args.root, args.source, args.endpoint.rstrip("/"), args.case or list(CASES), args.local_zip)


if __name__ == "__main__":
    main()
