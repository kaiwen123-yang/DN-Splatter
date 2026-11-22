"""Index every original A3 local triangle in world-space tiles, without duplicating meshes."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/capture_surface_quality_protocol.json"
PLAN = json.loads(PROTOCOL.read_bytes())
RUN = ROOT / ".local/runs" / PLAN["run_id"]
TILE = .25


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, allow_nan=False) + "\n")
    tmp.replace(path)


def ply(path):
    with path.open("rb") as f:
        lines = []
        while True:
            line = f.readline(512)
            if not line or len(lines) > 100:
                raise ValueError(f"Invalid PLY header {path}")
            lines.append(line.strip().decode("ascii"))
            if lines[-1] == "end_header":
                break
        offset = f.tell()
    nv, nf = (int(next(l for l in lines if l.startswith("element "+k+" ")).split()[-1]) for k in ("vertex","face"))
    if not nf and not nv:
        return np.empty((0,3)), np.empty((0,3),dtype=np.int32)
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"Unsupported nonempty mesh format {path}")
    start = lines.index(f"element vertex {nv}") + 1
    end = lines.index(f"element face {nf}")
    types = {"double":"<f8","float":"<f4","uchar":"u1"}
    fields = [(s.split()[2], types[s.split()[1]]) for s in lines[start:end]]
    dtype = np.dtype(fields)
    if [x[0] for x in fields[:3]] != ["x","y","z"]:
        raise ValueError("Unexpected vertex layout")
    vertices = np.memmap(path, mode="r", offset=offset, shape=(nv,), dtype=dtype)
    xyz = np.ndarray((nv,3), dtype=dtype.fields["x"][0], buffer=vertices, strides=(dtype.itemsize,dtype.fields["x"][0].itemsize))
    faces = np.memmap(path, mode="r", offset=offset+vertices.nbytes, shape=(nf,), dtype=[("n","u1"),("ids","<i4",(3,))])
    if lines[end+1] not in ("property list uchar uint vertex_indices","property list uchar int vertex_indices") or offset+vertices.nbytes+faces.nbytes != path.stat().st_size:
        raise ValueError("Unexpected face layout")
    return xyz, faces["ids"]


def key(c):
    c = np.asarray(c, dtype=np.int64)
    if (np.abs(c) >= 2**20).any():
        raise ValueError("Coordinate outside lossless grid-key range")
    c = c + 2**20
    return (c[...,0]<<42) | (c[...,1]<<21) | c[...,2]


def index_source(path, out, record):
    marker = out / "tiles.json"
    if marker.is_file():
        old = read(marker)
        if old["source"] != record or sha(out/"faces.npy") != old["order_sha256"]:
            raise ValueError(f"Invalid existing tile index {out}")
        return old
    if sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Source mesh identity {path}")
    v, f = ply(path)
    keys = np.empty(len(f), dtype=np.int64)
    for i in range(0,len(f),131072):
        idx = f[i:i+131072]
        if np.any(idx<0) or np.any(idx>=len(v)):
            raise ValueError("Invalid source triangle")
        keys[i:i+len(idx)] = key(np.floor(v[idx].mean(axis=1)/TILE))
    order = np.argsort(keys,kind="stable").astype("i4")
    distinct, counts = np.unique(keys[order],return_counts=True)
    ends = np.cumsum(counts)
    out.mkdir(parents=True,exist_ok=True)
    tmp=out/"faces.partial.npy"
    np.save(tmp,order)
    tmp.replace(out/"faces.npy")
    result = dict(source=record, tile_size_m=TILE, triangles=len(f), vertices=len(v),
                  order_sha256=sha(out/"faces.npy"), tiles=list(zip(distinct.tolist(),(ends-counts).tolist(),ends.tolist())))
    write(marker,result)
    return result


def index_all(label):
    export=RUN/"export"/label
    identity=sha(export/"identity.json")
    ledger=read(export/"submaps.json")
    if ledger["identity"]!=identity or ledger["frames"]!=1478:
        raise ValueError("Frozen input identity")
    groups=read(export/"identity.json")["groups"]
    if len(groups)!=len(ledger["receipts"]):
        raise ValueError("Missing submaps")
    data={}
    for item,expected in zip(groups,ledger["receipts"]):
        gid=item["id"]
        receipt=ROOT/expected["path"]
        if sha(receipt)!=expected["sha256"]:
            raise ValueError("Submap receipt hash")
        src=read(receipt)
        if src["identity"]!=identity or src["group_id"]!=gid:
            raise ValueError("Submap ID")
        for branch in ("model","sensor"):
            row=src["meshes"][branch]
            info=index_source(ROOT/row["path"],export/"tiles"/branch/f"{gid:04d}",row)
            data[branch]=data.get(branch,0)+info["triangles"]
        progress=dict(run_id=PLAN["run_id"],phase="A3_index",model=label,groups=gid+1,total=len(groups),
                      triangles=data,pid=os.getpid(),identity=identity)
        write(RUN/"state"/"A3_progress.json",progress)
        print(f"Indexed {gid+1}/{len(groups)} original model and sensor submaps: {data}",flush=True)
    write(export/"tiles"/"complete.json",dict(groups=len(groups),frames=1478,triangles=data,identity=identity))


if __name__=="__main__":
    import sys
    if sys.argv[1:] not in (["B2"],["B3"]):
        raise SystemExit("usage: capture_surface_tiles.py B2|B3")
    index_all(sys.argv[1])
