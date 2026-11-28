"""Verify the frozen VR training depths and export the exact natural-hole treatments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        h=hashlib.sha256()
        for b in iter(lambda:stream.read(1<<20),b""):
            h.update(b)
        return h.hexdigest()


def publish(path, doc):
    path.parent.mkdir(exist_ok=True,parents=True)
    raw=(json.dumps(doc,allow_nan=False,separators=(",",":"))+"\n").encode("utf8")
    if path.exists():
        if path.read_bytes()!=raw:
            raise ValueError(f"frozen VR treatment changed: {path}")
        return
    temp=path.with_name(path.name+f".{os.getpid()}.pending")
    temp.write_bytes(raw)
    temp.replace(path)


def prepare(run,scene):
    registry=json.loads((ROOT/"reports/vr-room-iphone-30k/frames.json").read_bytes())["frames"]["train"]
    if len(registry)!=376 or len({r["rgb"] for r in registry})!=376:
        raise ValueError("frozen VR train split differs")
    rows=[]
    shape=None
    total=outside_span=0
    for r in registry:
        p=scene/r["depth"]
        with Image.open(p) as im:
            z=np.asarray(im)
        if z.dtype!=np.uint16 or list(z.shape[::-1])!=r["depth_size"] or r["rgb_size"]!=r["depth_size"]:
            raise ValueError(f"VR sensor grid changed: {p.name}")
        if shape is None:
            shape=z.shape
        missing=int(np.count_nonzero(z==0))
        saturated=int(np.count_nonzero(z==65535))
        if z.shape!=shape or missing or saturated:
            raise ValueError(f"VR input differs from the verified dense 376-frame release: {p.name}")
        total+=z.size
        outside_span+=int(np.count_nonzero((z<300)|(z>6000)))
        rows.append({"id":Path(r["rgb"]).stem,"raw_depth":r["depth"],"raw_depth_sha256":sha(p),
                     "raw_pixels":z.size,"natural_holes":missing,"accepted_pixels":0,"saturated":saturated})
    base=run/"f_vr/supervision"
    base.mkdir(parents=True,exist_ok=True)
    empty=base/"shared_dense_frame.npz"
    if not empty.exists():
        # Source-independent empty arrays are identical for all 376 frames. Each
        # manifest independently binds its real RGB-D frame and checksum.
        with tempfile.NamedTemporaryFile(dir=base,delete=False) as f:
            fp=Path(f.name)
            bits=np.zeros(shape,np.bool_)
            zeros=np.zeros(shape,np.float32)
            np.savez_compressed(f,selected_depth=zeros,accepted=bits,natural_hole=bits,
                                selected_source=np.zeros(shape,np.uint8),risk=np.full(shape,np.nan,np.float32),
                                visibility_state=np.ones(shape,np.uint8),reject_reason=np.zeros(shape,np.uint8),
                                support_group_bits=np.zeros(shape,np.uint16))
        os.replace(fp,empty)
    with np.load(empty,allow_pickle=False) as p:
        if p["accepted"].shape!=shape or p["accepted"].any() or p["natural_hole"].any():
            raise ValueError("dense-frame packet changed")
    h=sha(empty)
    for variant in ("VR1","VR2"):
        dest=base/variant
        dest.mkdir(exist_ok=True)
        packet=dest/"dense_frame.npz"
        if not packet.exists():
            os.link(empty,packet)
        if sha(packet)!=h:
            raise ValueError("incorrect shared empty packet")
        completed=[dict(r,path=packet.name,sha256=h,bytes=packet.stat().st_size) for r in rows]
        doc={"variant":variant,"scope":"train","units":"m","shape":list(shape),"frames":completed,
             "actual_depth_pixels":total,"natural_holes":0,"correctly_retained_outside_03_6m":outside_span,
             "scored_quality_pixels":0,"changing_supervision_pixels":0,
             "method":"sensor-only gate" if variant=="VR1" else "Q1 selector with an empty eligible set",
             "attribution":"no natural holes in this release; a quality model cannot change the input or loss",
             "artificial_hiding_used_for_DN":False}
        publish(dest/"manifest.json",doc)
    print(json.dumps({"frames":len(rows),"hw":shape,"raw_pixels":total,"holes":0,"outside_03_6m":outside_span,
                      "VR1_sha":sha(base/"VR1/manifest.json"),"VR2_sha":sha(base/"VR2/manifest.json")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,default=ROOT/".local/runs/interactive-world-20261128-v1")
    parser.add_argument("--scene",type=Path)
    args=parser.parse_args()
    prepare(args.run_root,args.scene or args.run_root/"restore/tree/.local/data/room_datasets/vr_room")
