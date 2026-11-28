"""Build one read-only set of exact media reference tiles for all models sharing the scene."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from interactive_dn_cache import atomic_json
from interactive_dn_supervision import digest
from interactive_eval_render import media_plan,panel_depth


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/".local/runs/interactive-world-20261128-v1"


def raw_tiles(scene,frame):
    image=cv2.imread(str(scene/frame["file_path"]))
    if image is None:
        raise IOError(frame["file_path"])
    depth=np.load(scene/frame["depth_file_path"],mmap_mode="r",allow_pickle=False)
    return np.stack((cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA),
                     cv2.resize(panel_depth(depth),(640,360),interpolation=cv2.INTER_AREA)))


def prepare(scene,frames,seq,hidden,workers):
    identity=dict(version=1,files_sha256=digest(scene/"training_manifest.json"),
                  timeline_sha256=digest(ROOT/"reports/capture-full-dn-30k/20261119-v1/selection.json"),
                  pixels="640x360 BGR8 INTER_AREA after historical full-resolution depth LUT")
    folder=RUN/"shared"/("eval-reference-"+identity["files_sha256"])
    folder.mkdir(parents=True,exist_ok=True)
    info=folder/"manifest.json"
    names=[f'images/{r["id"]}.png' for i,r in enumerate(seq) if i not in hidden]
    def one(name):
        frame=frames[name]
        rgb,depth=frame["file_path"],frame["depth_file_path"]
        if any(digest(scene/s)!=files[s] for s in (rgb,depth)):
            raise IOError("Frozen reference input hash differs: "+name)
        target=folder/(Path(name).stem+".npy")
        tiles=raw_tiles(scene,frame)
        with tempfile.NamedTemporaryFile(dir=folder,delete=False) as f:
            np.save(f,tiles,allow_pickle=False)
            tmp=f.name
        os.replace(tmp,target)
        return dict(id=Path(name).stem,path=target.name,bytes=target.stat().st_size,sha256=digest(target),
                    rgb_sha256=files[rgb],sensor_z_sha256=files[depth])
    with (folder/"writer.lock").open("a+b") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if not info.exists():
            files=json.loads((scene/"training_manifest.json").read_bytes())["files"]
            start=time.monotonic()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                records=list(pool.map(one,names))
            atomic_json(info,dict(identity=identity,visible=len(records),withheld=len(hidden),
                    logical_bytes=sum(r["bytes"] for r in records),build_seconds=time.monotonic()-start,frames=records))
        manifest=json.loads(info.read_bytes())
        if manifest["identity"]!=identity or [r["id"] for r in manifest["frames"]]!=[Path(n).stem for n in names]:
            raise ValueError("Different scene or publication timeline in shared reference cache")
    return folder,manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",type=Path,default=RUN/"restore/tree/.local/runs/capture-full-quality-v3-20261122-v1/inputs")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    meta=json.loads((args.scene/"transforms.json").read_bytes())
    frames={f["file_path"]:f for f in meta["frames"]}
    seq,hidden,_=media_plan(ROOT)
    folder,manifest=prepare(args.scene,frames,seq,hidden,args.workers)
    old_times=[];new_times=[]
    for row in manifest["frames"][::max(1,len(manifest["frames"])//32)]:
        name=f'images/{row["id"]}.png'
        start=time.perf_counter()
        raw=raw_tiles(args.scene,frames[name])
        old_times.append(time.perf_counter()-start)
        start=time.perf_counter()
        packed=np.load(folder/row["path"],mmap_mode="r",allow_pickle=False).copy()
        new_times.append(time.perf_counter()-start)
        if not np.array_equal(raw,packed) or digest(folder/row["path"])!=row["sha256"]:
            raise ValueError("A reference display tile differs: "+name)
    out=RUN/"f_evaluation/eval_performance/reference.json"
    atomic_json(out,dict(manifest=str(folder/"manifest.json"),manifest_sha256=digest(folder/"manifest.json"),
        visible=manifest["visible"],withheld=manifest["withheld"],bytes=manifest["logical_bytes"],build_seconds=manifest["build_seconds"],
        readback_frames=len(old_times),exact_bgr_pixels=len(old_times)*2*640*360,
        old_median_s=float(np.median(old_times)),new_median_s=float(np.median(new_times))))
