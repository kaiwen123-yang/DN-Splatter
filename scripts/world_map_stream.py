"""Run the frozen 1,478-image causal Map stream with bounded, ordered I/O."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import torch

from lingbot_common import P0, ROOT, RUN as HIST, read, require, save, sha
from lingbot_map_run import audit


SEQUENCE = "full_train_1478"
OUT = ROOT / ".local/runs/interactive-world-20261128-v1/b_map/ordered_stream"
SOURCE = ROOT / ".local/upstream/lingbot-map-v2"
sys.path.insert(0, str(SOURCE))
from lingbot_map2.models import load_model
from lingbot_map2.utils.load_fn import image_paths, load_images, uniform_keyframe_ids


def sampled_resources(stop, rows):
    while not stop.is_set():
        values = dict(monotonic=time.monotonic())
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            k,_,v=line.partition(":")
            if k in ("Pss", "Rss", "Private_Clean", "Private_Dirty"):
                values[k+"_bytes"]=1024*int(v.split()[0])
        values["USS_bytes"] = values["Private_Clean_bytes"] + values["Private_Dirty_bytes"]
        path=Path("/sys/fs/cgroup/memory.current")
        values["cgroup_total_bytes"]=int(path.read_text()) if path.exists() else None
        q=subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],
                         capture_output=True,text=True)
        if q.returncode==0:
            used,util=q.stdout.splitlines()[0].split(",")
            values.update(device_used_MiB=int(used),gpu_util_percent=int(util))
        statline=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()
        values["io_wait_ticks"]=int(statline[39])
        rows.append(values)
        stop.wait(2)


def ordered_load(paths, width, height, workers, capacity):
    def decode(i):
        start=time.perf_counter()
        frame=load_images([paths[i]],width,height)
        return i,frame,time.perf_counter()-start
    with ThreadPoolExecutor(max_workers=workers) as pool:
        waiting=deque(pool.submit(decode,i) for i in range(min(len(paths),capacity)))
        for index in range(len(paths)):
            i,tensor,service=waiting.popleft().result()
            require(i==index, "Input order changed")
            if index+capacity < len(paths):
                waiting.append(pool.submit(decode,index+capacity))
            yield i,tensor,service


def compare(out, old):
    manifest=read(out/"manifest.json")
    other=read(old/"manifest.json")
    require(all(manifest[k]==other[k] for k in ("image_paths","persistent_raw_ids","completed_outputs")),
            "Frozen paths, persistent frames or output ordering changed")
    fields={}
    for name in manifest["completed_outputs"]:
        with np.load(out/name,allow_pickle=False) as a, np.load(old/name,allow_pickle=False) as b:
            require(set(a.files)==set(b.files),"Output fields changed")
            for field in a.files:
                aa,bb=a[field],b[field]
                require(aa.shape==bb.shape and aa.dtype==bb.dtype,"Shape or type changed: "+field)
                if field in {"raw_indices","frame_indices"}:
                    require(np.array_equal(aa,bb),"Raw/stream ID mismatch")
                rec=fields.setdefault(field,dict(samples=0,finite_pairs=0,unequal_nonfinite=0,
                                                max_abs=0.0,sum_abs=0.0,exact_values=0))
                same=(aa==bb)|(np.isnan(aa)&np.isnan(bb))
                finite=np.isfinite(aa)&np.isfinite(bb)
                diff=np.abs(aa[finite].astype(np.float64)-bb[finite])
                rec["samples"]+=aa.size
                rec["finite_pairs"]+=int(finite.sum())
                rec["unequal_nonfinite"]+=int((~same&~finite).sum())
                rec["exact_values"]+=int(same.sum())
                rec["max_abs"]=max(rec["max_abs"],float(diff.max(initial=0)))
                rec["sum_abs"]+=float(diff.sum())
    for rec in fields.values():
        rec["mean_abs"]=rec.pop("sum_abs")/rec["finite_pairs"] if rec["finite_pairs"] else None
    return fields


def execute(workers,capacity,writes):
    start=time.perf_counter()
    old=read(HIST/"map"/SEQUENCE/"complete.json")
    paths=image_paths(None,str(P0/(SEQUENCE+".json")))
    ids=read(P0/"selection.json")["sequences"][SEQUENCE]
    require(len(paths)==1478 and [p.stem for p in paths]==ids["ids"],"Not the strict train sequence")
    weights=HIST/"weights/map2"
    identity=dict(input=sha(P0/(SEQUENCE+".json")),freeze=sha(P0/"freeze.json"),
                  weight=sha(weights/"model_weights.pt"),config=sha(weights/"config.json"),
                  source=subprocess.check_output(["git","-C",str(SOURCE),"rev-parse","HEAD"],text=True).strip(),
                  implementation=sha(Path(__file__)),workers=workers,decode_capacity=capacity,write_capacity=writes)
    for new,previous in [("input","image_list"),("weight","weights"),("freeze","freeze"),("config","config"),("source","upstream")]:
        require(identity[new]==old["identity"][previous],f"Historical input identity mismatch: {new}")
    if (OUT/"verified.json").is_file():
        receipt=read(OUT/"verified.json")
        require(receipt["identity"]==identity,"Existing result has a different identity")
        seen=audit(Path(receipt["predictions"]),SEQUENCE)
        require(seen["chunks"]==read(Path(receipt["audit_manifest"]))["chunks"],"Saved chunks changed")
        print("Verified existing ordered stream",flush=True)
        return
    attempt=OUT/("attempt-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out=attempt/"predictions"
    out.mkdir(parents=True)
    stop=threading.Event()
    resources=[]
    monitor=threading.Thread(target=sampled_resources,args=(stop,resources))
    monitor.start()
    save(attempt/"identity.json",identity)
    try:
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        torch.manual_seed(42)
        device=torch.device("cuda:0")
        t=time.perf_counter()
        model=load_model(weights/"model_weights.pt",config=weights/"config.json",device=str(device),
                         depth=True,compile=True)
        load_seconds=time.perf_counter()-t
        persistent=uniform_keyframe_ids(len(paths),prefix=model.config.scale_frames)
        require(persistent==ids["persistent_raw_ids"],"Uniform320 keyframe IDs changed")
        t=time.perf_counter()
        model.warmup(paths[:min(model.config.scale_frames+2,model.config.max_frames)])
        compile_warmup=time.perf_counter()-t
        saved=[]
        def write(first,values):
            before=time.perf_counter()
            name=f"{first:06d}_{first+len(values['raw_indices']):06d}.npz"
            np.savez(out/name,**values)
            return name,time.perf_counter()-before
        futures=deque()
        records=[]
        written=0
        persistent_set=set(persistent)
        run_start=time.perf_counter()
        prev=run_start
        with model.create_stream() as stream, ThreadPoolExecutor(max_workers=1) as writer:
            for raw,tensor,service in ordered_load(paths,model.width,model.height,workers,capacity):
                before=time.perf_counter()
                pred=stream.online(tensor,persistent=raw in persistent_set)
                after=time.perf_counter()
                if pred is not None:
                    n=len(pred["poses_c2w"])
                    require(n==(model.config.scale_frames if written==0 else 1) and written+n==raw+1,"Bad output order")
                    values={k:np.array(v,copy=True) for k,v in pred.items()}
                    values["raw_indices"]=np.arange(written,written+n)
                    if len(futures)>=writes:
                        saved.append(futures.popleft().result())
                    futures.append(writer.submit(write,written,values))
                    written+=n
                now=time.perf_counter()
                records.append(dict(raw_index=raw,decode_s=service,wait_input_s=before-prev,
                                    stream_s=after-before,interval_s=now-prev))
                prev=now
                if (raw+1)%100==0:
                    print(f"{raw+1}/{len(paths)}",flush=True)
            saved.extend(f.result() for f in futures)
        torch.cuda.synchronize()
        run_seconds=time.perf_counter()-run_start
        require(written==len(paths),"Incomplete output")
        save(out/"manifest.json",dict(status="complete",image_paths=list(map(str,paths)),persistent_raw_ids=persistent,
             completed_outputs=[p for p,_ in saved],input_count=len(paths),input_wh=[model.width,model.height]))
        stable=[r["interval_s"] for r in records[model.config.scale_frames:]]
        total=time.perf_counter()-start
        timing=dict(identity_and_load_to_last_save_s=total,load_model_s=load_seconds,compile_warmup_s=compile_warmup,
                    streaming_through_final_save_s=run_seconds,end_to_end_fps=len(paths)/total,
                    stream_fps=len(paths)/run_seconds,stable_interarrival_p50_s=float(np.median(stable)),
                    stable_interarrival_p95_s=float(np.quantile(stable,.95)),save_service_s=sum(s for _,s in saved),
                    allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    upstream_historical_elapsed_s=read(Path(old["timing_file"]))["elapsed_s"])
        save(attempt/"timing.json",timing)
        save(attempt/"per_frame_timing.json",records)
        index=audit(out,SEQUENCE)
        save(attempt/"audit.json",index)
        diffs=compare(out,Path(old["audit"]["directory"]))
        save(OUT/"verified.json",dict(identity=identity,predictions=str(out),audit_manifest=str(attempt/"audit.json"),
             frames=written,persistent_frames=len(persistent),chunks=len(saved),field_differences=diffs,timing=timing))
    finally:
        stop.set()
        monitor.join()
        save(attempt/"resource_samples.json",resources)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--capacity",type=int,default=8)
    parser.add_argument("--writes",type=int,default=4)
    args=parser.parse_args()
    require(min(args.workers,args.capacity,args.writes)>0,"Nonpositive queue or thread count")
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"process.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(args.workers,args.capacity,args.writes)
