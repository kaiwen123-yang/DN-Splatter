"""Execute each official Map CLI stream in a fresh process and audit its raw IDs."""
import argparse
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import textwrap
import time
from unittest.mock import patch

import numpy as np

from lingbot_common import P0, RUN, SEQ, member, read, require, save, sha


def audit(directory, sequence):
    metadata=read(directory/"manifest.json")
    plan=read(P0/"selection.json")["sequences"][sequence]
    files=metadata["completed_outputs"]
    require(metadata["status"]=="complete" and metadata["image_paths"]==read(P0/(sequence+".json")),
            "Stream did not complete on the frozen list")
    require(metadata["persistent_raw_ids"]==plan["persistent_raw_ids"], "Keyframe plan changed")
    n=len(plan["ids"])
    indexed={}
    chunks=[]
    for name in files:
        with np.load(directory/name,allow_pickle=False) as data:
            ids=data["raw_indices"]
            require(all(len(data[k])==len(ids) for k in ("poses_c2w","intrinsics","depth","depth_conf","frame_indices")),
                    "Chunk cardinality "+name)
            for offset,raw in enumerate(ids):
                i=int(raw)
                require(0<=i<n and i not in indexed,"Duplicate or out-of-bounds raw index")
                require(data["depth"][offset].squeeze().shape==(288,512),"Map depth canvas")
                indexed[i]=dict(id=plan["ids"][i],raw_index=i,frame_index=int(data["frame_indices"][offset]),
                                chunk=name,offset=offset)
        chunks.append(member(directory/name))
    require(set(indexed)==set(range(n)),"Missing Map raw indices")
    return dict(sequence=sequence,directory=str(directory),input_frames=n,predicted_frames=len(indexed),
                persistent_frames=len(plan["persistent_raw_ids"]),npz_files=len(chunks),missing=[],duplicates=[],
                frames=[indexed[i] for i in range(n)],chunks=chunks)


def instrumented_cli(source,weight,config,sequence,attempt):
    """Timing hooks wrap two tensor copies. All model/cache/prediction statements stay upstream."""
    import torch
    from lingbot_map2.aggregator import stream
    from lingbot_map2.models.gcr_stream import GCRStream
    from lingbot_map2.utils import cli
    stats={}
    rows=[]
    active=False
    current={}
    sync=torch.cuda.synchronize
    def timed(fn,key,target):
        def call(*args,**kwargs):
            sync()
            start=time.perf_counter()
            result=fn(*args,**kwargs)
            sync()
            dest=target()
            dest[key]=dest.get(key,0.)+time.perf_counter()-start
            return result
        return call
    def host_to_device(image,device):
        return timed(lambda: image.detach().to(device,copy=True),"h2d_s",lambda:current)()
    def device_to_host(items):
        return timed(lambda:{key:value.cpu().numpy() for key,value in items.items()},"d2h_s",lambda:current)()
    def replace(fn,old,new,extra):
        code=textwrap.dedent(inspect.getsource(fn))
        require(code.count(old)==1,"Pinned Map timing hook has changed")
        scope={**fn.__globals__,**extra}
        exec(compile(code.replace(old,new),inspect.getsourcefile(fn),"exec"),scope)
        return scope[fn.__name__]
    decode=replace(stream.decode_predictions,"{key: value.cpu().numpy() for key, value in result.items()}",
                   "device_to_host(result)",{"device_to_host":device_to_host})
    online_inner=replace(stream.Stream._online,"image.detach().to(self.model.device, copy=True)",
                         "host_to_device(image,self.model.device)",{"host_to_device":host_to_device,"decode_predictions":decode})
    original=stream.Stream.online
    prepare=GCRStream.prepare
    def measure_prepare(self,*args,**kwargs):
        return timed(prepare,"input_s",lambda:current)(self,*args,**kwargs) if active else prepare(self,*args,**kwargs)
    def online(self,frame,**kwargs):
        nonlocal current,active
        if not active:
            return original(self,frame,**kwargs)
        current=dict(raw_index=len(rows))
        result=timed(original,"online_s",lambda:current)(self,frame,**kwargs)
        current["model_step_s"]=max(0.,current["online_s"]-sum(current.get(k,0.) for k in ("input_s","h2d_s","d2h_s")))
        current["returned_frames"]=0 if result is None else len(result["poses_c2w"])
        rows.append(current)
        return result
    warmup=GCRStream.warmup
    def measure_warmup(self,*args,**kwargs):
        nonlocal active
        result=timed(warmup,"warmup_compile_s",lambda:stats)(self,*args,**kwargs)
        active=True
        return result
    out=attempt/"predictions"
    arguments=[str(source/"infer.py"),"--checkpoint",str(weight),"--config",str(config),
               "--image-list",str(P0/(sequence+".json")),"--output",str(out)]
    save(attempt/"command.json",dict(argv=arguments,pid=os.getpid(),start=datetime.now(timezone.utc).isoformat()))
    start=time.perf_counter()
    def save_timing():
        stats.update(per_call=rows,elapsed_s=time.perf_counter()-start,peak_vram_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save(attempt/"timing.json",stats)
    try:
        with patch.object(cli,"load_model",timed(cli.load_model,"load_s",lambda:stats)), \
             patch.object(GCRStream,"warmup",measure_warmup), patch.object(GCRStream,"prepare",measure_prepare), \
             patch.object(stream.Stream,"_online",online_inner), patch.object(stream.Stream,"online",online), \
             patch.object(np,"savez",timed(np.savez,"save_s",lambda:current)), patch.object(sys,"argv",arguments):
            runpy.run_path(str(source/"infer.py"),run_name="__main__")
    finally:
        save_timing()


def run(source,weights):
    weight=weights/"model_weights.pt"
    config=weights/"config.json"
    digest=sha(weight)
    require(digest==read(config)["sha256"]=="d3ae62a09d3caaeb0947c4372f6fb23a4c0e85304867fa9702948ba030310e27",
            "Map release digest")
    for seq in SEQ:
        folder=RUN/"map"/seq
        folder.mkdir(exist_ok=True,parents=True)
        identity=dict(freeze=sha(P0/"freeze.json"),weights=digest,config=sha(config),image_list=sha(P0/(seq+".json")),
                      upstream="078b54485830c74ab1d3797667073e3e8353f3b9",compiled=True)
        receipt=folder/"complete.json"
        if receipt.is_file():
            status=read(receipt)
            require(status["identity"]==identity,"Map resume identity changed")
            require(audit(Path(status["audit"]["directory"]),seq)==status["audit"],"Saved NPZ changed")
            continue
        attempt=folder/("attempt-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        attempt.mkdir()
        save(attempt/"identity.json",identity)
        cmd=[sys.executable,__file__,str(source),str(weights),"--sequence",seq,"--attempt",str(attempt)]
        result=subprocess.run(cmd,env={**os.environ,"PYTHONPATH":str(source)+os.pathsep+os.environ.get("PYTHONPATH","")})
        save(attempt/"exit.json",dict(returncode=result.returncode))
        result.check_returncode()
        save(receipt,dict(identity=identity,audit=audit(attempt/"predictions",seq),timing_file=str(attempt/"timing.json")))
        print("Verified Map",seq,flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("source",type=Path)
    arg.add_argument("weights",type=Path)
    arg.add_argument("--sequence",choices=SEQ)
    arg.add_argument("--attempt",type=Path)
    a=arg.parse_args()
    if a.sequence:
        instrumented_cli(a.source.resolve(),a.weights/"model_weights.pt",a.weights/"config.json",a.sequence,a.attempt)
    else:
        run(a.source.resolve(),a.weights.resolve())
