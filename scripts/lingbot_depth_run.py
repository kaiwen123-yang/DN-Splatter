"""Run the published v0.5 network on frozen, full-resolution metric hall inputs."""
import argparse
import os
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lingbot_common import P0, RUN, by_id, depth, member, read, require, rgb, save, sha


def main(weight):
    from lingbot_depth_backend import configure
    configure()
    from mdm.model.v2 import MDMModel
    from mdm.model.dinov2_rgbd.layers.block import XFORMERS_AVAILABLE
    require(XFORMERS_AVAILABLE, "Default nested-token path requires xFormers")
    freeze = sha(P0/"freeze.json")
    digest = sha(weight)
    require(digest=="b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23", "Wrong v0.5 checkpoint")
    identity = dict(freeze=freeze,weights=digest,upstream="f3a237e434ae987bc38281476d6cfb5df3e4d739",
                    wrapper=sha(Path(__file__)), batch_size=1,resolution_level=9,apply_mask=True,enable_depth_mask=True)
    output = RUN/"depth"
    output.mkdir(exist_ok=True)
    manifest, frames = by_id()
    sel = read(P0/"selection.json")
    torch.manual_seed(42)
    cv2.setNumThreads(1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    model = MDMModel.from_pretrained(weight).to("cuda").eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter()-started
    k=np.array(manifest["K_cv"],np.float32)
    k[0]/=manifest["image_wh"][0]
    k[1]/=manifest["image_wh"][1]
    k_gpu=torch.from_numpy(k).unsqueeze(0).to("cuda")
    samples=[s["id"] for s in sel["samples"]]
    queue=[(fid,tag) for fid in samples for tag in ("unmasked","random_points","blocks")]
    queue.extend((fid,"unmasked") for fid in sel["sequences"]["full_train_1478"]["ids"] if fid not in samples)
    ledger=[]
    with torch.inference_mode():
        for fid,tag in queue:
            row=frames[fid]
            out=output/tag/fid
            receipt=out/"receipt.json"
            if receipt.is_file():
                r=read(receipt)
                require(r["identity"]==identity and all(sha(out/v["name"])==v["sha256"] for v in r["outputs"]),
                        "Depth resume identity "+fid)
                ledger.append(r)
                continue
            t0=time.perf_counter()
            image=cv2.cvtColor(rgb(row),cv2.COLOR_BGR2RGB)
            z=depth(row)
            if tag!="unmasked":
                with np.load(P0/"masks"/(fid+".npz")) as m: z[m[tag+"_input"]]=0
            t1=time.perf_counter()
            image_t=torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to("cuda",dtype=torch.float32)/255
            z_t=torch.from_numpy(z).unsqueeze(0).to("cuda",dtype=torch.float32)
            torch.cuda.synchronize()
            t2=time.perf_counter()
            pred=model.infer(image_t,depth_in=z_t,intrinsics=k_gpu,resolution_level=9,apply_mask=True)
            torch.cuda.synchronize()
            t3=time.perf_counter()
            result=pred["depth"][0].float().cpu().numpy()
            finite=np.isfinite(result)&(result>0)
            explicit="mask" in pred
            model_mask=pred["mask"][0].cpu().numpy().astype(bool) if explicit else None
            torch.cuda.synchronize()
            t4=time.perf_counter()
            require(result.shape==(720,1280) and (not explicit or model_mask.shape==result.shape), "Depth shape")
            out.mkdir(parents=True,exist_ok=True)
            np.save(out/"depth_refined_m.npy",result,allow_pickle=False)
            masks={"finite_positive":finite}
            if explicit: masks["model_mask"]=model_mask
            np.savez_compressed(out/"validity.npz",**masks)
            files=[]
            for name in ("depth_refined_m.npy","validity.npz"):
                v=member(out/name)
                files.append(dict(name=name,sha256=v["sha256"],bytes=v["bytes"]))
            end=time.perf_counter()
            r=dict(id=fid,mode=tag,identity=identity,outputs=files,model_mask_returned=explicit,
                   output_valid=int((finite & model_mask if explicit else finite).sum()),
                   loading_s=t1-t0,h2d_s=t2-t1,gpu_s=t3-t2,d2h_s=t4-t3,
                   save_hash_s=end-t4,end_to_end_s=end-t0,pid=os.getpid())
            save(receipt,r)
            ledger.append(r)
            if len(ledger)%30==0 or len(ledger)==1:
                save(output/"progress.json",dict(pid=os.getpid(),completed=len(ledger),total=len(queue),last=[fid,tag]))
                print("depth",len(ledger),"/",len(queue),fid,tag,flush=True)
            del pred, image_t, z_t
    save(output/"manifest.json",dict(identity=identity,records=ledger,unmasked=len({r["id"] for r in ledger if r["mode"]=="unmasked"}),
         masked=len([r for r in ledger if r["mode"]!="unmasked"]),model_load_s=load_s,num_tokens=int(model.num_tokens_range[-1]),
         peak_vram_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("weight",type=Path)
    main(p.parse_args().weight)
