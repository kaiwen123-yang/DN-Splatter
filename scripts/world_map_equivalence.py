"""Audit ordered Map preprocessing and compare the identical initialization of archived streams on CPU."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from world_map_stream import HIST, OUT, P0, ROOT, SEQUENCE, SOURCE, image_paths, ordered_load
from lingbot_common import read, require, save, sha
from lingbot_map2.aggregator.base import join_images
from lingbot_map2.models.gcr_stream import GCRStream


def compare_tensor(lhs,rhs):
    a,b=lhs.detach().numpy(),rhs.detach().numpy()
    same_dtype=a.dtype==b.dtype==np.float32
    require(lhs.device.type==rhs.device.type=="cpu" and a.shape==b.shape and same_dtype,
            "CPU tensor shape or dtype differs")
    bits=a.view(np.int32)==b.view(np.int32)
    return dict(values=a.size,unequal_bits=int(a.size-bits.sum()),max_abs=float(np.abs(a-b).max(initial=0)),
                same_stride=lhs.stride()==rhs.stride(),same_storage_offset=lhs.storage_offset()==rhs.storage_offset(),
                shape=list(a.shape),stride=list(lhs.stride()),other_stride=list(rhs.stride()))


def digest_tensor(t):
    return hashlib.sha256(t.numpy().tobytes(order="C")).hexdigest()


def scan_inputs(out,metadata,paths):
    w,h=metadata["input_wh"]
    cpu=GCRStream(None,SimpleNamespace(width=w,height=h))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    prior_frames=[]
    queued_frames=[]
    drift=[]
    shape_stride=Counter()
    offsets=errors=total=0
    largest=0.
    checksum=hashlib.sha256()
    members=[]
    spec=read(OUT/"verified.json")["identity"]
    for index,prepared,_ in ordered_load(paths,w,h,spec["workers"],spec["decode_capacity"]):
        with torch.inference_mode():
            path_tensor=cpu.prepare(paths[index])
            queue_tensor=cpu.prepare(prepared)
        r=compare_tensor(path_tensor,queue_tensor)
        path_digest=digest_tensor(path_tensor)
        queue_digest=digest_tensor(queue_tensor)
        if r["unequal_bits"] or not r["same_stride"] or not r["same_storage_offset"]:
            drift.append(index)
        checksum.update(bytes.fromhex(path_digest)); checksum.update(bytes.fromhex(queue_digest))
        total+=r["values"]; errors+=r["unequal_bits"]; offsets+=not r["same_storage_offset"]
        largest=max(largest,r["max_abs"])
        shape_stride[(tuple(r["shape"]),tuple(r["stride"]),tuple(r["other_stride"]))]+=1
        members.append(dict(raw_index=index,id=paths[index].stem,official=path_digest,prefetch=queue_digest,
                            unequal_float32_bit_patterns=r["unequal_bits"],max_abs=r["max_abs"],
                            same_storage_offset=r["same_storage_offset"]))
        if index < 8:
            prior_frames.append(path_tensor); queued_frames.append(queue_tensor)
        if (index+1)%200==0:
            print("CPU tensor audit",index+1,"differing",len(drift),flush=True)
    with gzip.GzipFile(filename=str(out/"per_frame.jsonl.gz"),mode="wb",mtime=0) as f:
        for row in members:
            f.write((json.dumps(row,separators=(",",":"))+"\n").encode())
    with torch.inference_mode():
        joined=compare_tensor(join_images(prior_frames),join_images(queued_frames))
    return dict(frames=len(members),rgb_scalars=total,unequal_bit_patterns=errors,max_abs=largest,
                frames_differing=drift,storage_offset_mismatches=offsets,
                shape_and_strides=[dict(shape=s,official=a,prefetch=b,frames=n) for (s,a,b),n in shape_stride.items()],
                group_digest=checksum.hexdigest(),per_frame_sha256=sha(out/"per_frame.jsonl.gz"),
                prefix_join=joined),cpu,prior_frames


def sensitivity(cpu, paths, reference, cv_default):
    result=[]
    # Only the diagnostic process is affected by these local settings.
    for opencv,torch_threads in ((1,1),(1,3),(cv_default,3)):
        cv2.setNumThreads(opencv)
        torch.set_num_threads(torch_threads)
        for inference in (False,True):
            with torch.inference_mode(inference):
                rows=[compare_tensor(reference[i],cpu.prepare(p)) for i,p in enumerate(paths[:len(reference)])]
            result.append(dict(opencv_threads=opencv,torch_threads=torch_threads,inference_mode=inference,
                         n_frames=len(rows),values=sum(r["values"] for r in rows),
                         unequal_bits=sum(r["unequal_bits"] for r in rows),max_abs=max(r["max_abs"] for r in rows),
                         stride_mismatch=sum(not r["same_stride"] for r in rows)))
    return result


def prefix_stats(folders):
    names=list(folders)
    meta={k:read(p/"manifest.json") for k,p in folders.items()}
    first=meta[names[0]]["image_paths"][:8]
    for m in meta.values():
        require(m["image_paths"][:8]==first and m["persistent_raw_ids"][:8]==list(range(8)) and
                m["completed_outputs"][0]=="000000_000008.npz","Comparable eight-frame causal prefix")
    raw={k:{} for k in names}
    for k,p in folders.items():
        with np.load(p/"000000_000008.npz",allow_pickle=False) as z:
            for key in z.files:
                raw[k][key]=z[key]
    rows=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            fields={}
            for key,x in raw[a].items():
                y=raw[b][key]
                require(x.shape==y.shape and x.dtype==y.dtype and np.isfinite(x).all() and np.isfinite(y).all(),
                        "Prefix field shape, dtype or finiteness")
                diff=np.abs(x.astype(np.float64)-y.astype(np.float64))
                fields[key]=dict(elements=x.size,exact=int((x==y).sum()),mean_abs=float(diff.mean()),
                                 max_abs=float(diff.max()),per_frame_mean_abs=diff.reshape(8,-1).mean(1).tolist())
            rows.append(dict(a=a,b=b,fields=fields))
    return dict(ids=[Path(p).stem for p in first],common_inputs=8,
                first_chunk_sha256={k:sha(v/"000000_000008.npz") for k,v in folders.items()},
                pairs=rows,role="Identical first eight images and initialization decisions only; "
                "full-all contributes no subsequent frame or training prior.")


def main(out, cpu_budget):
    out.mkdir(parents=True,exist_ok=True)
    affinity=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,affinity[:cpu_budget])
    cv_default=cv2.getNumThreads()
    verified=read(OUT/"verified.json")
    current=Path(verified["predictions"])
    manifest=read(current/"manifest.json")
    paths=image_paths(None,str(P0/(SEQUENCE+".json")))
    require(list(map(str,paths))==manifest["image_paths"] and len(paths)==1478,"Strict train list")
    inputs,cpu,reference=scan_inputs(out,manifest,paths)
    response=sensitivity(cpu,paths,reference,cv_default)
    folders={"ordered_full_train":current}
    for name in ("full_train_1478","short_train_256","full_all_1846_exploratory"):
        d=read(HIST/"map"/name/"complete.json")
        require(d["identity"]["weights"]==verified["identity"]["weight"] and
                d["identity"]["upstream"]==verified["identity"]["source"],"Comparative model identity")
        folders["archived_"+name]=Path(d["audit"]["directory"])
    prefixes=prefix_stats(folders)
    save(out/"cpu_equivalence.json",dict(inputs=inputs,thread_and_inference_sensitivity=response,
         identical_prefix_outputs=prefixes,torch_version=torch.__version__,opencv_version=cv2.__version__,
         cpu_affinity_count=len(os.sched_getaffinity(0)),cpu_only=not torch.cuda.is_initialized(),
         upstream=verified["identity"]["source"],image_list_sha256=sha(P0/(SEQUENCE+".json")),
         compared_loader_implementation_sha256=sha(ROOT/"scripts/world_map_stream.py"),
         official_prepare_implementation_sha256=sha(SOURCE/"lingbot_map2/models/gcr_stream.py"),
         official_resize_implementation_sha256=sha(SOURCE/"lingbot_map2/utils/load_fn.py"),
         verification_script_sha256=sha(Path(__file__)),replay="existing causal prefix arrays only; no GPU model loaded"))
    print("CPU",inputs["frames"],"frames",inputs["rgb_scalars"],"scalars,",inputs["unequal_bit_patterns"],
          "bit mismatches, first 8 depth MAE vs archived full",prefixes["pairs"][0]["fields"]["depth"]["mean_abs"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT.parent/"equivalence")
    parser.add_argument("--cpu-budget",type=int,default=2)
    a=parser.parse_args()
    require(a.cpu_budget>0,"Positive CPU budget")
    main(a.out,a.cpu_budget)
