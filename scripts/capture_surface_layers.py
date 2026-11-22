"""Certify source-supported local layers and select their owners in the shared 1 cm grid."""
from collections import defaultdict
import os
import time

import numpy as np

from capture_surface_device import Evidence
from capture_surface_tiles import ROOT, RUN, key, ply, read, sha, write


def seven(t):
    a,b,c=t[:,0],t[:,1],t[:,2]
    return np.stack((a,b,c,(a+b)/2,(b+c)/2,(c+a)/2,t.mean(axis=1)),axis=1)


def children(t):
    a,b,c,ab,bc,ca,_=seven(t).transpose(1,0,2)
    return np.stack([np.stack(x,axis=1) for x in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]],axis=1).reshape(-1,3,3)


def normal_area(t):
    n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    length=np.linalg.norm(n,axis=1)
    return n/np.maximum(length[:,None],1.e-30), length/2


def certify(e, triangles, group, original, stats, split=True):
    samples=seven(triangles)
    all_points=samples.reshape(-1,3)
    # Local TSDF triangles share vertices and edge midpoints. Each unique (position,source) is voted once.
    records=np.empty(len(all_points),dtype=[("g","i4"),("xyz","f8",(3,))])
    records["g"]=np.repeat(group,7);records["xyz"]=all_points
    unique,inv=np.unique(records,return_inverse=True);inv=inv.reshape(-1,7)
    votes,err,w=e.vote(unique["xyz"],unique["g"])
    stats["votes_scored_unique"]+=len(unique)
    stats["vote_class_counts"]+=votes.sum(axis=0)
    passed=(votes[:,0]>=2)&(votes[:,0]>votes[:,1])
    ok=passed[inv]
    n,area=normal_area(triangles)
    all_ok=ok.all(axis=1)&(area>1e-16)
    valid=inv[all_ok]
    parts=[(triangles[all_ok],group[all_ok],original[all_ok],votes[valid,0].min(axis=1),
           err[valid].mean(axis=1),w[valid[:,-1]])]
    if split:
        mixed=ok.any(axis=1)&~ok.all(axis=1)&(area>1e-16)
        stats["boundary_parents"]+=int(mixed.sum())
        if mixed.any():
            parts+=certify(e,children(triangles[mixed]),np.repeat(group[mixed],4),np.repeat(original[mixed],4),stats,False)
    return parts


def own(e, parts):
    t,g,ids,s,res,w=[np.concatenate(x) for x in zip(*parts)]
    if not len(t):return t,g,ids,w,0.
    mid=t.mean(axis=1)
    cells=key(np.floor(mid/.01))
    normals,area=normal_area(t)
    order=np.lexsort((ids,g,res,-s,cells))
    args=[np.ascontiguousarray(x[order]) for x in (cells,mid,normals,g)]
    keep=np.zeros(len(t),"u1")
    e.lib.choose_owners(len(t),*[a.ctypes.data for a in args],keep.ctypes.data)
    chosen=order[keep.astype(bool)]
    return t[chosen],g[chosen],ids[chosen],w[chosen],float(area[chosen].sum())


def certify_branch(label,branch):
    export=RUN/"export"/label
    identity=sha(export/"identity.json")
    ledger=read(export/"tiles/complete.json")
    if ledger["identity"]!=identity:raise ValueError("Tile ledger identity")
    out=export/("layers_"+branch);out.mkdir(exist_ok=True)
    evidence=Evidence(export)
    code=[sha(__file__),sha(ROOT/"scripts/capture_surface_device.py"),sha(ROOT/"scripts/capture_surface_evidence.cu")]
    signature=dict(camera_identity=identity,tolerance=sha(export/"tolerances.json"),source_triangles=ledger["triangles"][branch],
                   implementation=code,method="1cm cell; signed normal <=30 deg, reciprocal plane offset <=5mm; min 7-site support, mean residual, gid; mixed boundaries split into 4, recheck all seven")
    if (out/"identity.json").exists() and read(out/"identity.json")!=signature:raise ValueError("Existing layer identity mismatch")
    write(out/"identity.json",signature)
    tiles=defaultdict(list); sources=[]
    for g in range(ledger["groups"]):
        d=export/"tiles"/branch/f"{g:04d}";idx=read(d/"tiles.json")
        v,f=ply(ROOT/idx["source"]["path"]);order=np.load(d/"faces.npy",mmap_mode="r")
        sources.append((v,f,order))
        for k,a,b in idx["tiles"]:tiles[k].append((g,a,b))
    keys=sorted(tiles)
    total_faces=done_tiles=owned=0; receipts=[]
    # Shards limit open files while keeping interrupted work bounded.
    for start in range(0,len(keys),32):
        target=out/f"shard-{start:07d}.npz";receipt=target.with_suffix(".json")
        if receipt.is_file():
            row=read(receipt)
            if row["sha256"]!=sha(target) or row["identity"]!=sha(out/"identity.json"):raise ValueError("Layer shard hash")
        else:
            stats=dict(input_faces=0,boundary_parents=0,votes_scored_unique=0,vote_class_counts=np.zeros(4,"i8"))
            saved=[];area=0.
            for k in keys[start:start+32]:
                parts=[]
                for g,a,b in tiles[k]:
                    v,f,order=sources[g]
                    for first in range(a,b,16000):
                        ids=np.asarray(order[first:min(b,first+16000)])
                        t=np.asarray(v[f[ids]],dtype="f8");stats["input_faces"]+=len(ids)
                        parts+=certify(evidence,t,np.full(len(t),g,"i4"),ids,stats)
                t,g,ids,w,subarea=own(evidence,parts);area+=subarea
                saved.append((t,g,ids,w))
            t,g,ids,w=[np.concatenate(a) for a in zip(*saved)]
            tmp=target.with_suffix(".partial.npz")
            np.savez(tmp,triangles=t,source_group=g,source_face=ids,witness=w)
            tmp.replace(target)
            stats["vote_class_counts"]=stats["vote_class_counts"].tolist()
            row=dict(**stats,tiles=len(saved),owned_faces=len(t),owned_area_m2=area,sha256=sha(target),identity=sha(out/"identity.json"))
            write(receipt,row)
        receipts.append(receipt.name)
        total_faces+=row["input_faces"];done_tiles+=row["tiles"];owned+=row["owned_faces"]
        progress=dict(run_id=RUN.name,phase="A3_cross_view_fusion",branch=branch,pid=os.getpid(),utc_epoch=time.time(),
                     tiles=done_tiles,total_tiles=len(keys),source_faces=total_faces,total_source_faces=signature["source_triangles"],owned_faces=owned)
        write(RUN/"state"/"A3_fusion_progress.json",progress)
        if len(receipts)%10==1:print(progress,flush=True)
    if total_faces!=ledger["triangles"][branch]:raise ValueError("Incomplete face coverage")
    write(out/"complete.json",dict(ledger=receipts,source_faces=total_faces,owned_faces=owned,identity=sha(out/"identity.json")))


if __name__=="__main__":
    import sys
    if len(sys.argv)!=3 or sys.argv[1] not in ("B2","B3") or sys.argv[2] not in ("model","sensor"):raise SystemExit("usage: MODEL BRANCH")
    certify_branch(*sys.argv[1:])
