"""Fresh official LingBot depth/feature calls for each frozen RGB-D condition."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from interactive_depth_candidates import HallFrame, condition_z, export_selected, finalize
from interactive_depth_geometry import Footprint, atomic_npz
from interactive_depth_inputs import Hall, MODEL_REV, MODEL_SHA, OLD, REPS, RUN, digest_json
from lingbot_common import ROOT, read, require, save, sha


class FrozenMDM:
    def __init__(self, weight, upstream=None):
        import torch
        from lingbot_depth_backend import configure
        upstream = Path(upstream or ROOT/".local/upstream/lingbot-depth").resolve()
        require(subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()==MODEL_REV,
                "MDM source revision")
        tracked = subprocess.check_output(["git","-C",str(upstream),"status","--porcelain","--untracked-files=no"],text=True)
        require(not tracked.strip(),"Changed tracked MDM source")
        sys.path.insert(0,str(upstream))
        configure()
        from mdm.model.v2 import MDMModel
        from mdm.model.dinov2_rgbd.layers.block import XFORMERS_AVAILABLE
        require(XFORMERS_AVAILABLE and sha(weight)==MODEL_SHA,"MDM backend or checkpoint mismatch")
        torch.manual_seed(42)
        self.model=MDMModel.from_pretrained(str(weight)).to("cuda").eval().requires_grad_(False)
        self.tokens=int(self.model.num_tokens_range[-1])
        self.calls=0

    def predict(self, rgb_uint8, visible_z_m, K_cv):
        import torch
        h,w=visible_z_m.shape
        require(rgb_uint8.shape==(h,w,3) and rgb_uint8.dtype==np.uint8 and
                np.isfinite(visible_z_m).all() and (visible_z_m>=0).all(), "RGB/depth condition")
        k=np.asarray(K_cv,np.float32).copy()
        k[0]/=w
        k[1]/=h
        rgb=torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(2,0,1)[None].to("cuda",torch.float32)/255.
        z=torch.from_numpy(np.array(visible_z_m,dtype=np.float32,order="C",copy=True))[None].to("cuda")
        K=torch.from_numpy(k)[None].to("cuda")
        torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            depth=self.model.infer(rgb,depth_in=z,intrinsics=K,resolution_level=9,apply_mask=True)["depth"]
            torch.cuda.synchronize()
            half=time.perf_counter()
            features,cls=self.model.infer_feat(rgb,depth_in=z,intrinsics=K,resolution_level=9,apply_mask=True)
            ht=round((self.tokens*h/w)**.5)
            wt=round((self.tokens*w/h)**.5)
            require(depth.shape==(1,h,w) and features.ndim==4 and features.shape[0]==1 and
                    tuple(features.shape[-2:])==(ht,wt) and cls.ndim==2 and cls.shape[0]==1,"Official spatial API changed")
            d=depth[0].float().cpu().numpy().copy()
            s=features[0].float().cpu().numpy().copy()
            torch.cuda.synchronize()
        self.calls+=1
        return d,s,dict(image_hw=[h,w],normalized_intrinsics=k.tolist(),token_hw=[ht,wt],
                        encoder_rgb_resize_hw=[ht*14,wt*14], spatial_shape=list(features.shape),
                        cls_token_shape=list(cls.shape), feature_dtype=str(features.dtype), apply_mask=True,
                        depth_s=half-start, feature_and_return_s=time.perf_counter()-half,
                        pixel_to_token="floor((pixel+.5)*token_dim/image_dim); explicit patch containment",
                        allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved())


def patch_ids(flat, shape, grid):
    y,x=np.divmod(flat,shape[1])
    ty=np.floor((y+.5)*grid[0]/shape[0]).astype(np.int32)
    tx=np.floor((x+.5)*grid[1]/shape[1]).astype(np.int32)
    return np.minimum(ty,grid[0]-1)*grid[1]+np.minimum(tx,grid[1]-1)


def token_file(depth_m, spatial, flat):
    actual=patch_ids(flat,depth_m.shape,spatial.shape[-2:])
    which,indices=np.unique(actual,return_inverse=True)
    tokens=spatial.reshape(spatial.shape[0],-1)[:,which].T.copy()
    words=tokens.view(np.uint32)
    if not (words & np.uint32(65535)).any():
        # Lossless packed bfloat16; keep float32 if this backend produces it.
        tokens=(words >> 16).astype(np.uint16)
    require(np.isfinite(spatial).all(),"Nonfinite frozen local tokens")
    return dict(mdm_depth=depth_m.ravel()[flat], token_grid_ids=which, token_for_pixel=indices.astype(np.uint16),
                token_values=tokens,token_hw=np.asarray(spatial.shape[-2:],np.int16))


def unpack(array):
    if array.dtype==np.uint16:
        return (array.astype(np.uint32)<<16).view(np.float32)
    require(array.dtype==np.float32,"Unknown spatial token dtype")
    return array


def stratified(reference, side, seed, cap=4096):
    if len(reference)<=cap:
        return np.arange(len(reference),dtype=np.int32)
    rng=np.random.default_rng(seed)
    strata=3*np.searchsorted([1.,2.,4.],reference)+side
    groups=[rng.permutation(np.flatnonzero(strata==s)) for s in np.unique(strata)]
    # Equal shares first, redistribute unused slots among nonempty strata.
    sizes=np.zeros(len(groups),int)
    while sizes.sum()<cap:
        active=np.flatnonzero([n<len(g) for n,g in zip(sizes,groups)])
        left=cap-int(sizes.sum())
        share=max(1,left//len(active))
        for a in active:
            take=min(share,len(groups[a])-sizes[a],left)
            sizes[a]+=take
            left-=take
            if not left:
                break
    return np.sort(np.concatenate([g[:n] for g,n in zip(groups,sizes)])).astype(np.int32)


def artifact(path):
    return dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path))


def condition_cache(out,frame,kind,reps,model,binding):
    target=out/"cases"/frame.row["id"]/kind
    receipt=target/"receipt.json"
    if receipt.exists():
        r=read(receipt)
        require(r["identity"]==binding and all((target/f["path"]).stat().st_size==f["bytes"] and
                sha(target/f["path"])==f["sha256"] for f in r["outputs"]),"Condition resume identity")
        return r
    data=frame.condition(None if kind=="natural" else kind)
    flat=np.flatnonzero(data["score"]).astype(np.int32)
    fid=frame.row["id"]
    files=[]
    if kind!="natural":
        d,obs,local=data["geometry"].table(flat)
        reference=frame.full.ravel()[flat]
        side_map=np.zeros(frame.full.shape,np.uint8)
        for fg,bg,_ in frame.pairs:
            side_map.flat[bg]=2
            side_map.flat[fg]=1
        side=side_map.ravel()[flat]
        seed=np.random.SeedSequence((42,int(fid.split("_")[-1]),frame.hall.freeze["masks"].index(kind)))
        ids=stratified(reference,side,seed)
        rng=np.random.default_rng(seed.spawn(1)[0])
        rep_for_sample=np.empty(len(ids),np.uint8)
        rep_for_sample[rng.permutation(len(ids))]=np.arange(len(ids))%len(reps)
        nn_parent=data["geometry"].nn_src.ravel()[flat]
        parent=np.column_stack((np.where(nn_parent>=0,data["winner"].ravel()[np.maximum(nn_parent,0)],-1),
                data["foot_src"].ravel()[flat],frame.other.near_id.ravel()[flat],frame.other.far_id.ravel()[flat]))
        path=target/"source_and_score.npz"
        atomic_npz(path,flat=flat,reference=reference,edge_side=side,sample_index=ids,
                   sample_rep=rep_for_sample,depths=d[:,:4],local=local,observations=obs,
                   parent=parent,near_frame=frame.other.near_frame.ravel()[flat],far_frame=frame.other.far_frame.ravel()[flat],
                   audit_bits=np.asarray(frame.other.audit_bits,np.uint8),donor_groups=frame.other.donor_groups,
                   hidden_input=np.packbits(data["hidden"],axis=None),
                   hidden_native=np.packbits(data["native_hide"],axis=None) if data["native_hide"] is not None else np.empty(0,np.uint8))
        files.append(artifact(path))
    meta=[]
    for rep in reps:
        visible=condition_z(data,rep)
        require(not visible[data["hidden"]].any(), "Structured input mask not completely hidden")
        signature=hashlib.sha256(np.ascontiguousarray(visible).view(np.uint8)).hexdigest()
        dep,spatial,measured=model.predict(frame.rgb,visible,frame.hall.k)
        path=target/(rep+".npz")
        arrays=token_file(dep,spatial,flat)
        atomic_npz(path,**arrays)
        files.append(artifact(path))
        meta.append(dict(representation=rep,visible_z_sha256=signature,visible_pixels=int((visible>0).sum()),
                         actual_depth_forwards=1,actual_feature_forwards=1,**measured))
        if kind=="natural":
            supervision=export_selected(out/"supervision/R2",frame,data,dep)
    r=dict(identity=binding,frame=fid,kind=kind,score_pixels=len(flat),group=frame.row["group"],fold=frame.row["fold"],
           hidden_canvas=int(data["hidden"].sum()),hidden_native=int(data["native_hide"].sum()) if data["native_hide"] is not None else None,
           representations=meta,outputs=files,source_image_sha256=frame.row["rgb_sha256"],
           source_raw_sha256=frame.row["depth_sha256"], source_native_sha256=frame.row.get("native_sha256"),
           source_donors=frame.row["donors"],depth_model_sha256=MODEL_SHA,source_revision=MODEL_REV)
    if kind=="natural":
        r["supervision"]=supervision
    save(receipt,r)
    return r


def run(args):
    cv2.setNumThreads(1)
    os.environ.setdefault("OMP_NUM_THREADS","1")
    hall=Hall(args.out)
    fp=Footprint(hall)
    model=FrozenMDM(args.weight,args.upstream)
    binding=digest_json(dict(freeze=sha(args.out/"freeze.json"),model=MODEL_SHA,revision=MODEL_REV,
            implementation=[sha(Path(__file__).with_name(n)) for n in ("interactive_depth_geometry.py",
                 "interactive_depth_candidates.py","interactive_depth_mdm.py")]))
    rows=[]
    reps=hall.freeze["representations"]
    use_rep="footprint_aware" if "footprint_aware" in reps else "full_range_sparse"
    kinds=["natural"] if args.stage=="natural" else hall.freeze["masks"]
    for row in hall.rows:
        frame=HallFrame(hall,fp,row)
        for kind in kinds:
            r=condition_cache(args.out,frame,kind,[use_rep] if kind=="natural" else reps,model,binding)
            if kind=="natural":
                p=args.out/"supervision/R2"/r["supervision"]["path"]
                require(sha(p)==r["supervision"]["sha256"],"R2 supervision identity")
            rows.append(r)
        if len(rows)==len(kinds) or len(rows)%(20*len(kinds))==0:
            print(args.stage,len(rows),"cases,",sum(len(r["representations"]) for r in rows),"conditions",row["id"],flush=True)
    if args.stage=="natural":
        finalize(args.out,[dict(identity=binding,files={"supervision/R2":r["supervision"]}) for r in rows],hall.freeze)
    save(args.out/(args.stage+"_manifest.json"),dict(identity=binding, model_sha256=MODEL_SHA,frames=len(hall.rows),
         cases=len(rows),model_conditions=sum(len(r["representations"]) for r in rows),
         model_calls_this_process=model.calls,records=[dict(frame=r["frame"],kind=r["kind"],score_pixels=r["score_pixels"],
              receipt_sha256=sha(args.out/"cases"/r["frame"]/r["kind"]/"receipt.json")) for r in rows]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["natural","masked"])
    p.add_argument("--out",type=Path,default=RUN)
    p.add_argument("--upstream",type=Path)
    p.add_argument("--weight",type=Path,default=OLD/"weights/depth/model.pt")
    run(p.parse_args())
