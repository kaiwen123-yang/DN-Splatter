"""Propose train-only RGB/time edges and audit both pose initializations on them."""
import argparse
from functools import lru_cache
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from scipy.optimize import least_squares

from lingbot_common import ROOT, RUN as HIST, P0, SRC, by_id, depth, read, require, save, sha


WORLD=ROOT/".local/runs/interactive-world-20261128-v1"
OUT=WORLD/"b_map/pose"
MATCH=WORLD/"b_map/matching"
SPECIAL={"frame_01304","frame_01305","frame_01514"}
def frozen():
    data=read(WORLD/"a_depth/freeze.json")
    rows=data["rows"]
    require(len(rows)==1478 and all(r["split"]=="train" for r in rows),"Strict train set required")
    return data,rows


def edges():
    data,rows=frozen()
    keys=[r["id"] for r in rows]
    descriptors=[]
    for r in rows:
        im=cv2.imread(str(SRC/r["rgb"]),cv2.IMREAD_COLOR)
        require(im is not None,r["id"])
        small=cv2.resize(im,(16,12),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        vec=np.r_[small.ravel()-.5,np.diff(gray,axis=1).ravel(),np.diff(gray,axis=0).ravel()]
        descriptors.append(vec/max(1e-9,np.linalg.norm(vec)))
    desc=np.stack(descriptors)
    sim=desc@desc.T
    propose={}
    for i,r in enumerate(rows):
        roles={"fit":[],"audit":[],"temporal":[]}
        number=8 if r["id"] in SPECIAL else 3
        for order in (-1,1):
            j=i+order*4
            if 0<=j<len(rows) and rows[j]["depth_sha256"]!=r["depth_sha256"]:
                roles["temporal"].append(keys[j])
        groups=set()
        times=r["rgb_timestamp_ns"]
        for j in np.argsort(-sim[i],kind="stable"):
            other=rows[j]
            g=other["group"]
            role="fit" if other["fold"] in [0,2,4] else "audit"
            if (keys[j] in SPECIAL or len(roles[role])>=number or g in groups or g==r["group"] or
                abs(times-other["rgb_timestamp_ns"])<1_000_000_000 or
                other["depth_sha256"]==r["depth_sha256"] or other["rgb_sha256"]==r["rgb_sha256"]):
                continue
            roles[role].append(keys[j])
            groups.add(g)
            if min(len(roles["fit"]),len(roles["audit"]))==number:
                break
        propose[r["id"]]=roles
    save(OUT/"edges.json",dict(freeze=sha(WORLD/"a_depth/freeze.json"),ids=keys,targets=propose,
        retrieval="normalized 16x12 RGB and gradients; pose-independent. Fit/audit anchors have disjoint source folds and time groups. Temporal edges are diagnostic."))


def packet_path(i,j):
    return OUT/"matches"/(min(i,j)+"_"+max(i,j)+".npz")


def matching():
    import torch
    lib=MATCH/"lib/mast3r"
    sys.path.insert(0,str(lib))
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.utils.image import load_images
    from world_match_sources import check
    _,rows=frozen()
    frames={r["id"]:r for r in rows}
    plan=read(OUT/"edges.json")
    require(plan["freeze"]==sha(WORLD/"a_depth/freeze.json"),"Frozen inputs changed")
    for filename in ("config.json","model.safetensors"):
        check(MATCH/"weights"/filename,filename)
    pairs=sorted({tuple(sorted([a,b])) for a,d in plan["targets"].items() for vs in d.values() for b in vs},
                 key=lambda x:(not bool(set(x)&SPECIAL),x))
    identity=dict(source="f5209afc300cec36239a7ac992263f36847bbba0",edges=sha(OUT/"edges.json"),
                  release="06e7259f34c3060f322df5cb0c7b9094f57e41fc",size=512,subsample=8,method="reciprocal dot")
    stored=read(OUT/"matcher.json") if (OUT/"matcher.json").is_file() else None
    require(stored in (None,identity),"Matcher identity changed; preserve the earlier attempt")
    save(OUT/"matcher.json",identity)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    net=AsymmetricMASt3R.from_pretrained(str(MATCH/"weights"),local_files_only=True).to("cuda").eval()
    @lru_cache(maxsize=24)
    def image(name):
        d=load_images([str(SRC/frames[name]["rgb"])],size=512,verbose=False)[0]
        require(list(d["img"].shape[-2:])==[288,512],"Unexpected crop, inverse geometry must be rederived")
        return dict(img=d["img"].to("cuda"),true_shape=torch.as_tensor(d["true_shape"],device="cuda"),instance=[name])
    (OUT/"matches").mkdir(exist_ok=True)
    total=0
    with torch.inference_mode():
        for n,(i,j) in enumerate(pairs):
            path=packet_path(i,j)
            if not path.exists():
                a,b=net(image(i),image(j))
                da,db=a["desc"][0],b["desc"][0]
                u,v=fast_reciprocal_NNs(da,db,subsample_or_initxy1=8,device="cuda",dist="dot",block_size=4096)
                u,v=np.asarray(u),np.asarray(v)
                good=((u>=3)&(u<np.array([509,285]))&(v>=3)&(v<np.array([509,285]))).all(axis=1)
                u,v=u[good],v[good]
                score=(da[u[:,1],u[:,0]]*db[v[:,1],v[:,0]]).sum(-1).cpu().numpy()
                with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
                    np.savez_compressed(tmp,u=u.astype(np.int16),v=v.astype(np.int16),score=score)
                os.replace(tmp.name,path)
            with np.load(path,allow_pickle=False) as pack:
                require(pack["u"].shape==pack["v"].shape==(len(pack["score"]),2),"Invalid match file")
                total+=len(pack["score"])
            if (n+1)%50==0:
                save(OUT/"match_progress.json",dict(completed_pairs=n+1,pairs=len(pairs),reciprocal_matches=total))
                print(f"Matched {n+1}/{len(pairs)}",flush=True)
    save(OUT/"matches_complete.json",dict(pairs=len(pairs),reciprocal_matches=total,targets=len(frames),identity=identity))


def metric_depth(z,uv):
    x,y=np.rint(uv).astype(int).T
    dy,dx=np.mgrid[-2:3,-2:3]
    tile=z[y[:,None]+dy.ravel(),x[:,None]+dx.ravel()].copy()
    tile[(~np.isfinite(tile))|(tile<.3)|(tile>6)]=np.nan
    valid=np.isfinite(tile).sum(1)>=4
    out=np.full(len(uv),np.nan)
    if valid.any():
        block=tile[valid]
        mid=np.nanmedian(block,axis=1)
        ok=(np.nanmax(block,axis=1)-np.nanmin(block,axis=1))<=np.maximum(.05,.03*mid)
        out[np.flatnonzero(valid)[ok]]=mid[ok]
    return out


def camera_vec(pose):
    w=np.linalg.inv(pose)
    return np.r_[cv2.Rodrigues(w[:3,:3])[0].reshape(3),w[:3,3]]


def from_vec(x):
    w=np.eye(4)
    w[:3,:3]=cv2.Rodrigues(x[:3])[0]
    w[:3,3]=x[3:]
    return np.linalg.inv(w)


def solve(shard=0,shards=1,partial=False):
    from world_pose_validation import validate_native, independent_viewpoints
    data,rows=frozen()
    frames={r["id"]:r for r in rows}
    plan=read(OUT/"edges.json")
    if not partial:
        require(read(OUT/"matches_complete.json")["targets"]==1478,"Incomplete learned matching coverage")
    native_init={r["id"]:np.asarray(r["native_c2w"]) for r in read(HIST/"map/full_train_1478/evaluation/trajectory.json")}
    fit=read(WORLD/"b_map/scale/fit.json")["camera_alignment"]
    R=np.array(fit["R"]); t=np.array(fit["t_m"]); scale=fit["scale"]
    K=np.array(data["K_cv"]); invK=np.linalg.inv(K)
    cached=lru_cache(maxsize=24)(lambda i:depth(frames[i]))
    native=Path(data["packet"])
    rec=[]
    accepted={}
    outdir=OUT/"targets";outdir.mkdir(exist_ok=True)
    ordered=sorted(frames,key=lambda n:(n not in SPECIAL,n))
    for name in ordered[shard::shards]:
        path=outdir/(name+".json")
        if path.exists():
            result=read(path)
            rec.append(result)
            if result["accepted"]:
                accepted[name]=result["selected_c2w_cv"]
            continue
        proposed=plan["targets"][name]
        if partial and not all(packet_path(name,j).exists() for j in proposed["fit"]+proposed["audit"]):
            continue
        poses={i:np.array(frames[i]["c2w_cv"]) for i in proposed["fit"]+proposed["audit"]}
        prior=np.array(frames[name]["c2w_cv"])
        def build(role):
            parts=[]
            for j in proposed[role]:
                with np.load(packet_path(name,j),allow_pickle=False) as pack:
                    u,v=pack["u"],pack["v"]
                    if j<name:
                        u,v=v,u
                    score=pack["score"]
                # Disjoint 16x16 matching-grid tiles also separate repeated target features.
                tile=((u[:,0]//16)+3*(u[:,1]//16))%2
                good=tile==(0 if role=="fit" else 1)
                u=(u[good].astype(float)+.5)*2.5-.5
                v=(v[good].astype(float)+.5)*2.5-.5
                z=metric_depth(cached(j),v)
                valid=np.isfinite(z)
                world=(np.c_[v[valid],np.ones(valid.sum())]@invK.T)*z[valid,None]
                world=world@poses[j][:3,:3].T+poses[j][:3,3]
                parts.extend(zip(score[good][valid],u[valid],world,[j]*len(world)))
            # Deterministic spatial coverage and bounded solve cost. Retain the full audit set.
            parts.sort(key=lambda x:-float(x[0]))
            if role=="fit" and len(parts)>1200:
                parts=parts[:1200]
            xyz=np.array([r[2] for r in parts]).reshape(-1,3)
            uv=np.array([r[1] for r in parts]).reshape(-1,2)
            return xyz,uv,[r[3] for r in parts]
        xyz,uv,donor=build("fit"); ax,au,ad=build("audit")
        def delta(x,points,px):
            pc=points@cv2.Rodrigues(x[:3])[0].T+x[3:]
            zz=np.maximum(.05,pc[:,2])
            proj=(pc@K.T)[:,:2]/zz[:,None]
            return proj-px,pc[:,2]
        def assess(pose):
            d,z=delta(camera_vec(pose),ax,au)
            zz=metric_depth(cached(name),au)
            ok=np.isfinite(zz)&(z>0)
            err=np.linalg.norm(d,axis=1)
            return dict(audit_matches=len(err),positive=int((z>0).sum()),
                        median_px=float(np.median(err)) if len(err) else None,
                        metric_pairs=int(ok.sum()),median_abs_z_m=float(np.median(np.abs(z[ok]-zz[ok]))) if ok.any() else None)
        result=dict(id=name,accepted=False,fit_matches=len(xyz),audit=assess(prior),donors=proposed,initializations={})
        common=None
        if len(xyz)>=20 and len(ax)>0:
            cv2.setRNGSeed(42)
            ok,_,_,ix=cv2.solvePnPRansac(xyz,uv,K,None,iterationsCount=3000,reprojectionError=4.,
                                        confidence=.999,flags=cv2.SOLVEPNP_EPNP)
            if ok and ix is not None and len(ix)>=20:
                common=ix.ravel()
        if common is not None:
            X,U=xyz[common],uv[common]
            donor_ids=[donor[i] for i in common]
            independent=independent_viewpoints(X,donor_ids,poses,frames)
            m=native_init[name].copy()
            m[:3,:3]=R@m[:3,:3];m[:3,3]=R@(scale*m[:3,3])+t
            for label,start in (("B2",prior),("Map",m)):
                def residual(x):
                    xy,z=delta(x,X,U)
                    return np.r_[xy.ravel(),np.minimum(z-.05,0)*100]
                opt=least_squares(residual,camera_vec(start),loss="huber",f_scale=2.,max_nfev=60)
                pose=from_vec(opt.x); er,pz=delta(opt.x,X,U)
                item=dict(fit_inliers=len(X),fit_groups=len({frames[j]["group"] for j in donor_ids}),
                          fit_median_px=float(np.median(np.linalg.norm(er,axis=1))),
                          fit_positive=int((pz>0).sum()),audit=assess(pose),independent_viewpoints=independent,
                          candidate_c2w_cv=pose.tolist(),nfev=int(opt.nfev))
                result["initializations"][label]=item
            label=min(result["initializations"],key=lambda k:result["initializations"][k]["fit_median_px"])
            best=result["initializations"][label]; before=result["audit"]; after=best["audit"]
            observed=(after["metric_pairs"]>=before["metric_pairs"]>0 and after["median_abs_z_m"]<before["median_abs_z_m"]
                      and after["median_px"]<before["median_px"] and after["positive"]==after["audit_matches"])
            gates=observed and independent is not None and best["fit_groups"]>=2 and best["fit_median_px"]<=2.5 and best["fit_positive"]==len(X)
            result.update(fit_selected=label,passes_feature_audit=bool(gates))
            if name in SPECIAL and gates:
                pairs=[j for j in proposed["audit"] if j in ad]
                native_result=validate_native(native,frames,name,pairs,prior,np.array(best["candidate_c2w_cv"]))
                result["native_audit"]=native_result
                if native_result["accepted"]:
                    result.update(accepted=True,selected_c2w_cv=best["candidate_c2w_cv"])
                    accepted[name]=result["selected_c2w_cv"]
        save(path,result);rec.append(result)
    if partial:
        return len(rec), len(ordered[shard::shards])
    # The coupled 1304/1305 correction is published only if both independent gates passed.
    pair={"frame_01304","frame_01305"}
    if pair&accepted.keys() and not pair<=accepted.keys():
        for k in pair:
            accepted.pop(k,None)
    save(OUT/"accepted.json",dict(accepted_frame_ids=sorted(accepted),accepted_c2w_cv=accepted,audit_independent=True,
         total_targets=len(rec),normal_controls=len(rec)-len(SPECIAL),normal_controls_applied=0,
         reason="Local corrections are restricted to the three preregistered frames. All other train frames are common controls.",
         fit_solutions=sum(bool(r["initializations"]) for r in rec),matches_sha256=sha(OUT/"matches_complete.json"),
         hard_results=[r for r in rec if r["id"] in SPECIAL]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("phase",choices=("edges","match","solve","all"))
    stage=p.parse_args().phase
    OUT.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    with (OUT/"worker.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stage in ("all","edges"): edges()
        if stage in ("all","match"): matching()
        if stage in ("all","solve"): solve()
