"""Independent native-depth and viewpoint gates on common physical source rays."""
from functools import lru_cache
import json
import numpy as np
from scipy.spatial import cKDTree


def independent_viewpoints(xyz,donors,poses,frames):
    neighbors=cKDTree(xyz).query_pairs(.05,output_type="ndarray")
    for ia,ib in neighbors:
        a,b=donors[ia],donors[ib]
        if frames[a]["group"]==frames[b]["group"]:
            continue
        p=(xyz[ia]+xyz[ib])*.5
        da,db=poses[a][:3,3]-p,poses[b][:3,3]-p
        baseline=float(np.linalg.norm(da-db))
        denom=np.linalg.norm(da)*np.linalg.norm(db)
        if denom>0 and baseline>=.1:
            ang=float(np.rad2deg(np.arccos(np.clip(np.dot(da,db)/denom,-1.,1.))))
            if ang>=3:
                return dict(a=a,b=b,baseline_m=baseline,ray_angle_deg=ang)
    return None


def validate_native(packet,frames,name,anchors,old,new):
    calibration=json.loads((packet/"calibration.json").read_bytes())
    k=calibration["cameras"]["depth"]
    assert not np.any(k["D_sdk_k1_k2_k3_k4_k5_k6_p1_p2"])
    ex=next(x for x in calibration["static_extrinsics"] if (x["source_sensor"],x["destination_sensor"])==("depth","color"))
    T=np.eye(4);T[:3,:3]=ex["R"];T[:3,3]=np.asarray(ex["translation_mm"])/1000.
    yy,xx=np.mgrid[4:k["height"]:8,4:k["width"]:8]
    yy,xx=yy.ravel(),xx.ravel()
    @lru_cache(maxsize=8)
    def z(i):
        with np.load(packet/frames[i]["native_path"],allow_pickle=False) as f:
            return f["depth"].astype(np.float64)*.001
    def residuals(a,b,pa,pb):
        src=z(a)[yy,xx]
        valid=(src>=.3)&(src<=6.)
        idx=np.flatnonzero(valid);d=src[valid]
        xyz=np.stack(((xx[valid]-k["cx"])*d/k["fx"],(yy[valid]-k["cy"])*d/k["fy"],d),axis=1)
        rel=np.linalg.solve(pb@T,pa@T)
        q=xyz@rel[:3,:3].T+rel[:3,3]
        ok=q[:,2]>0;idx,q=idx[ok],q[ok]
        u=np.rint(q[:,0]*k["fx"]/q[:,2]+k["cx"]).astype(int)
        v=np.rint(q[:,1]*k["fy"]/q[:,2]+k["cy"]).astype(int)
        ok=(u>=0)&(v>=0)&(u<k["width"])&(v<k["height"])
        pix=(v[ok]*k["width"]+u[ok]);idx,q=idx[ok],q[ok]
        order=np.lexsort((q[:,2],pix)); take=order[np.unique(pix[order],return_index=True)[1]]
        ref=z(b).ravel()[pix[take]]
        ok=(ref>=.3)&(ref<=6)
        ar=np.full(len(yy),np.nan)
        ar[idx[take[ok]]]=q[take[ok],2]-ref[ok]
        return ar
    checks=[]
    for j in anchors:
        pj=np.array(frames[j]["c2w_cv"])
        for a,b,x,y,nx,ny in [(name,j,old,pj,new,pj),(j,name,pj,old,pj,new)]:
            before=residuals(a,b,x,y); after=residuals(a,b,nx,ny)
            common=np.isfinite(before)&np.isfinite(after)
            nc=int(common.sum()); n0=int(np.isfinite(before).sum()); n1=int(np.isfinite(after).sum())
            hits0=int((np.abs(before[common])<=.1).sum());hits1=int((np.abs(after[common])<=.1).sum())
            checks.append(dict(source=a,target=b,common=nc,before_valid=n0,after_valid=n1,
                before_correct=hits0,after_correct=hits1,pass_gate=bool(nc and n1>=n0 and hits1>hits0)))
    return dict(pairs=checks,accepted=bool(checks and all(r["pass_gate"] for r in checks)))
