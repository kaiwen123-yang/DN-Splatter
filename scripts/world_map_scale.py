"""Fit single and time-linear depth scales on train groups, audit untouched groups."""
from datetime import datetime, timezone
import fcntl
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from lingbot_common import ROOT, RUN as HIST, P0, by_id, depth, read, render_z, require, save, sha
from lingbot_map_evaluate import fit_rigid, load_prediction, sample


OUT=ROOT/".local/runs/interactive-world-20261128-v1/b_map/scale"
SEQ="full_train_1478"


def freeze_roles(ids, frames):
    t=np.array([frames[k]["rgb_timestamp_ns"] for k in ids],dtype=np.int64)
    seconds=(t-t[0])*1e-9
    groups=np.floor(seconds/10).astype(int)
    universe=np.unique(groups)
    perm=np.random.default_rng(42).permutation(universe)
    selected=set(int(k) for k in perm[:int(round(.6*len(perm)))])
    roles=np.where(np.isin(groups,list(selected)),"fit","audit").astype(object)
    original=roles.copy()
    seen={}
    for i,k in enumerate(ids):
        row=frames[k]
        for field in ("rgb_sha256","depth_sha256"):
            seen.setdefault((field,row[field]),set()).add(original[i])
    for i,k in enumerate(ids):
        other=seconds[original!=original[i]]
        if any(len(seen[(field,frames[k][field])])>1 for field in ("rgb_sha256","depth_sha256")):
            roles[i]="excluded_duplicate"
        elif np.min(np.abs(other-seconds[i]))<1.0:
            roles[i]="excluded_near_boundary"
        elif frames[k]["suspect_reference_pose"] and roles[i]=="fit":
            roles[i]="excluded_suspect_from_fit"
    return seconds,groups,roles


def stats(errors,den):
    e=np.concatenate(errors) if errors else np.empty(0)
    return dict(reference_pixels=den,finite_pairs=len(e),invalid_or_unobserved=den-len(e),
                mae_m=float(np.mean(e,dtype=np.float64)) if len(e) else None,
                p95_m=float(np.quantile(e,.95)) if len(e) else None,
                within_10cm=int(np.sum(e<=.10)),correct_coverage_10cm=float(np.sum(e<=.10)/den) if den else None)


def run():
    start=datetime.now(timezone.utc).isoformat()
    manifest,frames=by_id()
    old=read(HIST/"map"/SEQ/"complete.json")["audit"]
    idx=old["frames"]
    ids=[r["id"] for r in idx]
    require(len(ids)==1478 and all(frames[k]["split"]=="train" for k in ids),"Non-training data in scale fit")
    path=Path(old["directory"])
    seconds,groups,roles=freeze_roles(ids,frames)
    freeze=[dict(id=k,seconds=float(t),group=int(g),role=r) for k,t,g,r in zip(ids,seconds,groups,roles)]
    save(OUT/"roles.json",freeze)
    canvas=sample(np.load(P0/"canvas.npy"))
    def measured(i):
        z=sample(depth(frames[ids[i]]))
        mask=canvas&np.isfinite(z)&(z>=.3)&(z<=6.)
        return z,mask
    ratios=[]
    poses=[]
    counts=[]
    for i,r in enumerate(idx):
        pred,_,_,pose=load_prediction(path,r)
        raw,mask=measured(i)
        good=mask&np.isfinite(pred)&(pred>0)
        ratios.append(float(np.median(raw[good]/pred[good])) if good.any() else np.nan)
        counts.append(int(good.sum()))
        poses.append(pose)
    ratios=np.array(ratios)
    fit=(roles=="fit")&np.isfinite(ratios)&(ratios>0)
    require(fit.sum()>1 and np.unique(groups[fit]).size>2,"Insufficient independent fit groups")
    global_s=float(np.median(ratios[fit]))
    mid=float(np.mean(seconds[fit]))
    unit=float(np.ptp(seconds[fit]))
    tx=(seconds-mid)/unit
    # Two coefficients only, fitted on per-frame ratios. The curve is used for
    # camera-axis depth. All cross-view camera translations retain one global scale.
    result=least_squares(lambda q:q[0]+q[1]*tx[fit]-np.log(ratios[fit]),
                         [np.log(global_s),0.],loss="huber",f_scale=.03)
    variable_s=np.exp(result.x[0]+result.x[1]*tx)
    original_s=read(HIST/"map"/SEQ/"evaluation/alignment.json")["scale_m_per_model_unit"]
    scales={"historical_prefix32":np.full(len(ids),original_s),"fit_global":np.full(len(ids),global_s),"fit_log_linear":variable_s}
    K=np.array(manifest["K_cv"],dtype=float)
    K[:2]*=.4
    K[:2,2]+=(.4-1)/2
    poses=np.asarray(poses)
    refs=np.stack([frames[k]["c2w_cv"] for k in ids])
    rot,trans,condition=fit_rigid(poses[fit,:3,3]*global_s,refs[fit,:3,3])
    save(OUT/"fit.json",dict(started=start,split_sha256=sha(OUT/"roles.json"),fit_frames=int(fit.sum()),
         fit_groups=len(set(groups[fit])),reference="raw sensor optical-axis Z, 0.3–6 m",
         global_m_per_unit=global_s,log_linear_coefficients=result.x.tolist(),time_center_s=mid,time_unit_s=unit,
         camera_alignment=dict(R=rot.tolist(),t_m=trans.tolist(),scale=global_s,condition=condition,
            reference="B2 fit centers for initial gauge only"),original_prefix32_m_per_unit=original_s,
         per_frame_ratio=[dict(id=k,observed=n,median_m_per_unit=(float(r) if np.isfinite(r) else None))
                          for k,n,r in zip(ids,counts,ratios)]))
    aud=np.flatnonzero(roles=="audit")
    all_errors={s:[] for s in scales}
    total=0
    group_records=[]
    for g in sorted(set(groups[aud])):
        ge={s:[] for s in scales}
        den=0
        for i in aud[groups[aud]==g]:
            pred,_,_,_=load_prediction(path,idx[i])
            raw,mask=measured(i)
            valid=mask&np.isfinite(pred)&(pred>0)
            den+=int(mask.sum())
            for label,curve in scales.items():
                ge[label].append(np.abs(pred[valid]*np.float32(curve[i])-raw[valid]))
        total+=den
        group_records.append(dict(group=int(g),conditions={s:stats(e,den) for s,e in ge.items()}))
        for label in scales:
            all_errors[label].extend(ge[label])
    paired={s:[] for s in scales}
    pair_den=0
    pair_ids=[]
    # Freeze one directed partner per audit frame, at least one second away.
    # Select by time alone, independent of both B2 and the scale residuals.
    for i in aud:
        allowed=[j for j in aud if j>i and 1.0<=seconds[j]-seconds[i]<=2.5 and
                 frames[ids[j]]["depth_sha256"]!=frames[ids[i]]["depth_sha256"]]
        if not allowed:
            continue
        j=min(allowed,key=lambda j:(abs(seconds[j]-seconds[i]-1.5),int(j)))
        pred,_,ki,pi=load_prediction(path,idx[i])
        raw,mask=measured(j)
        pj=poses[j].copy()
        pair_den+=int(mask.sum())
        pair_ids.append([ids[i],ids[j]])
        for label,curve in scales.items():
            a=pi.copy()
            b=pj.copy()
            translation_scale=original_s if label=="historical_prefix32" else global_s
            a[:3,3]*=translation_scale
            b[:3,3]*=translation_scale
            mapped=render_z(pred*curve[i],ki,a,K,b,raw.shape)
            valid=mask&np.isfinite(mapped)&(mapped>0)
            paired[label].append(np.abs(mapped[valid]-raw[valid]).astype(np.float32))
    save(OUT/"audit_pairs.json",pair_ids)
    save(OUT/"audit.json",dict(sequence=SEQ,all_train_frames=len(ids),roles={r:int((roles==r).sum()) for r in sorted(set(roles))},
         fit_global_scale=global_s,log_linear_coefficients=result.x.tolist(),fit_sha256=sha(OUT/"fit.json"),
         roles_sha256=sha(OUT/"roles.json"),pairs_sha256=sha(OUT/"audit_pairs.json"),
         direct={s:stats(e,total) for s,e in all_errors.items()},cross_view={s:stats(e,pair_den) for s,e in paired.items()},
         directed_audit_pairs=len(pair_ids),audit_time_groups=group_records,
         reference="Same capture sensor consistency. Center gauge is never evaluated as ground truth.",
         code_sha256=sha(Path(__file__))))


if __name__=="__main__":
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"worker.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
