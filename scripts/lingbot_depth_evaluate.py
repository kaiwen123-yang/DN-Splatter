"""Evaluate fixed-mask recovery, observed-value changes, and separate multiview hole evidence."""
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from lingbot_common import P0, RUN, by_id, depth, masks, read, render_z, require, save
from lingbot_metrics import Accumulator, distance_masks, parts, ratio, summary, visibility, write_csv


def prediction(fid,tag="unmasked"):
    folder=RUN/"depth"/tag/fid
    z=np.load(folder/"depth_refined_m.npy",allow_pickle=False)
    with np.load(folder/"validity.npz") as m:
        finite=m["finite_positive"]
        require(np.array_equal(finite,np.isfinite(z)&(z>0)),"Stored validity mask "+fid)
        valid=finite & m["model_mask"] if "model_mask" in m else finite
    return z,valid


def cross_views(frames,selection,k,canvas):
    comparisons=[]
    holes=[]
    for fid,neighbours in selection["pairs"].items():
        row=frames[fid]
        ref=depth(row)
        obs,primary,_,reliable=masks(ref,canvas)
        pred,v=prediction(fid)
        empty=canvas & ~obs
        incoming=[]
        for sid in neighbours:
            other=frames[sid]
            other_z=depth(other)
            other_ref=masks(other_z,canvas)[-1]
            sensor=np.where(other_ref,other_z,0)
            incoming.append(render_z(sensor,k,other["c2w_cv"],k,row["c2w_cv"],ref.shape))
            for source,z in (("raw",np.where(primary,ref,0)),("refined",np.where(canvas&v,pred,0))):
                reproj=render_z(z,k,row["c2w_cv"],k,other["c2w_cv"],other_z.shape)
                comparisons.append(dict(id=fid,target=sid,source=source,suspect_pose=row["suspect_reference_pose"] or other["suspect_reference_pose"],
                                        **visibility(reproj,other_z,other_ref)))
        supported=np.zeros_like(empty)
        agree=np.zeros_like(empty)
        candidate_conflict=np.zeros_like(empty)
        farther=np.zeros_like(empty)
        if len(incoming)==2:
            a,b=incoming
            both=(a>0)&(b>0)
            tol=np.maximum(.05,.03*np.minimum(a,b))
            supported=empty & both & (np.abs(a-b)<=tol)
            surface=(a+b)*.5
            agree=supported & v & (np.abs(pred-surface)<=np.maximum(.05,.03*surface))
            candidate_conflict=supported & v & (pred<surface-np.maximum(.05,.03*surface))
            farther=supported & v & (pred>surface+np.maximum(.05,.03*surface))
        else:
            surface=np.zeros_like(ref)
        usable=supported & v
        holes.append(dict(id=fid,neighbours=len(neighbours),natural_holes=int(empty.sum()),predicted_holes=int((empty&v).sum()),
                          two_view_sensor_support=int(supported.sum()),evaluated_predictions=int(usable.sum()),
                          agreement=int(agree.sum()),closer_than_support=int(candidate_conflict.sum()),farther_than_support=int(farther.sum()),
                          no_two_view_evidence=int((empty & ~supported).sum()),suspect_reference_pose=row["suspect_reference_pose"],
                          agreement_on_supported_predictions=ratio(agree.sum(),usable.sum()),
                          mean_abs_residual_m=float(np.abs(pred[usable]-surface[usable]).mean()) if usable.any() else None))
        if fid in {s["id"] for s in selection["samples"]}:
            out=RUN/"depth"/"hole_evidence"
            out.mkdir(exist_ok=True)
            np.savez_compressed(out/(fid+".npz"),other_sensor_z=surface,observed_in_two=supported,
                                agrees=agree,closer=candidate_conflict,farther=farther)
    return comparisons,holes


def main():
    cv2.setNumThreads(1)
    manifest,frames=by_id()
    sel=read(P0/"selection.json")
    completed=read(RUN/"depth/manifest.json")
    require(completed["unmasked"]==1478 and completed["masked"]==60,"Incomplete Depth inventory")
    canvas=np.load(P0/"canvas.npy")
    out=RUN/"depth/evaluation"
    out.mkdir(parents=True,exist_ok=True)
    acc=Accumulator()
    rows=[]
    for sample in sel["samples"]:
        fid,group=sample["id"],sample["group"]
        raw=depth(frames[fid])
        for mode in ("random_points","blocks"):
            with np.load(P0/"masks"/(fid+".npz")) as m:
                hide,fixed=m[mode+"_input"],m[mode+"_score"]
            hidden=raw.copy()
            hidden[hide]=0
            present=np.isfinite(hidden)&(hidden>0)&canvas
            if present.any():
                nearest=distance_transform_edt(~present,return_distances=False,return_indices=True)
                nn=hidden[tuple(nearest)]
            else: nn=np.zeros_like(raw)
            pred,valid=prediction(fid,mode)
            nn_valid=np.isfinite(nn)&(nn>0)
            common=valid&nn_valid
            for method,z,v in (("lingbot",pred,valid),("nearest",nn,nn_valid)):
                for label,subset in [("all",fixed),("common_valid",fixed&common)]+[
                        (name,fixed&mask) for name,mask in distance_masks(raw)]:
                    r=acc.add([f"{mode}/{method}/{label}",f"{mode}/{method}/{group}/{label}"],parts(raw,z,subset,v))
                    rows.append(dict(id=fid,group=group,mode=mode,method=method,stratum=label,
                                     fixed_original_pixels=int(fixed.sum()),**r))
    masked_results=acc.finish()
    write_csv(out/"masked_per_frame.csv.gz",rows)
    acc=Accumulator()
    observed_rows=[]
    hole_rows=[]
    for fid in sel["sequences"]["full_train_1478"]["ids"]:
        raw=depth(frames[fid])
        pred,v=prediction(fid)
        observed,primary,band,reliable=masks(raw,canvas)
        strata=[("all",primary),("boundary",primary&band),("interior",reliable),
                ("outside_primary",observed&~primary)]+[(n,primary&m) for n,m in distance_masks(raw)]
        for name,mask in strata:
            r=acc.add([name],parts(raw,pred,mask,v))
            observed_rows.append(dict(id=fid,stratum=name,**r))
        holes=canvas&~observed
        filled=holes&v
        vals=pred[filled]
        q=np.percentile(vals,[5,50,95]) if len(vals) else [None]*3
        hole_rows.append(dict(id=fid,holes=int(holes.sum()),filled=int(filled.sum()),
                              nonfinite=int((holes&~np.isfinite(pred)).sum()),positive_outside_range=int(((vals<.3)|(vals>6)).sum()),
                              coverage=ratio(filled.sum(),holes.sum()),p05_m=q[0],median_m=q[1],p95_m=q[2]))
    write_csv(out/"measurement_change_per_frame.csv.gz",observed_rows)
    write_csv(out/"natural_holes_per_frame.csv.gz",hole_rows)
    measured=acc.finish()
    pairs,independent=cross_views(frames,sel,np.array(manifest["K_cv"]),canvas)
    write_csv(out/"cross_view_pairs.csv.gz",pairs)
    write_csv(out/"natural_holes_two_view.csv.gz",independent)
    counts={k:sum(r[k] for r in independent) for k in ("natural_holes","predicted_holes","two_view_sensor_support",
            "evaluated_predictions","agreement","closer_than_support","farther_than_support","no_two_view_evidence")}
    totals={k:sum(r[k] for r in hole_rows) for k in ("holes","filled","nonfinite","positive_outside_range")}
    save(out/"metrics.json",dict(masked_recovery=masked_results,measurement_change=measured,
         raw_hole_completion=dict(**totals,coverage=ratio(totals["filled"],totals["holes"]),frames=1478),
         two_view_holes=dict(**counts,diagnostic_frames=len(independent),pairs=len(pairs)//2,
            support_fraction=ratio(counts["two_view_sensor_support"],counts["natural_holes"]),
            supported_agreement=ratio(counts["agreement"],counts["evaluated_predictions"])),
         caveat="Other train views supply measured Z, projected through B2 poses. Surfaces with no two-view agreement remain unverified."))


if __name__=="__main__":
    main()
