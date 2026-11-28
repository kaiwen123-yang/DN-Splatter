"""Fit each stream's single metric scale and rigid alignment on its own frozen prefix."""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from lingbot_common import P0, RUN, SEQ, by_id, depth, masks, read, render_z, require, save
from lingbot_metrics import Accumulator, parts, ratio, summary, visibility, write_csv


def sample(array):
    # Inverse of the official AREA cover-resize, applied to pixel centres.
    # RGB is 1280x720 in all inputs, so neither dimension is cropped.
    y=np.rint((np.arange(288)+.5)/.4-.5).astype(int)
    x=np.rint((np.arange(512)+.5)/.4-.5).astype(int)
    return array[np.ix_(y,x)]


def load_prediction(directory,row):
    i=row["offset"]
    with np.load(directory/row["chunk"],allow_pickle=False) as data:
        require(int(data["raw_indices"][i])==row["raw_index"],"Map chunk changed")
        return (data["depth"][i].squeeze(),data["depth_conf"][i].squeeze(),
                data["intrinsics"][i],data["poses_c2w"][i].astype(float))


def fit_rigid(a,b):
    ca,cb=a.mean(0),b.mean(0)
    u,s,vt=np.linalg.svd((a-ca).T@(b-cb))
    signs=np.eye(3)
    signs[2,2]=np.sign(np.linalg.det(vt.T@u.T))
    rot=vt.T@signs@u.T
    trans=cb-rot@ca
    return rot,trans,dict(map_baselines_singular_values_m=np.linalg.svd(a-ca,compute_uv=False).tolist(),
        reference_baselines_singular_values_m=np.linalg.svd(b-cb,compute_uv=False).tolist(),
        cross_singular_values=s.tolist(),near_collinear=bool(s[0]==0 or s[1]<.01*s[0]),
        near_stationary=bool(np.linalg.norm(np.ptp(b,axis=0))<.02))


def angle(rot):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(rot)-1)/2,-1.,1.))))


def evaluate(seq,frames,selection,k_sensor,canvas):
    root=RUN/"map"/seq
    audit=read(root/"complete.json")["audit"]
    directory=Path(audit["directory"])
    positions={r["id"]:r for r in audit["frames"]}
    out=root/"evaluation"
    out.mkdir(exist_ok=True)
    cal=[]
    for chosen in selection["calibration"]:
        fid=chosen["id"]
        if fid not in positions:
            cal.append(dict(id=fid,support=0,ratio=None,reason="missing original selected frame"))
            continue
        pred,_,_,pose=load_prediction(directory,positions[fid])
        z=sample(depth(frames[fid]))
        with np.load(chosen["mask"]["path"]) as m: reliable=sample(m["static_support"])
        use=reliable & np.isfinite(pred) & (pred>0)
        ratios=z[use]/pred[use]
        cal.append(dict(id=fid,support=len(ratios),ratio=float(np.median(ratios)) if len(ratios) else None,
            ratio_distribution=summary(ratios),map_center=pose[:3,3].tolist(),
            reference_center=np.array(frames[fid]["c2w_cv"])[:3,3].tolist()))
    usable=[r for r in cal if r["ratio"] is not None]
    s=float(np.median([r["ratio"] for r in usable])) if usable else None
    rotation=translation=diagnosis=None
    if s and len(usable)>=3:
        rotation,translation,diagnosis=fit_rigid(s*np.array([r["map_center"] for r in usable]),
                                               np.array([r["reference_center"] for r in usable]))
    transform=dict(scale_m_per_model_unit=s,calibration_frames=cal,frame_ratio_distribution=summary([r["ratio"] for r in usable]),
                   rotation=rotation.tolist() if rotation is not None else None,
                   translation_m=translation.tolist() if translation is not None else None,rigid_condition=diagnosis,
                   source="median of within-frame sensor Z / native map Z medians; same IDs, separate fit for this attempt")
    save(out/"alignment.json",transform)
    ktarget=k_sensor.copy()
    ktarget[:2]*=.4
    ktarget[:2,2]+=(.4-1)/2
    rows=[]
    trajectory=[]
    temporal=defaultdict(list)
    acc=Accumulator()
    total=finite_count=confidence_count=0
    last=None
    start=frames[audit["frames"][0]["id"]]["rgb_timestamp_ns"]
    for info in audit["frames"]:
        fid=info["id"]
        frame=frames[fid]
        pred,conf,k,pose=load_prediction(directory,info)
        measured=sample(depth(frame))
        fixed=sample(canvas)&np.isfinite(measured)&(measured>=.3)&(measured<=6)
        valid=np.isfinite(pred)&(pred>0)
        filtered=valid & np.isfinite(conf)&(conf>1.5)
        total+=valid.size
        finite_count+=int(valid.sum())
        confidence_count+=int(filtered.sum())
        t=(frame["rgb_timestamp_ns"]-start)*1e-9
        ref=np.array(frame["c2w_cv"])
        fit=None
        if rotation is not None:
            fit=pose.copy()
            fit[:3,:3]=rotation@pose[:3,:3]
            fit[:3,3]=rotation@(s*pose[:3,3])+translation
        r=dict(id=fid,raw_index=info["raw_index"],frame_index=info["frame_index"],timestamp_ns=frame["rgb_timestamp_ns"],
               split=frame["split"],seconds=t,finite_positive=int(valid.sum()),confidence_gt_1p5=int(filtered.sum()),
               canvas_pixels=int(valid.size),ref_centre_difference_m=None,ref_rotation_difference_deg=None,
               map_step_m=None,reference_step_m=None,map_step_deg=None,reference_step_deg=None,
               step_vector_difference_m=None,scale_needed_native=None,
               K_fx=float(k[0,0]),K_fy=float(k[1,1]),K_cx=float(k[0,2]),K_cy=float(k[1,2]))
        if fit is not None:
            r["ref_centre_difference_m"]=float(np.linalg.norm(fit[:3,3]-ref[:3,3]))
            r["ref_rotation_difference_deg"]=angle(fit[:3,:3].T@ref[:3,:3])
        if last is not None:
            prev_pose,prev_ref,prev_fit,prev_t=last
            r["dt_s"]=t-prev_t
            r["map_step_m"]=float(s*np.linalg.norm(pose[:3,3]-prev_pose[:3,3])) if s else None
            r["reference_step_m"]=float(np.linalg.norm(ref[:3,3]-prev_ref[:3,3]))
            r["map_step_deg"]=angle(pose[:3,:3].T@prev_pose[:3,:3])
            r["reference_step_deg"]=angle(ref[:3,:3].T@prev_ref[:3,:3])
            if fit is not None:
                r["step_vector_difference_m"]=float(np.linalg.norm(fit[:3,3]-prev_fit[:3,3]-ref[:3,3]+prev_ref[:3,3]))
        last=(pose,ref,fit,t)
        trajectory.append(dict(id=fid,native_c2w=pose.tolist(),aligned_c2w=fit.tolist() if fit is not None else None,
                               reference_c2w=ref.tolist()))
        overlap=fixed&valid
        if overlap.any():
            r["scale_needed_native"]=float(np.median(measured[overlap]/pred[overlap]))
        if s:
            stat=acc.add(["all",f"seconds_{int(t//30)*30}"],parts(measured,s*pred,fixed,valid))
            r.update(stat)
        temporal[int(t//30)*30].append(r)
        rows.append(r)
    overlap_rows=[]
    for fid,others in selection["pairs"].items():
        if fid not in positions: continue
        z,_,k,p=load_prediction(directory,positions[fid])
        for other in others:
            if other not in positions: continue
            target,_,k2,p2=load_prediction(directory,positions[other])
            if s:
                ps=p.copy(); pt=p2.copy()
                ps[:3,3]*=s; pt[:3,3]*=s
                for label,target_z,target_k,refmask in (
                        ("map_to_map",target*s,k2,np.isfinite(target)&(target>0)),
                        ("map_to_other_sensor",sample(depth(frames[other])),ktarget,sample(masks(depth(frames[other]),canvas)[-1]))):
                    rz=render_z(z*s,k,ps,target_k,pt,target.shape)
                    overlap_rows.append(dict(id=fid,target=other,comparison=label,**visibility(rz,target_z,refmask)))
    write_csv(out/"per_frame.csv.gz",rows)
    write_csv(out/"overlap_pairs.csv.gz",overlap_rows)
    save(out/"trajectory.json",trajectory)
    segments=[]
    for t,items in temporal.items():
        segments.append(dict(start_s=t,frames=len(items),scale_needed=summary([r["scale_needed_native"] for r in items]),
               fixed_scale_depth_mae_m=summary([r.get("mae_m") for r in items]),ref_diff_m=summary([r["ref_centre_difference_m"] for r in items])))
    jump=sorted((r for r in rows if r["map_step_m"] is not None),key=lambda r:r["map_step_m"],reverse=True)[:10]
    angles=sorted((r for r in rows if r["map_step_deg"] is not None),key=lambda r:r["map_step_deg"],reverse=True)[:10]
    save(out/"metrics.json",dict(sequence=seq,attempt=str(directory.parent),input_frames=audit["input_frames"],predicted_frames=len(rows),
        persistent_frames=audit["persistent_frames"],npz_files=audit["npz_files"],model_scale=s,
        calibration_count=len(usable),rigid_condition=diagnosis,pooled_depth=acc.finish(),time_bins=segments,
        centre_difference_m=summary([r["ref_centre_difference_m"] for r in rows]),
        rotation_difference_deg=summary([r["ref_rotation_difference_deg"] for r in rows]),
        reference_path_m=float(sum(r["reference_step_m"] or 0 for r in rows)),
        map_path_m=float(sum(r["map_step_m"] or 0 for r in rows)) if s else None,
        top_translation_steps=jump,top_rotation_steps=angles,
        all_pixels=total,positive_finite=finite_count,confidence_gt_1p5=confidence_count,
        default_filter_coverage=ratio(confidence_count,total),known_difficult=[r for r in rows if r["id"] in selection["known_difficult"]],
        loop_closure=dict(value=None,reason="No independently established loop correspondences in the frozen input protocol."),
        overlap_pairs=len(overlap_rows)//2))
    print("Evaluated Map",seq,flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("sequence",nargs="?",choices=SEQ)
    choice=arg.parse_args().sequence
    cv2.setNumThreads(1)
    manifest,frames=by_id()
    selection=read(P0/"selection.json")
    for seq in ((choice,) if choice else SEQ):
        evaluate(seq,frames,selection,np.array(manifest["K_cv"]),np.load(P0/"canvas.npy"))
