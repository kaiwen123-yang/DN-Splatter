"""Replay the complete public-position hall timeline against causal RGB-D background anchors."""
import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import json
from pathlib import Path
import time

import cv2
import numpy as np

from interactive_physics_memory import EvidenceMemory
from interactive_push_assets import write_json
from lingbot_common import HARD, P0, ROOT, SRC, member, read, sha


OUT=ROOT/".local/runs/interactive-world-20261128-v1/d_world/static_hall"
TOL=.025  # Same geometric state threshold as EvidenceMemory.update_depth.
DY,DX=np.mgrid[-2:3,-2:3]
VARIANTS=("M0","M1","M2")


def reliable(z,v,u):
    tiles=z[v[:,None]+DY.ravel(),u[:,None]+DX.ravel()]
    use=np.isfinite(tiles)&(tiles>=.3)&(tiles<=6.)
    low=np.where(use,tiles,np.inf).min(axis=1)
    high=np.where(use,tiles,-np.inf).max(axis=1)
    return (use.sum(axis=1)>=4)&(high-low<=np.maximum(.05,.03*z[v,u]))


def seed_rays(z,canvas):
    """One visible interior source sample per 6x4 image tile in each new ten-second group."""
    h,w=z.shape
    mid=canvas&np.isfinite(z)&(z>=.3)&(z<=6.)
    mask=np.zeros((h,w),bool)
    for row in range(4):
        for col in range(6):
            cy=int((row+.5)*h/4);cx=int((col+.5)*w/6)
            # Fixed search window, with no fill across empty rays.
            y,x=np.nonzero(mid[cy-12:cy+13,cx-12:cx+13])
            if not len(y):continue
            y+=cy-12;x+=cx-12
            good=reliable(z,y,x)
            if not good.any():continue
            y,x=y[good],x[good]
            i=np.argmin((y-cy)**2+(x-cx)**2);mask[y[i],x[i]]=True
    return mask


def observe(z,canvas,K,pose,keys,records):
    n=len(keys)
    status=np.full(n,0,np.int8)
    u,v=np.zeros(n,int),np.zeros(n,int)
    world=np.array([records[k]["xyz"] for k in keys])
    pc=(world-pose[:3,3])@pose[:3,:3]
    h,w=z.shape
    front=np.flatnonzero(pc[:,2]>.01)
    if len(front):
        projected=pc[front]@K.T
        ux,vy=np.rint(projected[:,:2]/projected[:,2,None]).astype(int).T
        u[front],v[front]=ux,vy
        visible=(ux>=2)&(ux<w-2)&(vy>=2)&(vy<h-2)
        front=front[visible];front=front[canvas[v[front],u[front]]]
    if len(front):
        pixel=v[front]*w+u[front]
        order=np.lexsort((pc[front,2],pixel))
        first=front[order[np.unique(pixel[order],return_index=True)[1]]]
        status[front]=1  # hidden behind another projected anchor in this pixel
        status[first]=2  # sensor unknown, out of range, or depth-edge ambiguity
        measured=z[v[first],u[first]]
        ok=np.isfinite(measured)&(measured>=.3)&(measured<=6.)
        hit=first[ok];hit=hit[reliable(z,v[hit],u[hit])]
        delta=z[v[hit],u[hit]]-pc[hit,2]
        status[hit]=np.where(delta>TOL,5,np.where(delta<-TOL,4,3))
    return status,u,v


def worker():
    OUT.mkdir(parents=True,exist_ok=True)
    complete=OUT/"results.json"
    if complete.is_file():
        s=read(complete)
        if s["positions"]!=1846 or sha(OUT/"frames.jsonl")!=s["frame_audit"]["sha256"]:
            raise RuntimeError("The previous static diagnostic receipt changed")
        print("Verified previous complete static diagnostic",flush=True)
        return
    inputs=read(P0/"input_manifest.json")
    rows=inputs["frames"];K=np.array(inputs["K_cv"]); inv=np.linalg.inv(K)
    ids=read(P0/"selection.json")["sequences"]["full_all_1846_exploratory"]["ids"]
    if len(ids)!=1846 or ids!=[r["id"] for r in rows]:
        raise ValueError("Frozen full-timeline IDs differ")
    opt=ROOT/"reports/capture-full-quality-v3/20261122-v1/video.json"
    redacted=set(read(opt)["movies"]["B2_RGB_reference"]["redacted_frames"])
    canvas=np.load(P0/"canvas.npy",allow_pickle=False)
    groups=set(); birth={};votes=Counter();by_split=defaultdict(Counter);versions={v:Counter() for v in VARIANTS}
    memories={v:EvidenceMemory(v) for v in VARIANTS}
    protected=[]
    fpath=OUT/"frames.jsonl"
    started=time.monotonic()
    with fpath.open("w",encoding="utf8") as trace:
        for i,r in enumerate(rows):
            since=(r["rgb_timestamp_ns"]-rows[0]["rgb_timestamp_ns"])*1e-9;g=int(since//10)
            name=r["id"];pose=np.array(r["c2w_cv"]);cam={"K":K,"T_world_camera_cv":pose}
            rec={"id":name,"raw_position":i,"split":r["split"],"group":g,"seconds":since,
                 "emitted_unknown":False,"suspect_pose":bool(r["suspect_reference_pose"] or name in HARD)}
            tally=Counter(positions=1)
            # Check every received depth file against the frozen acquisition manifest.
            src=SRC/r["depth"]
            if sha(src)!=r["depth_sha256"]:
                raise ValueError("Optical Z digest mismatch for "+name)
            keys=list(memories["M0"].cells)
            if i in redacted or rec["suspect_pose"]:
                reason="publication_redaction" if i in redacted else "suspect_reference_pose"
                tally[reason]+=1;rec.update(emitted_unknown=True,reason=reason)
                for mem in memories.values():
                    mem.update_depth(i,cam,None,None,g,object_id="static_background",anchor_valid=False)
                if rec["suspect_pose"]:protected.append(name)
            else:
                z=np.load(src,allow_pickle=False)
                primary=canvas&np.isfinite(z)&(z>=.3)&(z<=6.)
                tally["raw_primary_pixels"]=int(primary.sum())
                tally["canvas_pixels"]=int(canvas.sum())
                frame_metrics={}
                if keys:
                    states,u,v=observe(z,canvas,K,pose,keys,memories["M0"].cells)
                    cross=np.array([birth[k]!=g for k in keys])
                    for c,label in enumerate(("outside_or_behind","occluded_by_anchor","unknown_sensor_or_edge",
                                               "consistent","foreground_occlusion","see_through")):
                        tally["cross_"+label]=int(np.sum((states==c)&cross))
                    support=np.flatnonzero(states==3)
                    regions={(int(x)//160,int(y)//120) for x,y in zip(u[support],v[support])}
                    trusted=len(support)>=3 and len(regions)>=2
                    rec["anchor_gate_passed"]=trusted
                    if trusted:
                        tally["registered_frames"]=1
                        obs=np.zeros_like(z)
                        mm=np.zeros_like(z,dtype=bool)
                        hits=np.flatnonzero(states>=3)
                        obs[v[hits],u[hits]]=z[v[hits],u[hits]]
                        # Reaffirm only a matching existing voxel. New anchors are admitted once per group below.
                        if len(support):
                            xyz=(np.c_[u[support],v[support],np.ones(len(support))]@inv.T)*z[v[support],u[support],None]
                            xyz=xyz@pose[:3,:3].T+pose[:3,3]
                            here=np.all(np.floor(xyz/.008).astype(int)==np.asarray(keys)[support],axis=1)
                            good=support[here];mm[v[good],u[good]]=True
                        for name_v,mem in memories.items():
                            counts=Counter()
                            before=[mem.cells[k]["state"] for k in keys]
                            mem.update_depth(i,cam,obs,mm,g,object_id="static_background")
                            for ix in np.flatnonzero(cross&(states>=3)):
                                a=before[ix];b=mem.cells[keys[ix]]["state"]
                                counts["independent_comparisons"]+=1
                                if states[ix]==5:
                                    counts["clear_ray_comparisons"]+=1
                                    counts["contradicted_on_clear_ray"]+=int(b=="contradicted")
                                    counts["new_static_retraction_events"]+=int(a!="contradicted" and b=="contradicted")
                                if states[ix]==4:
                                    counts["foreground_occlusions"]+=1
                                    counts["new_retraction_on_occlusion"]+=int(a!="contradicted" and b=="contradicted")
                                if states[ix]==3 and a=="contradicted":
                                    counts["previously_contradicted_seen_consistently"]+=1
                                    counts["reconfirmed_on_consistent_ray"]+=int(b=="confirmed")
                            versions[name_v].update(counts);frame_metrics[name_v]=dict(counts)
                    else:
                        tally["weak_anchor_frames"]=1;rec.update(emitted_unknown=True,reason="insufficient_common_static_anchors")
                        for mem in memories.values():
                            mem.update_depth(i,cam,None,None,g,object_id="static_background",anchor_valid=False)
                else:
                    rec.update(emitted_unknown=True,reason="no_prior_background_anchor")
                    tally["bootstrap_frames"]=1
                rec["variants"]=frame_metrics
                if g not in groups:
                    seed=EvidenceMemory("M0");mask=seed_rays(z,canvas)
                    seed.update_depth(i,cam,z,mask,g,object_id="static_background")
                    if seed.cells:
                        groups.add(g)
                        for key,val in seed.cells.items():
                            if key not in birth:
                                birth[key]=g
                                for mem in memories.values():mem.cells[key]=copy.deepcopy(val)
                                tally["seed_anchors"]+=1
            votes.update(tally);by_split[r["split"]].update(tally)
            rec.update(counts=dict(tally),anchor_count_after=len(birth))
            trace.write(json.dumps(rec,separators=(",",":"))+"\n")
            if (i+1)%100==0:
                trace.flush();write_json(OUT/"progress.json",{"positions":i+1,"total":len(rows),"anchors":len(birth)})
                print(f"{i+1}/1846, anchors={len(birth)}",flush=True)
    write_json(complete,{"scope":"real_hall_static_assumption_consistency","positions":len(rows),
        "input_roles":"all frames in chronology; analysis-only, includes publication holdouts; export to training disabled",
        "pose_source":"fixed captured B2 OpenCV camera poses, no pose fitting, no independently measured camera truth",
        "counts":dict(votes),"counts_by_original_split":dict(by_split),"methods":versions,
        "source_groups":len(groups),"anchors":len(birth),"hard_unknown_ids":protected,
        "source_sampling":"up to 24 background voxels per ten-second group from its first eligible view. 6x4 tile centers, scoped 25x25 pixel search. All 1846 time positions retained.",
        "comparison":"Cross-group means target frame and anchor birth belong to different ten-second acquisition groups. Each admitted in-canvas ray uses a finite primary sensor reading and local 5x5 continuity support; at least three consistent static anchors spanning two image regions for updates.",
        "threshold_m":TOL,"labels":"A new contradicted surface is a suspected false update under the static-scene assumption. Out-of-view, foreground occlusion, missing depth and uncertain camera pose provide no clearing evidence. Actual object motion and pose error remain unseparated.",
        "object_identity":"No independently labeled movable hall object. M2 surface decisions share M1 visibility, without object-pose version evidence.",
        "elapsed_s":time.monotonic()-started,"frame_audit":member(fpath),"inputs":member(P0/"input_manifest.json"),
        "selection":member(P0/"selection.json"),"redaction":member(opt),"implementation":member(Path(__file__))})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"worker.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        worker()
