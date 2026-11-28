"""Visibility-aware surface evidence and immutable object-pose history."""

import argparse
import copy
import time
from pathlib import Path

import numpy as np

from interactive_push_assets import write_json


class EvidenceMemory:
    def __init__(self,variant):
        if variant not in ("M0","M1","M2"):
            raise ValueError(variant)
        self.variant=variant
        self.reset()

    def reset(self):
        self.cells={}
        self.versions=[]
        self.events=[]
        self.groups={}
        self.last_step=-1

    def update_depth(self, step, camera, depth, mask, source_group, object_id="movable", anchor_valid=True):
        """Input is a classified measured RGB-D view; an absence of depth provides no free-space evidence."""
        if step<self.last_step:
            raise ValueError("Out-of-order causal observation")
        self.last_step=step
        if not anchor_valid:
            self.events.append((step,"insufficient_static_anchors"))
            return
        k=np.array(camera["K"]);t=np.array(camera["T_world_camera_cv"])
        if self.variant!="M0":
            for cell,record in self.cells.items():
                p=(record["xyz"]-t[:3,3])@t[:3,:3]
                if p[2]<=0:continue
                h=k@p;u,v=np.rint(h[:2]/h[2]).astype(int)
                if not (0<=u<depth.shape[1] and 0<=v<depth.shape[0]): continue
                observed=depth[v,u]
                if not np.isfinite(observed) or observed<=0:continue
                if observed>p[2]+.025:
                    record["state"]="contradicted"
                elif observed<p[2]-.025 and record["state"]!="contradicted":
                    record["state"]="occluded"
        y,x=np.nonzero(mask & np.isfinite(depth) & (depth>0))
        z=depth[y,x]
        pts=np.column_stack(((x-k[0,2])*z/k[0,0],(y-k[1,2])*z/k[1,1],z))@t[:3,:3].T+t[:3,3]
        keys=np.floor(pts/.008).astype(int)
        _,indexes=np.unique(keys,axis=0,return_index=True)
        for i in indexes:
            key=tuple(keys[i]);old=self.cells.get(key)
            groups=set(old["source_groups"]) if old else set()
            groups.add(source_group)
            self.cells[key]={"xyz":pts[i],"object_id":object_id,"state":"confirmed","last_seen":step,
                             "source_groups":groups,"version":len(self.versions)+1}

    def pose_observation(self, step, pose, source_group, valid, object_id="movable", independent=True):
        if step<self.last_step:
            raise ValueError("Future observation in stream")
        self.last_step=step
        if not valid:
            self.events.append((step,"occluded_or_unreliable"));return None
        pose=np.asarray(pose,float)
        if not np.isfinite(pose).all():
            return None
        if not independent:
            self.events.append((step,"hypothesized"));return None
        record={"step":int(step),"object_id":object_id,"source_group":str(source_group),"pose":pose.tolist(),
                "version":len(self.versions)+2,"state":"confirmed"}
        self.versions.append(record)
        return record

    def as_of(self,step):
        if self.variant!="M2":
            raise ValueError("Rollback requires object-version history")
        return next((copy.deepcopy(v) for v in reversed(self.versions) if v["step"]<=step),None)


def sanity(out):
    # The reference is an independent orthographic-motion program, not a render of the candidate scene.
    k=np.array([[90.,0.,31.],[0.,90.,23.],[0.,0.,1.]])
    cam={"K":k,"T_world_camera_cv":np.eye(4)}
    y,x=np.indices((48,64))
    report={}
    for variant in ("M0","M1","M2"):
        m=EvidenceMemory(variant);nfalse=nstale=0;nf=0; delete_hidden=0
        start=time.perf_counter()
        for frame in range(32):
            col=18 if frame<10 else 18+min(20,2*(frame-10))
            truth=(abs(x-col)<5)&(abs(y-23)<5)
            foreground=np.zeros_like(truth)
            if 13<=frame<19:foreground=(x>24)&(x<43)&(y>15)&(y<32)
            depth=np.where(foreground,.57,np.where(truth,.75,1.)).astype(np.float32)
            measured=truth&~foreground
            yy,xx=np.nonzero(measured)
            m.update_depth(frame,cam,depth,measured,f"cameraA-{frame}")
            if len(xx):
                p=[float((np.mean(xx)-31)*.75/90),float((np.mean(yy)-23)*.75/90),0.]
                m.pose_observation(frame,p,frame,True)
            present=np.zeros_like(truth)
            for rec in m.cells.values():
                p=rec["xyz"];u,v=np.rint((k@p)[:2]/p[2]).astype(int)
                if rec["state"]!="contradicted": present[v,u]=True
                elif truth[v,u] and foreground[v,u]: delete_hidden+=1
            nfalse+=int(np.count_nonzero(present & ~truth & ~foreground))
            nstale+=int(np.count_nonzero(~truth & ~foreground))
            nf+=int(np.count_nonzero(foreground&truth))
        before=len(m.versions)
        history=m.as_of(9) if variant=="M2" else None
        m.reset()
        report[variant]={"frames":32,"false_old_geometry_pixel_times":nfalse,"visible_background_pixel_times":nstale,
             "hidden_true_surface_pixel_times":nf,"contradicted_hidden_cell_times":delete_hidden,
             "pose_versions":before,"rollback_as_of_9":history,"cells_after_reset":len(m.cells),
             "events_after_reset":len(m.events),"elapsed_s":time.perf_counter()-start}
    write_json(out,{"scope":"simulation_sanity","teacher":"frozen independent analytic depth and moving rectangle",
                    "kinematic_motion_known_only_to_evaluator":True,"results":report})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("out",type=Path)
    sanity(p.parse_args().out)
