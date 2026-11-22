"""Resident native training images and exact frustum rejection for the A3 evidence pass."""
import ctypes as ct
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from capture_surface_repeatability import edge_distance
from capture_surface_submaps import PACKET, FLIP, camera_parameters
from capture_surface_tiles import ROOT, RUN, read, sha


class Evidence:
    def __init__(self, export):
        identity=read(export/"identity.json")
        tol=read(export/"tolerances.json")
        if tol["camera_sha256"]!=identity["cameras"]["sha256"]:
            raise ValueError("Tolerance camera identity")
        for k in ("cameras","selection","calibration"):
            if sha(ROOT/identity[k]["path"])!=identity[k]["sha256"]:
                raise ValueError("Camera/source digest: "+k)
        src=Path(__file__).with_name("capture_surface_evidence.cu")
        lib=RUN/"control"/("evidence-"+sha(src)[:16]+".so")
        if not lib.exists():
            tmp=lib.with_suffix(".pending.so")
            subprocess.run([str(Path(os.environ["CUDA_HOME"])/"bin/nvcc"),"-O3","-arch=sm_120","--shared",
                            "--fmad=false","-Xcompiler","-fPIC",str(src),"-o",str(tmp)],check=True)
            tmp.replace(lib)
        self.lib=ct.CDLL(str(lib))
        p=ct.c_void_p; i=ct.c_int; d=ct.c_double
        self.lib.votes.argtypes=[i,p,p,p,i,p,p,p,p,i,i,d,d,d,d,p,i,p,p,p]
        self.lib.votes.restype=i
        self.lib.choose_owners.argtypes=[i,p,p,p,p,p]
        self.lib.choose_owners.restype=None
        cam=read(ROOT/identity["cameras"]["path"])
        cal=read(ROOT/identity["calibration"]["path"])
        _,_,extr=camera_parameters(cam,cal)
        self.k=cal["cameras"]["depth"]
        self.h,self.w=self.k["height"],self.k["width"]
        frames={f["file_path"]:np.asarray(f["transform_matrix"])@FLIP@extr for f in cam["frames"]}
        packets={r["id"]:r["depth"]["file"] for r in read(ROOT/identity["selection"]["path"])}
        names=[]; gids=[]
        for g in identity["groups"]:
            for name in sorted(g["names"]):
                names.append(name);gids.append(g["id"])
        self.ng=len(identity["groups"])
        if self.ng>256 or len(names)!=1478:raise ValueError("Unexpected camera population")
        self.poses=np.array([frames[name] for name in names])
        self.gids=np.asarray(gids,dtype="i4")
        self.cam=torch.tensor(self.poses,device="cuda")
        self.group=torch.tensor(self.gids,device="cuda")
        self.depth=torch.empty((len(names),self.h,self.w),dtype=torch.int16,device="cuda")
        self.edge=torch.empty((len(names),self.h,self.w),dtype=torch.float32,device="cuda")
        self.terms=torch.tensor([x["empirical_term_m"] for x in tol["bins"]],dtype=torch.float64,device="cuda")
        cv2.setNumThreads(1)
        for i,name in enumerate(names):
            with np.load(PACKET/packets[Path(name).stem]) as f: raw=np.ascontiguousarray(f["depth"],dtype="u2")
            self.depth[i].copy_(torch.from_numpy(raw.view("i2")))
            self.edge[i].copy_(torch.from_numpy(edge_distance(raw.astype("f4")/1000.)))
        self.centers=self.poses[:,:3,3].copy()
        # Six inward world-space halfspaces. Discard only cameras with the entire AABB outside.
        fx,fy,cx,cy=[self.k[k] for k in ("fx","fy","cx","cy")]
        local=np.array([[fx,0,cx+.5],[-fx,0,self.w-.5-cx],[0,fy,cy+.5],
                        [0,-fy,self.h-.5-cy],[0,0,1],[0,0,-1.]])
        self.plane=np.einsum("vij,kj->vki",self.poses[:,:3,:3],local)
        self.offset=-np.einsum("vki,vi->vk",self.plane,self.centers) + [0,0,0,0,-.3,6.]
        print("Resident full native depth and interior distances:",len(names),"training views",flush=True)

    def vote(self, pts, sources):
        pts=np.ascontiguousarray(pts,dtype="f8")
        sources=np.broadcast_to(sources,(len(pts),)).astype("i4",copy=True)
        if not len(pts):return np.empty((0,4),"i4"),np.empty(0),np.empty(0,"i4")
        hi,lo=pts.max(axis=0),pts.min(axis=0)
        bound=self.plane@((hi+lo)/2)+np.abs(self.plane)@((hi-lo)/2)+self.offset
        views=np.flatnonzero((bound>=-1e-8).all(axis=1)).astype("i4")
        v=torch.as_tensor(views,device="cuda")
        result=[];residual=[];witness=[]
        for first in range(0,len(pts),131072):
            points=torch.as_tensor(pts[first:first+131072],device="cuda")
            src=torch.as_tensor(sources[first:first+131072],device="cuda")
            n=len(points)
            count=torch.empty((n,4),dtype=torch.int32,device="cuda")
            error=torch.empty(n,dtype=torch.float64,device="cuda"); view=torch.empty_like(src)
            torch.cuda.synchronize()
            rc=self.lib.votes(n,points.data_ptr(),src.data_ptr(),v.data_ptr(),len(views),self.group.data_ptr(),
                 self.cam.data_ptr(),self.depth.data_ptr(),self.edge.data_ptr(),self.h,self.w,
                 *[self.k[k] for k in ("fx","fy","cx","cy")],self.terms.data_ptr(),self.ng,
                 count.data_ptr(),error.data_ptr(),view.data_ptr())
            if rc:raise RuntimeError("CUDA native vote error "+str(rc))
            result.append(count.cpu().numpy());residual.append(error.cpu().numpy());witness.append(view.cpu().numpy())
        return np.concatenate(result),np.concatenate(residual),np.concatenate(witness)
