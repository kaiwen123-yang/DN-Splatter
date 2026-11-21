"""Rebuild RGB-D seeds and select independent primary-training SfM tracks in the corrected gauge."""
from collections import Counter
from pathlib import Path
import subprocess

import cv2
import numpy as np
import open3d as o3d

from capture_quality_paths import B1, GEOM, GEORUN, P, SCENE, check, read, record, write
from capture_rectify import edges, files

FLIP=np.diag([1.,-1.,-1.,1.])


def copy_inputs():
    source=B1/"inputs"
    check(record(source/"geometry_source_manifest.json")["sha256"]==P["baseline"]["original_geometry_manifest_sha256"],
          "B1 original geometry")
    SCENE.mkdir(parents=True,exist_ok=True)
    for name in ("images","depth","normals_from_pretrain","source_frames.json","splits.json","training_inputs.json"):
        # Separate inodes even on filesystems without copy-on-write support.
        subprocess.run(["cp","-a","--reflink=auto",source/name,SCENE],check=True)
    for name in ("images","depth","normals_from_pretrain"):
        members=sorted((source/name).iterdir())
        check(len(members)==1846 and all(not (SCENE/name/p.name).samefile(p) for p in members),name)
    write(GEOM/"input_copy.json",dict(source=str(source),counts=dict(train=1478,val=184,test=184)))


def sensor_seeds(meta, frames):
    Ks=np.array(read(GEORUN/"rect/grids.json")["640"]["K_cv"])
    coords,colors,counts=[],[],{}
    for name in meta["train_filenames"]:
        stem=Path(name).stem
        pose=np.asarray(frames[name]["transform_matrix"])@FLIP
        z=np.load(GEORUN/"rect/640/depth"/(stem+".npy"))
        occ=cv2.imread(str(GEORUN/"rect/640/occlusion"/(stem+".png")),0)
        good=(z>=.3)&(z<=6)&(occ==0)&~cv2.dilate(edges(z).astype("u1"),np.ones((3,3),"u1")).astype(bool)
        y,x=np.nonzero(good[::4,::4]); y*=4; x*=4
        raw=np.column_stack(((x-Ks[0,2])/Ks[0,0],(y-Ks[1,2])/Ks[1,1],np.ones(len(x))))*z[y,x,None]
        coords.append(raw@pose[:3,:3].T+pose[:3,3])
        rgb=cv2.imread(str(GEORUN/"rect/640"/name))
        colors.append(rgb[y,x,::-1]/255.)
        counts[stem]=len(x)
    cloud=o3d.geometry.PointCloud()
    cloud.points=o3d.utility.Vector3dVector(np.concatenate(coords))
    cloud.colors=o3d.utility.Vector3dVector(np.concatenate(colors))
    return cloud.voxel_down_sample(.03), counts


def sparse_seeds(meta,frames,sensor):
    with np.load(GEOM/"tracks.npz") as f:
        ids,raw,obs=(f[k] for k in ("point_ids","xyz","obs"))
    poses=read(GEORUN/"metric/poses.json")
    origin=np.asarray(poses["sfm_metric_world_to_local"])
    xyz=(raw*poses["scale"])@origin[:3,:3].T+origin[:3,3]
    n=len(xyz)
    fid,pi=obs[:,:2].astype(int).T
    N=len(pi)
    err=np.empty(N); pos=np.empty(N,bool); ray=np.empty((N,3)); support=np.zeros(N,bool); conflict=np.zeros(N,bool)
    rgb=np.empty((N,3),dtype="u1"); counted=np.zeros(N,bool)
    bad={int(k) for k,v in read(GEOM/"pose.json")["results"].items() if not v["accepted"]}
    reliable=~np.isin(fid,list(bad))
    check(np.all(fid[1:]>=fid[:-1]),"frame-sorted training tracks")
    for f,lo,number in zip(*np.unique(fid,return_index=True,return_counts=True)):
        sl=slice(lo,lo+number)
        name=f"images/frame_{f:05d}.png"
        pose=np.asarray(frames[name]["transform_matrix"])@FLIP
        pc=(xyz[pi[sl]]-pose[:3,3])@pose[:3,:3]
        zs=pc[:,2]; pos[sl]=zs>0
        uv=obs[sl,2:4]
        predicted=pc[:,:2]/np.where(zs!=0,zs,np.nan)[:,None]*[meta["fl_x"],meta["fl_y"]]+[meta["cx"],meta["cy"]]
        err[sl]=np.linalg.norm(predicted-uv,axis=1)
        vec=xyz[pi[sl]]-pose[:3,3]
        ray[sl]=vec/np.maximum(1.e-20,np.linalg.norm(vec,axis=1))[:,None]
        x,y=np.floor(uv).astype(int).T
        inside=(x>=0)&(x<meta["w"])&(y>=0)&(y<meta["h"])
        x,y=x.clip(0,meta["w"]-1),y.clip(0,meta["h"]-1)
        color=cv2.imread(str(SCENE/name))
        rgb[sl]=color[y,x,::-1]
        counted[sl]=inside
        z=np.load(SCENE/frames[name]["depth_file_path"])[y,x]
        valid=inside & np.isfinite(z)&(z>=.3)&(z<=6)
        tau=np.maximum(.05,.03*z)
        support[sl]=valid&(np.abs(zs-z)<=tau)
        conflict[sl]=valid&(zs<z-tau)
    def summed(a): return np.bincount(pi,weights=a,minlength=n)
    count=np.bincount(pi,minlength=n)
    order=np.lexsort((np.nan_to_num(err,nan=np.inf),pi))
    start=np.cumsum(count)-count
    def quantile(q):
        index=np.maximum(count-1,0)*q
        a=(start+np.floor(index).astype(int)).clip(0,N-1)
        b=(start+np.ceil(index).astype(int)).clip(0,N-1)
        frac=index-np.floor(index)
        return err[order[a]]*(1-frac)+err[order[b]]*frac
    checked=np.ones(n,bool); reasons=Counter()
    def reject(label,condition):
        reasons[label]=int((checked&~condition).sum()); checked[:] &= condition
    reject("less_than_three_primary_views",count>=3)
    reject("invalid_or_nonpositive",np.isfinite(xyz).all(1)&(summed(pos)==count)&(summed(np.isfinite(err))==count))
    reject("reprojection_median",quantile(.5)<=1.)
    reject("reprojection_p95",quantile(.95)<=2.)
    reject("sensor_conflicts",summed(support)>=summed(conflict))
    reject("only_unresolved_region",summed(reliable)>0)
    reject("no_primary_training_color",summed(counted)>0)
    cos=np.cos(np.deg2rad(1.))
    first=ray[order[start.clip(0,N-1)]]
    passes=summed((ray*first[pi]).sum(1)<=cos)>0
    for j in np.flatnonzero(checked&~passes):
        directions=ray[order[start[j]:start[j]+count[j]]]
        passes[j]=np.min(directions@directions.T)<=cos
    reject("insufficient_ray_angle",passes)
    # The sensor-first voxel preference and deterministic point-ID tie break apply in the same metric grid.
    def keys(a):
        return np.ascontiguousarray(np.floor(a/.03).astype("<i4")).view(np.dtype("V12")).ravel()
    candidates=np.flatnonzero(checked)
    overlap=np.isin(keys(xyz[candidates]),keys(sensor))
    reasons["sensor_voxel_overlap"]=int(overlap.sum())
    candidates=candidates[~overlap]
    _,unique=np.unique(keys(xyz[candidates]),return_index=True)
    reasons["duplicate_sfm_voxel"]=len(candidates)-len(unique)
    keep=candidates[np.sort(unique)]
    weights=summed(counted)[keep]
    colors=np.column_stack([summed(rgb[:,c]*counted)[keep]/weights/255. for c in range(3)])
    np.savez_compressed(GEOM/"added_tracks.npz",id=ids[keep],xyz_m=xyz[keep],count=count[keep],
                        support=summed(support)[keep],conflict=summed(conflict)[keep])
    return xyz[keep],colors,dict(input_tracks=n,kept=len(keep),rejected=dict(reasons),raw_unit="SfM",
           applied_frozen_scale_once=poses["scale"],world_to_local=origin.tolist(),local_BA=False)


def build():
    meta=read(SCENE/"transforms.json")
    frames={f["file_path"]:f for f in meta["frames"]}
    check([len(meta[k+"_filenames"]) for k in ("train","val","test")]==[1478,184,184],"frozen splits")
    cloud,counts=sensor_seeds(meta,frames)
    sensor=len(cloud.points)
    extra,colors,stats=sparse_seeds(meta,frames,np.asarray(cloud.points))
    if len(extra):
        add=o3d.geometry.PointCloud()
        add.points=o3d.utility.Vector3dVector(extra); add.colors=o3d.utility.Vector3dVector(colors)
        cloud+=add
    check(o3d.io.write_point_cloud(str(SCENE/meta["ply_file_path"]),cloud),"combined PLY export")
    geom=read(B1/"inputs/geometry.json")
    geom.update(sensor_points=sensor,init_points=len(cloud.points),init_input_count=sum(counts.values()),
                sfm_points=len(extra),init_from="corrected primary train sensor seeds followed by qualified training SfM tracks")
    write(SCENE/"geometry.json",geom)
    members=files(SCENE,["training_manifest.json"])
    inventory={r["path"]:r for r in read(B1/"inputs/training_manifest.json")["files"]}
    pixel_files=[r for r in members if Path(r["path"]).parts[0] in ("images","depth","normals_from_pretrain")]
    check(len(pixel_files)==3*1846 and all(r==inventory[r["path"]] for r in pixel_files),"B1 pixel input hashes")
    write(SCENE/"training_manifest.json",dict(files=members))
    write(GEOM/"initialization.json",dict(sensor_points=sensor,sfm_points=len(extra),total_points=len(cloud.points),
          sensor_samples_per_frame=counts,sfm=stats,ply=record(SCENE/meta["ply_file_path"]),
          input_manifest=record(SCENE/"training_manifest.json")))


if __name__=="__main__":
    import sys
    GEOM.mkdir(parents=True,exist_ok=True)
    {"copy":copy_inputs,"build":build}[sys.argv[1]]()
