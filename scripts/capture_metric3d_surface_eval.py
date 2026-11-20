"""Raycast final surfaces, z-buffer model samples and report accuracy with fixed sensor denominators."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import open3d as o3d
import torch

from capture_metric3d_common import (BASE, B0_REPORT, HOLDOUT, RAW_PREFIX, REPORT, RUN, SCENE,
                                     demand, model_paths, read, selection, sha, write)
from capture_s01_eval import color
from capture_s01_mesh import shaded, titled
from capture_rectify import imwrite
from vr_room_run import CPU

MESHES = [f"{m}_{type_}" for m in ("B0","B1") for type_ in ("raw_mesh","supported_mesh")]
GROUPS = {"common_four_meshes": MESHES,
          "common_raw_models": ["B0_raw_mesh","B1_raw_mesh"],
          "common_supported_models": ["B0_supported_mesh","B1_supported_mesh"],
          "common_B0_exports": ["B0_raw_mesh","B0_supported_mesh"],
          "common_B1_exports": ["B1_raw_mesh","B1_supported_mesh"]}
WORK = RUN/"surface_evaluation"


def compressed(path,data):
    with gzip.open(path,"wt",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,separators=(",",":"),allow_nan=False)


def world_camera(frame):
    return np.asarray(frame["transform_matrix"],np.float64) @ np.diag([1.,-1.,-1.,1.])


def rays(xx, yy, cam, k):
    fx,fy,cx,cy=k
    d=np.stack(((xx+.5-cx)/fx,(yy+.5-cy)/fy,np.ones_like(xx)),-1)
    v=np.broadcast_to(cam[:3,3],d.shape)
    return o3d.core.Tensor(np.concatenate((v,d@cam[:3,:3].T),axis=-1).astype(np.float32))


def project_cloud(xyz, cam, k, h, w, crop):
    """Nearest camera Z and deterministic winning PLY index in each pixel."""
    x0,y0,x1,y1=crop
    shape=(y1-y0,x1-x0)
    depth=torch.full((shape[0]*shape[1],),float("inf"),device="cuda",dtype=torch.float64)
    winner=torch.full(depth.shape,torch.iinfo(torch.int64).max,device="cuda",dtype=torch.int64)
    R=torch.as_tensor(cam[:3,:3],device="cuda")
    t=torch.as_tensor(cam[:3,3],device="cuda")
    for start in range(0,len(xyz),400000):
        pc=(xyz[start:start+400000]-t)@R
        x,y,z=pc.unbind(-1)
        u,v=k[0]*x/z+k[2],k[1]*y/z+k[3]
        valid=(z>0)&(u>=x0)&(u<x1)&(v>=y0)&(v<y1)
        ids=torch.nonzero(valid).flatten()
        pix=(v[ids].long()-y0)*shape[1]+u[ids].long()-x0
        zz=z[ids]
        local=torch.full_like(depth,float("inf"))
        local.scatter_reduce_(0,pix,zz,reduce="amin",include_self=True)
        closer=local<depth
        winner[closer]=torch.iinfo(torch.int64).max
        depth=torch.minimum(depth,local)
        best=zz==depth[pix]
        winner.scatter_reduce_(0,pix[best],ids[best]+start,reduce="amin",include_self=True)
    winner[~torch.isfinite(depth)]=-1
    return depth.reshape(shape).cpu().numpy(),winner.reshape(shape).int().cpu().numpy()


def raw_sources(label, cloud, meta, frames):
    run,_=model_paths(label)
    rows=read(run/"mesh/sampling.json")["views"]
    lookup={}
    source_view=np.empty(len(cloud),np.int16)
    source_pixel=np.full(len(cloud),-1,np.int32)
    start=0
    max_offset=0.
    for i,(row,name) in enumerate(zip(rows,meta["train_filenames"],strict=True)):
        end=start+row["samples"]
        stem=Path(name).stem
        demand(row["id"]==stem,"raw cloud ordering")
        lookup[stem]=(start,end)
        cam=world_camera(frames[stem])
        pc=(cloud[start:end]-cam[:3,3])@cam[:3,:3]
        z=pc[:,2]
        with np.errstate(invalid="ignore",divide="ignore"):
            u=meta["fl_x"]*pc[:,0]/z+meta["cx"]
            v=meta["fl_y"]*pc[:,1]/z+meta["cy"]
        inside=np.isfinite(u)&np.isfinite(v)&(u>=0)&(u<meta["w"])&(v>=0)&(v<meta["h"])&(z>0)
        ids=np.flatnonzero(inside)
        x,y=u[ids].astype(int),v[ids].astype(int)
        residual=np.maximum(np.abs(u[ids]-(x+.5)),np.abs(v[ids]-(y+.5)))
        # Uncertain reconstructions retain their known view and an unknown pixel.
        safe=residual<.1
        source_view[start:end]=i
        source_pixel[start+ids[safe]]=y[safe]*meta["w"]+x[safe]
        if label=="B1":
            recorded=np.load(run/"mesh/source_pixels"/(stem+".npy"))
            demand(len(recorded)==end-start,"recorded B1 sampling")
            source_pixel[start:end]=recorded
        if len(ids):
            max_offset=max(max_offset,float(residual.max()))
        start=end
    demand(start==len(cloud),"raw cloud point count")
    np.savez_compressed(WORK/(label+"_raw_sources.npz"),view=source_view,pixel=source_pixel)
    return lookup,dict(ordered_views=len(rows),point_count=start,unresolved_pixels=int((source_pixel<0).sum()),
                       maximum_reprojected_center_offset_px=max_offset,
                       method=("Original sampled indices captured during B1 export" if label=="B1" else
                               "Known B0 legacy per-view PLY blocks, pixel recovered within 0.1 pixel of the native center."))


def geometry(meta,frames):
    scenes, clouds, sources, intervals={}, {}, {}, {}
    global_tiles=[]
    for label in ("B0","B1"):
        run,_=model_paths(label)
        sup=RUN/"supported"/label
        for kind,path in (("raw",run/"mesh"/(RAW_PREFIX+"poisson_mesh.ply")),("pre_crop",sup/"before_crop.ply"),("supported",sup/"supported.ply")):
            mesh=o3d.io.read_triangle_mesh(str(path))
            key=label+"_"+kind+"_mesh"
            if not len(mesh.triangles):
                scenes[key]=None
                global_tiles.append(titled(np.full((240,320,3),246,np.uint8),key+" empty"))
                continue
            scene=o3d.t.geometry.RaycastingScene(nthreads=CPU)
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            scenes[key]=scene
            v=np.asarray(mesh.vertices)
            center=(v.min(0)+v.max(0))*.5
            distance=np.linalg.norm(v.max(0)-v.min(0))
            ext_ray=scene.create_rays_pinhole(48,center.astype("f4"),(center+np.array([.55,-.45,-.8])*distance*1.1).astype("f4"),
                                                np.array([0,-1,0],np.float32),320,240)
            gray,_=shaded(scene,ext_ray,mesh)
            global_tiles.append(titled(gray,key))
        for kind,path in (("raw",run/"mesh"/(RAW_PREFIX+"pcd.ply")),("fused",sup/"fused.ply")):
            pcd=o3d.io.read_point_cloud(str(path))
            xyz=np.asarray(pcd.points)
            if kind=="raw":
                intervals[label],sources[label]=raw_sources(label,xyz,meta,frames)
            clouds[label+"_"+kind+"_pcd"]=torch.as_tensor(xyz,dtype=torch.float64,device="cuda")
    imwrite(REPORT/"previews/surfaces_global.jpg",np.vstack((np.hstack(global_tiles[:3]),np.hstack(global_tiles[3:]))))
    return scenes,clouds,sources,intervals


def observations(stem, split, frame, meta, models, crop, local=False):
    x0,y0,x1,y1=crop
    img=np.s_[y0:y1,x0:x1]
    k=[meta[n] for n in ("fl_x","fl_y","cx","cy")]
    cam=world_camera(frame)
    ref=np.load(SCENE/frame["depth_file_path"],mmap_mode="r")[img]
    mask=np.isfinite(ref)&(ref>=.3)&(ref<=6.)
    # Cast only needed rays on the held-out reference denominator. Local previews cover the crop.
    yy,xx=np.mgrid[y0:y1,x0:x1]
    pix=np.flatnonzero(mask)
    query=rays(xx if local else xx[mask],yy if local else yy[mask],cam,k) if local or len(pix) else None
    scenes,clouds,_,intervals=models
    zall,winners={},{}
    for key,scene in scenes.items():
        own_shape=ref.shape if local else (len(pix),)
        zall[key]=(scene.cast_rays(query,nthreads=CPU)["t_hit"].numpy()
                   if scene is not None and query is not None else np.full(own_shape,np.inf,np.float32))
    for key,xyz in clouds.items():
        z,idx=project_cloud(xyz,cam,k,meta["h"],meta["w"],crop)
        zall[key]=z if local else z[mask]
        winners[key]=idx if local else idx[mask]
    for label in ("B0","B1"):
        run,_=model_paths(label)
        path=run/"video"/(stem+"_depth.npy") if local else run/"renders"/split/"depth"/(stem+".npy")
        z=np.load(path,mmap_mode="r")[img]
        zall[label+"_model"]=z if local else z[mask]
        if local:
            i,j=intervals[label][stem]
            sub=clouds[label+"_raw_pcd"][i:j]
            arr,ids=project_cloud(sub,cam,k,meta["h"],meta["w"],crop)
            zall[label+"_raw_own"]=arr
            winners[label+"_raw_own"]=np.where(ids>=0,ids+i,-1)
            with np.load(RUN/"supported"/label/"views"/(stem+".npz")) as src:
                arr,ids=project_cloud(torch.as_tensor(src["xyz"],device="cuda"),cam,k,meta["h"],meta["w"],crop)
                zall[label+"_supported_own"]=arr
                pixels=src["pixel"]
                found=ids>=0
                original=np.full(ids.shape,-1,np.int32)
                original[found]=pixels[ids[found]]
                winners[label+"_supported_own_source_pixel"]=original
    destination=WORK/("local" if local else split)
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination/(stem+"_winners.npz"),reference_flat_pixel=pix,x0=x0,y0=y0,**winners)
    return ref,mask,zall


def stats(ref,z,valid):
    z=z[valid].astype(np.float64)
    d=ref[valid].astype(np.float64)
    err=z-d
    absolute=np.abs(err)
    tau=np.maximum(.05,.03*d)
    counts=dict(reference=len(ref),hits=len(z),misses=len(ref)-len(z),correct_tau=int((absolute<=tau).sum()),
                correct_05=int((absolute<=.05).sum()),correct_10=int((absolute<=.1).sum()),correct_20=int((absolute<=.2).sum()),
                foreground=int((err < -tau).sum()),nearer_than_03=int((z<.3).sum()))
    sums=dict(abs_sum=float(absolute.sum()),squared_sum=float(np.dot(err,err)),signed_sum=float(err.sum()))
    return counts|sums,absolute


def measures(total, errors):
    n=total["reference"]
    hits=total["hits"]
    result=dict(total)
    for k in ("hits","correct_tau","correct_05","correct_10","correct_20","foreground","nearer_than_03"):
        result[k+"_fraction_all_reference"]=total[k]/n if n else None
    for k in ("foreground","nearer_than_03"):
        result[k+"_fraction_hits"]=total[k]/hits if hits else None
    result["MAE_hit_m"]=total["abs_sum"]/hits if hits else None
    result["RMSE_hit_m"]=float(np.sqrt(total["squared_sum"]/hits)) if hits else None
    result["signed_mean_hit_m"]=total["signed_sum"]/hits if hits else None
    quantile=np.percentile(errors,[50,95]) if hits else (None,None)
    result["P50_hit_m"],result["P95_hit_m"]=quantile
    return result


def variants(zall, include_all=False):
    valid={k:np.isfinite(z)&(z>0) for k,z in zall.items()}
    for k in zall:
        yield k,zall[k],valid[k]
    groups=dict(GROUPS)
    if include_all:
        for label in ("B0","B1"):
            for kind,pc,mesh in (("raw","raw_pcd","raw_mesh"),("supported","fused_pcd","supported_mesh")):
                groups[f"local_{label}_{kind}"]=[label+"_"+v for v in ("model",pc,mesh)]
        groups["local_common_all_models_clouds_meshes"]=[k for k in zall if "_own" not in k and "pre_crop" not in k]
    for group,names in groups.items():
        keep=np.logical_and.reduce([valid[k] for k in names])
        for k in names:
            yield group+"__"+k,zall[k],keep


def aggregate(rows,errors):
    sums={k:sum(row[k] for row in rows) for k in ("reference","hits","misses","abs_sum","squared_sum","signed_sum",
          "correct_tau","correct_05","correct_10","correct_20","foreground","nearer_than_03")}
    demand(sums["hits"]==len(errors),"pooled hit cardinality")
    pooled=measures(sums,errors)
    keys=[k for k in pooled if k.endswith(("_m","_reference","_hits"))]
    equal={}
    for k in keys:
        vals=[r[k] for r in rows if r[k] is not None]
        equal[k]=float(np.mean(vals)) if vals else None
    return dict(frames=len(rows),frames_with_reference=sum(r["reference"]>0 for r in rows),
                frames_with_hit=sum(r["hits"]>0 for r in rows),pixel_pooled=pooled,equal_frame=equal)


def main():
    WORK.mkdir(exist_ok=True)
    meta=read(SCENE/"transforms.json")
    frames={Path(r["file_path"]).stem:r for r in meta["frames"]}
    models=geometry(meta,frames)
    times={r["id"]:r["relative_s"] for r in selection()}
    summary={}
    for split in HOLDOUT:
        names=[Path(x).stem for x in meta[split+"_filenames"]]
        demand(len(names)==184,split)
        records=[]
        with tempfile.TemporaryDirectory(dir=WORK,prefix=split+"_quantiles_") as work:
            tmp=Path(work)
            for stem in names:
                ref,mask,zall=observations(stem,split,frames[stem],meta,models,(0,0,meta["w"],meta["h"]))
                period=int(times[stem]//30)*30
                row=dict(id=stem,relative_s=times[stem],bin_start=period,metrics={})
                for key,z,keep in variants(zall):
                    counts,error=stats(ref[mask],z,keep)
                    row["metrics"][key]=measures(counts,error)
                    with (tmp/f"{key}_{period}.bin").open("ab") as f:
                        error.tofile(f)
                records.append(row)
            summaries={}
            for key in records[0]["metrics"]:
                pieces,by_window=[],[]
                for period in sorted({r["bin_start"] for r in records}):
                    err=np.fromfile(tmp/f"{key}_{period}.bin",dtype=np.float64)
                    values=[r["metrics"][key] for r in records if r["bin_start"]==period]
                    by_window.append(dict(start_s=period,**aggregate(values,err)))
                    pieces.append(err)
                group=[r["metrics"][key] for r in records]
                summaries[key]=aggregate(group,np.concatenate(pieces))
                summaries[key]["windows_30s"]=by_window
                candidates=[r for r in records if r["metrics"][key]["reference"]]
                summaries[key]["lowest_correct_coverage_frame"]=min(candidates,key=lambda r:r["metrics"][key]["correct_tau_fraction_all_reference"])["id"] if candidates else None
            compressed(REPORT/f"surface_{split}_frames.json.gz",records)
            write(REPORT/f"surface_{split}_metrics.json",summaries)
            summary[split]={k:{q:v for q,v in s.items() if q!="windows_30s"} for k,s in summaries.items()}
    local={}
    for rec in read(B0_REPORT/"mesh_validation.json")["local_views"]:
        stem=f"frame_{rec['train_id']:05d}"
        ref,mask,zall=observations(stem,"train",frames[stem],meta,models,tuple(rec["xyxy"]),True)
        stats_local={}
        for key,z,keep in variants({k:a[mask] for k,a in zall.items()},True):
            counts,errors=stats(ref[mask],z,keep)
            stats_local[key]=measures(counts,errors)
        tiles=[]
        for label in ("B0","B1"):
            imgs=[("Sensor Z",ref)]+[(k,zall[label+"_"+k]) for k in ("model","raw_own","raw_pcd","raw_mesh","supported_own","fused_pcd","supported_mesh")]
            tiles.append(np.hstack([titled(color(z,.3,6,np.isfinite(z)&(z>0)),label+" "+key) for key,z in imgs]))
        imwrite(REPORT/"previews"/("surfaces_"+rec["object"]+".jpg"),np.vstack(tiles))
        local[rec["object"]]=dict(stem=stem,xyxy=rec["xyxy"],metrics=stats_local)
    write(REPORT/"surface_local.json",local)
    write(REPORT/"surface_summary.json",dict(splits=summary,raw_point_source_recovery=models[2],
          correct_fraction_denominator="all finite optical-axis sensor reference pixels in [0.3,6] m",
          conditional_errors="finite positive first intersection or z-buffer winner",
          common_intersection_definitions=GROUPS,
          provenance="Each stored PCD winning index resolves via raw_sources.npz or all source contributions in fusion_work/members_*.npz.",
          independent_geometry_ground_truth=False))


if __name__=="__main__":
    main()
