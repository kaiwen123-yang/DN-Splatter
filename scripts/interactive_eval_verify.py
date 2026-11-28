"""Numerical and stage timing checks on frozen server inputs; performs no training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import cv2
import numpy as np

from interactive_dn_cache import atomic_json
from interactive_dn_supervision import digest


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/".local/runs/interactive-world-20261128-v1"
OUT=RUN/"f_evaluation/eval_performance"


def rays_check(threads):
    from capture_metric3d_surface_eval import rays
    from interactive_dn_evaluate import FLIP,intersections,ray_directions,ray_scene
    folder=RUN/"f_vr/VR0C/mesh"
    meshes={k:folder/k/"mesh_cull.ply" for k in ("strict376","official418")}
    scenes={k:ray_scene(p,threads) for k,p in meshes.items()}
    metadata=json.loads((folder/"strict376/train_cameras.json").read_bytes())
    frames=metadata["frames"][::31][:12]
    metrics={"old":[],"new":[]}
    resolved=examined=0
    for i,frame in enumerate(frames):
        intr=[frame.get(k,metadata.get(k)) for k in ("fl_x","fl_y","cx","cy")]
        h,w=[frame.get(k,metadata.get(k)) for k in ("h","w")]
        yy,xx=np.mgrid[0:h:2,0:w:2]
        directions=ray_directions(yy,xx,intr)
        cam=np.asarray(frame["transform_matrix"])@FLIP
        def old():
            return {k:sc.cast_rays(rays(xx,yy,cam,intr),nthreads=threads)["t_hit"].numpy() for k,sc in scenes.items()}
        def new():
            return intersections(scenes,directions,frame,threads)
        result={}
        for name,op in ((("old",old),("new",new)) if i%2==0 else (("new",new),("old",old))):
            before=time.perf_counter()
            result[name]=op()
            if i:
                metrics[name].append(time.perf_counter()-before)
        for k in meshes:
            if not np.array_equal(result["old"][k],result["new"][k],equal_nan=True):
                raise ValueError(f"Ray hit depth changed in frame {i}, mesh {k}")
            resolved+=int(np.isfinite(result["old"][k]).sum())
            examined+=xx.size
    if not resolved:
        raise ValueError("Real mesh/camera check had no hits")
    atomic_json(OUT/"rays.json",dict(camera_count=len(frames),total_pixels_both_meshes=examined,finite_hits=resolved,
                per_pixel_bitwise_equal=True,unit="s for two mesh traversals and ray creation; static local grid reused",
                old_median=float(np.median(metrics["old"])),new_median=float(np.median(metrics["new"])),
                old_total=sum(metrics["old"]),new_total=sum(metrics["new"]),samples=metrics,threads=threads,
                meshes={k:{"path":str(p),"sha256":digest(p)} for k,p in meshes.items()}))


def checkpoint_check(node):
    import torch
    import capture_quality_eval as ev
    import capture_quality_tsdf as tsdf
    from dn_splatter.dn_model import DNSplatterModel
    from interactive_dn_cache import shared_seed_geometry
    from interactive_eval_render import DisplayCache,display_camera,equal_display_cameras,media_plan,panel_depth
    import open3d as o3d
    dest=RUN/"c_dn/R0C"
    scene=RUN/"restore/tree/.local/runs/capture-full-quality-v3-20261122-v1/inputs"
    ev.SCENE,ev.RUN,ev.EVAL=scene,dest,dest/"evaluation"
    rec=json.loads((dest/f"evaluation/val/step-{node:09d}.json").read_bytes())
    with shared_seed_geometry(DNSplatterModel,RUN/"shared/initial-geometry"):
        _,pipe,ck,_=ev.load_model(Path(rec["config"]),node,"val")
    if digest(ck)!=rec["checkpoint_sha256"]:
        raise ValueError("Formal validation checkpoint hash differs")
    meta=json.loads((scene/"transforms.json").read_bytes())
    lookup={f["file_path"]:f for f in meta["frames"]}
    seq,hidden,rois=media_plan(ROOT)
    ds=pipe.datamanager.train_dataset
    names=[f.relative_to(scene).as_posix() for f in ds.image_filenames]
    match=equal_display_cameras(ds,names,meta,lookup)
    cache=DisplayCache(OUT/f"step-{node:09d}",meta,lookup,seq,hidden,rois,{"checkpoint_sha256":rec["checkpoint_sha256"]})
    intr=o3d.camera.PinholeCameraIntrinsic(meta["w"],meta["h"],meta["fl_x"],meta["fl_y"],meta["cx"]-.5,meta["cy"]-.5)
    old,new=tsdf.volume(),tsdf.volume()
    eligible=[n for n in names if n in match and n in cache.visible]
    if len(eligible)<3:
        raise ValueError("No common train/display cameras for the quantitative check")
    chosen=[eligible[0],min(eligible,key=lambda n:abs(cache.visible[n]-120)),min(eligible,key=lambda n:abs(cache.visible[n]-240))]
    evidence=[]
    with torch.no_grad(),ThreadPoolExecutor(max_workers=1) as executor:
        jobs=[]
        for name in chosen:
            i=names.index(name)
            frame=lookup[name]
            outputs=pipe.model.get_outputs_for_camera(camera=ds.cameras[i:i+1].to(pipe.device))
            with torch.inference_mode():
                explicit=pipe.model.get_outputs_for_camera(camera=display_camera(meta,frame,pipe.device))
            def host(v):return v.detach().cpu().numpy()
            z0=host(explicit["depth"]).reshape(meta["h"],meta["w"])
            z1=host(outputs["depth"]).reshape(z0.shape)
            rgb0=np.rint(host(explicit["rgb"].clamp(0,1))*255).astype(np.uint8)[...,::-1]
            cache.write(name,outputs,"identical train camera")
            with np.load(cache.folder/cache.rows[name]["file"],allow_pickle=False) as packet:
                rgb_equal=np.array_equal(packet["rgb"],cv2.resize(rgb0,(640,360),interpolation=cv2.INTER_AREA))
                depth_equal=np.array_equal(packet["depth"],cv2.resize(panel_depth(z0),(640,360),interpolation=cv2.INTER_AREA))
                if "roi_z" in packet:
                    _,(x0,y0,x1,y1)=rois[cache.visible[name]]
                    if not np.array_equal(packet["roi_z"],z0[y0:y1,x0:x1]):
                        raise ValueError("ROI optical Z differs")
            raw_equal={key:torch.equal(outputs[key],explicit[key]) for key in ("rgb","accumulation","depth","normal")}
            if not (rgb_equal and depth_equal and all(raw_equal.values())):
                raise ValueError(f"Native and explicit render differ for {name}: {raw_equal}")
            w2c=np.linalg.inv(np.asarray(frame["transform_matrix"])@np.diag([1.,-1.,-1.,1.]))
            direct=tsdf.integrate(old,np.where(host(explicit["accumulation"]).reshape(z0.shape)>=.5,z0,0),intr,w2c)
            jobs.append((direct,executor.submit(tsdf.integrate,new,np.where(host(outputs["accumulation"]).reshape(z1.shape)>=.5,z1,0),intr,w2c)))
            evidence.append(dict(frame=name,raw_equal=raw_equal,display_rgb_equal=rgb_equal,display_depth_equal=depth_equal,valid=direct))
        if any(expected!=future.result() for expected,future in jobs):
            raise ValueError("Joint integration valid counts changed")
    a,b=old.extract_triangle_mesh(),new.extract_triangle_mesh()
    verts_equal=np.array_equal(np.asarray(a.vertices),np.asarray(b.vertices))
    faces_equal=np.array_equal(np.asarray(a.triangles),np.asarray(b.triangles))
    if not (verts_equal and faces_equal and len(a.triangles)>0):
        raise ValueError("Ordered integration mesh differs on identical real frames")
    atomic_json(OUT/"render.json",dict(checkpoint=str(ck),checkpoint_sha256=rec["checkpoint_sha256"],
        cameras_equal=len(match),visible_cameras_reusable=len(eligible),train_frames=len(names),
        raw_frames=evidence,tsdf_subset_vertices=len(a.vertices),tsdf_subset_faces=len(a.triangles),
        mesh_vertices_bitwise_equal=verts_equal,mesh_faces_exact=faces_equal,
        planned_new_forward_count=len(names)+len(cache.visible)-len(eligible),
        historical_forward_count=len(names)*2+len(cache.visible)))


def media_check(node):
    import hashlib
    import av
    from interactive_dn_evaluate import inspect_video,stream
    from interactive_eval_render import media_plan
    from lingbot_record_hall import write_frame
    seq,hidden,_=media_plan(ROOT)
    snapshot=OUT/f"step-{node:09d}/media/render_cache"
    raw_reference=next((RUN/"shared").glob("eval-reference-*/manifest.json"))
    refdir=raw_reference.parent
    indices=[0,120,240,min(hidden)]
    times=[(int(seq[i]["actual_rgb_timestamp_ns"])-int(seq[0]["actual_rgb_timestamp_ns"])+500)//1000 for i in indices]
    short=[i*40000 for i in range(len(indices))]
    held={i for i,j in enumerate(indices) if j in hidden}
    def canvas_at(j):
        canvas=np.full((768,1920,3),22,np.uint8)
        if j in hidden:
            canvas[48:]=45
            return canvas
        fid=seq[j]["id"]
        rgb,ref=np.load(refdir/(fid+".npy"),mmap_mode="r",allow_pickle=False)
        with np.load(snapshot/(fid+".npz"),allow_pickle=False) as frame:
            panels=(rgb,frame["rgb"],ref,frame["depth"])
            for k,x in enumerate(panels):
                y0,x0=48+(k//3)*360,(k%3)*640
                canvas[y0:y0+360,x0:x0+640]=x
        return canvas
    decoded={}
    checks={}
    for mode in ("serial","pipeline"):
        paths=[OUT/f"mux-{mode}-{label}.mp4" for label in ("full","summary")]
        opts={"movflags":"+faststart","video_track_timescale":"1000000"}
        with av.open(str(paths[0]),"w",format="mp4",options=opts) as a,av.open(str(paths[1]),"w",format="mp4",options=opts) as b,ThreadPoolExecutor(max_workers=1) as executor:
            sa,sb=stream(a,(1920,768)),stream(b,(1920,768))
            def encode(canvas,i):
                write_frame(a,sa,canvas,times[i])
                cv2.putText(canvas,"interface QA",(870,30),0,.68,(240,238,205),1,cv2.LINE_AA)
                write_frame(b,sb,canvas,short[i])
            pending=None
            for i,j in enumerate(indices):
                canvas=canvas_at(j)
                if pending is not None:
                    pending.result()
                if mode=="pipeline":
                    pending=executor.submit(encode,canvas,i)
                else:
                    encode(canvas,i)
            if pending is not None:
                pending.result()
            for mux,s in ((a,sa),(b,sb)):
                for pkt in s.encode():
                    mux.mux(pkt)
        for label,path,ts in zip(("full","summary"),paths,(times,short),strict=True):
            checks[mode+label]=inspect_video(path,ts,held)
            with av.open(str(path)) as mux:
                hashes=[hashlib.sha256(f.to_ndarray(format="bgr24")).hexdigest() for f in mux.decode(video=0)]
            if mode=="serial":
                decoded[label]=hashes
            elif hashes!=decoded[label]:
                raise ValueError("Mux pipeline changed decoded frame pixels: "+label)
    atomic_json(OUT/"mux.json",dict(frames=indices,interface_only=True,scope="rendered RGB, Z, reference and the original hidden-frame mask",
        original_and_pipelined_decoded_frames_bitwise_equal=True,checks=checks,decoded_sha256=decoded))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("rays","checkpoint","mux"))
    parser.add_argument("--threads",type=int,default=3)
    parser.add_argument("--node",type=int,default=119999)
    a=parser.parse_args()
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=="rays":
        rays_check(a.threads)
    elif a.mode=="checkpoint":
        checkpoint_check(a.node)
    else:
        media_check(a.node)
