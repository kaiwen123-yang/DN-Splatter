"""Same-export hall mesh comparison, local structure measurements and full-timeline publication."""
from contextlib import ExitStack
from fractions import Fraction
import json
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import open3d as o3d
import torch

from interactive_dn_cache import atomic_json
from interactive_dn_supervision import digest

ROOT=Path(__file__).resolve().parents[1]
FLIP=np.diag([1.,-1.,-1.,1.])
TIMEBASE=Fraction(1,1000000)


def read(p):
    return json.loads(p.read_bytes())


def artifact(p):
    return {"path":str(p),"bytes":p.stat().st_size,"sha256":digest(p)}


def export(dest, scene, config, step):
    import capture_quality_eval as ev
    import capture_quality_tsdf as tsdf
    import capture_metric3d_mesh as supported
    import capture_s01_eval as older
    base=dest/"meshes"
    folder=base/"supported"
    folder.mkdir(parents=True,exist_ok=True)
    check=read(dest/"state/selection.json")
    from interactive_dn_train import freeze
    freeze(base/"identity.json",{"step":step,"config":artifact(config),
                                 "selection_sha256":digest(dest/"state/selection.json"),"support":supported.SPEC})
    def selected(split="val"):
        return ev.load_model(config,step,split)
    with ExitStack() as stack:
        for obj,key,value in ((tsdf,"SCENE",scene),(tsdf,"RUN",dest),(tsdf,"MESH",base),
                              (tsdf,"load_selected",selected),(supported,"model_paths",lambda label:(dest,scene)),
                              (older,"final_setup",lambda split:selected("val"))):
            stack.enter_context(patch.object(obj,key,value))
        if not (base/"model_tsdf.json").is_file():
            tsdf.model()
        for action in ("sample","fuse","poisson","crop"):
            if not (folder/(action+".json")).is_file():
                getattr(supported,action)(dest.name,folder)
    paths={"model_tsdf":base/"model_tsdf.ply","supported_poisson":folder/"supported.ply"}
    freeze(base/"export_freeze.json",{"models":{k:artifact(v) for k,v in paths.items()},
                                      "checkpoint_selection":check["selected_step"],"unit":"m"})
    return paths


def ray_scene(path, threads):
    mesh=o3d.io.read_triangle_mesh(str(path))
    if not len(mesh.triangles):
        return None
    sc=o3d.t.geometry.RaycastingScene(nthreads=threads)
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return sc


def z_at(sc,y,x,meta,frame,threads):
    from capture_metric3d_surface_eval import rays
    if sc is None or not x.size:
        return np.full(x.shape,np.inf,np.float32)
    camera=np.asarray(frame["transform_matrix"])@FLIP
    intr=[frame.get(k,meta.get(k)) for k in ("fl_x","fl_y","cx","cy")]
    return sc.cast_rays(rays(x,y,camera,intr),nthreads=threads)["t_hit"].numpy()


def batch_edges(ref,z):
    """Score observed neighbouring depth changes and failed/missing predicted edges."""
    result=dict(reference_edges=0,predicted_edges=0,matched_reference_edges_1px=0,
                matched_predicted_edges_1px=0,collapsed_reference_edges=0,missing_prediction_at_reference_edge=0)
    for axis in (0,1):
        a,b=np.moveaxis(ref,axis,0)[:-1],np.moveaxis(ref,axis,0)[1:]
        x,y=np.moveaxis(z,axis,0)[:-1],np.moveaxis(z,axis,0)[1:]
        source=(np.isfinite(a)&np.isfinite(b)&(a>=.3)&(b>=.3)&(a<=6)&(b<=6))
        pred=source&np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
        target=source & (np.abs(a-b)>np.maximum(.05,.03*np.minimum(a,b)))
        delta=np.zeros_like(x)
        np.subtract(x,y,out=delta,where=pred)
        proposed=pred & (np.abs(delta)>np.maximum(.05,.03*np.minimum(x,y)))
        kernel=np.ones((3,3),np.uint8)
        nearby_real=cv2.dilate(target.astype(np.uint8),kernel).astype(bool)
        nearby_pred=cv2.dilate(proposed.astype(np.uint8),kernel).astype(bool)
        result["reference_edges"]+=int(target.sum())
        result["predicted_edges"]+=int(proposed.sum())
        result["matched_reference_edges_1px"]+=int((target&nearby_pred).sum())
        result["matched_predicted_edges_1px"]+=int((proposed&nearby_real).sum())
        result["collapsed_reference_edges"]+=int((target&pred&(np.abs(delta)<.5*np.abs(a-b))).sum())
        result["missing_prediction_at_reference_edge"]+=int((target&~pred).sum())
    return result


def widths(mask, valid):
    """Only finite foreground runs bounded by observed background at both ends."""
    vals=[]
    for row,good in zip(mask,valid):
        switches=np.diff(np.r_[False,row,False].astype(np.int8))
        left,right=np.where(switches==1)[0],np.where(switches==-1)[0]
        ok=(left>0)&(right<len(row))
        left,right=left[ok],right[ok]
        ok=good[left-1]&good[right]
        vals.extend((right[ok]-left[ok]).tolist())
    return {"bounded_runs":len(vals),"median_width_px":float(np.median(vals)) if vals else None,
            "P90_width_px":float(np.percentile(vals,90)) if vals else None}


def local_stats(ref,pred):
    valid=np.isfinite(ref)&(ref>=.3)&(ref<=6)
    out={"measured_pixels":int(valid.sum()),"edge_counts":batch_edges(ref,pred)}
    if valid.sum()<2:
        return dict(out,two_mode_data=None)
    v=ref[valid].astype(np.float64)
    centres=np.percentile(v,[20,80])
    for _ in range(24):
        side=v<=np.mean(centres)
        if side.all() or not side.any():
            return dict(out,two_mode_data=None)
        centres=np.array([v[side].mean(),v[~side].mean()])
    if centres[1]-centres[0] <= max(.1,.06*centres[0]):
        return dict(out,two_mode_data=None)
    mid=float(centres.mean())
    def summary(a):
        ok=valid&np.isfinite(a)&(a>0)
        foreground=ok&(a<=mid)
        n,_,st,_=cv2.connectedComponentsWithStats(foreground.astype(np.uint8),connectivity=8)
        return {"observed_pairs":int(ok.sum()),"closer_mode_pixels":int(foreground.sum()),"connected_components":n-1,
                "largest_component_fraction":float(st[1:,cv2.CC_STAT_AREA].max()/foreground.sum()) if n>1 else None,
                "horizontal":widths(foreground,ok),"vertical":widths(foreground.T,ok.T)}
    return dict(out,two_mode_data={"sensor_mode_centres_m":centres.tolist(),"threshold_m":mid,"reference":summary(ref),
                                    "prediction":summary(pred),"interpretation":"projected two-mode connectivity on measured pixels"})


def evaluate_meshes(dest,scene,casts,meta,frames,threads):
    from capture_metric3d_surface_eval import stats,measures
    # Fixed denominator pooled over all 184 frames. Bound memory by retaining only
    # per-method hit errors for one split, matching the historical percentile.
    for split,count in (("val",29222080),("test",29100252)):
        for k,sc in casts.items():
            target=dest/"evaluation"/f"{split}_{k}.json"
            if target.exists():
                continue
            rows=[]; totals={}; edges={}; errs=[]
            for name in meta[split+"_filenames"]:
                frame=frames[name]
                ref=np.load(scene/frame["depth_file_path"],mmap_mode="r",allow_pickle=False)
                y,x=np.where(np.isfinite(ref)&(ref>=.3)&(ref<=6))
                z=z_at(sc,y,x,meta,frame,threads)
                c,e=stats(ref[y,x],z,np.isfinite(z)&(z>0))
                for field,val in c.items():
                    totals[field]=totals.get(field,0)+val
                dense=np.full(ref.shape,np.nan,np.float32); dense[y,x]=z
                ec=batch_edges(ref,dense)
                for field,val in ec.items():
                    edges[field]=edges.get(field,0)+val
                errs.append(e)
                rows.append({"id":Path(name).stem,"reference_pixels":len(x),"counts":c,"edges":ec})
            if len(rows)!=184 or sum(r["reference_pixels"] for r in rows)!=count:
                raise ValueError("hall fixed surface denominator changed")
            atomic_json(target,{"split":split,"method":k,"frames":rows,"edge_counts":edges,
                                "pooled":measures(totals,np.concatenate(errs)),"reference_pixels":count})


def panel_depth(values):
    keep=np.isfinite(values)&(values>0)
    code=np.clip((np.where(keep,values,.3)-.3)*(255/5.7),0,255).astype(np.uint8)
    img=cv2.applyColorMap(code,cv2.COLORMAP_TURBO)
    img[~keep]=0
    return img


def stream(mux,size):
    import av
    codec="h264_nvenc" if "h264_nvenc" in av.codecs_available else "libx264"
    s=mux.add_stream(codec,rate=25)
    s.width,s.height=size
    s.pix_fmt="yuv420p"
    s.time_base=s.codec_context.time_base=TIMEBASE
    s.codec_context.max_b_frames=0
    s.codec_context.thread_count=min(int(os.environ.get("DN_CPU_BUDGET",1)),4)
    s.options={"preset":"p4","cq":"20"} if codec=="h264_nvenc" else {"preset":"fast","crf":"20"}
    return s


def inspect_video(path, times, hidden):
    import av
    n=0; max_redaction_error=0
    with av.open(str(path)) as mux:
        for i,frame in enumerate(mux.decode(video=0)):
            if i>=len(times) or frame.pts*frame.time_base != times[i]*TIMEBASE:
                raise ValueError(f"decoded timestamp mismatch at {i}")
            if i in hidden:
                img=frame.to_ndarray(format="bgr24")
                for y in (78,438):
                    region=img[y+13:y+240,16:-16].astype(np.int16)
                    max_redaction_error=max(max_redaction_error,int(np.abs(region-45).max()))
            n+=1
    if n!=len(times) or max_redaction_error>10:
        raise ValueError("decoded timeline or historical full-panel privacy mask differs")
    return dict(frames=n,verified_pts=n,redacted_frames=len(hidden),max_mask_error=max_redaction_error)


def media(dest,scene,config,step,meta,frames,casts,threads):
    import av
    from nerfstudio.cameras.cameras import Cameras,CameraType
    import capture_quality_eval as ev
    from capture_metric3d_common import selection
    from lingbot_record_hall import write_frame
    folder=dest/"media"; folder.mkdir(exist_ok=True)
    if (folder/"receipt.json").is_file():
        return
    _,pipe,ck,_=ev.load_model(config,step,"val")
    seq=selection()
    if len(seq)!=1846:
        raise ValueError("full hall timeline missing")
    stamps=[int(r["actual_rgb_timestamp_ns"]) for r in seq]
    times=[(t-stamps[0]+500)//1000 for t in stamps]
    short=[40000*i for i in range(len(seq))]
    hidden={i for i,r in enumerate(seq) if 349<=r["relative_s"]<=353.7 or r.get("annotation")=="late dynamic interval"}
    if len(hidden)!=30 or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError("public timeline timing or withholding interval changed")
    boxes=read(ROOT/"reports/capture-full-dn-30k/20261119-v1/mesh_validation.json")["local_views"]
    rois={int(r["train_id"]):(r["object"],r["xyxy"]) for r in boxes}
    rois[1420]=("foam_teeth",[720,224,1104,608])
    masks=set(rois)|{1304,1305,1514}
    local={}
    h,w=meta["h"],meta["w"]
    yy,xx=np.mgrid[0:h:2,0:w:2]
    file1,file2=folder/"full.pending.mp4",folder/"summary.pending.mp4"
    opts={"movflags":"+faststart","video_track_timescale":"1000000"}
    with av.open(str(file1),"w",format="mp4",options=opts) as full, av.open(str(file2),"w",format="mp4",options=opts) as brief,torch.inference_mode():
        size=(1920,768)
        st1,st2=stream(full,size),stream(brief,size)
        for i,r in enumerate(seq):
            canvas=np.full((768,1920,3),22,np.uint8)
            fid=r["id"]; frame=frames[f"images/{fid}.png"]
            if i not in hidden:
                P=np.asarray(frame["transform_matrix"],np.float32)
                c=Cameras(camera_to_worlds=torch.tensor(P)[None,:3,:],**{k:meta[k] for k in ("cx","cy")},
                          fx=meta["fl_x"],fy=meta["fl_y"],width=w,height=h,camera_type=CameraType.PERSPECTIVE).to(pipe.device)
                out=pipe.model.get_outputs_for_camera(camera=c)
                rgb=np.rint(out["rgb"].clamp(0,1).cpu().numpy()*255).astype(np.uint8)[...,::-1]
                z=out["depth"].reshape(h,w).cpu().numpy()
                ref=np.load(scene/frame["depth_file_path"],mmap_mode="r",allow_pickle=False)
                photo=cv2.imread(str(scene/frame["file_path"]))
                meshes={k:z_at(s,yy,xx,meta,frame,threads) for k,s in casts.items()}
                tiles=(photo,rgb,panel_depth(ref),panel_depth(z),panel_depth(meshes["supported_poisson"]),panel_depth(meshes["model_tsdf"]))
                for j,tile in enumerate(tiles):
                    y=48+(j//3)*360; x=j%3*640
                    canvas[y:y+360,x:x+640]=cv2.resize(tile,(640,360),interpolation=cv2.INTER_AREA)
                if i in rois:
                    label,(x0,y0,x1,y1)=rois[i]
                    gy,gx=np.mgrid[y0:y1,x0:x1]
                    crop=ref[y0:y1,x0:x1]
                    models={"model":z[y0:y1,x0:x1],**{k:z_at(s,gy,gx,meta,frame,threads) for k,s in casts.items()}}
                    local[label]={"id":fid,"xyxy":[x0,y0,x1,y1],"results":{k:local_stats(crop,v) for k,v in models.items()}}
                    cv2.imwrite(str(folder/(label+".jpg")),np.hstack((photo[y0:y1,x0:x1],panel_depth(crop),*[panel_depth(a) for a in models.values()])))
            else:
                canvas[48:]=45
            titles=("Reference RGB","DN RGB","Sensor Z [0.3, 6 m]","DN Z","Supported Poisson Z","Model TSDF Z")
            if i not in hidden:
                for j,title in enumerate(titles):
                    x=j%3*640; y=48+j//3*360
                    cv2.rectangle(canvas,(x,y),(x+255,y+27),(24,24,24),-1)
                    cv2.putText(canvas,title,(x+8,y+20),0,.58,(255,255,255),1,cv2.LINE_AA)
            label=f"{dest.name}  {step+1} updates   {fid}   {r['requested_split']}   source {r['relative_s']:.3f} s"
            cv2.putText(canvas,label,(14,31),0,.80,(235,235,235),1,cv2.LINE_AA)
            if i in masks:
                cv2.imwrite(str(folder/(fid+".jpg")),cv2.resize(canvas,(960,384)))
            write_frame(full,st1,canvas,times[i])
            caption=("Reference / novel rendering share one camera" if i<len(seq)//3 else
                     "Depth panels share one metric colour scale" if i<2*len(seq)//3 else
                     "Mesh depths are ray intersections; black means no predicted surface")
            cv2.putText(canvas,caption,(870,30),0,.68,(240,238,205),1,cv2.LINE_AA)
            write_frame(brief,st2,canvas,short[i])
        for mux,s in ((full,st1),(brief,st2)):
            for packet in s.encode(): mux.mux(packet)
    artifacts={}
    for label,path,ts in (("full",file1,times),("summary",file2,short)):
        check=inspect_video(path,ts,hidden)
        output=folder/(label+".mp4");path.replace(output)
        artifacts[label]=dict(artifact(output),validation=check,last_pts_seconds=ts[-1]/1e6)
    atomic_json(folder/"local_geometry.json",local)
    atomic_json(folder/"receipt.json",{"checkpoint":artifact(ck),"views":1846,"public_withheld":len(hidden),"movies":artifacts})


def run(dest,scene,config,step):
    cv2.setNumThreads(1)
    threads=int(os.environ["DN_CPU_BUDGET"])
    paths=export(dest,scene,config,step)
    casts={k:ray_scene(p,threads) for k,p in paths.items()}
    meta=read(scene/"transforms.json")
    frames={f["file_path"]:f for f in meta["frames"]}
    evaluate_meshes(dest,scene,casts,meta,frames,threads)
    media(dest,scene,config,step,meta,frames,casts,threads)
    receipts=[dest/"evaluation/test_complete.json",dest/"meshes/export_freeze.json",dest/"media/receipt.json",
              dest/"media/local_geometry.json",*[dest/"evaluation"/f"{s}_{k}.json" for s in ("val","test") for k in casts]]
    atomic_json(dest/"evaluation/complete.json",{"variant":dest.name,"selected_step":step,"artifacts":[artifact(p) for p in receipts]})
