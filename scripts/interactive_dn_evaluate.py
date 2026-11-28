"""Same-export hall mesh comparison, local structure measurements and full-timeline publication."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime,timezone
from fractions import Fraction
import gc
import json
import os
from pathlib import Path
import threading
from unittest.mock import patch

import cv2
import numpy as np
import open3d as o3d
import torch

from interactive_dn_cache import atomic_json
from interactive_dn_supervision import digest
from interactive_eval_render import panel_depth,prepare_train_and_media

ROOT=Path(__file__).resolve().parents[1]
FLIP=np.diag([1.,-1.,-1.,1.])
TIMEBASE=Fraction(1,1000000)


def read(p):
    return json.loads(p.read_bytes())


def artifact(p):
    return {"path":str(p),"bytes":p.stat().st_size,"sha256":digest(p)}


def export(dest, scene, config, step,stage):
    import capture_metric3d_mesh as supported
    base=dest/"meshes"
    folder=base/"supported"
    folder.mkdir(parents=True,exist_ok=True)
    check=read(dest/"state/selection.json")
    from interactive_dn_train import freeze
    freeze(base/"identity.json",{"step":step,"config":artifact(config),
                                 "selection_sha256":digest(dest/"state/selection.json"),"support":supported.SPEC})
    render=prepare_train_and_media(dest,scene,config,step,ROOT,folder,stage)
    gc.collect()
    torch.cuda.empty_cache()
    with patch.object(supported,"model_paths",lambda label:(dest,scene)):
        for action in ("fuse","poisson","crop"):
            if not (folder/(action+".json")).is_file():
                stage("supported_"+action,action!="crop",False)
                getattr(supported,action)(dest.name,folder)
    paths={"model_tsdf":base/"model_tsdf.ply","supported_poisson":folder/"supported.ply"}
    freeze(base/"export_freeze.json",{"models":{k:artifact(v) for k,v in paths.items()},
                                      "checkpoint_selection":check["selected_step"],"unit":"m"})
    return paths,render


def ray_scene(path, threads):
    mesh=o3d.io.read_triangle_mesh(str(path))
    if not len(mesh.triangles):
        return None
    sc=o3d.t.geometry.RaycastingScene(nthreads=threads)
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return sc


def intrinsics(meta,frame):
    return tuple(frame.get(k,meta.get(k)) for k in ("fl_x","fl_y","cx","cy"))


def ray_directions(y,x,intr):
    fx,fy,cx,cy=intr
    return np.stack(((x+.5-cx)/fx,(y+.5-cy)/fy,np.ones_like(x)),axis=-1)


def intersections(casts,directions,frame,threads):
    camera=np.asarray(frame["transform_matrix"])@FLIP
    data=np.empty((*directions.shape[:-1],6),np.float32)
    data[...,:3]=camera[:3,3]
    data[...,3:]=directions@camera[:3,:3].T
    rays=o3d.core.Tensor.from_numpy(data)
    return {k:(sc.cast_rays(rays,nthreads=threads)["t_hit"].numpy() if sc is not None and data.size
               else np.full(directions.shape[:-1],np.inf,np.float32)) for k,sc in casts.items()}


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
    for split,count in (("val",29222080),("test",29100252)):
        todo={k:sc for k,sc in casts.items() if not (dest/"evaluation"/f"{split}_{k}.json").is_file()}
        if not todo:
            continue
        records={k:([],{},{},[]) for k in todo}
        for name in meta[split+"_filenames"]:
            frame=frames[name]
            ref=np.load(scene/frame["depth_file_path"],mmap_mode="r",allow_pickle=False)
            y,x=np.where(np.isfinite(ref)&(ref>=.3)&(ref<=6))
            directions=ray_directions(y,x,intrinsics(meta,frame))
            for k,z in intersections(todo,directions,frame,threads).items():
                rows,totals,edges,errs=records[k]
                c,e=stats(ref[y,x],z,np.isfinite(z)&(z>0))
                for field,val in c.items():
                    totals[field]=totals.get(field,0)+val
                dense=np.full(ref.shape,np.nan,np.float32); dense[y,x]=z
                ec=batch_edges(ref,dense)
                for field,val in ec.items():
                    edges[field]=edges.get(field,0)+val
                errs.append(e)
                rows.append({"id":Path(name).stem,"reference_pixels":len(x),"counts":c,"edges":ec})
        for k,(rows,totals,edges,errs) in records.items():
            if len(rows)!=184 or sum(r["reference_pixels"] for r in rows)!=count:
                raise ValueError("hall fixed surface denominator changed")
            atomic_json(dest/"evaluation"/f"{split}_{k}.json",{"split":split,"method":k,"frames":rows,"edge_counts":edges,
                                "pooled":measures(totals,np.concatenate(errs)),"reference_pixels":count})


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


def media(dest,scene,step,render,casts,threads):
    import av
    from lingbot_record_hall import write_frame
    from interactive_eval_reference import prepare as reference_panels
    cache,meta,frames,seq,hidden,rois,ck=render
    folder=dest/"media"; folder.mkdir(exist_ok=True)
    if (folder/"receipt.json").is_file():
        return
    stamps=[int(r["actual_rgb_timestamp_ns"]) for r in seq]
    times=[(t-stamps[0]+500)//1000 for t in stamps]
    short=[40000*i for i in range(len(seq))]
    if len(hidden)!=30 or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError("public timeline timing or withholding interval changed")
    refdir,refmanifest=reference_panels(scene,frames,seq,hidden,max(1,min(3,threads)))
    refrows={row["id"]:row for row in refmanifest["frames"]}
    masks=set(rois)|{1304,1305,1514}
    local={}
    h,w=meta["h"],meta["w"]
    yy,xx=np.mgrid[0:h:2,0:w:2]
    # One camera-space grid is shared by all frames using these intrinsics.
    last_intr=grid=None
    file1,file2=folder/"full.pending.mp4",folder/"summary.pending.mp4"
    opts={"movflags":"+faststart","video_track_timescale":"1000000"}
    with av.open(str(file1),"w",format="mp4",options=opts) as full, av.open(str(file2),"w",format="mp4",options=opts) as brief,ThreadPoolExecutor(max_workers=1) as writer:
        size=(1920,768)
        st1,st2=stream(full,size),stream(brief,size)
        def encode(canvas,i):
            write_frame(full,st1,canvas,times[i])
            caption=("Reference / novel rendering share one camera" if i<len(seq)//3 else
                     "Depth panels share one metric colour scale" if i<2*len(seq)//3 else
                     "Mesh depths are ray intersections; black means no predicted surface")
            cv2.putText(canvas,caption,(870,30),0,.68,(240,238,205),1,cv2.LINE_AA)
            write_frame(brief,st2,canvas,short[i])
        pending=None
        for i,r in enumerate(seq):
            canvas=np.full((768,1920,3),22,np.uint8)
            fid=r["id"]; name=f"images/{fid}.png";frame=frames[name]
            if i not in hidden:
                refrow=refrows[fid]
                refpath=refdir/refrow["path"]
                if digest(refpath)!=refrow["sha256"]:
                    raise IOError("Shared reference panels changed: "+fid)
                photo_tile,ref_tile=np.load(refpath,mmap_mode="r",allow_pickle=False)
                current=intrinsics(meta,frame)
                if current!=last_intr:
                    grid=ray_directions(yy,xx,current);last_intr=current
                meshes=intersections(casts,grid,frame,max(1,threads-1))
                with np.load(cache.folder/cache.rows[name]["file"],allow_pickle=False) as saved:
                    rgb_tile,z_tile=saved["rgb"],saved["depth"]
                    roi_z=saved["roi_z"] if i in rois else None
                tiles=(photo_tile,rgb_tile,ref_tile,z_tile,panel_depth(meshes["supported_poisson"]),panel_depth(meshes["model_tsdf"]))
                for j,tile in enumerate(tiles):
                    y=48+(j//3)*360; x=j%3*640
                    canvas[y:y+360,x:x+640]=tile if tile.shape[:2]==(360,640) else cv2.resize(tile,(640,360),interpolation=cv2.INTER_AREA)
                if i in rois:
                    label,(x0,y0,x1,y1)=rois[i]
                    gy,gx=np.mgrid[y0:y1,x0:x1]
                    crop=np.load(scene/frame["depth_file_path"],mmap_mode="r",allow_pickle=False)[y0:y1,x0:x1]
                    photo=cv2.imread(str(scene/frame["file_path"]))
                    models={"model":roi_z,**intersections(casts,ray_directions(gy,gx,current),frame,max(1,threads-1))}
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
            if pending is not None:
                pending.result()
            pending=writer.submit(encode,canvas,i)
        if pending is not None:
            pending.result()
        for mux,s in ((full,st1),(brief,st2)):
            for packet in s.encode(): mux.mux(packet)
    artifacts={}
    for label,path,ts in (("full",file1,times),("summary",file2,short)):
        check=inspect_video(path,ts,hidden)
        output=folder/(label+".mp4");path.replace(output)
        artifacts[label]=dict(artifact(output),validation=check,last_pts_seconds=ts[-1]/1e6)
    atomic_json(folder/"local_geometry.json",local)
    atomic_json(folder/"receipt.json",{"checkpoint":artifact(ck),"views":1846,"public_withheld":len(hidden),"movies":artifacts})


@contextmanager
def measured_stages(dest,threads):
    state={}
    lock=threading.Lock()
    finished=threading.Event()
    def sample():
        report=dict(state)
        report["sampled"]=datetime.now(timezone.utc).isoformat()
        report["reserved_bytes"]=torch.cuda.memory_reserved()
        report["cgroup_bytes"]=int(Path("/sys/fs/cgroup/memory.current").read_text())
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            k,_,v=line.partition(":")
            if k in ("Pss","Rss","Private_Clean","Private_Dirty"):
                report[k+"_bytes"]=int(v.split()[0])*1024
        report["USS_bytes"]=report["Private_Clean_bytes"]+report["Private_Dirty_bytes"]
        for k in ("Pss_bytes","Rss_bytes","USS_bytes","reserved_bytes"):
            state["peak_"+k]=max(state.get("peak_"+k,0),report[k])
            report["peak_"+k]=state["peak_"+k]
        atomic_json(dest/"state/evaluation_stage.json",report)
        return report
    def log(report,event):
        with (dest/"state/evaluation_events.jsonl").open("a") as f:
            f.write(json.dumps({**report,"event":event})+"\n")
    def stage(name,cpu_only,gpu_released):
        with lock:
            if state:
                log(sample(),"end")
            state.clear()
            state.update(stage=name,pid=os.getpid(),started=datetime.now(timezone.utc).isoformat(),threads=threads,
                         cpu_only=cpu_only,gpu_released=gpu_released)
            log(sample(),"start")
        print("Evaluation stage",name,"threads",threads,flush=True)
    def heartbeat():
        while not finished.wait(10):
            with lock:
                if state:
                    sample()
    worker=threading.Thread(target=heartbeat,daemon=True,name="evaluation-stage-resources")
    worker.start()
    try:
        yield stage
    finally:
        finished.set()
        worker.join()


def run(dest,scene,config,step):
    cv2.setNumThreads(1)
    threads=int(os.environ.get("IW_EVAL_CPU_THREADS",os.environ["DN_CPU_BUDGET"]))
    capacity=len(os.sched_getaffinity(0))
    quota,period=Path("/sys/fs/cgroup/cpu.max").read_text().split()
    if quota!="max":
        capacity=min(capacity,max(1,int(quota)//int(period)))
    if not 1<=threads<=capacity:
        raise ValueError("Invalid allocated evaluation CPU threads")
    os.environ["DN_CPU_BUDGET"]=str(threads)
    with measured_stages(dest,threads) as stage:
        stage("geometry_identity",False,False)
        paths,render=export(dest,scene,config,step,stage)
        gc.collect()
        torch.cuda.empty_cache()
        stage("surface_scores",True,True)
        casts={k:ray_scene(p,threads) for k,p in paths.items()}
        evaluate_meshes(dest,scene,casts,render[1],render[2],threads)
        stage("full_timeline_encoding",False,True)
        media(dest,scene,step,render,casts,threads)
        receipts=[dest/"evaluation/test_complete.json",dest/"meshes/export_freeze.json",dest/"media/receipt.json",
                  dest/"media/local_geometry.json",*[dest/"evaluation"/f"{s}_{k}.json" for s in ("val","test") for k in casts]]
        atomic_json(dest/"evaluation/complete.json",{"variant":dest.name,"selected_step":step,"artifacts":[artifact(p) for p in receipts]})
        stage("completed",True,True)
