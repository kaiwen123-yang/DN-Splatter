"""Share each selected-checkpoint render across the historical train-only exporters and media."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import open3d as o3d
import torch

from interactive_dn_cache import atomic_json, shared_seed_geometry
from interactive_dn_supervision import digest


def panel_depth(values):
    keep=np.isfinite(values)&(values>0)
    code=np.clip((np.where(keep,values,.3)-.3)*(255/5.7),0,255).astype(np.uint8)
    img=cv2.applyColorMap(code,cv2.COLORMAP_TURBO)
    img[~keep]=0
    return img


def media_plan(root):
    import json
    from capture_metric3d_common import selection
    seq=selection()
    hidden={i for i,r in enumerate(seq) if 349<=r["relative_s"]<=353.7 or r.get("annotation")=="late dynamic interval"}
    if len(seq)!=1846 or len(hidden)!=30:
        raise ValueError("Full hall timeline or original withholding rule differs")
    boxes=json.loads((root/"reports/capture-full-dn-30k/20261119-v1/mesh_validation.json").read_bytes())["local_views"]
    rois={int(r["train_id"]):(r["object"],r["xyxy"]) for r in boxes}
    rois[1420]=("foam_teeth",[720,224,1104,608])
    return seq,hidden,rois


def display_camera(meta, frame, device):
    from nerfstudio.cameras.cameras import Cameras,CameraType
    pose=np.asarray(frame["transform_matrix"],np.float32)
    return Cameras(camera_to_worlds=torch.from_numpy(pose)[None,:3,:],fx=meta["fl_x"],fy=meta["fl_y"],
        cx=meta["cx"],cy=meta["cy"],width=meta["w"],height=meta["h"],camera_type=CameraType.PERSPECTIVE).to(device)


def equal_display_cameras(ds, names, meta, frames):
    """Check every camera quantity consumed by DN in eval mode before sharing display pixels."""
    from nerfstudio.cameras.cameras import CameraType
    cameras=ds.cameras
    poses=cameras.camera_to_worlds.detach().cpu().numpy()
    good=np.all(poses==np.array([frames[n]["transform_matrix"][:3] for n in names],np.float32),axis=(1,2))
    for key,expected in (("width",meta["w"]),("height",meta["h"]),("fx",meta["fl_x"]),("fy",meta["fl_y"]),
                         ("cx",meta["cx"]),("cy",meta["cy"]),("camera_type",CameraType.PERSPECTIVE.value)):
        values=getattr(cameras,key).detach().cpu().numpy().reshape(-1)
        good &= values==np.asarray(expected,dtype=values.dtype)
    distortion=cameras.distortion_params
    if distortion is not None:
        good &= np.all(distortion.detach().cpu().numpy()==0,axis=-1)
    return {name for name,valid in zip(names,good,strict=True) if valid}


class DisplayCache:
    """Lossless final-size RGB/depth display panels; retain raw model Z for the declared ROIs."""
    def __init__(self,dest,meta,frames,seq,hidden,rois,identity):
        import json
        from interactive_dn_train import freeze
        self.folder=dest/"media"/"render_cache"
        self.folder.mkdir(parents=True,exist_ok=True)
        self.meta=meta
        self.frames=frames
        self.visible={f'images/{r["id"]}.png':i for i,r in enumerate(seq) if i not in hidden}
        self.rois=rois
        self.rows={}
        self.reused=0
        self.generated=0
        freeze(self.folder/"identity.json",dict(version=1,**identity))
        for name in self.visible:
            path=self.folder/(Path(name).stem+".npz")
            rec=path.with_suffix(".json")
            if rec.is_file():
                row=json.loads(rec.read_bytes())
                if not path.is_file() or row["bytes"]!=path.stat().st_size or row["sha256"]!=digest(path):
                    raise ValueError(f"Saved display panels differ from the committed frame receipt: {name}")
                self.rows[name]=row
                self.reused+=1

    def needs(self,name):
        return name in self.visible and name not in self.rows

    def write(self,name,pred,source):
        h,w=self.meta["h"],self.meta["w"]
        rgb=np.rint(pred["rgb"].clamp(0,1).detach().cpu().numpy()*255).astype(np.uint8)[...,::-1]
        z=pred["depth"].detach().reshape(h,w).cpu().numpy()
        arrays={"rgb":cv2.resize(rgb,(640,360),interpolation=cv2.INTER_AREA),
                "depth":cv2.resize(panel_depth(z),(640,360),interpolation=cv2.INTER_AREA)}
        i=self.visible[name]
        if i in self.rois:
            _,(x0,y0,x1,y1)=self.rois[i]
            arrays["roi_z"]=np.ascontiguousarray(z[y0:y1,x0:x1])
        path=self.folder/(Path(name).stem+".npz")
        with tempfile.NamedTemporaryFile(dir=self.folder,delete=False) as f:
            np.savez(f,**arrays)
            temp=f.name
        os.replace(temp,path)
        row=dict(id=Path(name).stem,index=i,file=path.name,bytes=path.stat().st_size,sha256=digest(path),source=source)
        atomic_json(path.with_suffix(".json"),row)
        self.rows[name]=row
        self.generated+=1

    def publish(self):
        if self.rows.keys()!=self.visible.keys():
            raise ValueError("Incomplete full-timeline model display cache")
        atomic_json(self.folder/"manifest.json",dict(redacted=30,visible=len(self.rows),type="lossless display tiles with raw ROI optical Z",
            frames=[self.rows[name] for name in self.visible]))


def prepare_train_and_media(dest,scene,config,step,root,out,stage):
    import capture_quality_eval as ev
    import capture_quality_tsdf as tsdf
    import capture_s01_eval as older
    import capture_metric3d_mesh as supported
    from capture_quality_paths import read,record
    from dn_splatter.dn_model import DNSplatterModel
    from contextlib import ExitStack
    base=dest/"meshes"
    meta=read(scene/"transforms.json")
    frames={f["file_path"]:f for f in meta["frames"]}
    seq,hidden,rois=media_plan(root)
    selected=read(dest/"state/selection.json")
    node=dest/"evaluation"/"val"/f"step-{step:09d}.json"
    val=read(node)
    ck=Path(val["checkpoint"])
    if selected["selected_step"]!=step or val["step"]!=step or digest(ck)!=val["checkpoint_sha256"]:
        raise ValueError("Frozen selected checkpoint for geometry or video differs")
    baseline=root/"reports/capture-full-dn-30k/20261119-v1"
    identity=dict(checkpoint_sha256=val["checkpoint_sha256"],config_sha256=digest(config),
        scene_sha256=digest(scene/"transforms.json"),timeline_sha256=digest(baseline/"selection.json"),
        fixed_roi_sha256=digest(baseline/"mesh_validation.json"))
    cache=DisplayCache(dest,meta,frames,seq,hidden,rois,identity)
    missing_volume=not (base/"model_tsdf.json").is_file()
    missing_sample=not (out/"sample.json").is_file()
    if not (missing_volume or missing_sample or any(cache.needs(n) for n in cache.visible)):
        cache.publish()
        return cache,meta,frames,seq,hidden,rois,ck
    stage("selected_checkpoint_renders",False,False)
    with shared_seed_geometry(DNSplatterModel,dest.parents[1]/"shared/initial-geometry"):
        loaded=ev.load_model(config,step,"val")
    _,pipe,actual,_=loaded
    if Path(actual).resolve()!=ck.resolve():
        raise ValueError("Loaded checkpoint differs from the frozen evaluation")
    ds=pipe.datamanager.train_dataset
    names=[p.relative_to(scene).as_posix() for p in ds.image_filenames]
    if names!=meta["train_filenames"] or len(names)!=1478:
        raise ValueError("Train-camera order or membership changed")
    can_share=equal_display_cameras(ds,names,meta,frames)
    # The original sampler skips a view only when both files exist, and still
    # advances its PCG64 for every image. We leave its implementation untouched.
    def sample_missing(name):
        stem=Path(name).stem
        return not all((out/"views"/(stem+s)).is_file() for s in (".npz",".json"))
    missing_names=[name for name in names if sample_missing(name)]
    joint=missing_sample and missing_volume and len(missing_names)==len(names)
    vol=tsdf.volume() if joint else None
    intr=o3d.camera.PinholeCameraIntrinsic(meta["w"],meta["h"],meta["fl_x"],meta["fl_y"],meta["cx"]-.5,meta["cy"]-.5)
    tracked=deque()
    views=[]
    flip=np.diag([1.,-1.,-1.,1.])
    original_render=pipe.model.get_outputs_for_camera
    def render_during(operation,order,integrator=None):
        it=iter(order)
        def tapped(*args,**kwargs):
            name=next(it)
            pred=original_render(*args,**kwargs)
            if integrator is not None:
                if len(tracked)>=2:
                    k,job=tracked.popleft()
                    views.append(dict(id=Path(k).stem,valid=job.result()))
                shape=(meta["h"],meta["w"])
                z=pred["depth"].detach().cpu().numpy().reshape(shape)
                a=pred["accumulation"].detach().cpu().numpy().reshape(shape)
                pose=np.asarray(frames[name]["transform_matrix"])@flip
                # np.where owns its storage. The single consumer updates the same volume in dataset order.
                job=integrator.submit(tsdf.integrate,vol,np.where(a>=.5,z,0),intr,np.linalg.inv(pose))
                tracked.append((name,job))
            if name in can_share and cache.needs(name):
                cache.write(name,pred,"identical train camera")
            return pred
        with patch.object(pipe.model,"get_outputs_for_camera",tapped):
            operation()
        if next(it,None) is not None:
            raise ValueError("The native exporter omitted a scheduled camera")

    with ExitStack() as stack,torch.no_grad():
        for obj,k,v in ((tsdf,"SCENE",scene),(tsdf,"RUN",dest),(tsdf,"MESH",base),(tsdf,"load_selected",lambda *a:loaded),
                        (supported,"model_paths",lambda label:(dest,scene)),(older,"final_setup",lambda *a:loaded)):
            stack.enter_context(patch.object(obj,k,v))
        if joint:
            with ThreadPoolExecutor(max_workers=1,thread_name_prefix="ordered-tsdf") as executor:
                render_during(lambda:supported.sample(dest.name,out),names,executor)
                while tracked:
                    name,job=tracked.popleft()
                    views.append(dict(id=Path(name).stem,valid=job.result()))
            tsdf.finish(vol,"model",dict(views=views,alpha_min=.5,checkpoint=record(ck),sensor_value_substitution=False))
        else:
            if missing_volume:
                render_during(tsdf.model,names)
            if missing_sample:
                render_during(lambda:supported.sample(dest.name,out),missing_names)
        with torch.inference_mode():
            for name in cache.visible:
                if cache.needs(name):
                    pred=original_render(camera=display_camera(meta,frames[name],pipe.device))
                    cache.write(name,pred,"explicit media camera")
    cache.publish()
    atomic_json(base/"shared_render.json",dict(joint_tsdf_sampling=joint,train_views=len(names),
        camera_equivalent_train_views=len(can_share),new_tiles=cache.generated,reused_tiles=cache.reused,
        display_train_views=sum(row["source"]=="identical train camera" for row in cache.rows.values()),
        display_explicit_views=sum(row["source"]=="explicit media camera" for row in cache.rows.values()),
        source_identity=identity))
    return cache,meta,frames,seq,hidden,rois,ck
