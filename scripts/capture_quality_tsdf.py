"""Integrate train-only native sensor or predicted optical Z, and reuse the B1 supported Poisson."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import open3d as o3d

from capture_quality_paths import GEOM, GEORUN, RUN, SCENE, STATE, check, read, record, write
from capture_rectify import grid_setup
from capture_registration_review import project, zbuffer

MESH=RUN/"meshes"
FLIP=np.diag([1.,-1.,-1.,1.])


def load_selected(split="val"):
    from capture_quality_eval import load_model
    sel=read(STATE/"selection.json")
    return load_model(Path(sel["config"]),sel["selected_step"],split)


def volume():
    return o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.01,sdf_trunc=.04,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,depth_sampling_stride=1)


def integrate(vol,z,intr,w2c):
    z=np.ascontiguousarray(np.where(np.isfinite(z)&(z>=.3)&(z<=6),z,0),dtype=np.float32)
    im=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.zeros(z.shape,dtype=np.uint8)),
        o3d.geometry.Image(z),depth_scale=1.,depth_trunc=6.000001,convert_rgb_to_intensity=False)
    vol.integrate(im,intr,w2c)
    return int(np.count_nonzero(z))


def finish(vol,label,evidence):
    path=MESH/(label+"_tsdf.ply")
    mesh=vol.extract_triangle_mesh()
    if len(mesh.triangles):
        mesh.compute_vertex_normals()
        check(o3d.io.write_triangle_mesh(str(path),mesh),"TSDF mesh write")
    else:
        from capture_metric3d_mesh import empty_ply
        empty_ply(path,True)
    write(MESH/(label+"_tsdf.json"),dict(mesh=record(path),vertices=len(mesh.vertices),triangles=len(mesh.triangles),
          voxel_length_m=.01,sdf_trunc_m=.04,depth_scale=1.,depth_trunc=6.000001,depth_sampling_stride=1,**evidence))


def sensor():
    cv2.setNumThreads(1)
    meta=read(SCENE/"transforms.json")
    frames={Path(f["file_path"]).stem:f for f in meta["frames"]}
    train={Path(n).stem for n in meta["train_filenames"]}
    packet=GEORUN/"transfer"
    source={r["id"]:r for r in read(packet/"selection.json")}
    cal=read(packet/"calibration.json")
    dep=cal["cameras"]["depth"]
    rays,t,grids=grid_setup(cal)
    cam,_,_,valid,_=grids["1280"]
    ext=next(e for e in cal["static_extrinsics"] if (e["source_sensor"],e["destination_sensor"])==("depth","color"))
    T=np.eye(4); T[:3,:3]=np.asarray(ext["R"]); T[:3,3]=t
    intr=o3d.camera.PinholeCameraIntrinsic(dep["width"],dep["height"],dep["fx"],dep["fy"],dep["cx"],dep["cy"])
    def counts(stem):
        with np.load(packet/source[stem]["depth"]["file"]) as a: raw=a["depth"]
        live=(raw!=0)&(raw!=65535)
        native=(raw>=300)&(raw<=6000)&live
        pc=rays[live]*(raw[live,None]*.001)+t
        z,inside=zbuffer(project(pc,cam),pc[:,2],cam)
        target=valid&(z>=.3)&(z<=6.)
        expected=np.load(SCENE/frames[stem]["depth_file_path"],mmap_mode="r")
        # The frozen RGB-grid depth was converted to float32 before range clipping.
        zr=z.astype(np.float32)
        target=valid&(zr>=.3)&(zr<=6.)
        check(int(target.sum())==int(np.count_nonzero(expected)),f"frozen RGB-Z support {stem}")
        return dict(id=stem,native_non_sentinel=int(live.sum()),native_03_6=int(native.sum()),
                    native_pixels=raw.size,positive_rgb_Z=int((pc[:,2]>0).sum()),in_rgb_fov=int(inside.sum()),
                    collisions=int(inside.sum()-np.count_nonzero(z)),outside_rectified_rgb=int(np.count_nonzero(z[~valid])),
                    rgb_Z_support=int(target.sum()),rgb_pixels=z.size)
    workers=min(int(os.environ["DN_CPU_BUDGET"]),8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        coverage=list(pool.map(counts,sorted(frames)))
    write(GEOM/"depth_coverage.json",dict(workers=workers,frames=coverage,axes="Native depth optical Z and RGB optical Z"))
    vol=volume(); views=[]
    for stem in sorted(train):
        with np.load(packet/source[stem]["depth"]["file"]) as a: z=a["depth"].astype(np.float32)/1000.
        pose=np.asarray(frames[stem]["transform_matrix"])@FLIP@T
        views.append(dict(id=stem,valid=integrate(vol,z,intr,np.linalg.inv(pose))))
        if len(views)%100==0: print("sensor TSDF",len(views),flush=True)
    check(len(views)==1478,"sensor train population")
    finish(vol,"sensor",dict(views=views,T_rgb_depth=T.tolist(),native_K=dep["K"],independent_ground_truth=False,
           depth_grid=[dep["width"],dep["height"]],depth_distortion="native calibrated zero coefficients"))


def model():
    import torch
    _,pipe,checkpoint,_=load_selected()
    ds=pipe.datamanager.train_dataset
    meta=read(SCENE/"transforms.json")
    frames={f["file_path"]:f for f in meta["frames"]}
    names=[p.relative_to(SCENE).as_posix() for p in ds.image_filenames]
    check(names==meta["train_filenames"] and len(names)==1478,"model train population")
    intr=o3d.camera.PinholeCameraIntrinsic(meta["w"],meta["h"],meta["fl_x"],meta["fl_y"],meta["cx"]-.5,meta["cy"]-.5)
    vol=volume(); views=[]
    with torch.no_grad():
        for i,name in enumerate(names):
            cam=ds.cameras[i:i+1].to(pipe.device)
            out=pipe.model.get_outputs_for_camera(camera=cam)
            z=out["depth"].detach().cpu().numpy().reshape(meta["h"],meta["w"])
            a=out["accumulation"].detach().cpu().numpy().reshape(z.shape)
            pose=np.asarray(frames[name]["transform_matrix"])@FLIP
            views.append(dict(id=Path(name).stem,valid=integrate(vol,np.where(a>=.5,z,0),intr,np.linalg.inv(pose))))
            if (i+1)%100==0: print("model TSDF",i+1,flush=True)
    finish(vol,"model",dict(views=views,alpha_min=.5,checkpoint=record(checkpoint),sensor_value_substitution=False))


def supported(action):
    import capture_metric3d_mesh as old
    import capture_s01_eval as legacy_eval
    out=MESH/"supported"; out.mkdir(exist_ok=True)
    iden=dict(checkpoint=read(STATE/"selection.json"),geometry=record(SCENE/"transforms.json"))
    if (out/"identity.json").exists(): check(read(out/"identity.json")==iden,"supported cache identity")
    else: write(out/"identity.json",iden)
    with patch.object(old,"model_paths",lambda _: (RUN,SCENE)),patch.object(legacy_eval,"final_setup",lambda _:load_selected()):
        getattr(old,action)("B2",out)
    if action=="crop": write(out/"protocol.json",dict(spec=old.SPEC,parameter_changes_vs_B1={},identity=iden))


if __name__=="__main__":
    MESH.mkdir(parents=True,exist_ok=True)
    action=sys.argv[1]
    if action in ("sensor","model"): {"sensor":sensor,"model":model}[action]()
    else: supported(action)
