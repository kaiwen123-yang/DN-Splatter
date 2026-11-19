"""Publish geometry-only DN data after reviewing metric evidence or verify it with GDataset."""
import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from capture_rectify import describe, edges, files, read, save, sha


def export(run,out):
    poses=read(run/"metric/poses.json")
    splits=read(run/"transfer/splits.json")
    proof=read(run/"metric/cross_view.json")
    if not proof["pairs"] or {e["id"] for e in splits["entries"]}!=set(poses["poses"]):
        raise ValueError("Incomplete geometry or frozen splits")
    stage=out.with_name(out.name+".building")
    if out.exists():
        raise FileExistsError(out)
    stage.mkdir(parents=True,exist_ok=False)
    for folder in ("images","depth"):
        (stage/folder).mkdir()
    grid=read(run/"rect/grids.json")
    K=grid["1280"]["K_colmap_ns"]
    Ks=np.array(grid["640"]["K_cv"])
    meta=dict(w=1280,h=720,fl_x=K[0][0],fl_y=K[1][1],cx=K[0][2],cy=K[1][2],
              camera_model="PINHOLE",k1=0,k2=0,k3=0,k4=0,p1=0,p2=0,ply_file_path="init_train.ply",frames=[])
    xyz=[]; colors=[]; coverage={}
    for entry in splits["entries"]:
        stem=entry["id"]; image=f"images/{stem}.png"; path=f"depth/{stem}.npy"
        os.link(run/"rect/1280"/image,stage/image)
        z=np.load(run/"rect/1280"/path)
        z[(z<.3)|(z>6)]=0
        coverage[stem]=float(np.count_nonzero(z)/z.size)
        assert z.dtype==np.float32 and np.isfinite(z).all() and coverage[stem]>0
        np.save(stage/path,z,allow_pickle=False)
        c2w=np.array(poses["poses"][stem])
        cgl=c2w@np.diag([1,-1,-1,1])
        assert np.allclose(cgl[:3,:3].T@cgl[:3,:3],np.eye(3)) and np.isclose(np.linalg.det(cgl[:3,:3]),1)
        meta["frames"].append(dict(file_path=image,depth_file_path=path,transform_matrix=cgl.tolist()))
        if entry["split"]!="train":
            continue
        z=np.load(run/"rect/640"/path)
        occ=cv2.imread(str(run/"rect/640/occlusion"/(stem+".png")),0)
        mask=(z>=.3)&(z<=6)&(occ==0)&~cv2.dilate(edges(z).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        yy,xx=np.nonzero(mask[::4,::4])
        yy*=4;xx*=4
        pc=np.column_stack(((xx-Ks[0,2])/Ks[0,0],(yy-Ks[1,2])/Ks[1,1],np.ones(len(xx))))*z[yy,xx,None]
        xyz.append(pc@c2w[:3,:3].T+c2w[:3,3])
        rgb=cv2.imread(str(run/"rect/640"/image))
        colors.append(rgb[yy,xx,::-1]/255.)
    cloud=o3d.geometry.PointCloud()
    cloud.points=o3d.utility.Vector3dVector(np.concatenate(xyz))
    cloud.colors=o3d.utility.Vector3dVector(np.concatenate(colors))
    reduced=cloud.voxel_down_sample(.03)
    if not o3d.io.write_point_cloud(str(stage/"init_train.ply"),reduced):
        raise IOError("PLY write failed")
    back=o3d.io.read_point_cloud(str(stage/"init_train.ply"))
    pts=np.asarray(back.points)
    assert len(pts)==len(reduced.points)>0 and back.has_colors() and np.isfinite(pts).all()
    # Transform all sparse world points consistently as well; they are distinct from RGB-D initialization.
    with np.load(run/"sfm/observations.npz") as obs:
        origin=np.array(poses["sfm_metric_world_to_local"])
        world=(obs["xyz"]*poses["scale"])@origin[:3,:3].T+origin[:3,3]
        np.savez_compressed(run/"metric/sparse_world_metres.npz",xyz=world,point_ids=obs["point_ids"])
    for split,ids in splits["splits"].items():
        meta[split+"_filenames"]=[f"images/{i}.png" for i in ids]
    save(stage/"transforms.json",meta)
    for actual,filename in (("selection.json","source_frames.json"),("splits.json","splits.json")):
        shutil.copyfile(run/"transfer"/actual,stage/filename)
    scene_scale=int(np.ceil(np.abs(pts).max()))
    settings=dict(depth_unit_scale_factor=1.,auto_scale_poses=False,scale_factor=1.,orientation_method="none",
                  center_method="none",downscale_factor=1,scene_scale=scene_scale,normal_format="opengl")
    save(stage/"geometry.json",dict(state="geometry_prepared_pending_normals_and_training",
         packet_manifest_sha256=sha(run/"transfer/manifest.json"),parser="normal-nerfstudio",parser_settings=settings,
         camera_res_scale_factor=1,num_downscales=0,depth="float32 color-axis Z metres, inclusive 0.3–6 m, holes zero. Full range in native packet and rectified intermediate.",
         depth_positive_fraction=describe(list(coverage.values())),depth_positive_fraction_by_id=coverage,
         init_from="Frozen train split only. 640 pinhole Z, 0.3–6 m; remove competing occlusion hits and 1px dilated near-side discontinuities. Sample every fourth pixel along both axes, voxel-average XYZ and RGB at 0.03 m.",
         init_input_count=len(cloud.points),init_points=len(pts),init_xyz_min_m=pts.min(0).tolist(),init_xyz_max_m=pts.max(0).tolist(),
         normal_priors=False,training_started=False,poisson=False))
    save(stage/"manifest.json",dict(files=files(stage)))
    stage.rename(out)


def validate(run,out):
    import torch
    from dn_splatter.data.normal_nerfstudio import NormalNerfstudioConfig
    from dn_splatter.data.dn_dataset import GDataset
    options=read(out/"geometry.json")
    cfg=NormalNerfstudioConfig(data=out,**options["parser_settings"],load_normals=False,load_pcd_normals=False)
    meta=read(out/"transforms.json")
    frames={f["file_path"]:f for f in meta["frames"]}
    parser=cfg.setup()
    checks={}
    with torch.no_grad():
        for split in ("train","val","test"):
            outputs=parser.get_dataparser_outputs(split=split)
            data=GDataset(outputs,scale_factor=1)
            names=[p.relative_to(out).as_posix() for p in data.image_filenames]
            assert names==meta[split+"_filenames"] and outputs.dataparser_scale==1
            assert np.allclose(outputs.dataparser_transform[:3],np.eye(4)[:3])
            assert len(outputs.metadata["points3D_xyz"])==options["init_points"]
            for i,name in enumerate(names):
                item=data.get_data(i); f=frames[name]
                expected=np.load(out/f["depth_file_path"])
                rgb=cv2.imread(str(out/name))[:,:,::-1]
                assert np.array_equal(np.uint8(np.rint(item["image"].numpy()*255)),rgb)
                assert np.array_equal(item["sensor_depth"].numpy().squeeze(),expected)
                assert np.allclose(outputs.cameras.camera_to_worlds[i],f["transform_matrix"][:3],atol=2e-6)
                for prop,key in (("cx","cx"),("cy","cy"),("fx","fl_x"),("fy","fl_y")):
                    assert np.isclose(float(getattr(outputs.cameras,prop)[i].item()),meta[key])
            checks[split]=len(data)
    if files(out,["manifest.json"])!=read(out/"manifest.json")["files"]:
        raise ValueError("Final dataset hash mismatch")
    save(run/"metric/dataset_check.json",dict(reader="NormalNerfstudio + GDataset.get_data, historical DN; all RGB/depth compared to file pixels; K, c2w, init point count, exact frozen split lists, unit scale and member hashes checked.",
         load_normals=False,load_pcd_normals=False,counts=checks,dataset=str(out),manifest_sha256=sha(out/"manifest.json")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["export","validate"])
    p.add_argument("run",type=Path)
    p.add_argument("dataset",type=Path)
    args=p.parse_args()
    (export if args.stage=="export" else validate)(args.run.resolve(),args.dataset.resolve())
