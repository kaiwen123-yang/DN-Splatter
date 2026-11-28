"""Official VR/Faro evaluation plus comparable fixed-mask metrics and train-only mesh culling."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image

from interactive_dn_cache import atomic_json
from interactive_dn_supervision import digest

ROOT = Path(__file__).resolve().parents[1]


def faro_counts(reference, prediction):
    gt = np.asarray(reference)
    pred = np.asarray(prediction)
    if gt.shape != pred.shape:
        raise ValueError("Faro and prediction grids disagree")
    reference_mask = np.isfinite(gt) & (gt > .1)
    valid = reference_mask & np.isfinite(pred) & (pred > 0)
    g,p = gt[valid].astype(np.float64),pred[valid].astype(np.float64)
    ae = np.abs(g-p)
    logs = np.log(p)-np.log(g)
    return {"pixels":gt.size,"finite_positive_gt":int(np.count_nonzero(np.isfinite(gt)&(gt>0))),
            "reference":int(reference_mask.sum()),"invalid_prediction":int((reference_mask&~valid).sum()),
            "absolute_error_sum":float(ae.sum()),"square_error_sum":float(ae@ae),
            "square_log_error_sum":float(logs@logs),"correct_5cm":int((ae<=.05).sum()),
            "correct_10cm":int((ae<=.1).sum()),"correct_20cm":int((ae<=.2).sum())}


def finish(counts):
    n=counts["reference"]-counts["invalid_prediction"]
    mse=counts["square_error_sum"]/n if n else None
    conditional={"MAE_m":counts["absolute_error_sum"]/n if n else None,"MSE_m2":mse,
                 "RMSE_m":float(np.sqrt(mse)) if n else None,
                 "log_RMSE":float(np.sqrt(counts["square_log_error_sum"]/n)) if n else None}
    return {**counts,"valid_pairs":n,"conditional":conditional,
            "fixed_reference":conditional if not counts["invalid_prediction"] else {k:None for k in conditional},
            "coverage":{k:counts[k]/counts["reference"] if counts["reference"] else None
                        for k in ("correct_5cm","correct_10cm","correct_20cm")}}


def score_depth(dest,scene,renders):
    manifest=json.loads((ROOT/"reports/vr-room-iphone-30k/frames.json").read_bytes())["frames"]
    scale=json.loads((renders/"dataparser_transforms.json").read_bytes())["scale"]
    groups={}
    raw=renders/"final_renders/pred/depth/raw"
    expected=set()
    for group in ("within","with"):
        rows=[]
        totals={}
        for frame in manifest[group]:
            rgb=Path(frame["rgb"])
            label=rgb.parts[1]+"_"+rgb.stem
            expected.add(label)
            # Historical save_depth writes millimetre codes for the model's scaled Z.
            pred=np.load(raw/(label+".npy"),allow_pickle=False)
            if pred.ndim == 3 and pred.shape[2] == 1:
                pred=pred[...,0]
            with Image.open(scene/rgb.parent.parent/"reference_depth"/(rgb.stem+".png")) as im:
                gt=np.asarray(im).astype(np.float32)*.001
            result=faro_counts(gt,pred*(.001/float(scale)))
            rows.append(dict(id=label,**finish(result)))
            for k,v in result.items():
                totals[k]=totals.get(k,0)+v
        groups[group]={"frame_count":len(rows),"pooled":finish(totals),"frames":rows}
    if {p.stem for p in raw.glob("*.npy")} != expected:
        raise ValueError("rendered Faro names differ from the two registered holdouts")
    atomic_json(dest/"evaluation/faro_common_mask.json",{"mask":"finite Faro Z strictly greater than 0.1 m",
                "dataparser_scale":scale,"prediction_metres_factor":.001/float(scale),"trajectories":groups})


def strict_mesh(scene,mesh,output):
    from dn_splatter.eval import eval_mesh_mushroom_vis_cull as official
    import trimesh
    reg=json.loads((ROOT/"reports/vr-room-iphone-30k/registered_frames.json").read_bytes())["registered_rgb_paths"]["train"]
    names={Path(p).name for p in reg}
    cam_file=scene/"iphone/long_capture/transformations_colmap.json"
    cameras=json.loads(cam_file.read_bytes())
    frames=[r for r in cameras["frames"] if Path(r["file_path"]).name in names]
    if len(frames)!=376 or {Path(r["file_path"]).name for r in frames}!=names:
        raise ValueError("strict crop requires all 376 training cameras")
    output.mkdir(parents=True,exist_ok=True)
    filtered=output/"train_cameras.json"
    atomic_json(filtered,{**cameras,"frames":frames})
    signatures=[tuple(r[k] if k in r else cameras[k] for k in ("h","w","fl_x","fl_y","cx","cy")) for r in frames]
    native=official.cull_mesh
    def cull(dataset_path,mesh,transformation_files,test_ids,**options):
        if len(set(signatures))==1:
            return native(dataset_path,mesh,filtered,[],**options)
        # Varying intrinsics need rendering and projection with each frame's own K.
        mesh.remove_unreferenced_vertices()
        verts,faces=mesh.vertices,mesh.faces
        if options["subdivide"]:
            verts,faces=trimesh.remesh.subdivide_to_size(verts,faces,max_edge=options["max_edge"],max_iter=10)
        obs=np.zeros(len(verts),dtype=np.int32)
        invalid=np.zeros_like(obs)
        for r,(h,w,fx,fy,cx,cy) in zip(frames,signatures):
            K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32)
            cv=np.asarray(r["transform_matrix"],dtype=np.float32)@np.diag([1.,-1.,-1.,1.])
            with Image.open(dataset_path/r["depth_file_path"]) as im:
                sensor=np.asarray(im,dtype=np.float32)*.001
            z=official.render_depth_maps_doublesided(mesh,[cv],h,w,K,far=10.)[0]
            seen,missing=official.cull_from_one_pose(verts,cv,h,w,K,z,sensor,
                    remove_missing_depth=options["remove_missing_depth"],remove_occlusion=options["remove_occlusion"])
            obs+=seen
            invalid+=missing
        keep=(obs[faces]>3).any(axis=1) & ~(invalid[faces]>.7*obs[faces]).all(axis=1)
        out=trimesh.Trimesh(verts,faces[keep],process=False)
        out.remove_unreferenced_vertices()
        return out
    with patch.object(official,"cull_mesh",cull):
        np.random.seed(42)
        official.main(scene,mesh,device="iphone",output=output,output_same_as_pred_mesh=False)
    atomic_json(output/"membership.json",{"camera_file_sha256":digest(cam_file),"train_cameras_sha256":digest(filtered),
                "registered":len(frames),"distinct_intrinsics":len(set(signatures)),"seed":42})


def evaluate(dest,scene,cfg,ck):
    from importlib import metadata
    src=Path(os.environ.get("DN_SOURCE_DIR",ROOT/".local/upstream/dn-splatter"))
    env=dict(os.environ,DN_PROJECT_ROOT=str(ROOT),DN_RUN_ROOT=str(dest),DN_SCENE_DIR=str(scene))
    evaluation=dest/"evaluation"
    evaluation.mkdir(parents=True,exist_ok=True)
    (dest/"logs").mkdir(exist_ok=True)
    cli=[sys.executable,ROOT/"scripts/vr_room_checkpoint.py",ck]
    def stage(name,args,target):
        if target.exists():
            return
        with (dest/"logs"/(name+".log")).open("a") as log:
            subprocess.run(list(map(str,args)),cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not target.is_file() or not target.stat().st_size:
            raise ValueError(f"{name} did not produce {target}")
    official=evaluation/"official_faro.json"
    renders=dest/"renders_official"
    renders.mkdir(exist_ok=True)
    shutil.copyfile(cfg.parent/"dataparser_transforms.json",renders/"dataparser_transforms.json")
    stage("official_faro",[*cli,"ns-eval","--load-config",cfg,"--output-path",official,
                          "--render-output-path",os.path.relpath(renders,src)],official)
    score_depth(dest,scene,renders)
    mesh=dest/"mesh/DepthAndNormalMapsPoisson_poisson_mesh.ply"
    mesh.parent.mkdir(exist_ok=True)
    stage("poisson",[*cli,"gs-mesh","dn","--load-config",cfg,"--output-dir",mesh.parent,
                      "--total-points","5000000","--poisson-depth","9"],mesh)
    old=dest/"mesh/official418"
    old.mkdir(exist_ok=True)
    from dn_splatter.eval import eval_mesh_mushroom_vis_cull as exp
    names=json.loads((scene/"iphone/long_capture/transformations_colmap.json").read_bytes())["frames"]
    if len(names)!=418:
        raise ValueError("official long camera count changed")
    if not (old/"mesh_metrics.json").exists():
        np.random.seed(42)
        exp.main(scene,mesh,device="iphone",output=old,output_same_as_pred_mesh=False)
    strict=dest/"mesh/strict376"
    if not (strict/"membership.json").exists():
        strict_mesh(scene,mesh,strict)
    products=[official,evaluation/"faro_common_mask.json",mesh,old/"mesh_metrics.json",strict/"mesh_metrics.json"]
    atomic_json(evaluation/"complete.json",{"updates":30000,"faro_training_views":0,"faro_tuning_views":0,
                "official_crop_views":418,"train_crop_views":376,"torch_version":metadata.version("torch"),
                "artifacts":[{"path":str(p),"bytes":p.stat().st_size,"sha256":digest(p)} for p in products]})
