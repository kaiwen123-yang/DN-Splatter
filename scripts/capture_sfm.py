"""Train-only calibrated COLMAP map. Extract held-out features into a separate DB."""
import argparse
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pycolmap as pc


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run",type=Path)
    ap.add_argument("stage",choices=["map","queries","match","features"])
    ap.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    ap.add_argument("--exhaustive",action="store_true")
    ap.add_argument("--bridge",action="store_true",help="Include only the sealed pose-only auxiliary RGB set")
    a=ap.parse_args()
    s=a.run/"sfm"; s.mkdir(exist_ok=True)
    root=a.run/"rect"/"1280"/"images"
    splits=read(a.run/"transfer"/"splits.json")["splits"]
    train=[x+".png" for x in splits["train"]]
    primary=set(train)
    if a.bridge:
        train += [r["id"]+".png" for r in read(a.run/"bridge"/"frames.json")]
    held=sorted(x+".png" for split in ("val","test") for x in splits[split])
    k=read(a.run/"rect"/"grids.json")["1280"]["K_colmap_ns"]
    params=[k[0][0],k[1][1],k[0][2],k[1][2]]
    n=int(os.environ["DN_CPU_BUDGET"])
    db=s/("queries.db" if a.stage=="queries" else "train.db")
    opt=pc.ImageReaderOptions(camera_model="PINHOLE",camera_params=",".join(map(str,params)))
    if a.bridge and db.is_file():
        with sqlite3.connect(db) as cx:
            camera_ids=cx.execute("select distinct camera_id from images").fetchall()
        assert len(camera_ids)==1
        opt.existing_camera_id=camera_ids[0][0]
    device=getattr(pc.Device,a.device)
    if a.stage in ("queries","features"):
        names=held if a.stage=="queries" else train
        fe=pc.FeatureExtractionOptions(num_threads=n,use_gpu=a.device=="cuda")
        pc.extract_features(str(db),str(root),image_names=names,camera_mode=pc.CameraMode.SINGLE,
                            camera_model="PINHOLE",reader_options=opt,extraction_options=fe,device=device)
        with sqlite3.connect(db) as cx:
            actual=cx.execute("select name, keypoints.rows from images join keypoints using(image_id)").fetchall()
            assert {r[0] for r in actual}==set(names)
            if not (a.run/"transfer/splits.json").is_file() or "s01_identities_verified" not in read(a.run/"transfer/splits.json")["review"]:
                assert all(r[1]>=4 for r in actual)
        return
    if a.stage=="match":
        mo=pc.FeatureMatchingOptions(num_threads=n,use_gpu=a.device=="cuda")
        if a.exhaustive:
            pc.match_exhaustive(str(db),matching_options=mo,device=device)
        else:
            po=pc.SequentialPairingOptions(overlap=10,quadratic_overlap=True,loop_detection=False,num_threads=n)
            pc.match_sequential(str(db),matching_options=mo,pairing_options=po,device=device)
        return
    suffix="bridge" if a.bridge else ("exhaustive" if a.exhaustive else "")
    out=s/("models_"+suffix if suffix else "models")
    if out.exists():
        from datetime import datetime
        out.rename(out.with_name(out.name+".interrupted-"+datetime.now().strftime("%Y%m%dT%H%M%S%f")))
    out.mkdir()
    op=pc.IncrementalPipelineOptions(num_threads=n,random_seed=17,ba_refine_focal_length=False,
        ba_refine_principal_point=False,ba_refine_extra_params=False,ba_refine_sensor_from_rig=False,image_names=train)
    op.mapper.abs_pose_refine_focal_length=False
    op.mapper.abs_pose_refine_extra_params=False
    op.mapper.num_threads=n
    models=pc.incremental_mapping(str(db),str(root),str(out),options=op)
    components=[]
    for key,rec in models.items():
        registered=sorted(im.name for im in rec.images.values() if im.has_pose)
        assert set(registered)<=set(train)
        assert len(rec.cameras)==1 and np.allclose(next(iter(rec.cameras.values())).params,params,rtol=0,atol=1e-9)
        components.append(dict(key=key,model=str(out/str(key)),registered=registered,
                               registered_primary=sorted(primary.intersection(registered)),points=rec.num_points3D()))
    save(s/("map_"+suffix+".json" if suffix else "map.json"),dict(train=train,components=components,
         pycolmap=pc.__version__,single_fixed_pinhole=params,sequential_overlap=10,quadratic_overlap=True,
         exhaustive=a.exhaustive,seed=17,cpu_threads=n))
    if not components and not a.exhaustive:
        raise RuntimeError("No training map recovered")


if __name__=="__main__":
    main()
