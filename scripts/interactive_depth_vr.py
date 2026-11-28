"""Structured missing-depth transfer on VR Room, independent of the zero-hole DN comparison."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from interactive_depth_candidates import CaseGeometry
from interactive_depth_geometry import atomic_npz,edge_pairs,primary
from interactive_depth_inputs import MODEL_REV,MODEL_SHA,OLD,RUN,digest_json
from interactive_depth_mdm import FrozenMDM,artifact,stratified,token_file
from interactive_depth_metrics import Scores
from interactive_depth_quality import (HEADS,FrozenQuality,best_networks,calibrate,choose_epochs,choose_pixel_prob,
        complete_batches,input_for,preprocessing,train)
from interactive_depth_transfer import empty_views
from lingbot_common import ROOT,read,require,save,sha


KINDS=("contiguous_block","geometric_boundary_band")
REP="full_range_sparse"
REGISTERED=ROOT/"reports/vr-room-iphone-30k/registered_frames.json"
SCENE=RUN.parent/"restore/tree/.local/data/room_datasets/vr_room"


def dataset(scene):
    split=read(REGISTERED)["registered_rgb_paths"]
    rows=[]
    for capture in ("long_capture","short_capture"):
        base=scene/"iphone"/capture
        # The ARKit file contains each acquired RGB-D frame's measured intrinsics;
        # the COLMAP file provides the frozen registered frame identities and camera transforms.
        physical=read(base/"transformations.json")
        colmap=read(base/"transformations_colmap.json")
        physical_by_name={Path(f["file_path"]).name:f for f in physical["frames"]}
        cameras={Path(f["file_path"]).name:f for f in colmap["frames"]}
        for name,files in split.items():
            for rel in files:
                if Path(rel).parts[1]!=capture:
                    continue
                fname=Path(rel).name
                f=physical_by_name[fname]
                item=cameras[fname]
                seq=int(Path(fname).stem.split("_")[-1])
                # Actual timestamps are not part of this release. Groups are stated in frame units.
                group=(seq-1)//20+(10000 if capture=="short_capture" else 0)
                key={k:f.get(k,physical.get(k)) for k in ("fl_x","fl_y","cx","cy","w","h")}
                require(all(v is not None for v in key.values()),"Per-frame capture calibration "+fname)
                k=[[key["fl_x"],0,key["cx"]],[0,key["fl_y"],key["cy"]],[0,0,1]]
                img=scene/rel
                dep=base/item["depth_file_path"]
                rows.append(dict(id=capture+"_"+Path(fname).stem,split=name,capture=capture,frame=seq,group=group,
                      image=str(img),depth=str(dep),image_sha256=sha(img),depth_sha256=sha(dep),K_cv=k,
                      wh=[key["w"],key["h"]],c2w=item["transform_matrix"]))
    fit_rows=[r for r in rows if r["split"]=="train"]
    groups=np.unique([r["group"] for r in fit_rows])
    cal=set(np.random.default_rng(42).permutation(groups)[::5])
    cal_rows=[s for s in fit_rows if s["group"] in cal]
    for row in rows:
        role="outer"
        if row["split"]=="train":
            role="calibration" if row["group"] in cal else "fit"
            if role=="fit":
                pose=np.array(row["c2w"])
                for s in cal_rows:
                    other=np.array(s["c2w"])
                    close=np.linalg.norm(pose[:3,3]-other[:3,3])<.1
                    facing=(np.trace(pose[:3,:3].T@other[:3,:3])-1)/2
                    if abs(s["frame"]-row["frame"])<=2 or (close and facing>=np.cos(np.deg2rad(3))):
                        role="excluded_near_calibration"
                        break
        row["role_for_outer"]=[role]
    require([sum(r["split"]==s for r in rows) for s in ("train","within","with")]==[376,42,186],"VR split")
    return dict(rows=rows,masks=list(KINDS),representations=[REP],registered_sha256=sha(REGISTERED),
                units="optical Z m; original mm multiplied by .001 exactly once",frame_group_size=20,
                camera="original per-frame ARKit RGB-D K; registered COLMAP frame identity; single-view inference",
                faro_input=False,coordinates="registered RGB pixel grid, no reprojection or crop",
                native_bundle="blocked: package exposes registered depth, no native iPhone beams/lineage")


def make(root,scene,weight):
    d=dataset(scene)
    config=root/"freeze.json"
    if config.exists():
        require(read(config)==d,"VR source identity changed")
    else:
        save(config,d)
    model=FrozenMDM(weight)
    identity=digest_json(dict(data=sha(config),code=sha(Path(__file__)),model=MODEL_SHA,revision=MODEL_REV))
    outputs=[]
    for row in d["rows"]:
        fid=row["id"]
        im=cv2.imread(row["image"],cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        native=cv2.imread(row["depth"],cv2.IMREAD_UNCHANGED)
        require(native.dtype==np.uint16 and native.shape==im.shape[:2] and list(native.shape[::-1])==row["wh"],
                "VR image/depth/K canvas")
        z=np.where((native!=0)&(native!=65535),native,0).astype(np.float32)*.001
        sides=edge_pairs(z)
        h,w=z.shape
        for j,kind in enumerate(KINDS):
            folder=root/"cases"/fid/kind
            rfile=folder/"receipt.json"
            if rfile.exists():
                r=read(rfile)
                require(r["identity"]==identity and all(sha(folder/v["path"])==v["sha256"] for v in r["outputs"]),
                        "VR condition resume")
                outputs.append(r)
                continue
            seed=np.random.SeedSequence((42,row["group"]//10000,row["frame"],j))
            rng=np.random.default_rng(seed)
            hidden=np.zeros(z.shape,bool)
            side=np.zeros(z.shape,np.uint8)
            for fg,bg,_ in sides:
                side.flat[bg]=2
                side.flat[fg]=1
            if kind=="contiguous_block":
                y,x=int(rng.integers(h-128+1)),int(rng.integers(w-128+1))
                hidden[y:y+128,x:x+128]=True
            else:
                hidden=cv2.dilate(np.uint8(side>0),np.ones((5,5),np.uint8)).astype(bool)
            flat=np.flatnonzero(hidden&primary(z)).astype(np.int32)
            visible=np.where(hidden,0.,z)
            geo=CaseGeometry(rgb,visible,np.zeros_like(z),empty_views(z.shape))
            ds,obs,local=geo.table(flat)
            reference=z.ravel()[flat]
            sample=stratified(reference,side.ravel()[flat],seed)
            base=folder/"source_and_score.npz"
            atomic_npz(base,flat=flat,reference=reference,edge_side=side.ravel()[flat],sample_index=sample,
                       sample_rep=np.zeros(len(sample),np.uint8),depths=ds[:,:4],local=local,observations=obs,
                       audit_bits=np.empty(0,np.uint8),hidden_input=np.packbits(hidden,axis=None))
            dep,tokens,timing=model.predict(rgb,visible,np.asarray(row["K_cv"]))
            outfile=folder/(REP+".npz")
            atomic_npz(outfile,**token_file(dep,tokens,flat))
            r=dict(identity=identity,id=fid,kind=kind,shape=list(z.shape),raw_missing=int((z==0).sum()),
                   hidden_input=int(hidden.sum()),score_pixels=len(flat),outputs=[artifact(base),artifact(outfile)],timing=timing)
            save(rfile,r)
            outputs.append(r)
        if len(outputs)==2 or len(outputs)%40==0:
            print("VR masked conditions",len(outputs),"/",len(d["rows"])*2,fid,flush=True)
    save(root/"masked_manifest.json",dict(identity=identity,frames=len(d["rows"]),cases=len(outputs),
         score_pixels=sum(v["score_pixels"] for v in outputs),official_depth_calls=len(outputs),official_feature_calls=len(outputs)))


@torch.no_grad()
def evaluate(root,data,model_dir,name):
    q=FrozenQuality(model_dir)
    out=root/(name+".json")
    by_split={s:Scores() for s in ("train","within","with")}
    for s in by_split:
        # Training-set rows under VR adaptation are descriptive only; never a validation claim.
        pop=dict(data,rows=[dict(r,role_for_outer=["outer"]) for r in data["rows"] if r["split"]==s])
        scores=by_split[s]
        for row,kind,rep,b,idx,d,ii,jj,g,m,_ in complete_batches(root,pop,0,"outer",q.prep):
            z=b["reference"][idx]
            prefix=kind,rep
            scores.raw_scores(prefix,d,z,b["edge_side"][idx])
            for n in HEADS:
                probs=np.zeros((*d.shape,3),np.float32)
                if len(ii):
                    logits=q.nets[n](input_for(n,g,m))/q.cal[rep][n]["temperature"]
                    probs[ii,jj]=logits.softmax(1).cumsum(1)[:,:3].cpu().numpy()
                scores.calibration(prefix,n,probs,d,z)
                choices,p=choose_pixel_prob(probs[:,:,1],primary(d))
                scores.chosen(prefix,row["group"],n,choices,p,d,z,q.cal[rep][n])
    save(out,dict(condition=name,calibration_sha256=sha(model_dir/"calibration.json"),
         reference="structured hidden sensor readings; Faro is never a source or label",
         independent_external_role="with = separate short-capture trajectory; within = same acquisition",
         split_reports={s:sc.report() for s,sc in by_split.items()}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["make","evaluate","adapt"])
    p.add_argument("--out",type=Path,default=RUN/"vr_transfer")
    p.add_argument("--scene",type=Path,default=SCENE)
    p.add_argument("--weight",type=Path,default=OLD/"weights/depth/model.pt")
    p.add_argument("--hall-fold",type=Path,default=RUN/"quality/fold_0")
    a=p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    if a.stage=="make":
        make(a.out,a.scene,a.weight)
    else:
        data=read(a.out/"freeze.json")
        actual=read(a.out/"masked_manifest.json")
        require(actual["cases"]==2*604,"Incomplete VR mask set")
        if a.stage=="evaluate":
            evaluate(a.out,data,a.hall_fold,"hall_fold0_zero_shot")
        else:
            folder=a.out/"quality/vr_train_only"
            prep=preprocessing(a.out,data,0,folder)
            train(a.out,data,0,folder,prep)
            epochs=choose_epochs(a.out,data,0,folder,prep)
            calibrate(a.out,data,0,folder,prep,epochs)
            evaluate(a.out,data,folder,"vr_train_only_adaptation")
