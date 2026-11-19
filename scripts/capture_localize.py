"""Localize held-out RGB against a frozen COLMAP training map and export observations."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np
import pycolmap as pc


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(v):
    return dict(zip(("min","p10","p50","p90","p95","max"),np.percentile(v,[0,10,50,90,95,100]).tolist())) if len(v) else None


def blob(db, table, image_id, dtype):
    rows,cols,data=db.execute(f"select rows,cols,data from {table} where image_id=?",(image_id,)).fetchone()
    return np.frombuffer(data or b"",dtype).reshape(rows,cols)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run",type=Path)
    ap.add_argument("--map",default="map.json")
    a=ap.parse_args()
    s=a.run/"sfm"
    maplog=json.loads((s/a.map).read_text())
    model=Path(max(maplog["components"],key=lambda r:len(r.get("registered_primary",r["registered"])))["model"])
    fingerprint={p.name:digest(p) for p in model.glob("*.bin")}
    rec=pc.Reconstruction(str(model))
    camera=next(iter(rec.cameras.values()))
    train=sqlite3.connect((s/"train.db").as_uri()+"?mode=ro",uri=True)
    query=sqlite3.connect((s/"queries.db").as_uri()+"?mode=ro",uri=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(17)
    train_names={im.name:im for im in rec.images.values() if im.has_pose}
    pts=rec.points3D
    point_ids=np.array(list(pts.keys()))
    xyz=np.array([pts[i].xyz for i in point_ids])
    centers={n:im.projection_center() for n,im in train_names.items()}
    poses={n:im.cam_from_world().matrix().tolist() for n,im in train_names.items()}
    reproj=[]
    observations=[]
    lengths=[]
    angles=[]
    for pi,pid in enumerate(point_ids):
        pt=pts[pid]
        track=pt.track.elements
        lengths.append(len(track))
        # Angle between the rays at the first and last temporal observation of this track.
        names=sorted(rec.images[e.image_id].name for e in track)
        l=centers[names[0]]-pt.xyz
        r=centers[names[-1]]-pt.xyz
        denom=np.linalg.norm(l)*np.linalg.norm(r)
        if denom>0:
            angles.append(float(np.degrees(np.arccos(np.clip(l@r/denom,-1,1)))))
        for e in track:
            im=rec.images[e.image_id]
            if im.name.startswith("aux_"):
                continue
            uv=im.points2D[e.point2D_idx].xy
            pc3=im.cam_from_world()*pt.xyz
            if pc3[2]<=0:
                continue
            pred=camera.img_from_cam(pc3)
            if pred is None or not np.isfinite(pred).all():
                continue
            err=np.linalg.norm(pred-uv)
            reproj.append(err)
            observations.append([int(Path(im.name).stem.split("_")[-1]),pi,*uv,pc3[2],err,len(track)])
    np.savez_compressed(s/"observations.npz",observations=np.array(observations),xyz=xyz,point_ids=point_ids)
    @lru_cache(maxsize=24)
    def index(name):
        im=train_names[name]
        ix=np.array([i for i,k in enumerate(im.points2D) if k.has_point3D()],dtype=int)
        if len(ix)<2:
            return None,None
        ids=np.array([im.points2D[i].point3D_id for i in ix])
        desc=blob(train,"descriptors",im.image_id,np.uint8)[ix].astype(np.float32)
        tree=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64))
        tree.add([desc])
        tree.train()
        return tree,ids
    quality=[]
    lookup={r["id"]+".png":int(r["actual_rgb_timestamp_ns"]) for r in json.loads((a.run/"transfer/selection.json").read_text())}
    primary=[n for n in train_names if n in lookup]
    def estimate(qid,name,limit):
        desc=blob(query,"descriptors",qid,np.uint8).astype(np.float32)
        uv=blob(query,"keypoints",qid,np.float32)[:,:2]
        neighbors=sorted(primary,key=lambda n:abs(lookup[n]-lookup[name]))[:limit]
        candidates=[]
        for n in neighbors:
            tree,ids=index(n)
            if tree is None:
                continue
            for match in tree.knnMatch(desc,k=2) if len(desc) else ():
                if len(match)!=2:
                    continue
                m,k=match
                if m.distance < .75*k.distance:
                    candidates.append((m.distance/max(k.distance,1),m.queryIdx,int(ids[m.trainIdx])))
        indices=[]; used_2d=set(); used_3d=set()
        for _,i,pid in sorted(candidates):
            if i not in used_2d and pid not in used_3d:
                used_2d.add(i); used_3d.add(pid); indices.append((i,pid))
        out=dict(name=name,train_neighbors=neighbors,unique_correspondences=len(indices),registered=False,inliers=0)
        if len(indices)>=6:
            u=np.array([uv[i] for i,pid in indices]); X=np.array([pts[pid].xyz for i,pid in indices])
            opt=pc.AbsolutePoseEstimationOptions()
            opt.ransac.max_error=4.0
            opt.ransac.random_seed=17
            result=pc.estimate_and_refine_absolute_pose(u,X,camera,opt,pc.AbsolutePoseRefinementOptions())
            if result and result["num_inliers"]>=20:
                rigid=result["cam_from_world"]
                inliers=np.array(result["inlier_mask"],dtype=bool)
                P=rigid*X[inliers]
                K=camera.calibration_matrix()
                q=P@K.T
                err=np.linalg.norm(q[:,:2]/q[:,2:]-u[inliers],axis=1)
                out.update(registered=True,inliers=int(inliers.sum()),inlier_reprojection_px=stats(err))
                poses[name]=rigid.matrix().tolist()
        return out
    for qid,name in query.execute("select image_id,name from images order by name"):
        out=estimate(qid,name,4)
        if not out["registered"] and a.map.startswith("map_full"):
            retry=estimate(qid,name,24)
            retry["first_attempt"]=out
            out=retry
        quality.append(out)
    assert fingerprint=={p.name:digest(p) for p in model.glob("*.bin")}
    # Verified image-match graph connectivity, independently of mapper submodel count.
    adj={i:set() for i, in train.execute("select image_id from images")}
    good_edges=0
    for pair, in train.execute("select pair_id from two_view_geometries where rows>=15"):
        x,y=pc.pair_id_to_image_pair(pair);adj[x].add(y);adj[y].add(x);good_edges+=1
    remaining=set(adj); graph=[]
    while remaining:
        stack=[remaining.pop()]; group=[]
        while stack:
            cur=stack.pop();group.append(cur);new=adj[cur]&remaining
            stack.extend(new);remaining-=new
        graph.append(len(group))
    summary=dict(map=maplog,map_files_sha256_before_and_after=fingerprint,
                 method="Train-only incremental SfM. Held-out SIFT to 4 nearest-time registered training views, FLANN ratio 0.75; unique 2D and 3D. Fixed camera LO-RANSAC PnP 4px, minimum 20 inliers; pose refinement only.",
                 train_registered=sorted(train_names),train_unregistered=sorted(set(maplog["train"])-set(train_names)),
                 graph_components=sorted(graph,reverse=True),verified_edges_at_least_15_matches=good_edges,
                 sparse_points=len(point_ids),observations=len(observations),track_length=stats(lengths),
                 first_last_track_ray_angle_degrees=stats(angles),train_reprojection_px=stats(reproj),
                 heldout=quality,poses_w2c_sfm_units=poses)
    (s/"pose_quality.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(len(train_names),"training poses and",sum(q["registered"] for q in quality),"held-out poses")


if __name__=="__main__":
    main()
