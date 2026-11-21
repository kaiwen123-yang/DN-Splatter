"""Relocalize the two candidate regions against fixed external primary training observations."""
from functools import lru_cache
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from capture_quality_paths import B1, GEOM, GEORUN, SCENE, check, read, write


FLIP = np.diag([1., -1., -1., 1.])
CANDIDATES = set(range(1294, 1307)) | {1514}


def pixdepth(z, uv):
    x, y = np.rint(uv).astype(int)
    if not (2 <= x < z.shape[1]-2 and 2 <= y < z.shape[0]-2):
        return np.nan
    block = z[y-2:y+3, x-2:x+3]
    vals = block[np.isfinite(block) & (block >= .3) & (block <= 6)]
    if len(vals) < 6 or np.ptp(vals) > max(.05, .03 * np.median(vals)):
        return np.nan
    return float(np.median(vals))


def main():
    meta = read(B1 / "inputs/transforms.json")
    train = {int(Path(n).stem[-5:]) for n in meta["train_filenames"]}
    frames = {int(Path(x["file_path"]).stem[-5:]): x for x in meta["frames"]}
    fixed = set(train) - CANDIDATES
    splits = {int(Path(n).stem[-5:]): s for s in ("train", "val", "test") for n in meta[s + "_filenames"]}
    selected = {r["id"]:r for r in read(GEORUN / "transfer/selection.json")}
    K = np.array(read(GEORUN / "rect/grids.json")["1280"]["K_cv"])
    ki = np.linalg.inv(K)
    dbs = [sqlite3.connect((GEORUN / "sfm" / n).as_uri()+"?mode=ro", uri=True)
           for n in ("train.db", "queries.db")]
    ids = [{name:idx for idx,name in db.execute("select image_id,name from images")} for db in dbs]
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    context = []
    for i in [*range(1288,1313),*range(1510,1519)]:
        name = Path(frames[i]["file_path"]).name
        row = selected[Path(name).stem]
        which = 0 if i in train else 1
        check(name in ids[which] and name == row["id"]+".png", "fixed RGB/SIFT association")
        context.append(dict(id=i, split=splits[i], rgb=row["rgb"]["file"], depth=row["depth"]["file"],
                            actual_rgb_timestamp_ns=row["actual_rgb_timestamp_ns"], feature_image_id=ids[which][name]))

    @lru_cache(48)
    def observation(i):
        which = 0 if i in train else 1
        name = Path(frames[i]["file_path"]).name
        db, idx = dbs[which], ids[which][name]
        def array(table, dtype):
            r,c,b = db.execute(f"select rows,cols,data from {table} where image_id=?", (idx,)).fetchone()
            return np.frombuffer(b or b"", dtype=dtype).reshape(r,c)
        uv = array("keypoints", np.float32)[:,:2].astype(float) - .5
        return uv, array("descriptors", np.uint8).astype(np.float32), np.load(GEORUN / "rect/640/depth" / (name[:-4]+".npy"))

    @lru_cache(40)
    def anchor(i):
        uv, desc, depth = observation(i)
        z = np.asarray([pixdepth(depth,(p+.5)*.5-.5) for p in uv])
        xyz = np.c_[uv,np.ones(len(uv))] @ ki.T * z[:,None]
        T = np.asarray(frames[i]["transform_matrix"]) @ FLIP
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64))
        matcher.add([desc])
        matcher.train()
        return matcher, xyz @ T[:3,:3].T + T[:3,3]

    results = {}
    order = sorted(CANDIDATES & train) + sorted(CANDIDATES - train)
    for i in order:
        q, desc, depth = observation(i)
        # Primary candidate source observations come only from fixed reliable external train frames.
        neighbors = sorted(fixed,key=lambda j:(abs(j-i),j))[:16]
        matches = []
        for j in neighbors:
            matcher, xyz = anchor(j)
            for pair in matcher.knnMatch(desc,k=2):
                if len(pair) != 2:
                    continue
                m,n = pair
                if m.distance < .75*n.distance and np.isfinite(xyz[m.trainIdx]).all():
                    matches.append((m.distance/max(1.,n.distance),m.queryIdx,j,xyz[m.trainIdx]))
        used, keep = set(), []
        for match in sorted(matches, key=lambda r:r[:3]):
            if match[1] not in used:
                used.add(match[1])
                keep.append(match)
        old = np.asarray(frames[i]["transform_matrix"]) @ FLIP
        info = dict(id=i,split=splits[i],anchors=neighbors,correspondences=len(keep),
                    old_c2w_cv=old.tolist(),new_c2w_cv=old.tolist(),accepted=False,inliers=0)
        if len(keep) >= 6:
            xyz = np.array([r[3] for r in keep])
            uv = np.array([q[r[1]] for r in keep])
            ok,rvec,tvec,inl = cv2.solvePnPRansac(xyz,uv,K,None,iterationsCount=3000,reprojectionError=4.,
                                                 confidence=.999,flags=cv2.SOLVEPNP_EPNP)
            if ok and inl is not None and len(inl)>=6:
                idx=inl.ravel()
                rvec,tvec=cv2.solvePnPRefineLM(xyz[idx],uv[idx],K,None,rvec,tvec)
                W=np.eye(4)
                W[:3,:3]=cv2.Rodrigues(rvec)[0]
                W[:3,3]=tvec.ravel()
                new=np.linalg.inv(W)
                actual=np.array([pixdepth(depth,(p+.5)*.5-.5) for p in uv[idx]])
                def residual(T):
                    pc=(xyz[idx]-T[:3,3])@T[:3,:3]
                    pr=pc@K.T
                    uvproj=pr[:,:2]/np.where(np.abs(pr[:,2:])>1e-9,pr[:,2:],np.nan)
                    valid=np.isfinite(actual)
                    error=np.linalg.norm(uvproj-uv[idx],axis=1)
                    return dict(median_px=float(np.nanmedian(error)),p95_px=float(np.nanpercentile(error,95)),
                                positive_fraction=float(np.mean(pc[:,2]>0)),rgbd_pairs=int(valid.sum()),
                                median_abs_Z_m=float(np.median(abs(pc[valid,2]-actual[valid]))) if valid.any() else None)
                before,after=residual(old),residual(new)
                supporters=sorted({keep[k][2] for k in idx})
                change=np.linalg.inv(old)@new
                info.update(inliers=len(idx),supporting_train_frames=supporters,before=before,proposed=after,
                            proposed_c2w_cv=new.tolist(),delta_m=float(np.linalg.norm(change[:3,3])),
                            delta_degrees=float(np.degrees(np.arccos(np.clip((np.trace(change[:3,:3])-1)/2,-1,1)))))
                accepted=(len(idx)>=20 and len(supporters)>=2 and after["median_px"]<=2.5 and
                          after["positive_fraction"]==1 and (before["positive_fraction"]<.5 or
                          before["median_px"]>after["median_px"]+.5))
                if accepted:
                    frames[i]["transform_matrix"]=(new@FLIP).tolist()
                    info.update(new_c2w_cv=new.tolist(),accepted=True)
                # Private diagnostic with measured features and both projections on the original image.
                image=cv2.imread(str(B1/"inputs"/frames[i]["file_path"]))
                for T,col in ((old,(0,0,230)),(new,(0,190,0))):
                    pc=(xyz[idx]-T[:3,3])@T[:3,:3]
                    v=pc@K.T
                    for target,proj in zip(uv[idx],v):
                        if proj[2]<=0: continue
                        point=proj[:2]/proj[2]
                        if np.isfinite(point).all() and np.max(np.abs(point))<10000:
                            cv2.line(image,tuple(target.round().astype(int)),tuple(point.round().astype(int)),col,1)
                cv2.imwrite(str(GEOM/f"pose_{i:05d}.jpg"),image)
        info["status"]="corrected" if info["accepted"] else "original_retained_unresolved"
        results[str(i)] = info
        print(i,info["status"],info["inliers"],flush=True)
    write(SCENE/"transforms.json",meta)
    write(GEOM/"pose.json",dict(context=context,results=results,metric_scale="unchanged",
          method="SIFT against fixed external primary training RGB-D map, metric PnP RANSAC and pose-only LM. No map BA or interpolation. Heldout estimated last. Residuals on the same proposed inliers."))


if __name__ == "__main__":
    GEOM.mkdir(parents=True,exist_ok=True)
    main()
