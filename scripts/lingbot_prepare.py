"""Verify the restored B2 inputs and freeze the hall experiments before inference."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lingbot_common import (FLIP, HARD, P0, PROTOCOL, ROOT, SEQ, SRC, depth, masks,
                            member, read, render_z, require, rgb, save, sha)


def make_canvas(cal, grid):
    c = cal["cameras"]["RGB"]
    w, h = grid["width"], grid["height"]
    mx, my = cv2.initUndistortRectifyMap(np.array(c["K"]), np.array(grid["D_cv_source"]),
                                         None, np.array(grid["K_cv"]), (w, h), cv2.CV_32FC1)
    return (mx >= 0) & (my >= 0) & (mx <= c["width"] - 1) & (my <= c["height"] - 1)


def best_roi(mask, size=160):
    # Count pixels within tiles on the original canvas, including the last complete tile.
    h, w = mask.shape
    integral = cv2.integral(mask.astype(np.uint8))
    ys = np.unique(np.r_[np.arange(0, h-size+1, 40), h-size])
    xs = np.unique(np.r_[np.arange(0, w-size+1, 40), w-size])
    y, x = np.ix_(ys, xs)
    counts = integral[y+size,x+size] - integral[y,x+size] - integral[y+size,x] + integral[y,x]
    iy, ix = np.unravel_index(np.argmax(counts), counts.shape)
    left, top = int(xs[ix]), int(ys[iy])
    return [left, top, left+size, top+size], int(counts[iy,ix])


def freeze_masks(row, canvas):
    raw = depth(row)
    _, _, _, ref = masks(raw, canvas)
    h, w = raw.shape
    seed = np.random.SeedSequence([42, int(row["id"].split("_")[1])])
    sr, br = (np.random.default_rng(s) for s in seed.spawn(2))
    indices = np.flatnonzero(ref)
    scatter = np.zeros_like(ref)
    scatter.flat[sr.choice(indices, round(.1 * len(indices)), replace=False)] = True
    block = np.zeros_like(ref)
    covered = 0
    rectangles = []
    while covered < .2 * len(indices):
        x, y = int(br.integers(0, w-63)), int(br.integers(0, h-63))
        sl = np.s_[y:y+64, x:x+64]
        covered += np.count_nonzero(ref[sl] & ~block[sl])
        block[sl] = True
        rectangles.append([x,y,64,64])
    path = P0 / "masks" / (row["id"] + ".npz")
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, reference=ref, random_points_input=scatter,
                        random_points_score=scatter, blocks_input=block, blocks_score=block & ref)
    return dict(file=path.name, sha256=sha(path), reference_pixels=len(indices),
                random_score=int(scatter.sum()), block_input=int(block.sum()),
                block_score=int(covered), rectangles=rectangles)


def main():
    cv2.setNumThreads(1)
    P0.mkdir(parents=True, exist_ok=True)
    if (P0 / "freeze.json").is_file():
        for row in read(P0 / "freeze.json")["files"]:
            require(sha(row["path"]) == row["sha256"], "Frozen file changed: " + row["path"])
        return
    inv = read(SRC / "archive_inventory.json")
    meta = read(SRC / "b2/transforms.json")
    splitdoc = read(SRC / "b2/splits.json")
    srcrows = {r["id"]:r for r in read(SRC / "geometry/transfer/selection.json")}
    splits = splitdoc["splits"]
    members = {r["path"]:r for r in read(SRC / "b2/training_manifest.json")["files"]}
    grid = read(SRC / "geometry/rect/grids.json")["1280"]
    k = np.array(grid["K_cv"])
    kdn = np.array([[meta["fl_x"],0,meta["cx"]],[0,meta["fl_y"],meta["cy"]],[0,0,1]])
    require(np.allclose(kdn, grid["K_colmap_ns"]) and np.allclose(kdn[:2,2]-.5,k[:2,2]), "K convention")
    require((meta["w"],meta["h"]) == (1280,720), "RGB canvas")
    canvas = make_canvas(read(SRC / "geometry/transfer/calibration.json"), grid)
    np.save(P0 / "canvas.npy", canvas, allow_pickle=False)
    role = {}
    for name, count in (("train",1478),("val",184),("test",184)):
        ids = splits[name]
        require(len(ids)==count and ids==[Path(v).stem for v in meta[name+"_filenames"]], "Split "+name)
        for fid in ids:
            require(fid not in role, "Duplicate split membership")
            role[fid] = name
    require(len(meta["frames"])==len(role)==len(srcrows)==1846, "Primary frame identity count")
    def inspect(f):
        fid = Path(f["file_path"]).stem
        s = srcrows[fid]
        require(s["requested_split"] == role[fid] and not s["drop_reason"], fid)
        img = "b2/" + f["file_path"]
        zname = "geometry/rect/1280/" + f["depth_file_path"]
        b2 = inv["b2/" + f["depth_file_path"]]
        require(b2["sha256"] == members[f["depth_file_path"]]["sha256"], "B2 depth manifest " + fid)
        require(sha(SRC/img)==members[f["file_path"]]["sha256"]==inv[img]["sha256"], "RGB hash " + fid)
        require(sha(SRC/zname)==inv[zname]["sha256"], "Full-range Z hash " + fid)
        with Image.open(SRC/img) as image:
            require(image.size==(1280,720), "RGB size " + fid)
        raw = np.load(SRC/zname, allow_pickle=False)
        require(raw.dtype==np.float32 and raw.shape==canvas.shape and np.isfinite(raw).all()
                and (raw>=0).all() and not np.any(raw[~canvas]), "Input Z " + fid)
        observed, primary, band, ref = masks(raw, canvas)
        restricted = np.where(primary,raw,np.float32(0))
        before = io.BytesIO()
        np.save(before, restricted, allow_pickle=False)
        require(hashlib.sha256(before.getbuffer()).hexdigest()==b2["sha256"], "B2 range projection " + fid)
        pose = np.asarray(f["transform_matrix"]) @ FLIP
        require(np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=2e-5) and
                np.isclose(np.linalg.det(pose[:3,:3]),1,atol=2e-5), "C2W " + fid)
        natural = canvas & ~observed
        # A 7x7 empty neighbourhood distinguishes extended gaps from sparse projected pixels.
        interior_holes = cv2.erode(natural.astype("uint8"), np.ones((7,7),np.uint8)).astype(bool)
        hr, nh = best_roi(interior_holes)
        er, ne = best_roi(primary & band)
        return dict(id=fid, split=role[fid], rgb_timestamp_ns=int(s["actual_rgb_timestamp_ns"]),
                    source_rgb_index=s["rgb_zero_based_index"], camera="Gemini335_RGB", rgb=img,
                    depth=zname, rgb_sha256=inv[img]["sha256"], depth_sha256=inv[zname]["sha256"],
                    frozen_b2_depth_sha256=b2["sha256"], c2w_cv=pose.tolist(),
                    valid_observed=int(observed.sum()), in_range=int(primary.sum()), reliable=int(ref.sum()),
                    out_of_range=int((observed & ~primary).sum()), natural_holes=int(natural.sum()),
                    hole_roi=hr, hole_pixels_7x7=int(interior_holes.sum()), edge_roi=er, edge_roi_pixels=ne,
                    hole_roi_pixels=nh, suspect_reference_pose=fid in HARD)
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = sorted(pool.map(inspect,meta["frames"]),key=lambda r:r["rgb_timestamp_ns"])
    require(len({r["id"] for r in frames})==1846 and np.all(np.diff([r["rgb_timestamp_ns"] for r in frames])>0),
            "Sorted camera times")
    metadata = [member(SRC / p) for p in inv if not "/frame_" in p]
    for f in metadata:
        require(f["sha256"] == inv[Path(f["path"]).relative_to(SRC).as_posix()]["sha256"], "Source metadata")
    first = frames[0]
    z = depth(first)
    y,x = np.nonzero(z>0)
    reproj = max(float(np.max(np.abs(((x-k[0,2])*z[y,x]/k[0,0])*k[0,0]/z[y,x]+k[0,2]-x))),
                 float(np.max(np.abs(((y-k[1,2])*z[y,x]/k[1,1])*k[1,1]/z[y,x]+k[1,2]-y))))
    from capture_rectify import edges
    edge = edges(z)
    canny = cv2.Canny(cv2.cvtColor(rgb(first),cv2.COLOR_BGR2GRAY),50,150)>0
    distance = cv2.distanceTransform((~canny).astype("uint8"),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    save(P0/"input_manifest.json",dict(frames=frames, sensor="Gemini 335", units="m", depth_axis="RGB optical Z",
         image_wh=[1280,720], pixel_centres="OpenCV integer", K_cv=k.tolist(), K_colmap_ns=kdn.tolist(),
         source_root=str(SRC), sources=metadata, archive_inventory=member(SRC/"archive_inventory.json"),
         same_B2_in_range_sha256_verified=1846, valid_canvas_pixels=int(canvas.sum()),
         first_frame_check=dict(id=first["id"],roundtrip_max_px=reproj, edges=int(edge.sum()),
             median_distance_to_rgb_canny_px=float(np.median(distance[edge])), registration="original D2C once")))
    train = [r for r in frames if r["split"]=="train"]
    lookup = {r["id"]:r for r in frames}
    fixed = read(ROOT/"reports/capture-full-dn-30k/20261119-v1/mesh_validation.json")["local_views"]
    roi = {f"frame_{r['train_id']:05d}":(r["object"],r["xyxy"]) for r in fixed}
    roi["frame_01420"]=("foam_teeth",[720,224,1104,608])
    samples = []
    def add(r, group, reason, rect):
        samples.append(dict(id=r["id"],group=group,reason=reason,roi_xyxy=list(map(int,rect))))
    for fid,(obj,box) in roi.items():
        require(lookup[fid]["split"]=="train", "Historical train ROI")
        add(lookup[fid],"edges_and_thin_structures",obj+" (frozen B3 crop)",box)
    for group,n,field,box in [("edges_and_thin_structures",6,"edge_roi_pixels","edge_roi"),
                              ("ordinary_complete",10,"reliable",None),("natural_holes",10,"hole_pixels_7x7","hole_roi")]:
        for segment in np.array_split(np.arange(len(train)), n):
            used = {r["id"] for r in samples}
            candidates = [train[i] for i in segment if train[i]["id"] not in used and train[i]["reliable"]]
            chosen = max(candidates,key=lambda r:r[field])
            add(chosen,group,"Highest "+field+" among available frames of this time stratum",
                chosen[box] if box else [160,90,1120,630])
    for sample in samples:
        sample["masks"]=freeze_masks(lookup[sample["id"]],canvas)
    short = train[:256]
    lists = {}
    for seq, rows in zip(SEQ,(short,train,frames)):
        ids = [r["id"] for r in rows]
        save(P0/(seq+".json"),[str((SRC/r["rgb"]).resolve()) for r in rows])
        interval=(len(ids)+319)//320
        dt=np.diff([r["rgb_timestamp_ns"] for r in rows])*1e-9
        lists[seq]=dict(ids=ids,persistent_raw_ids=list(range(8))+list(range(8,len(ids),interval)),
                        interval_s_min_median_max=[float(v) for v in np.percentile(dt,[0,50,100])])
    calibration=[]
    for segment in np.array_split(np.arange(2,98),32):
        best=None
        for i in segment:
            r,other=train[i],train[i-2]
            z=depth(r)
            ref=masks(z,canvas)[-1]
            projected=render_z(depth(other),k,other["c2w_cv"],k,r["c2w_cv"],z.shape)
            stable=ref & (projected>0) & (np.abs(projected-z)<=np.maximum(.05,.03*z))
            n=int(stable.sum())
            if n and (best is None or n>best[0]): best=(n,r,other,stable)
        if best:
            n,r,other,stable=best
            path=P0/"calibration"/(r["id"]+".npz")
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,static_support=stable)
            calibration.append(dict(id=r["id"],support=n,neighbour=other["id"],mask=member(path)))
    diag=sorted({r["id"] for r in samples}|set(HARD)|{train[i]["id"] for i in np.linspace(0,len(train)-1,32).astype(int)})
    pairs={}
    for fid in diag:
        r=lookup[fid]
        if r["split"]!="train": continue
        i=train.index(r)
        neighbours=[]
        for direction in (-1,1):
            eligible=[]
            for offset in range(2,11):
                j=i+direction*offset
                if not 0<=j<len(train): continue
                s=train[j]
                a,b=np.array(r["c2w_cv"]),np.array(s["c2w_cv"])
                angle=np.degrees(np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))
                baseline=float(np.linalg.norm(a[:3,3]-b[:3,3]))
                if .025<=baseline<=.80 and angle<25 and abs(s["rgb_timestamp_ns"]-r["rgb_timestamp_ns"])<=3e9:
                    eligible.append((abs(baseline-.15),s["id"]))
            if eligible: neighbours.append(min(eligible)[1])
        pairs[fid]=neighbours
    save(P0/"selection.json",dict(samples=samples,sequences=lists,calibration=calibration,
         calibration_rule="Best visible non-edge cross-view sensor agreement in each consecutive three-frame group near beginning of short stream.",
         pairs=pairs,known_difficult={fid:lookup[fid]["split"] for fid in HARD},
         video_train_ids=[r["id"] for r in train[:256]]+[r["id"] for r in train if "frame_01288"<=r["id"]<="frame_01315" or "frame_01508"<=r["id"]<="frame_01520"],
         map_canvas=dict(original_wh=[1280,720],resized_wh=[512,288],output_wh=[512,288],
                         scale_xy=[.4,.4],crop_left_top=[0,0],integer_center="dst=(src+0.5)*0.4-0.5",
                         sensor_sampling="nearest original pixel centre; no interpolation across holes")))
    save(P0/"freeze.json",dict(protocol_sha256=sha(PROTOCOL),
         files=[member(p) for p in sorted(P0.rglob("*")) if p.is_file() and p.name!="freeze.json"]))
    print("Frozen",len(frames),"primary frames,",len(samples),"samples,",len(calibration),"calibration frames",flush=True)


if __name__=="__main__":
    main()
