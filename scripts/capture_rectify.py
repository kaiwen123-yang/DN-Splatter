"""Build pinhole RGB and color-axis Z on two grids from a verified native packet."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from capture_registration_review import project, scaled_camera, zbuffer


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def describe(array):
    a = np.asarray(array)
    if not a.size:
        return None
    return dict(zip(("min","p10","p50","p90","p95","max"),
                    np.percentile(a,[0,10,50,90,95,100]).tolist()))


def sha(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for part in iter(lambda:f.read(2**20),b""):
            h.update(part)
    return h.hexdigest()


def files(root, exclude=()):
    return [dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in exclude]


def verify(root):
    expected = read(root/"manifest.json")["files"]
    if files(root, ["manifest.json"]) != expected:
        raise ValueError("Packet members, SHA256 or sizes differ")


def imwrite(path, image):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image):
        raise IOError(path)


def grid_setup(cal):
    color,dep = [cal["cameras"][key] for key in ("RGB","depth")]
    coeff = "D_sdk_k1_k2_k3_k4_k5_k6_p1_p2"
    assert color["distortion_model_enum"] == 4 and not np.any(dep[coeff])
    d = np.asarray(color[coeff])[[0,1,6,7,2,3,4,5]]
    ext = next(e for e in cal["static_extrinsics"] if (e["source_sensor"],e["destination_sensor"])==("depth","color"))
    v,u = np.indices((dep["height"],dep["width"]),dtype=float)
    rays = np.stack(((u-dep["cx"])/dep["fx"],(v-dep["cy"])/dep["fy"],np.ones_like(u)),-1)
    rays = rays @ np.array(ext["R"]).T
    translation = np.array(ext["translation_mm"])*.001
    grids = {}
    for size, scale in (("1280",1),("640",.5)):
        cam = scaled_camera(color,scale)
        k = [[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1]]
        kc = np.array(k,dtype=float)
        kc[:2,2] += .5
        mx,my = cv2.initUndistortRectifyMap(np.array(color["K"]),d,None,np.array(k),
                                           (cam["width"],cam["height"]),cv2.CV_32FC1)
        valid = (mx>=0)&(mx<=color["width"]-1)&(my>=0)&(my<=color["height"]-1)
        grids[size] = (cam,mx,my,valid,dict(K_cv=k,K_colmap_ns=kc.tolist(),width=cam["width"],height=cam["height"],
                                          D_cv_source=d.tolist(),target_model="PINHOLE",distortion_target=[0]*8))
    return rays,translation,grids


def project_frame(raw,rgb,rays,t,g):
    cam,mx,my,rgb_valid,_ = g
    im = cv2.remap(rgb,mx,my,cv2.INTER_LINEAR)
    im[~rgb_valid] = 0
    good = (raw!=0)&(raw!=65535)
    pc = rays[good]*(raw[good,None]*.001) + t
    uv = project(pc,cam)
    z,inside = zbuffer(uv,pc[:,2],cam)
    xy = np.rint(uv[inside]).astype(int)
    near = z[xy[:,1],xy[:,0]]
    occluded = pc[inside,2]-near > np.maximum(.05,.03*near)
    occ = np.zeros_like(z,dtype=bool)
    occ[xy[occluded,1],xy[occluded,0]] = True
    counts = dict(source_non_sentinel=int(good.sum()),behind=int((pc[:,2]<=0).sum()),
                  out_of_bounds=int(((pc[:,2]>0)&~inside).sum()),in_bounds=int(inside.sum()),
                  additional_hits_at_same_pixel=int(inside.sum()-np.count_nonzero(z)),
                  discarded_by_rgb_remap=int(np.count_nonzero(z[~rgb_valid])))
    z[~rgb_valid] = 0
    return im,z.astype(np.float32),occ & rgb_valid,counts


def edges(z):
    mask = np.zeros(z.shape,bool)
    for a,b in ((np.s_[:,:-1],np.s_[:,1:]),(np.s_[:-1,:],np.s_[1:,:])):
        x,y=z[a],z[b]
        change=(x>0)&(y>0)&(np.abs(x-y)>np.maximum(.05,.03*np.minimum(x,y)))
        mask[a] |= change & (x<y)
        mask[b] |= change & (y<x)
    return mask


def diagnostic(im,z,occ,counts,path):
    edge = edges(z)
    texture = cv2.Canny(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),50,150)>0
    distance = cv2.distanceTransform((~texture).astype("uint8"),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    bands = {}
    for hi in (3,6,10):
        window = (z>=.3)&(z<=hi)
        e = edge & window
        bands[str(hi)] = dict(target_coverage=float(window.mean()),edge_pixels=int(e.sum()),
                               nearest_canny_px=describe(distance[e]))
    disp = im.copy()
    disp[edge&(z>=.3)&(z<=6)] = (45,255,70)
    colored = cv2.applyColorMap(np.uint8(255*np.clip((z-.3)/5.7,0,1)),cv2.COLORMAP_TURBO)
    mask = (z>=.3)&(z<=6)
    overlay=im.copy()
    overlay[mask] = cv2.addWeighted(im,.48,colored,.52,0)[mask]
    imwrite(path,np.hstack((im,disp,overlay)))
    return dict(**counts,target_coverage=float((z>0).mean()),occlusion_pixels=int(occ.sum()),
                z_m=describe(z[z>0]),windows= bands)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("packet",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--all",action="store_true")
    a=p.parse_args()
    verify(a.packet)
    rows=[r for r in read(a.packet/"selection.json") if not r["drop_reason"]]
    retained=set(sum(read(a.packet/"splits.json")["splits"].values(),[]))
    rows=[r for r in rows if r["id"] in retained]
    samples=set(np.linspace(0,len(rows)-1,9).round().astype(int))
    cv2.setNumThreads(1)
    rays,t,grids=grid_setup(read(a.packet/"calibration.json"))
    save(a.out/"grids.json",{k:v[-1] for k,v in grids.items()})
    checks=[]
    done=0
    for i,row in enumerate(rows):
        if not a.all and i not in samples:
            continue
        stem=row["id"]
        rgb=cv2.imread(str(a.packet/row["rgb"]["file"]), cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        with np.load(a.packet/row["depth"]["file"],allow_pickle=False) as archive:
            raw=archive["depth"]
        for size,g in grids.items():
            im,z,occ,counts=project_frame(raw,rgb,rays,t,g)
            root=a.out/size
            imwrite(root/"images"/(stem+".png"),im)
            (root/"depth").mkdir(exist_ok=True)
            np.save(root/"depth"/(stem+".npy"),z,allow_pickle=False)
            imwrite(root/"coverage"/(stem+".png"),np.uint8(z>0)*255)
            imwrite(root/"occlusion"/(stem+".png"),np.uint8(occ)*255)
            if i in samples:
                checks.append(dict(id=stem,relative_s=row["relative_s"],grid=size,
                                   **diagnostic(im,z,occ,counts,a.out/"checks"/(stem+"_"+size+".jpg"))))
        done+=1
    save(a.out/"registration.json",dict(frame_count=done,checked_frames=len(samples),grids=read(a.out/"grids.json"),
        metric="Nearest positive color-axis Z in metres. Raw 0/65535 excluded. Rounded single-pixel projection; holes zero.",
        rgb="Bilinear inverse map with all Brown coefficients, no crop; both grids use the recorded color optical axes.",
        occlusion="Target has a farther competing sample separated by >max(0.05m,3% of visible depth). Coverage is separate.",
        edge="Adjacent valid projected pixels separated by >max(0.05m,3% of nearer Z). On the nearer side within each window, measure distance to nearest grayscale Canny(50,150) edge in pixels of this grid. Texture edges can be unrelated; no independent reference.",
        rows=checks))
    print("rectified",done,"pairs")


if __name__=="__main__":
    main()
