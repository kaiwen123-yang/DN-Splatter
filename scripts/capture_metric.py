"""Fit one train-only metric scale, then validate independent views on the 640 grid."""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from capture_rectify import describe, edges, imwrite, project, read, save, zbuffer


def median(values,weights):
    order=np.argsort(values)
    return float(values[order[np.searchsorted(np.cumsum(weights[order]),weights.sum()*.5)]])


def fit(samples):
    if not len(samples):
        return None
    # Repeated tracks and very textured frames cannot dictate the scale by observation count.
    frame,point,zm,zsfm=samples.T
    _,f=np.unique(frame,return_inverse=True)
    _,p=np.unique(point,return_inverse=True)
    # Inverse track multiplicity within this subset, then equal total weight per frame.
    weight=1/np.bincount(p)[p]
    weight/=np.bincount(f,weights=weight)[f]
    scale=median(zm/zsfm,weight)
    return dict(scale=scale,frames=int(f.max()+1),tracks=int(p.max()+1),observations=len(f),
                weighted_median_abs_residual_m=median(np.abs(zm-scale*zsfm),weight),
                absolute_residual_m=describe(np.abs(zm-scale*zsfm)))


def gather(run,obs):
    samples=[]
    for fid in np.unique(obs[:,0]).astype(int):
        arr=obs[(obs[:,0]==fid)&(obs[:,5]<=2)&(obs[:,6]>=3)&(obs[:,4]>0)]
        # Corner-based 1280 features -> 640 integer-center array coordinates, exactly once.
        x,y=np.rint(arr[:,2:4]*.5-.5).astype(int).T
        z=np.load(run/f"rect/640/depth/frame_{fid:05d}.npy")
        occ=cv2.imread(str(run/f"rect/640/occlusion/frame_{fid:05d}.png"),0)
        valid=(x>=2)&(x<z.shape[1]-2)&(y>=2)&(y<z.shape[0]-2)
        x,y=x[valid],y[valid]; arr=arr[valid]
        dx,dy=np.meshgrid(np.arange(-2,3),np.arange(-2,3))
        xx=x[:,None]+dx.ravel(); yy=y[:,None]+dy.ravel()
        values=z[yy,xx]
        supported=np.sum(values>0,axis=1)>=12
        if not supported.any():
            continue
        values=values[supported]
        nearby_occlusion=occ[yy[supported],xx[supported]].any(axis=1)
        values[values<=0]=np.nan
        depth=np.nanmedian(values,axis=1)
        smooth=(np.nanmax(values,axis=1)-np.nanmin(values,axis=1)<=np.maximum(.05,.03*depth))
        keep=(~nearby_occlusion)&smooth&(depth>=.3)&(depth<=10)
        a=arr[supported][keep]
        samples.append(np.column_stack((a[:,0],a[:,1],depth[keep],a[:,4])))
    return np.concatenate(samples)


def remap_points(z,mask,cam):
    v,u=np.nonzero(mask)
    return np.column_stack(((u-cam["cx"])/cam["fx"],(v-cam["cy"])/cam["fy"],np.ones(len(v))))*z[v,u,None]


def compare(run,ref,dst,poses,cam,out,colors):
    root=run/"rect/640"
    def frame(stem):
        return np.load(root/"depth"/(stem+".npy"))
    zs,zt=frame(ref),frame(dst)
    rgb=cv2.imread(str(root/"images"/(dst+".png")))
    tf=np.linalg.inv(poses[dst])@poses[ref]
    # All observed source depths participate in visibility ordering before range gates.
    points=remap_points(zs,zs>0,cam)
    pc=points@tf[:3,:3].T+tf[:3,3]
    pc=pc[pc[:,2]>0]
    pred,_=zbuffer(project(pc,cam),pc[:,2],cam)
    overlap=(pred>0)&(zt>0)
    diff=pred-zt
    band=cv2.dilate(edges(zt).astype("uint8"),np.ones((5,5),np.uint8)).astype(bool)
    occ=cv2.imread(str(root/"occlusion"/(dst+".png")),0)>0
    # Target foreground may hide the reprojected source. Keep unconditional results too.
    behind=(diff>np.maximum(.05,.03*zt))
    windows={}
    for hi in (3,6,10):
        inside=overlap&(zt>=.3)&(zt<=hi)
        usable=inside&~(band|occ|behind)
        both=inside&(pred>=.3)&(pred<=hi)
        windows[str(hi)]=dict(overlap=int(inside.sum()),outside_edge_and_visible=int(usable.sum()),
          both_depths_in_window=int(both.sum()),both_in_window_abs_m=describe(np.abs(diff[both])),
          projected_beyond_window=int((inside&(pred>hi)).sum()),
          projected_above_20m=int((inside&(pred>20)).sum()),
          farther_than_target=int((inside&behind).sum()),nearer_than_target=int((inside&(diff<-np.maximum(.05,.03*zt))).sum()),
          unconditional_abs_m=describe(np.abs(diff[inside])),visible_abs_m=describe(np.abs(diff[usable])),
          unconditional_signed_m=describe(diff[inside]))
    if colors:
        tag=(ref+"_"+dst)
        mask=overlap&(zt>=.3)&(zt<=6)
        cm=cv2.applyColorMap(np.uint8(255*np.minimum(np.abs(diff)/.20,1)),cv2.COLORMAP_INFERNO)
        view=rgb.copy()
        view[mask]=cv2.addWeighted(rgb,.35,cm,.65,0)[mask]
        cv2.putText(view,tag+" | abs dZ 0..0.20m",(9,26),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
        imwrite(out/(tag+".jpg"),view)
    return dict(source=ref,target=dst,translation_m=float(np.linalg.norm(tf[:3,3])),
                shared_pixels=int(overlap.sum()),target_grid_pixels=int(zt.size),windows=windows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run",type=Path)
    a=ap.parse_args()
    run=a.run; out=run/"metric"; out.mkdir(exist_ok=True)
    pose=read(run/"sfm/pose_quality.json")
    entries=read(run/"transfer/splits.json")["entries"]
    obs=np.load(run/"sfm/observations.npz")["observations"]
    sample=gather(run,obs)
    train={int(e["id"].split("_")[-1]) for e in entries if e["split"]=="train"}
    assert set(sample[:,0].astype(int))<=train
    estimates={str(hi):fit(sample[sample[:,2]<=hi]) for hi in (3,6,10)}
    # Prespecified near/mid indoor objects for scale & initialization; keep all native values.
    chosen=sample[sample[:,2]<=6]
    scale=estimates["6"]["scale"]
    if not np.isfinite(scale) or scale<=0:
        raise ValueError("Cannot recover a positive scale")
    times=np.array_split(np.array([int(e["id"].split("_")[-1]) for e in entries]),3)
    temporal=[fit(chosen[np.isin(chosen[:,0],t)]) for t in times]
    by_distance={f"{lo}-{hi}":fit(sample[(sample[:,2]>lo)&(sample[:,2]<=hi)]) for lo,hi in ((.3,3),(3,6),(6,10)) if np.any((sample[:,2]>lo)&(sample[:,2]<=hi))}
    def fixed(group):
        return describe(group[:,2]-scale*group[:,3])
    save(out/"scale_alignment.json",dict(scale_m_per_sfm_unit=scale,fit_range_m=[.3,6],windows=estimates,
         early_middle_late=temporal,disjoint_depth_bands=by_distance,
         fixed_scale_signed_residual_m_by_time=[fixed(chosen[np.isin(chosen[:,0],t)]) for t in times],
         fixed_scale_signed_residual_m_by_depth={str(hi):fixed(sample[(sample[:,2]>lo)&(sample[:,2]<=hi)]) for lo,hi in ((.3,3),(3,6),(6,10))},
         method="Train map observations only, track length >=3, reprojection <=2px, positive SfM Z. Round half-resolution corner-centred coordinates after subtracting 0.5. Take median of 5x5 Z_C with >=12 hits, no competing occluder, full local Z spread <=max(5cm,3%). Median of Z_C/Z_sfm; inverse track frequency then normalize each frame to equal total weight.",
         fit_and_export_policy="0.3–6 m fixed for primary fit and initialization before inspecting held-out depth. Overlapping/disjoint windows and thirds use training observations only.",
         units="metric sensor consistency; no external geometric reference"))
    poses={}
    for name,mat in pose["poses_w2c_sfm_units"].items():
        if name.startswith("aux_"):
            continue
        w2c=np.eye(4);w2c[:3]=mat
        c2w=np.linalg.inv(w2c)
        c2w[:3,3]*=scale
        poses[Path(name).stem]=c2w
    first=next(e["id"] for e in entries if e["id"] in poses and e["split"]=="train")
    origin=np.linalg.inv(poses[first])
    poses={k:origin@m for k,m in poses.items()}
    save(out/"poses.json",dict(frame="Metric OpenCV c2w in first registered training camera axes; right/down/forward.",
          origin_frame=first,sfm_metric_world_to_local=origin.tolist(),scale=scale,poses={k:m.tolist() for k,m in poses.items()}))
    grid=read(run/"rect/grids.json")["640"];K=grid["K_cv"]
    cam=dict(width=640,height=360,fx=K[0][0],fy=K[1][1],cx=K[0][2],cy=K[1][2])
    queries=[e["id"] for e in entries if e["split"]!="train"]
    train_names=[e["id"] for e in entries if e["split"]=="train" and e["id"] in poses]
    checks=[]
    for qi in np.linspace(0,len(queries)-1,9).round().astype(int) if queries else ():
        dst=queries[qi]; index=int(dst.split("_")[-1])
        if dst not in poses:
            checks.append(dict(target=dst,status="missing pose"));continue
        for offset in (0,(-15 if index>=15 else 15)):
            ref=min(train_names,key=lambda n:abs(int(n.split("_")[-1])-index-offset))
            checks.append(compare(run,ref,dst,poses,cam,out/"pairs",colors=offset!=0))
    # Earlier views revisit the tables after the initial turn. Disclose any low-overlap pair.
    for idx,target in ((15,88),(32,68)):
        ref=f"frame_{idx:05d}"; dst=f"frame_{target:05d}"
        if ref in poses and ref in train_names and dst in poses:
            checks.append(compare(run,ref,dst,poses,cam,out/"pairs",colors=True))
    ordered=[e["id"] for e in entries if e["id"] in poses]
    loc=np.array([poses[i][:3,3] for i in ordered])
    rot=np.array([poses[i][:3,:3] for i in ordered])
    jumps=np.linalg.norm(np.diff(loc,axis=0),axis=1)
    degrees=np.rad2deg(np.arccos(np.clip((np.einsum("nij,nij->n",rot[1:],rot[:-1])-1)/2,-1,1)))
    save(out/"cross_view.json",dict(method="Train source to held-out target 640 grid. Project all source depths, nearest-positive z-buffer. Windows based on measured target Z. Report all overlapping residuals; second table excludes near depth boundaries (2px dilation), collision mask and source behind target by >max(5cm,3%). Unconditional residuals retain disagreements.",
         pairs=checks,pose_count=len(ordered),trajectory_length_m=float(jumps.sum()),adjacent_translation_m=describe(jumps),
         adjacent_rotation_deg=describe(degrees),maximum_translation_pair=ordered[np.argmax(jumps):np.argmax(jumps)+2],
         maximum_rotation_pair=ordered[np.argmax(degrees):np.argmax(degrees)+2],trajectory_extent_xyz_m=np.ptp(loc,axis=0).tolist()))
    fig,ax=plt.subplots(1,2,figsize=(11,5.3),layout="constrained")
    sc=ax[0].scatter(loc[:,0],loc[:,2],c=np.arange(len(loc)),cmap="viridis",s=10)
    ax[0].plot(loc[:,0],loc[:,2],color="0.5",lw=.5,zorder=0)
    fig.colorbar(sc,ax=ax[0],label="Time-ordered frame index",shrink=.75)
    for idx in np.linspace(0,len(train_names)-1,9).round().astype(int):
        stem=train_names[idx]
        z=np.load(run/f"rect/640/depth/{stem}.npy")
        mask=(z>=.3)&(z<=6)
        cv_rgb=cv2.imread(str(run/f"rect/640/images/{stem}.png"))[:,:,::-1]
        p=remap_points(z,mask,cam)[::25]
        color=cv_rgb[mask][::25]/255
        world=p@poses[stem][:3,:3].T+poses[stem][:3,3]
        ax[1].scatter(world[:,0],world[:,2],c=color,s=.12)
    for panel,title in zip(ax,("Camera path","9 training views | RGB-D surfaces, 0.3–6 m")):
        panel.set(aspect="equal",xlabel="First camera X (m)",ylabel="First camera Z (m)",title=title)
    fig.savefig(out/"trajectory_fusion.jpg",dpi=165);plt.close(fig)
    print("Metric scale",scale,"cross-view pairs",len(checks))


if __name__=="__main__":
    main()
