"""Known-rope state revision from one masked RGB-D view, followed by official spring-mass dynamics."""

import argparse
import ast
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from interactive_push_assets import CASES,file_hash,write_json


# Independent positive RGB scribbles on frame 0 before the train/test split, on half-size previews.
SEEDS={
 "single_push_rope_1":[[(177,127),(202,126),(221,127),(263,127),(286,121),(308,122),(319,122)],
                       [(197,98),(201,105),(209,118),(217,130),(230,145)],
                       [(170,151),(181,145),(191,139),(200,133),(221,119),(243,103)]],
 "single_push_rope_4":[[(170,132),(193,130),(213,128),(255,129),(277,130),(305,130)],
                       [(202,104),(212,116),(225,130),(237,147)],
                       [(176,150),(190,142),(207,130),(225,117),(241,107)]]}
KERNEL_SHA="7deab9a25f4b8b8772f7df45c35571caf3767d014dd353cad151fe8eddceca1c"


def read(path):
    return json.loads(path.read_text())


def spec(root,case):
    d=root/"assets/rope/data/different_types"/case
    out=root/"phystwin"/case
    out.mkdir(parents=True,exist_ok=True)
    split=read(d/"split.json");m=read(d/"metadata.json")
    n,cut=CASES[case]
    if split["train"]!=[0,cut] or split["test"]!=[cut,n] or m["frame_num"]!=n:
        raise ValueError("Case identity or partition mismatch")
    with (d/"calibrate.pkl").open("rb") as f:
        cams=np.array(pickle.load(f))
    if cams.shape!=(3,4,4) or not np.allclose(np.linalg.det(cams[:,:3,:3]),1,atol=.0001):
        raise ValueError("Invalid RGB-D calibration")
    return d,out,cut,np.array(m["intrinsics"]),cams


def controls(root,case):
    d,out,cut,k,t=spec(root,case)
    # This separate extraction process releases only the recorded controller trajectory.
    with (d/"track_process_data.pkl").open("rb") as f:
        bundle=pickle.load(f)
        cp=np.asarray(bundle["controller_points"],np.float32).copy()
    if cp.ndim!=3 or cp.shape[-1]!=3 or not np.isfinite(cp).all():
        raise ValueError("Missing or nonfinite recorded control trajectory")
    np.save(out/"recorded_controls.npy",cp)
    write_json(out/"case_audit.json",{"case":case,"frames":len(cp),"train":[0,cut],"test":[cut,len(cp)],
        "cameras":len(t),"camera_K":k.tolist(),"extrinsics_sha256":file_hash(d/"calibrate.pkl")[0],
        "recorded_controls":list(cp.shape),"controller_trajectory_sha256":file_hash(out/"recorded_controls.npy")[0],
        "controller_condition":"recorded waypoints selected by official all-frame visibility, conditioned offline",
        "asset_identity":"thick brown twine" if case.endswith("_1") else "thin blue rope",
        "same_object_assumed":False,"object_fields_exported_from_official_trajectory_bundle":[]})


def image(d,c,i):
    rgb=cv2.imread(str(d/f"color/{c}/{i}.png"),cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape[:2]!=(480,848):raise ValueError("Missing RGB")
    return rgb


def depth(d,c,i):
    z=np.load(d/f"depth/{c}/{i}.npy",allow_pickle=False).astype(np.float32)/1000
    return np.where(np.isfinite(z)&(z>=.2)&(z<=1.5),z,0.)


def points(z,mask,k,t):
    y,x=np.nonzero((z>0)&np.isfinite(z)&mask)
    a=z[y,x]
    cam=np.column_stack(((x-k[0,2])*a/k[0,0],(y-k[1,2])*a/k[1,1],a))
    return cam@t[:3,:3].T+t[:3,3]


def project(nodes,k,t):
    cam=(nodes-t[:3,3])@t[:3,:3]
    p=cam@k.T
    return np.rint(p[:,:2]/p[:,2:]).astype(np.int32)


def rgb_model(d,case,c):
    seed=2*np.int32(SEEDS[case][c])
    im=image(d,c,0)
    colors=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(float)
    patches=[colors[y-1:y+2,x-1:x+2].reshape(-1,3) for x,y in seed]
    samples=np.concatenate(patches)
    center=np.median(samples,axis=0)
    scale=np.maximum(np.median(np.abs(samples-center),axis=0)*1.4826,[8.,3.,3.])
    return center,scale


def rgb_mask(im,prior,model,radius):
    coords=np.zeros(im.shape[:2],np.uint8)
    for p in prior:
        cv2.circle(coords,tuple(p),radius,1,-1)
    x,y,w,h=cv2.boundingRect(coords)
    result=np.zeros(coords.shape,bool)
    if not w or not h:return result
    colors=cv2.cvtColor(im[y:y+h,x:x+w],cv2.COLOR_BGR2LAB).astype(float)
    cost=np.sum(((colors-model[0])/model[1])**2,axis=-1)
    result[y:y+h,x:x+w]=(coords[y:y+h,x:x+w]>0)&(cost<12.)
    return result


def thin(cloud,size=.005):
    if len(cloud)<10:raise RuntimeError("Too few measured material pixels")
    keys=np.floor(cloud/size).astype(int)
    _,inv=np.unique(keys,axis=0,return_inverse=True)
    n=np.bincount(inv)
    pts=np.column_stack([np.bincount(inv,weights=cloud[:,i])/n for i in range(3)])
    dist=cKDTree(pts).query(pts,k=min(5,len(pts)))[0][:,-1]
    return pts[dist<.020]


def correct(nodes,cloud,neigh,iters=5):
    if len(cloud)<5:return nodes.copy()
    original=nodes.copy();x=nodes.copy();tree=cKDTree(cloud)
    for _ in range(iters):
        dist,idx=tree.query(x)
        delta=cloud[idx]-x
        delta*=np.clip((.042-dist)/.032,0,1)[:,None]
        delta=.40*delta+.60*np.mean(delta[neigh],axis=1)
        x+=delta*.48
        offsets=x-original
        x+=.18*(offsets[neigh].mean(1)-offsets)
    return x


def prepare(root,case):
    d,out,cut,k,t=spec(root,case)
    models=[rgb_model(d,case,c) for c in range(3)]
    coords=[2*np.int32(SEEDS[case][c]) for c in range(3)]
    initcloud=[]
    for c in range(3):
        im=image(d,c,0)
        p=np.zeros((480,848),np.uint8)
        cv2.polylines(p,[coords[c]],False,1,15)
        mask=rgb_mask(im,coords[c],models[c],44)&p.astype(bool)
        initcloud.append(points(depth(d,c,0),mask,k[c],t[c]))
    nodes=thin(np.vstack(initcloud),.006)
    if len(nodes)<15:raise RuntimeError("No common measured rope topology from raw prefix frame zero")
    neigh=cKDTree(nodes).query(nodes,k=min(8,len(nodes)))[1]
    poses=[nodes];cloud_counts=[[len(x) for x in initcloud]]
    for i in range(1,cut):
        pcl=[]
        for c in range(3):
            pr=project(poses[-1],k[c],t[c])
            mask=rgb_mask(image(d,c,i),pr,models[c],19)
            pcl.append(points(depth(d,c,i),mask,k[c],t[c]))
        cloud_counts.append([len(x) for x in pcl])
        poses.append(correct(poses[-1],thin(np.vstack(pcl)),neigh))
    poses=np.array(poses)
    cp=np.load(out/"recorded_controls.npy",allow_pickle=False)
    obj_edges=[]
    tree=cKDTree(nodes)
    for i,p in enumerate(nodes):
        ids=np.atleast_1d(tree.query(p,k=min(30,len(nodes)),distance_upper_bound=.020)[1])
        for j in ids:
            if i<j<len(nodes):obj_edges.append([i,j])
    control_edges=[]
    for i,p in enumerate(cp[0]):
        ids=np.atleast_1d(tree.query(p,k=min(50,len(nodes)),distance_upper_bound=.040)[1])
        for j in ids:
            if j<len(nodes):control_edges.append([len(nodes)+i,j])
    if not obj_edges or not control_edges:raise RuntimeError("Measured prefix does not support object/controller spring attachments")
    edges=np.array(obj_edges+control_edges);all_xyz=np.vstack((nodes,cp[0]))
    lengths=np.linalg.norm(all_xyz[edges[:,0]]-all_xyz[edges[:,1]],axis=1)
    good=lengths>1e-4
    np.savez_compressed(out/"prefix.npz",poses=poses,neigh=neigh,edges=edges[good],lengths=lengths[good])
    x0=poses[-1]+(poses[-1]-poses[-2])
    pr=project(x0,k[0],t[0]);im=image(d,0,cut)
    foreground=rgb_mask(im,pr,models[0],22)
    z=depth(d,0,cut)
    # Alternating contiguous columns fixed before reading hidden values; only the single causal camera enters state revision.
    y,x=np.indices(z.shape)
    query=(np.mod(x-pr[:,0].min(),48)<22)&foreground
    visible=np.where(query,0.,z).astype("f4")
    np.savez_compressed(out/"camera0_input.npz",rgb=im[:,:,::-1],visible_z_m=visible,K_cv=k[0].astype("f4"),
                        query_mask=query,foreground=foreground)
    nn_indices=distance_transform_edt(~((visible>0)&foreground),return_distances=False,return_indices=True)
    nn=np.zeros_like(z);nn[query]=visible[tuple(nn_indices)][query]
    np.save(out/"G-NN.npy",nn)
    write_json(out/"prefix.json",{"prefix":[0,cut],"raw_rgbd_only_for_object_prior":True,"manual_color_seeds_frame0":SEEDS[case],
        "official_future_filtered_object_trajectories_used":False,"official_pretrained_physics_used":False,
        "test_observation":["camera0 RGB", "camera0 structurally masked optical Z"],"test_frame":cut,
        "nodes":len(nodes),"object_springs":len(obj_edges),"control_springs":len(control_edges),"all_valid_springs":int(good.sum()),
        "prefix_measured_material_counts":cloud_counts,"initial_query_pixels":int(query.sum()),"test_mask_from_rgb_and_prefix_only":True,
        "visible_camera0_object_depths":int(np.count_nonzero((visible>0)&foreground)),
        "input_sha256":file_hash(out/"camera0_input.npz")[0],"prior_sha256":file_hash(out/"prefix.npz")[0]})


def infer(root,case,weights):
    import torch
    from lingbot_depth_backend import configure
    from mdm.model.v2 import MDMModel
    configure()
    if file_hash(weights)[0]!="b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23":
        raise ValueError("LingBot checkpoint identity mismatch")
    out=root/"phystwin"/case
    with np.load(out/"camera0_input.npz") as o:
        im=np.ascontiguousarray(o["rgb"]);z=o["visible_z_m"];k=o["K_cv"].copy();query=o["query_mask"]
    k[0]/=im.shape[1];k[1]/=im.shape[0]
    model=MDMModel.from_pretrained(weights).to("cuda").eval()
    with torch.inference_mode():
        result=model.infer(torch.from_numpy(im).permute(2,0,1)[None].cuda().float()/255,
            depth_in=torch.from_numpy(z)[None].cuda(),intrinsics=torch.from_numpy(k)[None].cuda(),resolution_level=9,apply_mask=True)
    pred=result["depth"].detach().cpu().numpy().reshape(z.shape)
    np.save(out/"G-LingBot.npy",np.where(query&np.isfinite(pred)&(pred>0),pred,0.).astype("f4"))


def kernels(root):
    import warp as wp
    path=root/"assets/official-kernel/spring_mass_warp.py"
    if file_hash(path)[0]!=KERNEL_SHA:raise ValueError("Unexpected official PhysTwin Warp kernel")
    source=path.read_text()
    names=["set_control_points","eval_springs","update_vel_from_force","integrate_ground_collision"]
    sections=[]
    lines=source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            sections.append("".join(lines[node.decorator_list[0].lineno-1:node.end_lineno]))
    if len(sections)!=4:raise ValueError("Pinned PhysTwin source definitions missing")
    target=root/"phystwin"/"prefix_warp_kernels.py"
    target.write_text("import warp as wp\n\n"+"\n\n".join(sections))
    wp.config.kernel_cache_dir=str(root/"phystwin/warp_cache")
    wp.init();wp.set_device("cpu")
    mod=importlib.util.spec_from_file_location("prefix_warp_kernels",target)
    k=importlib.util.module_from_spec(mod);mod.loader.exec_module(k)
    return wp,k


def simulate(wp,k,x,v,ctrl,edges,lengths,spring):
    def arr(a,typ):return wp.array(np.ascontiguousarray(a),dtype=typ,device="cpu")
    n=len(x);ps=arr(x,wp.vec3);vs=arr(v,wp.vec3)
    cs=arr(ctrl[0],wp.vec3);cv=wp.zeros_like(cs)
    z=wp.zeros_like(ps);vel=wp.zeros_like(ps);nextx=wp.zeros_like(ps);nextv=wp.zeros_like(ps)
    mass=arr(np.ones(n,np.float32),wp.float32);we=arr(edges.astype("i4"),wp.vec2i);wl=arr(lengths,wp.float32)
    stiff=arr(np.full(len(edges),np.log(spring)),wp.float32);elas=arr([.5],wp.float32);fric=arr([.3],wp.float32)
    output=[x.copy()]
    for j in range(len(ctrl)-1):
        a=arr(ctrl[j],wp.vec3);b=arr(ctrl[j+1],wp.vec3)
        for i in range(667):
            z.zero_()
            wp.launch(k.set_control_points,dim=len(ctrl[0]),inputs=[667,a,b,i,cs])
            wp.launch(k.eval_springs,dim=len(edges),inputs=[ps,vs,cs,cv,n,we,wl,stiff,100.,0.,1.e5,z])
            wp.launch(k.update_vel_from_force,dim=n,inputs=[vs,z,mass,.00005,3.,-1.,vel])
            wp.launch(k.integrate_ground_collision,dim=n,inputs=[ps,vel,elas,fric,.00005,-1.,nextx,nextv])
            ps,nextx=nextx,ps;vs,nextv=nextv,vs
        output.append(ps.numpy().copy())
    return np.array(output)


def predict(root,case,variant):
    d,out,cut,k,t=spec(root,case)
    with np.load(out/"prefix.npz") as a:
        poses=a["poses"];neigh=a["neigh"];edges=a["edges"];lengths=a["lengths"]
    ctrl=np.load(out/"recorded_controls.npy",allow_pickle=False)
    wp,kernels_=kernels(root)
    frozen=out/"physical_prior.json"
    if not frozen.is_file():
        profile=[]
        for stiffness in (15000.,30000.,55000.):
            errors=[]
            for s in (cut//3,2*cut//3):
                sim=simulate(wp,kernels_,poses[s],(poses[s]-poses[s-1])*30,ctrl[s:s+4],edges,lengths,stiffness)
                errors.append(float(np.linalg.norm(sim[-1]-poses[s+3],axis=-1).mean()))
            profile.append([stiffness,errors])
        if not np.isfinite(profile[0][1]).all():raise RuntimeError("Nonfinite common prefix dynamics")
        chosen=min(profile,key=lambda v:np.mean(v[1]))[0]
        write_json(frozen,{"chosen_spring":chosen,"prefix_fit_profile":profile,"source":"four upstream Warp kernels, single material",
            "kernel_sha256":KERNEL_SHA,"device":"CPU","prefix": [0,cut],"official_config":{"dt":.00005,"substeps":667,
            "dashpot":100.,"drag":3.,"ground_elasticity":.5,"ground_friction":.3,"reverse_z":True}})
    spring=read(frozen)["chosen_spring"]
    with np.load(out/"camera0_input.npz") as a:
        vis=a["visible_z_m"];fg=a["foreground"];q=a["query_mask"]
    if variant=="G-Q1":
        with np.load(out/"transfer.npz") as a:fill=a["G_Q1"]
    else:fill=np.load(out/f"{variant}.npy",allow_pickle=False)
    z=vis.copy();selected=q&np.isfinite(fill)&(fill>=.2)&(fill<=1.5);z[selected]=fill[selected]
    cloud=points(z,fg,k[0],t[0]);x0=poses[-1]+(poses[-1]-poses[-2])
    x=correct(x0,cloud,neigh);v=(poses[-1]-poses[-2])*30
    sim=simulate(wp,kernels_,x,v,ctrl[cut:cut+16],edges,lengths,spring)
    if sim.shape!=(16,len(x),3) or not np.isfinite(sim).all():raise ValueError("0.5-second forecast invalid")
    np.save(out/f"{variant}_prediction.npy",sim)
    # Evaluation starts only after the immutable forecast has been exported.
    with (d/"gt_track_3d.pkl").open("rb") as f:gt=np.asarray(pickle.load(f))
    keep=np.isfinite(gt[0]).all(axis=-1); idx=cKDTree(poses[0]).query(gt[0][keep])[1]
    rows=[]
    for j,frame in enumerate(sim):
        i=cut+j;ref=gt[i][keep];valid=np.isfinite(ref).all(axis=1)
        track=np.linalg.norm(frame[idx][valid]-ref[valid],axis=1)
        views={}
        for c in (1,2):
            info=read(d/f"mask/mask_info_{c}.json");oid=next(key for key,val in info.items() if val!="hand")
            mask=cv2.imread(str(d/f"mask/{c}/{oid}/{i}.png"),0)>0
            raw=points(depth(d,c,i),mask,k[c],t[c]);distance=cKDTree(frame).query(raw)[0] if len(raw) else []
            views[str(c)]={"valid_measured_object_points":len(raw),"observed_to_nodes_mean_m":float(np.mean(distance)) if len(raw) else None}
        rows.append({"frame":i,"valid_tracks":len(track),"track_mean_m":float(np.mean(track)) if len(track) else None,"views":views})
    write_json(out/f"{variant}_metrics.json",{"case":case,"variant":variant,"label":"known object correction and open-loop response conditional on recorded controls",
                "forecast_timestamps":16,"horizon_s":.5,"revision_count":1,"same_solver_iterations":5,
                "accepted_hidden_object_pixels":int((selected&fg).sum()),"object_pixels_after_revision":len(cloud),"track_and_external_geometry":rows,
                "shared_prior_sha256":file_hash(frozen)[0],"prediction_sha256":file_hash(out/f"{variant}_prediction.npy")[0]})
    if variant=="G-Q1":
        from interactive_push_audit import rope
        write_json(root/"phystwin/results_summary.json",rope(root))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["controls","prepare","infer","predict"])
    p.add_argument("root",type=Path);p.add_argument("case",choices=CASES);p.add_argument("--weights",type=Path)
    p.add_argument("--variant",choices=["G-NN","G-LingBot","G-Q1"]);a=p.parse_args();cv2.setNumThreads(1)
    if a.mode=="predict":predict(a.root,a.case,a.variant)
    elif a.mode=="infer":infer(a.root,a.case,a.weights)
    elif a.mode=="controls":controls(a.root,a.case)
    else:prepare(a.root,a.case)
