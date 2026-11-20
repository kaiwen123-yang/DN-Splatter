"""Train-only supported cloud, world-grid fusion, Poisson and face-center voting."""
import argparse
import os
from pathlib import Path

import numpy as np

from capture_metric3d_common import (BASE, P, RUN, SCENE, demand, member, model_paths, read, sha, write)

SPEC = P["supported_mesh"]
BUCKETS = 64
SAMPLE = np.dtype([("key","<i4",3), ("xyz","<f8",3), ("rgb","<f4",3),
                   ("normal","<f4",3), ("view","<u2"), ("pixel","<i4")])


def empty_ply(path, mesh=False):
    header = "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
    if mesh:
        header += "element face 0\nproperty list uchar int vertex_indices\n"
    path.write_text(header + "end_header\n")


def sample(label, out):
    run, dataset = model_paths(label)
    # Set these before importing the historical evaluation modules. All output stays under B1.
    os.environ.update(DN_RUN_ROOT=str(run), DN_SCENE_DIR=str(dataset))
    import torch
    from capture_s01_eval import final_setup
    _, pipe, ck, _ = final_setup("test")
    ds = pipe.datamanager.train_dataset
    meta = read(dataset / "transforms.json")
    names = [p.relative_to(dataset).as_posix() for p in ds.image_filenames]
    demand(names == meta["train_filenames"] and len(names)==1478, "training source ordering")
    h, w = meta["h"], meta["w"]
    frame_dir = out/"views"
    frame_dir.mkdir(exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(42))
    rows = []
    ck_hash = sha(ck)
    with torch.no_grad():
        for i,name in enumerate(names):
            stem = Path(name).stem
            priority = rng.random(h*w)
            rec = frame_dir/(stem+".json")
            target = frame_dir/(stem+".npz")
            if rec.is_file() and target.is_file():
                row = read(rec)
                demand(row["checkpoint_sha256"]==ck_hash and row["sha256"]==sha(target), stem)
                rows.append(row)
                continue
            cam = ds.cameras[i:i+1]
            pred = pipe.model.get_outputs_for_camera(camera=cam.to(pipe.device))
            def array(k, channels):
                return pred[k].detach().cpu().numpy().reshape(-1, channels)
            z = array("depth",1)[:,0]
            a = array("accumulation",1)[:,0]
            n = 2*array("normal",3)-1
            length = np.linalg.norm(n, axis=1)
            ref = np.load(dataset/"depth"/(stem+".npy"), mmap_mode="r").reshape(-1)
            valid = np.ones(h*w,dtype=bool)
            removed = {}
            for key, keep in (("numeric",np.isfinite(z)&np.isfinite(a)&np.isfinite(n).all(axis=1)&(length>0)),
                    ("alpha",a>=SPEC["alpha_min_inclusive"]), ("prediction_range",(z>=.3)&(z<=6.)),
                    ("sensor_range",np.isfinite(ref)&(ref>=.3)&(ref<=6.)),
                    ("agreement",np.abs(z-ref)<=np.maximum(.05,.03*ref))):
                removed[key] = int((valid & ~keep).sum())
                valid &= keep
            ids = np.flatnonzero(valid)
            candidates = len(ids)
            cap = SPEC["max_samples_per_view"]
            if len(ids)>cap:
                ids = ids[np.argpartition(priority[ids],cap-1)[:cap]]
            ids.sort()
            yy,xx = np.divmod(ids,w)
            cam_GL = cam.camera_to_worlds[0].cpu().numpy().astype(np.float64)
            Rcv = cam_GL[:3,:3] @ np.diag([1.,-1.,-1.])
            local = np.column_stack(((xx+.5-meta["cx"])*z[ids]/meta["fl_x"],
                                     (yy+.5-meta["cy"])*z[ids]/meta["fl_y"],z[ids]))
            xyz = local @ Rcv.T + cam_GL[:3,3]
            normal = (n[ids]/length[ids,None]) @ cam_GL[:3,:3].T
            rgb = array("rgb",3)[ids]
            demand(np.isfinite(xyz).all() and np.isfinite(rgb).all(), stem)
            np.savez(target, xyz=xyz, normal=normal.astype(np.float32), rgb=rgb, pixel=ids.astype(np.int32))
            row = dict(id=stem,view=i,pixels=h*w,candidates=candidates,selected=len(ids),removed_sequential=removed,
                       checkpoint_sha256=ck_hash,sha256=sha(target))
            write(rec,row)
            rows.append(row)
    write(out/"sample.json",dict(views=rows, selected=sum(r["selected"] for r in rows),
          rng=f"numpy {np.__version__} PCG64 seed42, all H*W uniforms every view", checkpoint_sha256=ck_hash))


def fuse(label,out):
    import open3d as o3d
    rows = read(out/"sample.json")["views"]
    buckets = out/"fusion_work"
    buckets.mkdir(exist_ok=True)
    # Rebuild these derived buckets on a retry, retaining all original per-frame samples.
    from contextlib import ExitStack
    with ExitStack() as stack:
        handles = [stack.enter_context((buckets/f"bucket_{k:02d}.bin").open("wb")) for k in range(BUCKETS)]
        for row in rows:
            with np.load(out/"views"/(row["id"]+".npz")) as sample:
                a = np.empty(len(sample["pixel"]),dtype=SAMPLE)
                a["xyz"],a["normal"],a["rgb"],a["pixel"],a["view"] = (sample["xyz"],sample["normal"],sample["rgb"],sample["pixel"],row["view"])
            a["key"] = np.floor(a["xyz"] / SPEC["voxel_size_m"]).astype(np.int32)
            k = a["key"].astype(np.int64)
            bucket = ((k[:,0]*73856093) ^ (k[:,1]*19349663) ^ (k[:,2]*83492791)) % BUCKETS
            perm = np.argsort(bucket,kind="stable")
            bounds = np.searchsorted(bucket[perm],np.arange(BUCKETS+1))
            sorted_a = a[perm]
            for i in np.flatnonzero(np.diff(bounds)):
                sorted_a[bounds[i]:bounds[i+1]].tofile(handles[i])
    total, invalid = 0,0
    chunks = []
    for i in range(BUCKETS):
        a = np.fromfile(buckets/f"bucket_{i:02d}.bin",dtype=SAMPLE)
        if not len(a):
            continue
        keys,inv,counts = np.unique(a["key"],axis=0,return_inverse=True,return_counts=True)
        inv = inv.reshape(-1)
        def mean(field):
            return np.column_stack([np.bincount(inv,weights=a[field][:,j])/counts for j in range(3)])
        xyz, rgb, norm = mean("xyz"), mean("rgb"), mean("normal")
        lengths = np.linalg.norm(norm,axis=1)
        good = np.isfinite(lengths)&(lengths>0)
        invalid += int((~good).sum())
        ids = np.full(len(keys),-1,np.int32)
        ids[good] = np.arange(total,total+int(good.sum()),dtype=np.int32)
        np.savez(buckets/f"members_{i:02d}.npz",voxel=ids[inv],source_view=a["view"],source_pixel=a["pixel"])
        part = buckets/f"fused_{i:02d}.npz"
        np.savez(part,xyz=xyz[good],rgb=rgb[good],normal=norm[good]/lengths[good,None],key=keys[good],count=counts[good])
        chunks.append((part,int(good.sum())))
        total += int(good.sum())
    path = out/"fused.ply"
    if total:
        cloud = o3d.geometry.PointCloud()
        for field, attr in (("xyz","points"),("rgb","colors"),("normal","normals")):
            array = np.empty((total,3),np.float64)
            start=0
            for part,n in chunks:
                with np.load(part) as arr:
                    array[start:start+n] = arr[field]
                start+=n
            setattr(cloud,attr,o3d.utility.Vector3dVector(array))
            del array
        demand(o3d.io.write_point_cloud(str(path),cloud),"write fused cloud")
    else:
        empty_ply(path)
    write(out/"fuse.json",dict(samples=sum(r["selected"] for r in rows), fused=total, invalid_voxel_normals=invalid,
          grid_m=.01, origin=[0,0,0], cloud=member(path), provenance="fusion_work/members_*.npz maps every selected source frame/pixel to fused PLY vertex index; -1 if invalid"))


def poisson(label,out):
    import open3d as o3d
    info = read(out/"fuse.json")
    path = out/"before_crop.ply"
    threads = int(os.environ["DN_CPU_BUDGET"])
    before_density=0
    if info["fused"]:
        cloud=o3d.io.read_point_cloud(str(out/"fused.ply"))
        mesh,dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud,depth=11,scale=1.1,n_threads=threads)
        density=np.asarray(dens)
        before_density=len(mesh.vertices)
        if len(density):
            mesh.remove_vertices_by_mask(density<np.quantile(density,.01))
        vertices,faces=len(mesh.vertices),len(mesh.triangles)
        cell=1.1*np.ptp(np.asarray(cloud.points),axis=0).max()/2048
        if faces:
            demand(o3d.io.write_triangle_mesh(str(path),mesh),"write pre-crop mesh")
        else:
            empty_ply(path,True)
    else:
        vertices,faces,cell=0,0,None
        empty_ply(path,True)
    write(out/"poisson.json",dict(depth=11,scale=1.1,threads=threads, density_trim=.01,
          vertices_before_density=before_density,vertices=vertices,faces=faces,nominal_cell_m=cell, mesh=member(path)))


def crop(label,out):
    import open3d as o3d
    import torch
    _, dataset = model_paths(label)
    meta=read(dataset/"transforms.json")
    source={f["file_path"]:f for f in meta["frames"]}
    faces=read(out/"poisson.json")["faces"]
    s=np.zeros(faces,np.int16)
    c=np.zeros(faces,np.int16)
    path=out/"supported.ply"
    if faces:
        mesh=o3d.io.read_triangle_mesh(str(out/"before_crop.ply"))
        v=np.asarray(mesh.vertices)
        tri=np.asarray(mesh.triangles)
        xyz=torch.empty((faces,3),dtype=torch.float64,device="cuda")
        for i in range(0,faces,250000):
            ids=tri[i:i+250000]
            xyz[i:i+len(ids)]=torch.as_tensor(v[ids].mean(axis=1),device="cuda")
        votes_s=torch.zeros(faces,device="cuda",dtype=torch.int16)
        votes_c=torch.zeros_like(votes_s)
        for name in meta["train_filenames"]:
            f=source[name]
            cam=np.asarray(f["transform_matrix"])@np.diag([1.,-1.,-1.,1.])
            R=torch.as_tensor(cam[:3,:3],device="cuda")
            t=torch.as_tensor(cam[:3,3],device="cuda")
            depth=torch.from_numpy(np.load(dataset/f["depth_file_path"])).cuda()
            for i in range(0,faces,500000):
                xyz_c=(xyz[i:i+500000]-t)@R
                x,y,z=xyz_c.unbind(-1)
                u,vv=meta["fl_x"]*x/z+meta["cx"],meta["fl_y"]*y/z+meta["cy"]
                eligible=(z>0)&(u>=0)&(u<meta["w"])&(vv>=0)&(vv<meta["h"])
                indices=torch.nonzero(eligible).flatten()
                d=depth[vv[indices].long(),u[indices].long()].double()
                valid=(d>=.3)&(d<=6.)&torch.isfinite(d)
                tz=torch.maximum(.03*d,torch.full_like(d,.05))
                err=z[indices]-d
                votes_s[i+indices]+=((err.abs()<=tz)&valid).to(torch.int16)
                votes_c[i+indices]+=((err < -tz)&valid).to(torch.int16)
        s,c=votes_s.cpu().numpy(),votes_c.cpu().numpy()
        keep=(s>=2)&(s>c)
        mesh.remove_triangles_by_mask(~keep)
        mesh.remove_unreferenced_vertices()
        if keep.any():
            demand(o3d.io.write_triangle_mesh(str(path),mesh),"write supported mesh")
        else:
            empty_ply(path,True)
    else:
        empty_ply(path,True)
    np.savez_compressed(out/"face_votes.npz",surface=s,conflict=c)
    write(out/"crop.json",dict(before=faces, retained=int(((s>=2)&(s>c)).sum()), insufficient=int((s<2).sum()),
          conflict_dominant=int(((s>=2)&(s<=c)).sum()), source_training_views=len(meta["train_filenames"]),
          surface_vote_histogram=np.bincount(s.astype(int)).tolist(),conflict_vote_histogram=np.bincount(c.astype(int)).tolist(),
          mesh=member(path), votes=member(out/"face_votes.npz")))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model",choices=("B0","B1"))
    ap.add_argument("operation",choices=("sample","fuse","poisson","crop"))
    args=ap.parse_args()
    out=RUN/"supported"/args.model
    out.mkdir(parents=True,exist_ok=True)
    {"sample":sample,"fuse":fuse,"poisson":poisson,"crop":crop}[args.operation](args.model,out)
