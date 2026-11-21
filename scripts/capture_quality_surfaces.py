"""Raycast the three B2 surfaces and frozen B1 at their own poses on paired sensor pixels."""
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d

from capture_metric3d_common import B0_REPORT, selection
from capture_metric3d_surface_eval import rays, stats, measures
from capture_quality_paths import B1, RUN, SCENE, check, read, record, write
from capture_s01_mesh import titled
from vr_room_run import CPU


def scene(path):
    mesh=o3d.io.read_triangle_mesh(str(path))
    if not len(mesh.triangles): return None
    raycast=o3d.t.geometry.RaycastingScene(nthreads=CPU)
    raycast.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return raycast


def main(split):
    root=RUN/"meshes"; ev=RUN/"evaluation"; ev.mkdir(exist_ok=True)
    paths=dict(sensor_tsdf=root/"sensor_tsdf.ply",model_tsdf=root/"model_tsdf.ply",
               supported_poisson=root/"supported/supported.ply",B1_supported=B1/"supported/B1/supported.ply")
    scenes={key:scene(path) for key,path in paths.items()}
    meta=read(SCENE/"transforms.json"); old=read(B1/"inputs/transforms.json")
    train={f["file_path"]:f for f in meta["frames"]}
    oldframes={f["file_path"]:f for f in old["frames"]}
    time={r["id"]:r["relative_s"] for r in selection()}
    flip=np.diag([1.,-1.,-1.,1.])
    intr=[meta[k] for k in ("fl_x","fl_y","cx","cy")]
    sums={}; errors={}; rows=[]
    def add(group,key,ref,z,valid):
        counts,err=stats(ref,z,valid)
        entry=sums.setdefault((group,key),{k:0 for k in counts})
        for k,v in counts.items(): entry[k]+=v
        errors.setdefault((group,key),[]).append(err)
        return counts,err
    names=meta[split+"_filenames"]
    check(len(names)==184 and names==old[split+"_filenames"],"paired mesh evaluation")
    for i,name in enumerate(names):
        frame=train[name]; stem=Path(name).stem
        ref=np.load(SCENE/frame["depth_file_path"])
        y,x=np.where(np.isfinite(ref)&(ref>=.3)&(ref<=6))
        ref=ref[y,x]
        query=rays(x,y,np.asarray(frame["transform_matrix"])@flip,intr) if len(x) else None
        oldq=rays(x,y,np.asarray(oldframes[name]["transform_matrix"])@flip,intr) if len(x) else None
        zall={k:(s.cast_rays(oldq if k=="B1_supported" else query,nthreads=CPU)["t_hit"].numpy()
                 if s is not None and query is not None else np.full(len(x),np.inf)) for k,s in scenes.items()}
        valid={k:np.isfinite(z)&(z>0) for k,z in zall.items()}
        common=np.logical_and.reduce(list(valid.values()))
        row=dict(id=stem,reference_pixels=len(x),relative_s=time[stem],surfaces={})
        for key,z in zall.items():
            counts,err=add("all",key,ref,z,valid[key])
            row["surfaces"][key]=measures(counts,err)
            add(f"time_{int(time[stem]//30)*30}",key,ref,z,valid[key])
            for a,b in ((.3,1),(1,2),(2,4),(4,6)):
                band=(ref>=a)&((ref<=b) if b==6 else (ref<b))
                add(f"range_{a}_{b}",key,ref[band],z[band],valid[key][band])
            add("common_all_four",key,ref,z,common)
            if key!="B1_supported":
                pair=valid[key]&valid["B1_supported"]
                add("common_B1_"+key,key,ref,z,pair)
                add("common_B1_"+key,"B1_supported",ref,zall["B1_supported"],pair)
        rows.append(row)
        if (i+1)%32==0: print(split,i+1,flush=True)
    groups={}
    for (group,k),totals in sums.items():
        groups.setdefault(group,{})[k]=measures(totals,np.concatenate(errors.pop((group,k))))
    write(ev/f"surface_frames_{split}.json",rows)
    if split=="val":
        options=groups["all"]
        candidates=["model_tsdf","supported_poisson"]
        eligible=[k for k in candidates if options[k]["hits_fraction_all_reference"]>=.9]
        def key(k):
            m=options[k]
            return (-m["correct_10_fraction_all_reference"],m["foreground_fraction_all_reference"],
                    m["MAE_hit_m"] if m["MAE_hit_m"] is not None else np.inf)
        chosen=min(eligible or candidates,key=key)
        write(ev/"export_selection.json",dict(selected=chosen,coverage_gate_passed=bool(eligible),eligible=eligible))
        previews=RUN/"previews"; previews.mkdir(exist_ok=True)
        for rec in read(B0_REPORT/"mesh_validation.json")["local_views"]:
            name=f"images/frame_{rec['train_id']:05d}.png"
            x0,y0,x1,y1=rec["xyxy"]; yy,xx=np.mgrid[y0:y1,x0:x1]
            rgb=cv2.imread(str(SCENE/name))[y0:y1,x0:x1]
            tiles=[titled(rgb.copy(),"Reference")]
            for k,s in scenes.items():
                frame=(oldframes if k=="B1_supported" else train)[name]
                q=rays(xx,yy,np.asarray(frame["transform_matrix"])@flip,intr)
                img=np.full_like(rgb,245)
                if s is not None:
                    hit=s.cast_rays(q,nthreads=CPU)
                    keep=np.isfinite(hit["t_hit"].numpy())
                    normals=hit["primitive_normals"].numpy()[keep]
                    shade=.24+.76*np.abs(normals@np.array([-.4,-.6,-.69])).clip(0,1)
                    img[keep]=(shade[:,None]*[204,212,219]).astype("u1")
                tiles.append(titled(img,k))
            cv2.imwrite(str(previews/f"surface_{rec['object']}.jpg"),np.hstack(tiles))
    write(ev/f"surfaces_{split}.json",dict(split=split,frames=len(rows),meshes={k:record(p) for k,p in paths.items()},
          groups=groups,reference="All paired RGB-camera sensor Z in [0.3,6] m; B1 uses its own original camera."))


if __name__=="__main__":
    check(sys.argv[1] in ("val","test"),"split")
    main(sys.argv[1])
