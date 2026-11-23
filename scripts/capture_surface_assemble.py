"""Assemble B3 global surfaces. Clip proven replacements in the original Poisson face plane."""
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import open3d as o3d

from capture_surface_device import Evidence
from capture_surface_layers import children, normal_area, seven
from capture_surface_ply import save_welded
from capture_surface_tiles import PLAN, ROOT, RUN, ply, read, sha, write


def eligible_in_witness(e, pts, views):
    import torch
    p=torch.as_tensor(pts.reshape(-1,3),device="cuda")
    i=torch.as_tensor(np.repeat(views,7).astype("i8"),device="cuda")
    a=e.cam[i]
    q=torch.bmm((p-a[:,:3,3]).unsqueeze(1),a[:,:3,:3]).squeeze(1)
    z=q[:,2]; k=e.k
    u=torch.round(k["fx"]*q[:,0]/z+k["cx"]).long()
    v=torch.round(k["fy"]*q[:,1]/z+k["cy"]).long()
    inside=(u>=0)&(u<e.w)&(v>=0)&(v<e.h)&(z>=.3)&(z<=6)
    u=u.clamp(0,e.w-1);v=v.clamp(0,e.h-1)
    raw=e.depth[i,v,u].long() & 65535
    measured=(raw.float()/1000).double()
    band=(raw>=1000).long()+(raw>=2000).long()+(raw>=4000).long()
    tau=torch.maximum(torch.clamp(.03*measured,min=.05),e.terms[band]).clamp(max=.20)
    good=inside&(raw>=300)&(raw<=6000)&(e.edge[i,v,u]>2)&((z-measured).abs()<=tau)
    return good.cpu().numpy().reshape(-1,7)


def classify(t,g,w,scene,e,level,output):
    """All seven samples must certify a patch. Mixed footprints are subdivided twice."""
    if not len(t):return
    pts=seven(t);seen=eligible_in_witness(e,pts,w)
    if level:
        count,_,_=e.vote(pts.reshape(-1,3),np.repeat(g,7))
        seen&=((count[:,0]>=2)&(count[:,0]>count[:,1])).reshape(-1,7)
    origin=np.broadcast_to(e.centers[w,None],pts.shape)
    vectors=pts-origin
    z=np.einsum("nsi,ni->ns",vectors,e.poses[w,:3,2])
    rays=np.concatenate((origin,vectors),axis=-1).astype("f4")
    hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=8)
    dist=hit["t_hit"].numpy().astype("f8")
    tid=hit["primitive_ids"].numpy().astype("i8")
    hituv=hit["primitive_uvs"].numpy()
    miss=~np.isfinite(dist)
    fill=(seen&miss).all(axis=1)
    if fill.any():output["fill"].append(t[fill])
    old_z=np.where(miss,0.,dist)*z
    # A different layer must differ by the calibrated construction tolerance; evaluation thresholds stay frozen.
    terms=e.terms.cpu().numpy()
    band=np.minimum(np.searchsorted([1.,2.,4.,6.],old_z,side="right"),3)
    tol=np.minimum(.20,np.maximum(np.maximum(.05,.03*old_z),terms[band]))
    distinct=seen&~miss&(np.abs((dist-1)*z)>tol)
    conflict=np.zeros_like(seen)
    rows,sites=np.nonzero(distinct)
    if len(rows):
        old=origin[rows,sites]+vectors[rows,sites]*dist[rows,sites,None]
        votes,_,_=e.vote(old,e.gids[w[rows]])
        conflict[rows,sites]=(votes[:,1]>=2)&(votes[:,1]>votes[:,0])
    replace=conflict.all(axis=1)&(tid==tid[:,:1]).all(axis=1)
    uv=hituv[:,:3].astype("f8")
    d1=uv[:,1]-uv[:,0];d2=uv[:,2]-uv[:,0]
    replace&=np.abs(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])>1e-12
    if replace.any():
        output["replace"].append(t[replace])
        output["ids"].append(tid[replace,0]);output["uv"].append(uv[replace])
    pending=(~fill)&(~replace)&((seen&miss)|conflict).any(axis=1)
    if level<2 and pending.any():
        indices=np.flatnonzero(pending)
        for start in range(0,len(indices),2048):
            i=indices[start:start+2048]
            classify(children(t[i]),np.repeat(g[i],4),np.repeat(w[i],4),scene,e,level+1,output)
    else:
        output["ambiguous_area"]+=float(normal_area(t[pending])[1].sum())


def cut_original(t, polygons):
    from shapely import make_valid
    from shapely.geometry import Polygon
    from shapely.ops import triangulate, unary_union
    import mapbox_earcut

    def polygon_parts(geometry):
        if geometry.is_empty:
            return
        if geometry.geom_type == "Polygon":
            yield geometry
        elif hasattr(geometry, "geoms"):
            for child in geometry.geoms:
                yield from polygon_parts(child)

    def valid_polygonal(geometry):
        if geometry.is_valid:
            return geometry
        # Interpolated cut edges can meet within a few ulps and leave nested shells.
        # make_valid may also return boundary linework, which has no surface area.
        return unary_union(list(polygon_parts(make_valid(geometry))))

    def earcut(piece):
        rings = [np.asarray(ring.coords)[:-1] for ring in (piece.exterior, *piece.interiors)]
        points = np.ascontiguousarray(np.concatenate(rings), dtype="f8")
        ends = np.cumsum([len(ring) for ring in rings]).astype("u4")
        triangles = points[mapbox_earcut.triangulate_float64(points, ends).reshape(-1, 3)]
        return triangles

    def oriented(triangles):
        u = triangles[:, 1] - triangles[:, 0]
        v = triangles[:, 2] - triangles[:, 0]
        cross = u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0]
        back = cross < 0
        triangles[back] = triangles[back][:, [0, 2, 1]]
        return triangles, float(np.abs(cross).sum()/2)

    parent = Polygon([(0., 0.), (1., 0.), (0., 1.)])
    cuts = [Polygon(p).intersection(parent) for p in polygons]
    removed = valid_polygonal(unary_union(cuts))
    rest = valid_polygonal(parent.difference(removed))
    if abs(rest.area + removed.area - parent.area) > 1e-7:
        raise ValueError("Clipped domain area conservation")
    faces = []
    for piece in polygon_parts(rest):
        if piece.area < 1e-14:
            continue
        tri, area = oriented(earcut(piece))
        if abs(area - piece.area) > 1e-7:
            # Each Delaunay cell is intersected with the actual polygon, retaining holes.
            parts = [earcut(part) for cell in triangulate(piece)
                     for part in polygon_parts(cell.intersection(piece)) if part.area >= 1e-14]
            tri, area = oriented(np.concatenate(parts) if parts else np.empty((0, 3, 2)))
            if abs(area - piece.area) > 1e-7:
                raise ValueError("Clipped area conservation")
        faces.extend(t[0] + tri[:, :, 0, None]*(t[1]-t[0]) + tri[:, :, 1, None]*(t[2]-t[0]))
    return np.asarray(faces, dtype="f8").reshape(-1, 3, 3), min(1., max(0., 1-2*rest.area))


def combined(label,branch):
    export=RUN/"export"/label
    layer=export/("layers_"+branch);src=read(layer/"complete.json")
    out=export/("fusion_"+branch);out.mkdir(exist_ok=True)
    path=out/"global.ply"
    base_run=(ROOT/".local/runs"/PLAN["baseline"]["run_id"]) if label=="B2" else RUN
    base=base_run/"meshes/supported/supported.ply"
    ident=dict(layers=sha(layer/"complete.json"),old_sha256=sha(base) if branch=="model" else None,
               implementation=[sha(__file__),sha(ROOT/"scripts/capture_surface_ply.py")])
    signature=out/"assembly_identity.json"
    if signature.is_file() and read(signature) != ident:
        previous = read(signature)
        # The original classifier and PLY writer agree byte-for-byte. Only final clipping changed.
        compatible = dict(ident, implementation=[
            "884d4c28556effae59c6719d0c8e69d2568499836f8c42dde1948adf8cb5808f", ident["implementation"][1]])
        if previous != compatible or (out/"complete.json").exists():
            raise ValueError("Existing global assembly identity changed")
        write(out/"assembly_identity_before_clip_repair.json", previous)
    write(signature,ident)
    if (out/"complete.json").is_file():
        if read(out/"complete.json")["mesh"]["sha256"]!=sha(path):raise ValueError("Global mesh readback")
        return
    block_paths=[]
    stats=dict(owned_area_m2=0.,fill_area_m2=0.,replacement_new_area_m2=0.,old_removed_area_m2=0.,old_retained_area_m2=0.,boundary_unresolved_area_m2=0.)
    if branch=="model":
        v,f=ply(base);scene=o3d.t.geometry.RaycastingScene(nthreads=8)
        scene.add_triangles(o3d.core.Tensor(np.asarray(v,dtype="f4")),o3d.core.Tensor(np.asarray(f,dtype="u4")))
        e=Evidence(export)
    for name in src["ledger"]:
        marker=layer/name;data=marker.with_suffix(".npz");old=read(marker)
        if sha(data)!=old["sha256"]:raise ValueError("Layer shard identity")
        stats["owned_area_m2"]+=old["owned_area_m2"]
        if branch=="sensor":
            block_paths.append(data);continue
        target=out/data.name;check=target.with_suffix(".json")
        if not check.exists():
            with np.load(data) as a:
                t,g,w=a["triangles"],a["source_group"],a["witness"]
            result=dict(fill=[],replace=[],ids=[],uv=[],ambiguous_area=0.)
            for i in range(0,len(t),4096):classify(t[i:i+4096],g[i:i+4096],w[i:i+4096],scene,e,0,result)
            def join(key,shape,dtype):
                return np.concatenate(result[key]) if result[key] else np.empty(shape,dtype)
            fill=join("fill",(0,3,3),"f8");replace=join("replace",(0,3,3),"f8")
            tmp=target.with_suffix(".partial.npz")
            np.savez(tmp,triangles=np.concatenate((fill,replace)),ids=join("ids",(0,),"i8"),uv=join("uv",(0,3,2),"f8"));tmp.replace(target)
            write(check,dict(sha256=sha(target),new_area=float(normal_area(replace)[1].sum()),
                       fill_area=float(normal_area(fill)[1].sum()),unresolved=result["ambiguous_area"]))
        item=read(check)
        if item["sha256"]!=sha(target):raise ValueError("Assembly shard readback")
        stats["fill_area_m2"]+=item["fill_area"];stats["replacement_new_area_m2"]+=item["new_area"]
        stats["boundary_unresolved_area_m2"]+=item["unresolved"];block_paths.append(target)
    if branch=="model":del scene,e
    def chunks():
        import tempfile
        with tempfile.TemporaryDirectory(prefix="old-cuts-",dir=out) as tmp:
            d=Path(tmp);dtype=np.dtype([("id","<i8"),("uv","<f8",(3,2))])
            for p in block_paths:
                with np.load(p) as a:
                    yield a["triangles"]
                    if branch=="model":
                        ids=a["ids"];uv=a["uv"];bands=ids//65536
                        for band in np.unique(bands):
                            mask=bands==band
                            records=np.empty(int(mask.sum()),dtype)
                            records["id"]=ids[mask];records["uv"]=uv[mask]
                            with (d/f"{band}.bin").open("ab") as dest:records.tofile(dest)
            if branch=="model":
                for first in range(0,len(f),65536):
                    t=np.asarray(v[f[first:first+65536]])
                    area=normal_area(t)[1];stats["old_retained_area_m2"]+=float(area.sum())
                    block=d/f"{first//65536}.bin"
                    write(RUN/"state"/"A3_assembly_progress.json",
                          dict(run_id=RUN.name, branch=branch, phase="old_face_clipping",
                               first_face=first, total_faces=len(f), block_faces=len(t)))
                    if not block.exists():yield t;continue
                    records=np.fromfile(block,dtype);groups=defaultdict(list)
                    for rec in records:groups[int(rec["id"])-first].append(rec["uv"])
                    live=np.ones(len(t),bool);parts=[]
                    for i,polygons in groups.items():
                        clipped,fraction=cut_original(t[i],polygons)
                        live[i]=False;parts.append(clipped)
                        removed=float(area[i]*fraction)
                        stats["old_removed_area_m2"]+=removed;stats["old_retained_area_m2"]-=removed
                    yield t[live]
                    if parts:yield np.concatenate(parts)
    mesh=save_welded(path,chunks())
    mesh["path"]=path.relative_to(ROOT).as_posix()
    write(out/"complete.json",dict(model=label,branch=branch,mesh=mesh,source_faces=src["source_faces"],
          owned_faces=src["owned_faces"],area=stats,assembly_identity=sha(signature)))
    print("Verified global",branch,mesh,flush=True)


if __name__=="__main__":
    if len(sys.argv)!=3 or sys.argv[1] not in ("B2","B3") or sys.argv[2] not in ("model","sensor"):raise SystemExit("usage: MODEL BRANCH")
    combined(*sys.argv[1:])
