"""Build a welded PLY with bounded RAM, preserving the winding of each face."""
import shutil
from pathlib import Path
import numpy as np

from capture_surface_tiles import read, write, sha

BUCKETS=64
CORNER=np.dtype([("xyz","<f8",3),("normal","<f4",3),("corner","<i8")])


def append(stream, array):
    with stream.open("ab") as f:array.tofile(f)


def hash3(keys):
    return (keys[:,0]*73856093 ^ keys[:,1]*19349663 ^ keys[:,2]*83492791) % BUCKETS


def bucket_write(directory, prefix, records, bucket):
    order=np.argsort(bucket,kind="stable"); b=bucket[order]
    starts=np.r_[0,np.flatnonzero(np.diff(b))+1,len(b)]
    for i,j in zip(starts[:-1],starts[1:]):
        if j>i:append(directory/f"{prefix}-{b[i]:02d}.bin",records[order[i:j]])


def save_welded(path, chunks):
    import tempfile
    # Exact XYZ joins avoid bridging distinct layers. Coincident oppositely oriented faces keep their winding.
    with tempfile.TemporaryDirectory(prefix="weld-",dir=path.parent) as work:
        d=Path(work); total=degenerate=0
        for triangles in chunks:
            for start in range(0,len(triangles),100000):
                t=np.asarray(triangles[start:start+100000],dtype="f8")
                normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
                length=np.linalg.norm(normal,axis=1)
                keep=np.isfinite(length)&(length>2e-16)
                degenerate+=len(t)-int(keep.sum());t=t[keep];normal=normal[keep]/length[keep,None]
                if not len(t):continue
                r=np.empty(t.size//3,CORNER);r["xyz"]=t.reshape(-1,3);r["xyz"][r["xyz"]==0]=0
                r["normal"]=np.repeat(normal,3,axis=0);r["corner"]=np.arange(3*total,3*(total+len(t)))
                # Hash the full IEEE position bits, mixing mantissa and exponent before selecting a bucket.
                bits=np.ascontiguousarray(r["xyz"]).view("u8")
                h=hash3(bits^(bits>>17)^(bits>>37))
                bucket_write(d,"corner",r,h);total+=len(t)
        vertices_path=d/"vertices.bin";faces_path=d/"faces.bin"
        vertices_path.touch();faces_path.touch();nv=nf=duplicates=0
        if total:
            remap=np.memmap(d/"remap.bin",mode="w+",shape=(3*total,),dtype="<i4")
            for file in sorted(d.glob("corner-*.bin")):
                a=np.fromfile(file,CORNER)
                pos=np.ascontiguousarray(a["xyz"])
                v,inv=np.unique(pos,axis=0,return_inverse=True);inv=inv.reshape(-1)
                remaining=np.arange(len(a)); unit=a["normal"].astype("f8")
                unit/=np.maximum(np.linalg.norm(unit,axis=1)[:,None],1e-15)
                while len(remaining):
                    group,first,local=np.unique(inv[remaining],return_index=True,return_inverse=True)
                    local=local.reshape(-1)
                    anchor=unit[remaining[first]]
                    link=(unit[remaining]*anchor[local]).sum(axis=1)>=np.cos(np.deg2rad(30.))
                    members=remaining[link];idx=local[link]
                    if nv+len(group)>=2**31:raise ValueError("PLY index overflow")
                    sums=np.column_stack([np.bincount(idx,weights=unit[members,k],minlength=len(group)) for k in range(3)])
                    sums/=np.maximum(np.linalg.norm(sums,axis=1)[:,None],1e-15)
                    append(vertices_path,np.column_stack((v[group],sums)).astype("<f8"))
                    remap[a["corner"][members]]=nv+idx;nv+=len(group)
                    remaining=remaining[~link]
            remap.flush()
            for first in range(0,total,200000):
                ids=np.array(remap[3*first:3*min(total,first+200000)]).reshape(-1,3)
                good=(ids[:,0]!=ids[:,1])&(ids[:,1]!=ids[:,2])&(ids[:,0]!=ids[:,2])
                degenerate+=int((~good).sum());ids=ids[good]
                if not len(ids):continue
                shift=ids.argmin(axis=1)
                canon=ids[np.arange(len(ids))[:,None],(np.arange(3)+shift[:,None])%3]
                bucket_write(d,"faceset",canon,hash3(canon.astype("i8")))
            del remap
            for file in sorted(d.glob("faceset-*.bin")):
                ids=np.fromfile(file,"<i4").reshape(-1,3);unique=np.unique(ids,axis=0)
                duplicates+=len(ids)-len(unique)
                faces=np.empty(len(unique),dtype=[("n","u1"),("ids","<u4",3)])
                faces["n"]=3;faces["ids"]=unique
                append(faces_path,faces);nf+=len(faces)
        header=["ply","format binary_little_endian 1.0","comment Created by Open3D",f"element vertex {nv}",
                *["property double "+k for k in ("x","y","z","nx","ny","nz")],f"element face {nf}",
                "property list uchar uint vertex_indices","end_header"]
        tmp=path.with_suffix(".pending.ply")
        with tmp.open("wb") as out:
            out.write(("\n".join(header)+"\n").encode("ascii"))
            for file in (vertices_path,faces_path):
                with file.open("rb") as f:shutil.copyfileobj(f,out,4<<20)
        tmp.replace(path)
    return dict(vertices=nv,triangles=nf,duplicate_faces=duplicates,degenerate_faces=degenerate,
                bytes=path.stat().st_size,sha256=sha(path))
