"""Read complete primary observation tracks from the fixed, hash-checked training reconstruction."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pycolmap


def main():
    root=Path(__file__).resolve().parents[1]
    def read(path):
        return json.loads(path.read_text())
    full=read(root/"configs/capture_full_test_protocol.json")
    plan=read(root/"configs/capture_quality_protocol.json")
    geometry=root/full["paths"]["geometry_run"]
    out=root/".local/runs"/plan["run_id"]/"geometry"
    evidence=read(geometry/"sfm/pose_quality.json")
    component=max(evidence["map"]["components"],key=lambda c:len(c["registered_primary"]))
    model=Path(component["model"])
    for path in model.glob("*.bin"):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==evidence["map_files_sha256_before_and_after"][path.name]
    rec=pycolmap.Reconstruction(str(model))
    ids=np.asarray(sorted(rec.points3D),dtype=np.int64)
    xyz=np.asarray([rec.points3D[int(i)].xyz for i in ids])
    train={int(Path(s).stem[-5:]) for s in read(out.parent/"inputs/transforms.json")["train_filenames"]}
    rows=[]
    for image in sorted(rec.images.values(),key=lambda im:im.name):
        if not image.has_pose or image.name.startswith("aux_"):
            continue
        fid=int(Path(image.name).stem[-5:])
        assert fid in train
        points=[p for p in image.points2D if p.has_point3D()]
        if points:
            uv=np.array([p.xy for p in points])
            pi=np.searchsorted(ids,[p.point3D_id for p in points])
            rows.append(np.column_stack([np.full(len(pi),fid),pi,uv]))
    assert len(train)==1478 and rows
    np.savez(out/"tracks.npz",point_ids=ids,xyz=xyz,obs=np.concatenate(rows))


if __name__=="__main__":
    main()
