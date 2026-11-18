"""Extract only the VR Room inputs and record the exact registered-frame split."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from zenodo_inputs import FILES


ROOT = Path(os.environ["DN_PROJECT_ROOT"])
DATA = Path(os.environ["DN_DATA_ROOT"])
SCENE = Path(os.environ["DN_SCENE_DIR"])
RUN = Path(os.environ["DN_RUN_ROOT"])
SOURCE = Path(os.environ["DN_SOURCE_DIR"])


def dump(name, obj):
    (RUN / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def hash_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def registered():
    folder = SCENE / "iphone"
    test = set((folder/"long_capture/test.txt").read_text().split())
    by_split = {"train": [], "within": [], "with": []}
    for capture in ("long_capture","short_capture"):
        base = folder/capture
        doc = json.loads((base/"transformations_colmap.json").read_text())
        for frame in sorted(doc["frames"], key=lambda f:f["file_path"]):
            rgb, depth = base/frame["file_path"], base/frame["depth_file_path"]
            if not rgb.is_file() or not depth.is_file():
                raise FileNotFoundError(f"Missing RGB/depth for {rgb.name}")
            split = "with" if capture=="short_capture" else "within" if rgb.stem in test else "train"
            by_split[split].append(str(rgb.relative_to(SCENE)))
    if not all(by_split.values()) or any(len(x)!=len(set(x)) for x in by_split.values()):
        raise ValueError("Missing or duplicate registered frame")
    dump("registered.json",{"registered_rgb_paths":by_split, "counts":{k:len(v) for k,v in by_split.items()},
         "normal_alignment_checked":False})


def alignment():
    import numpy as np
    from PIL import Image

    folder = SCENE/"iphone/long_capture"
    doc = json.loads((folder/"transformations_colmap.json").read_text())
    test = set((folder/"test.txt").read_text().split())
    train = sorted((f for f in doc["frames"] if Path(f["file_path"]).stem not in test),
                   key=lambda f:f["file_path"])
    def camera(frame):
        with Image.open(folder/frame["depth_file_path"]) as im:
            depth = np.asarray(im,dtype=np.float64)*0.001
        h,w = depth.shape
        meta = {**doc, **frame}
        k = np.array([[meta["fl_x"],0,meta["cx"]],[0,meta["fl_y"],meta["cy"]],[0,0,1.0]])
        k[:2] *= np.array([w/meta["w"],h/meta["h"]])[:,None]
        pose = np.asarray(frame["transform_matrix"])@np.diag([1,-1,-1,1])
        return depth,k,pose
    rows = []
    for i in (0,len(train)//2,len(train)-2):
        a,b = train[i:i+2]
        da,ka,ta = camera(a)
        db,kb,tb = camera(b)
        y,x = np.mgrid[:da.shape[0]:4,:da.shape[1]:4]
        valid = da[y,x]>0
        y,x = y[valid],x[valid]
        xyz = np.linalg.solve(ka,np.stack([x,y,np.ones_like(x)]))*da[y,x]
        rel = np.linalg.solve(tb,ta)
        xyz = rel[:3,:3]@xyz+rel[:3,3:4]
        xyz = xyz[:,xyz[2]>0]
        uv = np.rint((kb@xyz)[:2]/xyz[2]).astype(int)
        inside=(uv[0]>=0)&(uv[0]<db.shape[1])&(uv[1]>=0)&(uv[1]<db.shape[0])
        target=db[uv[1,inside],uv[0,inside]]
        errors=np.abs(xyz[2,inside][target>0]-target[target>0])
        if not len(errors):
            raise ValueError("No valid cross-view depth overlap")
        rows.append({"source":a["file_path"],"target":b["file_path"],"source_valid_samples":len(x),
                     "matched_samples":int(errors.size),"depth_size_wh":list(da.shape[::-1]),
                     "median_abs_axial_depth_difference_m":float(np.median(errors)),
                     "p90_abs_axial_depth_difference_m":float(np.quantile(errors,.9))})
    dump("input_alignment.json",{"method":"Every fourth sensor pixel in fixed adjacent registered training views; "
         "source depth backprojected in metres, OpenGL COLMAP poses converted to OpenCV, nearest target sensor pixel. "
         "Both depths must be positive. Occlusions are included.", "pairs":rows})


def extract(name):
    src = DATA / "downloads" / name
    rid, size, checksum = FILES[name]
    if src.stat().st_size != size or hash_file(src, "md5") != checksum:
        raise ValueError(f"Input integrity failure: {name}")
    kept = []
    if name.endswith(".ckpt"):
        target = SOURCE / "omnidata_ckpt" / name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.symlink_to(src)
        if not target.samefile(src) and hash_file(target, "md5") != checksum:
            raise ValueError("Existing Omnidata checkpoint differs")
        kept.append(str(target.relative_to(ROOT)))
    else:
        with tarfile.open(src, "r|gz") as tar:
            for member in tar:
                path = PurePosixPath(member.name)
                parts = path.parts
                if not member.isfile() or parts[:2] != ("room_datasets", "vr_room"):
                    continue
                if ".." in parts or path.is_absolute() or any(t.startswith("._") for t in parts):
                    continue
                if name == "reference_depth.tar.gz" and not (
                    len(parts) >= 6 and parts[2] == "iphone" and parts[4] == "reference_depth"
                ):
                    continue
                if path.name == "iphone_pointcloud.ply":
                    # Regenerate using the pinned training-only initializer.
                    continue
                dest = DATA.joinpath(*parts)
                dest.resolve().relative_to(SCENE.resolve())
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    tmp = dest.with_name(dest.name + ".extracting")
                    with tar.extractfile(member) as data, tmp.open("wb") as out:
                        shutil.copyfileobj(data, out, 1024 * 1024)
                    if tmp.stat().st_size != member.size:
                        raise IOError(str(path))
                    tmp.replace(dest)
                elif dest.stat().st_size != member.size:
                    raise ValueError(f"Existing project input size differs: {path}")
                kept.append(str(path))
    if not kept:
        raise ValueError(f"No VR Room members: {name}")
    dump("input-" + name + ".json", {"name": name, "url": f"https://zenodo.org/records/{rid}/files/{name}",
         "bytes": size, "zenodo_md5": checksum, "sha256": hash_file(src), "selected_members": kept})
    if name == "vr_room_iphone.tar.gz":
        registered()


def frames():
    from PIL import Image
    import numpy as np

    iphone = SCENE / "iphone"
    test = set((iphone / "long_capture/test.txt").read_text().split())
    result = {"train": [], "within": [], "with": []}
    for capture in ["long_capture", "short_capture"]:
        folder = iphone / capture
        doc = json.loads((folder / "transformations_colmap.json").read_text())
        def inspect_frame(frame):
            rgb = folder / frame["file_path"]
            depth = folder / frame["depth_file_path"]
            normal = folder / "normals_from_pretrain" / (rgb.stem + ".png")
            with Image.open(rgb) as im, Image.open(depth) as z, Image.open(normal) as n:
                if im.size != n.size:
                    raise ValueError(f"Normal resolution mismatch: {normal.name}")
                dims = (doc.get("w", frame.get("w")), doc.get("h", frame.get("h")))
                if dims != (None, None) and dims != im.size:
                    raise ValueError(f"RGB resolution mismatch: {rgb.name}")
                valid = float((np.asarray(z) > 0).mean())
                entry = {"rgb": str(rgb.relative_to(SCENE)), "depth": str(depth.relative_to(SCENE)),
                         "normal": str(normal.relative_to(SCENE)), "rgb_size": im.size, "depth_size": z.size,
                         "valid_depth_fraction": valid}
            if np.asarray(frame["transform_matrix"]).shape != (4,4):
                raise ValueError(rgb.name)
            return ("with" if capture == "short_capture" else "within" if rgb.stem in test else "train"), entry
        with concurrent.futures.ThreadPoolExecutor(int(os.environ["DN_CPU_BUDGET"])) as pool:
            for split, entry in pool.map(inspect_frame, sorted(doc["frames"], key=lambda f: f["file_path"])):
                result[split].append(entry)
    if not all(result.values()):
        raise ValueError("Empty VR Room split")
    for entries in result.values():
        if len(entries) != len({i["rgb"] for i in entries}):
            raise ValueError("Duplicate registered RGB frame")
    dump("frames.json", {"frames": result, "counts": {k: len(v) for k,v in result.items()},
         "unregistered_test_ids": sorted(test - {Path(x["rgb"]).stem for x in result["within"]}),
         "poses": "transformations_colmap.json", "sensor_depth_metres_per_unit": 0.001,
         "normal_prior": "Omnidata v2 low, historical parser opengl"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["extract", "frames", "registered", "alignment"])
    p.add_argument("name", nargs="?")
    args = p.parse_args()
    if args.action == "extract":
        extract(args.name)
    elif args.action == "frames":
        frames()
        alignment()
    elif args.action == "alignment":
        alignment()
    else:
        registered()
