"""Check frame coverage, recompute Faro metrics by trajectory, and render mesh QA views."""
import json
import math
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import torch
import torchvision.transforms.functional as tf
from dn_splatter.metrics import DepthMetrics
from dn_splatter.utils.utils import depth_path_to_tensor

RUN = Path(os.environ["DN_RUN_ROOT"])
SCENE = Path(os.environ["DN_SCENE_DIR"])
RENDERS = RUN / "renders_official"
IMAGES = RENDERS / "final_renders"
PREVIEW = RUN / "previews"


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def audit_split(split, frames, scale):
    metric = DepthMetrics()
    keys = ["mse", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    totals = np.zeros(len(keys))
    pixels = 0
    selected = {0, len(frames)//2, len(frames)-1}
    coverage = set()
    for index, frame in enumerate(frames):
        f = Path(frame["rgb"])
        capture = f.parts[1]
        label = capture + "_" + f.stem
        pred = depth_path_to_tensor(IMAGES / "pred/depth/raw" / (label + ".npy")).permute(2,0,1)
        gt = depth_path_to_tensor(SCENE / "iphone" / capture / "reference_depth" / (f.stem + ".png")).permute(2,0,1)
        if pred.shape != gt.shape:
            pred = tf.resize(pred, gt.shape[-2:], antialias=None)
        pred = pred / scale
        keep = gt > metric.tolerance
        if not torch.isfinite(pred).all() or not torch.isfinite(gt).all() or not keep.any():
            raise ValueError(f"Invalid reference or prediction for {label}")
        values = [torch.mean((pred-gt)**2), *metric(pred[None], gt[None])]
        if not all(torch.isfinite(value) for value in values):
            raise ValueError(f"Non-finite Faro metric at {label}")
        totals += [float(v) for v in values]
        pixels += int(keep.sum())
        paths = [IMAGES / folder / (label + ".png") for folder in
                 ("gt/rgb", "pred/rgb", "pred/depth/colorised", "pred/normal")]
        for p in paths:
            with Image.open(p) as im:
                im.verify()
        if index in selected:
            render_preview(label, paths)
        coverage.add(label)
    return {"frames": len(frames), "valid_depth_pixels": pixels,
            "mean_per_frame": dict(zip(keys, (totals/len(frames)).tolist()))}, coverage


def render_preview(label, paths):
    panels = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((310, 270))
            panels.append(im.copy())
    w, h = max(i.width for i in panels), max(i.height for i in panels)
    sheet = Image.new("RGB", (w*4,h+24),"white")
    draw = ImageDraw.Draw(sheet)
    for i, (panel, title) in enumerate(zip(panels, ("Reference RGB","Rendered RGB","Depth","Normal"))):
        sheet.paste(panel,(w*i,24))
        draw.text((w*i+4,5),title,fill="black")
    sheet.save(PREVIEW / (label + ".jpg"),quality=88)


def mesh_checks():
    res = {}
    for file in ("DepthAndNormalMapsPoisson_poisson_mesh.ply", "mesh_cull.ply"):
        mesh = o3d.io.read_triangle_mesh(str(RUN / "mesh" / file))
        verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        if not len(verts) or not len(faces) or not np.isfinite(verts).all() or faces.min()<0 or faces.max()>=len(verts):
            raise ValueError(f"Invalid Poisson surface: {file}")
        res[file] = {"vertices": len(verts), "triangles": len(faces),
                     "bounds": [verts.min(axis=0).tolist(), verts.max(axis=0).tolist()]}
        if file == "mesh_cull.ply":
            scene = o3d.t.geometry.RaycastingScene(nthreads=int(os.environ["DN_CPU_BUDGET"]))
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            mid = (verts.min(axis=0)+verts.max(axis=0))/2
            radius = np.linalg.norm(verts.max(axis=0)-verts.min(axis=0))
            for i, direction in enumerate(([1,-1,0.65], [-1,0.8,0.45])):
                ray = scene.create_rays_pinhole(50, mid.astype("float32"),
                    (mid+np.array(direction)*radius*0.7).astype("float32"),
                    np.array([0,0,1],dtype="float32"), 640, 480)
                hits = scene.cast_rays(ray, nthreads=int(os.environ["DN_CPU_BUDGET"]))
                z = hits["t_hit"].numpy()
                normals = hits["primitive_normals"].numpy()
                valid = np.isfinite(z)
                if valid.sum() < 100:
                    raise ValueError("Empty mesh preview")
                img = np.full((*z.shape,3),246,dtype="uint8")
                light = np.array([0.4,-0.45,0.8])
                shade = 0.2+0.8*np.abs(normals[valid]@light).clip(0,1)
                img[valid] = (shade[:,None]*np.array([198,205,211])).astype("uint8")
                Image.fromarray(img).save(PREVIEW / f"culled-mesh-{i+1}.jpg", quality=90)
    cloud = o3d.io.read_point_cloud(str(RUN / "mesh/DepthAndNormalMapsPoisson_pcd.ply"))
    res["points_entering_poisson_after_filtering"] = len(cloud.points)
    import re
    records = re.findall(r"DN sampled points before filtering:\s*([\d,]+)",(RUN/"logs/poisson.log").read_text(errors="replace"))
    if not records or not 0 < len(cloud.points) <= int(records[-1].replace(",","")):
        raise ValueError("Missing sample counts")
    res.update(target_samples=5000000, sampled_before_filtering=int(records[-1].replace(",","")),
               poisson_depth=9, poisson_threads=int(os.environ["DN_CPU_BUDGET"]), fscore_threshold_m=0.05)
    dump(RUN/"mesh/validation.json",res)


def main():
    PREVIEW.mkdir(exist_ok=True)
    frames = json.loads((RUN/"frames.json").read_text())["frames"]
    scale = json.loads((RENDERS/"dataparser_transforms.json").read_text())["scale"]
    if not math.isfinite(scale) or scale<=0:
        raise ValueError("Invalid dataparser scale")
    out = {"aggregation": "Equal weight per registered frame, independently within each trajectory. "
           "Official meter conversion and resize; MSE over all pixels; other metrics use GT >0.1 m. "
           "rmse_log retains the historical DepthMetrics formula.", "scale": scale, "splits": {}}
    expected = set()
    for split in ("within", "with"):
        values, names = audit_split(split, frames[split], scale)
        out["splits"][split] = values
        expected |= names
    disk = {p.stem for p in (IMAGES/"pred/depth/raw").glob("*.npy")}
    if disk != expected:
        raise ValueError("Frame sets differ between raw renders and registered holdout")
    dump(RUN/"evaluation/faro_audited.json",out)
    mesh_checks()


if __name__ == "__main__":
    main()
