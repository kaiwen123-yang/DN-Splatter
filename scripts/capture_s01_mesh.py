"""Run the historical Poisson exporter with an observational sampling audit."""
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

import cv2
import numpy as np
import open3d as o3d
import torch
from dn_splatter import export_mesh as exporter
from capture_rectify import imwrite, read, save, sha
from capture_s01_eval import final_setup
from capture_s01_inputs import FULL, REPORT, ensure
from vr_room_run import CPU, RUN, SCENE

MESH = RUN / "mesh"
PREFIX = "DepthAndNormalMapsPoisson_"


def export():
    MESH.mkdir(exist_ok=True)
    cfg, pipe, checkpoint, step = final_setup("test")
    ds = pipe.datamanager.train_dataset
    names = [p.relative_to(SCENE).as_posix() for p in ds.image_filenames]
    ensure(
        names == read(SCENE / "transforms.json")["train_filenames"] and (FULL or len(names) == 193),
        "mesh camera split",
    )
    original_render = pipe.model.get_outputs_for_camera
    original_pick = exporter.pick_indices_at_random
    rows = []
    current = {}

    def watch_render(camera, **kwargs):
        idx = len(rows)
        ensure(
            torch.allclose(
                camera.camera_to_worlds.cpu(), ds.cameras[idx : idx + 1].camera_to_worlds.cpu()
            ),
            "train camera order",
        )
        outputs = original_render(camera=camera, **kwargs)
        current.update(outputs)
        return outputs

    def sample(valid_mask, samples_per_frame):
        indices = original_pick(valid_mask, samples_per_frame)
        z = current["depth"].reshape(-1)
        a = current["accumulation"].reshape(-1)
        normals = 2 * current["normal"].reshape(-1, 3)[indices] - 1
        finite_z = z[torch.isfinite(z)]
        sz = z[indices]
        row = dict(
            id=Path(names[len(rows)]).stem,
            candidates=int((valid_mask != 0).sum()),
            samples=indices.numel(),
            alpha_zero_samples=int((a[indices] == 0).sum()),
            sampled_outside_sensor_range=int(((sz < 0.3) | (sz > 6)).sum()),
            normal_nonfinite=int((~torch.isfinite(normals).all(dim=-1)).sum()),
            normal_zero=int((normals.norm(dim=-1) == 0).sum()),
            render_nonfinite_depth=int(z.numel() - finite_z.numel()),
            render_z_min_m=float(finite_z.min()),
            render_z_max_m=float(finite_z.max()),
        )
        if os.environ.get("DN_CAPTURE_VARIANT") == "metric3d":
            provenance = MESH / "source_pixels"
            provenance.mkdir(exist_ok=True)
            np.save(provenance / (row["id"] + ".npy"), indices.detach().cpu().numpy().reshape(-1).astype(np.int32))
        rows.append(row)
        current.clear()
        if len(rows) == len(names):
            save(
                MESH / "sampling.json",
                dict(
                    views=rows,
                    seed=42,
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=sha(checkpoint),
                    step=step,
                    sample_target=5000000,
                ),
            )
        return indices

    model = exporter.DepthAndNormalMapsPoisson(
        load_config=Path(read(RUN / "training_complete.json")["config"]),
        output_dir=MESH,
        total_points=5000000,
        poisson_depth=9,
        normal_method="normal_maps",
    )
    ensure(
        not model.filter_edges_from_depth_maps
        and not model.outlier_removal
        and model.down_sample_voxel is None,
        "raw exporter flags",
    )
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    with patch.object(exporter, "eval_setup", return_value=(cfg, pipe, checkpoint, step)), patch.object(
        pipe.model, "get_outputs_for_camera", watch_render
    ), patch.object(exporter, "pick_indices_at_random", sample):
        model.main()
    ensure(len(rows) == len(names) and (MESH / (PREFIX + "poisson_mesh.ply")).is_file(), "raw mesh export")


def shaded(scene, ray, mesh):
    result = scene.cast_rays(ray, nthreads=CPU)
    valid = np.isfinite(result["t_hit"].numpy())
    light = np.array([-0.4, -0.6, -0.69])
    normals = result["primitive_normals"].numpy()[valid]
    shade = (0.24 + 0.76 * np.abs(normals @ light).clip(0, 1))[:, None]
    gray = np.full((*valid.shape, 3), 245, np.uint8)
    gray[valid] = np.uint8(shade * np.array([204, 212, 219]))
    color = gray.copy()
    if mesh.has_vertex_colors():
        tri = result["primitive_ids"].numpy()[valid]
        uv = result["primitive_uvs"].numpy()[valid]
        w = np.column_stack((1 - uv.sum(axis=1), uv))
        c = np.asarray(mesh.vertex_colors)[np.asarray(mesh.triangles)[tri]]
        color[valid] = np.uint8(np.clip((w[:, :, None] * c).sum(axis=1)[:, ::-1], 0, 1) * 255)
    return gray, color


def titled(img, label):
    return cv2.putText(
        cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT),
        label,
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def previews(mesh):
    scene = o3d.t.geometry.RaycastingScene(nthreads=CPU)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    v = np.asarray(mesh.vertices)
    mid = (v.min(0) + v.max(0)) * 0.5
    span = np.linalg.norm(v.max(0) - v.min(0))
    views = []
    for direction in (np.array([0.55, -0.45, -0.8]), np.array([-0.7, -0.55, 0.55])):
        ray = scene.create_rays_pinhole(
            48,
            mid.astype("float32"),
            (mid + direction * span * 1.1).astype("float32"),
            np.array([0, -1, 0], dtype="float32"),
            640,
            480,
        )
        gray, _ = shaded(scene, ray, mesh)
        views.append(titled(gray, "Complete raw mesh | external view " + str(len(views) + 1)))
    imwrite(REPORT / "previews/mesh_global.jpg", np.hstack(views))
    meta = read(SCENE / "transforms.json")
    frames = {Path(f["file_path"]).stem: f for f in meta["frames"]}
    # Crops are chosen on original train RGB images at 640x360; render only these image rays at native density.
    crops = [
        (120, "table", (387, 203, 590, 299)),
        (0, "pillar", (302, 142, 405, 357)),
        (240, "chair_legs", (107, 220, 248, 360)),
    ]
    records = []
    for fid, label, (x0, y0, x1, y1) in crops:
        if FULL and f"frame_{fid:05d}" not in frames:
            records.append(dict(object=label,train_id=fid,status="missing pose"))
            continue
        f = frames[f"frame_{fid:05d}"]
        ensure(f["file_path"] in meta["train_filenames"], "preview pose")
        x0, y0, x1, y1 = [n * 2 for n in (x0, y0, x1, y1)]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        local = np.stack(
            (
                (xx + 0.5 - meta["cx"]) / meta["fl_x"],
                (yy + 0.5 - meta["cy"]) / meta["fl_y"],
                np.ones_like(xx),
            ),
            -1,
        )
        c2w = np.array(f["transform_matrix"]) @ np.diag([1, -1, -1, 1])
        rays = np.concatenate(
            (np.broadcast_to(c2w[:3, 3], local.shape), local @ c2w[:3, :3].T), axis=-1
        )
        gray, color = shaded(scene, o3d.core.Tensor(rays.astype("float32")), mesh)
        rgb = cv2.imread(str(SCENE / f["file_path"]))[y0:y1, x0:x1]
        extra=[]
        support={}
        if FULL:
            from capture_s01_eval import color as depth_color
            crop=np.s_[y0:y1,x0:x1]
            name=f"frame_{fid:05d}"
            ref=np.load(SCENE/f["depth_file_path"])[crop]
            pred=np.load(RUN/"video"/(name+"_depth.npy"))[crop]
            norm=cv2.imread(str(RUN/"video"/(name+"_normal.png")))[crop]
            hit=scene.cast_rays(o3d.core.Tensor(rays.astype("float32")),nthreads=CPU)["t_hit"].numpy()
            cloud=o3d.io.read_point_cloud(str(MESH/(PREFIX+"pcd.ply")))
            # Project the complete sampled point cloud in chunks and keep the nearest positive Z.
            cloud_z=np.full(ref.shape,np.inf)
            pts=np.asarray(cloud.points)
            for start in range(0,len(pts),250000):
                pc=(pts[start:start+250000]-c2w[:3,3])@c2w[:3,:3]
                pc=pc[pc[:,2]>0]
                x=np.rint(meta["fl_x"]*pc[:,0]/pc[:,2]+meta["cx"]-.5).astype(int)-x0
                y=np.rint(meta["fl_y"]*pc[:,1]/pc[:,2]+meta["cy"]-.5).astype(int)-y0
                ok=(x>=0)&(y>=0)&(x<x1-x0)&(y<y1-y0)
                np.minimum.at(cloud_z,(y[ok],x[ok]),pc[ok,2])
            for label_z,z in (("Sensor Z",ref),("Model Z",pred),("Sample cloud Z",cloud_z),("Mesh Z",hit)):
                valid=np.isfinite(z)&(z>0)
                mask=valid&(ref>0)
                support[label_z]=dict(reference_pixels=int((ref>0).sum()),matched_pixels=int(mask.sum()),
                                      MAE_m=float(np.abs(z[mask]-ref[mask]).mean()) if mask.any() else None)
                extra.append((depth_color(z,.3,6,valid),label_z))
            extra.append((norm,"Model normal"))
        sheet = np.hstack(
            [
                titled(img, lbl)
                for img, lbl in [
                    (rgb, "Train RGB"),
                    *extra,
                    (color, "Mesh vertex color"),
                    (gray, "Mesh face shading"),
                ]
            ]
        )
        imwrite(REPORT / "previews" / ("mesh_" + label + ".jpg"), sheet)
        records.append(dict(object=label, train_id=fid, xyxy=[x0, y0, x1, y1],depth_comparison=support))
    return records


def validate():
    pcd = o3d.io.read_point_cloud(str(MESH / (PREFIX + "pcd.ply")))
    mesh = o3d.io.read_triangle_mesh(str(MESH / (PREFIX + "poisson_mesh.ply")))
    xyz, norm = np.asarray(pcd.points), np.asarray(pcd.normals)
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    rows = read(MESH / "sampling.json")["views"]
    count = sum(r["samples"] for r in rows)
    ensure(
        len(xyz) == count
        and count > 0
        and np.isfinite(xyz).all()
        and norm.shape == xyz.shape
        and np.isfinite(norm).all(),
        "sample cloud validity",
    )
    lengths = np.linalg.norm(norm, axis=1)
    zero = int((lengths == 0).sum())
    ensure(
        len(f) and len(v) and np.isfinite(v).all() and f.min() >= 0 and f.max() < len(v),
        "mesh validity",
    )
    _, sizes, areas = mesh.cluster_connected_triangles()
    mesh_normals = np.asarray(mesh.vertex_normals)
    ensure(np.isfinite(mesh_normals).all(), "mesh normals")
    result = dict(
        original_exporter="DepthAndNormalMapsPoisson",
        views=len(rows),
        samples=count,
        samples_by_view={r["id"]:r["samples"] for r in rows},
        target_samples=5000000,
        poisson_depth=9,
        threads=CPU,
        seed=42,
        density_low_quantile=0.01,
        depth_range_gate=False,
        alpha_gate=False,
        edge_filter=False,
        outlier_filter=False,
        sampled_nonfinite_normals=int((~np.isfinite(norm).all(1)).sum()),
        sampled_zero_normals=zero,
        sampled_norm_length_range=[float(lengths.min()), float(lengths.max())],
        sampled_alpha_zero=sum(r["alpha_zero_samples"] for r in rows),
        sampled_outside_sensor_range=sum(r["sampled_outside_sensor_range"] for r in rows),
        render_z_range_m=[
            min(r["render_z_min_m"] for r in rows),
            max(r["render_z_max_m"] for r in rows),
        ],
        vertices=len(v),
        faces=len(f),
        vertex_normals=len(mesh_normals),
        bounds_xyz_m=[v.min(0).tolist(), v.max(0).tolist()],
        faces_index_range=[int(f.min()), int(f.max())],
        connected_components=len(sizes),
        largest_component_faces=int(max(sizes)),
        largest_component_face_fraction=float(max(sizes)/len(f)),
        nominal_depth9_cell_m=float(1.1*np.ptp(xyz,axis=0).max()/512),
        largest_component_surface_area_m2=float(max(areas)),
        geometry_finite=True,
        pcd_path=str(MESH / (PREFIX + "pcd.ply")),
        mesh_path=str(MESH / (PREFIX + "poisson_mesh.ply")),
        local_views=previews(mesh),
    )
    save(REPORT / "mesh_validation.json", result)


if __name__ == "__main__":
    {"export": export, "validate": validate}[sys.argv[1]]()
