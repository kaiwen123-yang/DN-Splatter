"""Check the frozen B2 native TSDF integration against its own input views on CPU.

Run in the project environment. The default probe creates temporary in-memory
volumes. --full-mesh raycasts the published PLY in disjoint blocks on the same
pixels. Each mode prints its results to stdout.
"""
import gc
import json
import os
from pathlib import Path
import re
import resource
import sys

import cv2
import numpy as np
import open3d as o3d

from capture_quality_paths import GEORUN, RUN, SCENE
from capture_quality_tsdf import FLIP, integrate, volume
from capture_rectify import grid_setup, project_frame, read, sha


def make_intrinsic(shape, k):
    height, width = shape
    return o3d.camera.PinholeCameraIntrinsic(
        width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2])


def ray_query(reference, k, pose):
    y, x = np.nonzero(np.isfinite(reference) & (reference >= .3) & (reference <= 6))
    ref = reference[y, x].astype(np.float64)
    local = np.column_stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1], np.ones(len(y))))
    rays = np.concatenate((np.broadcast_to(pose[:3, 3], local.shape), local @ pose[:3, :3].T), axis=1)
    return ref, o3d.core.Tensor(rays.astype(np.float32))


def hit_metrics(hit, ref):
    valid = np.isfinite(hit) & (hit > 0)
    delta = hit[valid]-ref[valid]
    absolute = abs(delta)
    correct = int((absolute <= .1).sum())
    return dict(reference_pixels=len(ref), hits=int(valid.sum()), correct_10cm=correct,
                hit_fraction=float(valid.mean()), correct_10cm_fraction=correct/len(ref),
                foreground_conflict_fraction=float((delta < -np.maximum(.05, .03*ref[valid])).sum()/len(ref)),
                mae_hit_m=float(absolute.mean()) if len(absolute) else None,
                signed_mean_hit_m=float(delta.mean()) if len(delta) else None,
                p50_hit_m=float(np.median(absolute)) if len(absolute) else None,
                p95_hit_m=float(np.percentile(absolute, 95)) if len(absolute) else None)


def ray_metrics(scene, reference, k, pose):
    ref, rays = ray_query(reference, k, pose)
    return hit_metrics(scene.cast_rays(rays, nthreads=1)["t_hit"].numpy(), ref)


def existing_mesh_evaluation(path, queries):
    """Raycast all faces in disjoint chunks and take the closest hit on each ray."""
    header = bytearray()
    with path.open("rb") as stream:
        for line in stream:
            header.extend(line)
            if line.strip() == b"end_header":
                break
    n_vertices = int(re.search(rb"element vertex (\d+)", header)[1])
    n_faces = int(re.search(rb"element face (\d+)", header)[1])
    expected_header = ["ply", "format binary_little_endian 1.0", "comment Created by Open3D",
                       f"element vertex {n_vertices}", *["property double "+k for k in ("x","y","z","nx","ny","nz")],
                       f"element face {n_faces}", "property list uchar uint vertex_indices", "end_header"]
    if header.decode("ascii").splitlines() != expected_header:
        raise ValueError("Unexpected production mesh PLY schema")
    pos = np.memmap(path, dtype="<f8", mode="r", offset=len(header), shape=(n_vertices, 6))
    face_type = np.dtype([("n", "u1"), ("indices", "<u4", 3)])
    faces = np.memmap(path, dtype=face_type, mode="r", offset=len(header) + pos.nbytes, shape=(n_faces,))
    if len(header) + pos.nbytes + faces.nbytes != path.stat().st_size:
        raise ValueError("Truncated or trailing PLY bytes")
    chunk = 500_000
    names = list(queries)
    rays = o3d.core.Tensor(np.concatenate([queries[key][1].numpy() for key in names]))
    best = np.full(rays.shape[0], np.inf, np.float32)
    for start in range(0, n_faces, chunk):
        batch = faces[start:start+chunk]
        if (batch["n"] != 3).any():
            raise ValueError("Non-triangle in production mesh")
        ids, inverse = np.unique(batch["indices"].reshape(-1), return_inverse=True)
        vertices = o3d.core.Tensor(np.ascontiguousarray(pos[ids, :3], dtype="f4"))
        triangles = o3d.core.Tensor(inverse.reshape(-1,3).astype("u4"))
        scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(vertices, triangles)
        hit = scene.cast_rays(rays, nthreads=1)["t_hit"].numpy()
        np.minimum(best, hit, out=best)
        del scene, vertices, triangles, hit
    measurements = {}
    start = 0
    for stem, grid in names:
        ref = queries[stem, grid][0]
        measurements.setdefault(stem, {})[grid] = hit_metrics(best[start:start+len(ref)], ref)
        start += len(ref)
    return dict(mesh=path.name, vertices=n_vertices, faces=n_faces, all_faces_processed=True,
                face_chunk=chunk, measurements=measurements)


def evaluate(view_ids, frames, intr, outputs):
    vol = volume()
    valid = []
    for stem in view_ids:
        depth, pose = frames(stem)
        valid.append(integrate(vol, depth, intr, np.linalg.inv(pose)))
    mesh = vol.extract_triangle_mesh()
    del vol
    if not len(mesh.triangles):
        raise RuntimeError("Empty mesh in TSDF closure check")
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return dict(input_views=view_ids, valid_native_counts=valid,
                mesh_vertices=len(mesh.vertices), mesh_faces=len(mesh.triangles),
                measurements={key: ray_metrics(scene, *args) for key, args in outputs.items()})


def main(full_mesh=False):
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    cv2.setNumThreads(1)
    packet = GEORUN/"transfer"
    cal = read(packet/"calibration.json")
    rays, t, grids = grid_setup(cal)
    rgb_grid = grids["1280"]
    rgb_k = np.array(rgb_grid[-1]["K_cv"])
    native = cal["cameras"]["depth"]
    k = np.array(native["K"])
    intr = make_intrinsic((native["height"], native["width"]), k)
    extrinsic = next(row for row in cal["static_extrinsics"]
                     if (row["source_sensor"], row["destination_sensor"]) == ("depth", "color"))
    rel = np.eye(4)
    rel[:3, :3], rel[:3, 3] = extrinsic["R"], t
    meta = read(SCENE/"transforms.json")
    train = [Path(name).stem for name in meta["train_filenames"]]
    poses = {Path(row["file_path"]).stem: np.array(row["transform_matrix"]) @ FLIP for row in meta["frames"]}
    selection = {row["id"]: row for row in read(packet/"selection.json")}

    def raw(stem):
        with np.load(packet/selection[stem]["depth"]["file"]) as f:
            return f["depth"]

    def source(stem):
        return raw(stem).astype(np.float32)/1000, poses[stem] @ rel

    anchors = ["frame_00000", "frame_00330", "frame_01420"]
    if full_mesh:
        queries = {}
        for stem in anchors:
            if stem not in train:
                raise ValueError(stem)
            z, pose = source(stem)
            queries[stem, "native"] = ray_query(z, k, pose)
            ref = np.load(SCENE/"depth"/(stem+".npy"))
            queries[stem, "frozen_rgb"] = ray_query(ref, rgb_k, poses[stem])
        mesh_path = RUN/"meshes"/"sensor_tsdf.ply"
        result = existing_mesh_evaluation(mesh_path, queries)
        result["mesh_sha256"] = sha(mesh_path)
        if result["mesh_sha256"] != read(RUN/"meshes"/"sensor_tsdf.json")["mesh"]["sha256"]:
            raise ValueError("Production mesh SHA-256 differs from its receipt")
        result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        print(json.dumps(result, indent=2, allow_nan=False))
        return

    # A tilted plane with an off-centre principal point tests row/column layout
    # and metre-scale depth variation under a nonidentity rigid transform.
    plane_k = np.array([[105., 0., 70.], [0., 111., 46.], [0., 0., 1.]])
    yy, xx = np.indices((96, 144))
    plane = (2./(1.+.2*(xx-plane_k[0, 2])/plane_k[0, 0]
                         -.3*(yy-plane_k[1, 2])/plane_k[1, 1])).astype(np.float32)
    plane_pose = poses["frame_00330"] @ rel
    result = dict(run=RUN.name, numpy=np.__version__, open3d=o3d.__version__,
                  cpu_affinity_count=len(os.sched_getaffinity(0)),
                  settings=dict(voxel_m=.01, tsdf_trunc_m=.04, source="production volume() and integrate()",
                                ray_direction="unnormalized, optical-axis Z", evaluation="own training view"),
                  transforms_sha256=sha(SCENE/"transforms.json"), calibration_sha256=sha(packet/"calibration.json"),
                  production_integration_sha256=sha(Path(__import__("capture_quality_tsdf").__file__)),
                  synthetic=evaluate(["plane"], lambda _: (plane, plane_pose), make_intrinsic(plane.shape, plane_k),
                                     {"native": (plane, plane_k, plane_pose)}), frames={})
    for stem in anchors:
        index = train.index(stem)
        z, pose = source(stem)
        ref = np.load(SCENE/"depth"/(stem+".npy"))
        _, rebuilt, _, _ = project_frame(raw(stem), np.zeros((meta["h"], meta["w"], 3), np.uint8),
                                         rays, t, rgb_grid)
        rebuilt[(rebuilt < .3) | (rebuilt > 6.)] = 0
        if not np.array_equal(rebuilt, ref):
            raise ValueError(f"Frozen RGB-grid reference differs at {stem}")
        outputs = {"native": (z, k, pose), "frozen_rgb": (ref, rgb_k, poses[stem])}
        by_case = dict(rgb_reference_exact=True)
        for key, view_ids in (("self", [stem]),
                              ("neighborhood", train[max(0, index-4):index+5])):
            by_case[key] = evaluate(view_ids, source, intr, outputs)
            gc.collect()
        result["frames"][stem] = by_case
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--full-mesh"]):
        raise SystemExit("usage: capture_tsdf_closure_check.py [--full-mesh]")
    main(bool(sys.argv[1:]))
