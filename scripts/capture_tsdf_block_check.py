"""Recompute four small blocks of the frozen B2 native-depth TSDF with NumPy."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / ".local/capture-full/20261119-full-5hz-3982cb26"
RUN = "capture-full-quality-v2-20261121-v1"
REPORT = ROOT / "reports/capture-quality-review/2026-11-22/tsdf_block_check.json"
LAUNCH = ROOT / "reports/capture-full-quality-v2/launch-20261122/sensor_tsdf.json"
VOXEL = 0.01
SIDE = 16
TRUNC = 0.04
BLOCK = VOXEL * SIDE
CASES = [(120, 67, 355), (600, 243, 339), (900, 115, 323), (1700, 435, 275)]
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def native_image(packet, row):
    with np.load(packet / row["depth"]["file"], allow_pickle=False) as record:
        return record["depth"].astype(np.float32) / 1000.0


def back_project(x, y, z, k):
    fx, fy, cx, cy = k
    return np.stack(((x - cx) * z / fx, (y - cy) * z / fy, z), axis=-1)


def project(points, k):
    fx, fy, cx, cy = k
    return points[:, :2] / points[:, 2:3] * [fx, fy] + [cx, cy]


def frame_id(number):
    return f"frame_{number:05d}"


def block_queries(point):
    origin = np.floor(point / BLOCK) * BLOCK
    centers = (np.indices((SIDE,) * 3).reshape(3, -1).T + 0.5) * VOXEL + origin
    corners = np.array(list(itertools.product([-TRUNC, BLOCK + TRUNC], repeat=3))) + origin
    return origin, centers, corners


def search_rectangle(camera_corners, k, shape):
    # Only positive, in-range depth points can activate a block.
    z = camera_corners[:, 2]
    if z.max() < 0.3 or z.min() > 6.0:
        return None
    h, w = shape
    if z.min() <= 0:
        return 0, 0, w - 1, h - 1
    uv = project(camera_corners, k)
    x0, y0 = np.maximum(0, np.floor(uv.min(axis=0))).astype(int)
    x1, y1 = np.minimum([w - 1, h - 1], np.ceil(uv.max(axis=0))).astype(int)
    if x1 >= x0 and y1 >= y0:
        return x0, y0, x1, y1
    return None


def block_touched(image, rect, z_range, pose, origin, k):
    # Open3D activates a block if any observed point's ±sdf_trunc box overlaps it.
    x0, y0, x1, y1 = rect
    crop = image[y0:y1 + 1, x0:x1 + 1]
    lo, hi = z_range
    valid = (crop >= 0.3) & (crop <= 6.0) & (crop >= lo - 1e-6) & (crop <= hi + 1e-6)
    y, x = np.nonzero(valid)
    z = crop[y, x]
    xyz = back_project(x + x0, y + y0, z, k) @ pose[:3, :3].T + pose[:3, 3]
    return np.any(np.all((xyz >= origin - TRUNC) & (xyz < origin + BLOCK + TRUNC), axis=1))


def signed_updates(xyz, image, pose, k):
    camera = (xyz - pose[:3, 3]) @ pose[:3, :3]
    positive = np.flatnonzero(camera[:, 2] > 0)
    uv = project(camera[positive], k) + 0.5
    h, w = image.shape
    valid = ((uv >= 0.0001) & (uv < [w - 0.0001, h - 0.0001])).all(axis=1)
    ids = positive[valid]
    x, y = uv[valid].astype(int).T
    depth = image[y, x]
    ok = (depth >= 0.3) & (depth <= 6.0)
    ids, x, y = ids[ok], x[ok], y[ok]
    fx, fy, cx, cy = k
    ray_length = np.sqrt(1.0 + ((x - cx) / fx) ** 2 + ((y - cy) / fy) ** 2)
    return ids, (depth[ok] - camera[ids, 2]) * ray_length


def field_summary(total, weight):
    field = (total / np.maximum(weight, 1)).reshape((SIDE,) * 3)
    live = weight.reshape(field.shape) > 0
    crossing = 0
    for axis in range(field.ndim):
        lo = [slice(None)] * 3
        hi = lo.copy()
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        crossing += int(np.count_nonzero(live[lo] & live[hi] & (field[lo] * field[hi] < 0)))
    return {
        "observed_voxels": int(live.sum()),
        "min_tsdf": float(field[live].min()) if live.any() else None,
        "max_tsdf": float(field[live].max()) if live.any() else None,
        "observed_exact_zero_voxels": int((live & (field == 0)).sum()),
        "internal_grid_edges_with_sign_change": crossing,
    }


def point_summary(samples):
    sdf = np.asarray(samples)
    used = sdf > -TRUNC
    return {
        "valid_measurements_in_touched_blocks": len(sdf),
        "behind_measured_surface_skipped": int(np.count_nonzero(sdf <= -TRUNC)),
        "within_truncation_band": int(np.count_nonzero(np.abs(sdf) < TRUNC)),
        "free_space_positive_saturated": int(np.count_nonzero(sdf >= TRUNC)),
        "mean_accepted_tsdf": float(np.minimum(1, sdf[used] / TRUNC).mean()) if used.any() else None,
    }


def mesh_blocks(path, regions, receipt):
    # Scan all vertices of the frozen native TSDF PLY in bounded RAM.
    with path.open("rb") as stream:
        header = []
        for line in stream:
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        offset = stream.tell()
    count = int(next(line.split()[-1] for line in header if line.startswith("element vertex ")))
    anchor = header.index(f"element vertex {count}")
    properties = [f"property double {p}" for p in ("x", "y", "z", "nx", "ny", "nz")]
    if (header[1] != "format binary_little_endian 1.0"
            or header[anchor + 1:anchor + 7] != properties
            or not header[anchor + 7].startswith("element face ")
            or count != receipt["vertices"] or path.stat().st_size != receipt["mesh"]["bytes"]):
        raise ValueError("Unexpected native TSDF PLY structure or size")
    mesh = np.memmap(path, mode="r", dtype="<f8", offset=offset, shape=(count, 6))
    for region in regions:
        region["vertices_in_block"] = 0
        region["min_dist_sq"] = np.inf
    for i in range(0, count, 250000):
        xyz = mesh[i:i + 250000, :3]
        for region in regions:
            origin, point = region["origin"], region["point"]
            inside = np.all((xyz >= origin) & (xyz < origin + BLOCK), axis=1)
            region["vertices_in_block"] += int(inside.sum())
            delta = xyz - point
            region["min_dist_sq"] = min(region["min_dist_sq"], np.einsum("ij,ij->i", delta, delta).min())


def diagnose(capture):
    packet = capture / "transfer"
    artifacts = capture / RUN / "artifacts"
    transforms_path, calibration_path = artifacts / "inputs/transforms.json", packet / "calibration.json"
    meta, cal = read(transforms_path), read(calibration_path)
    rows = {row["id"]: row for row in read(packet / "selection.json")}
    native = cal["cameras"]["depth"]
    k = [native[x] for x in ("fx", "fy", "cx", "cy")]
    shape = (native["height"], native["width"])
    ext = next(e for e in cal["static_extrinsics"]
               if (e["source_sensor"], e["destination_sensor"]) == ("depth", "color"))
    sensor_to_rgb = np.eye(4)
    sensor_to_rgb[:3, :3] = ext["R"]
    sensor_to_rgb[:3, 3] = np.asarray(ext["translation_mm"]) / 1000.0
    poses = {Path(f["file_path"]).stem: np.asarray(f["transform_matrix"]) @ FLIP @ sensor_to_rgb
             for f in meta["frames"]}
    train = [Path(p).stem for p in meta["train_filenames"]]
    if len(set(train)) != 1478 or len(train) != 1478:
        raise ValueError("Expected the original 1,478 primary training frames")
    regions = []
    for frame, x, y in CASES:
        name = frame_id(frame)
        if name not in train:
            raise ValueError(f"Probe is outside training split: {name}")
        image = native_image(packet, rows[name])
        z = float(image[y, x])
        variation = float(np.ptp(image[y-3:y+4, x-3:x+4]))
        if not (0.6 < z < 4 and variation < 0.035):
            raise ValueError(f"Source depth differs from freeze: {name}")
        pose = poses[name]
        point = back_project(x, y, z, k) @ pose[:3, :3].T + pose[:3, 3]
        origin, centers, corners = block_queries(point)
        regions.append(dict(name=name, pixel=[x, y], z=z, variation=variation, point=point, origin=origin,
                            centers=centers, corners=corners, total=np.zeros(len(centers)),
                            weight=np.zeros(len(centers), dtype=int), local_total=np.zeros(len(centers)),
                            local_weight=np.zeros(len(centers), dtype=int), local=[], far=[], touched=0))
    for name in train:
        image = None
        pose = poses[name]
        for region in regions:
            corners = (region["corners"] - pose[:3, 3]) @ pose[:3, :3]
            rect = search_rectangle(corners, k, shape)
            if rect is None:
                continue
            if image is None:
                image = native_image(packet, rows[name])
            if not block_touched(image, rect, (corners[:, 2].min(), corners[:, 2].max()), pose, region["origin"], k):
                continue
            region["touched"] += 1
            local = abs(int(rows[name]["actual_depth_timestamp_ns"])
                        - int(rows[region["name"]]["actual_depth_timestamp_ns"])) <= 2_000_000_000
            ids, sdf = signed_updates(region["centers"], image, pose, k)
            ok = sdf > -TRUNC
            accepted, values = ids[ok], np.minimum(1, sdf[ok] / TRUNC)
            region["total"][accepted] += values
            region["weight"][accepted] += 1
            if local:
                region["local_total"][accepted] += values
                region["local_weight"][accepted] += 1
            _, at_source = signed_updates(region["point"][None], image, pose, k)
            region["local" if local else "far"].extend(at_source)
    receipt = read(LAUNCH)
    mesh_blocks(artifacts / "meshes/sensor_tsdf.ply", regions, receipt)
    probes = []
    for r in regions:
        probes.append(dict(frame=r["name"], native_xy=r["pixel"], native_Z_m=r["z"],
                           native_patch_7x7_depth_peak_to_peak_m=r["variation"], block_origin_world_m=r["origin"].tolist(),
                           world_point_m=r["point"].tolist(), touched_train_frames=r["touched"],
                           full_field=field_summary(r["total"], r["weight"]),
                           local_field_within_2s=field_summary(r["local_total"], r["local_weight"]),
                           full_mesh_vertices_in_block=r["vertices_in_block"],
                           nearest_full_mesh_vertex_m=float(np.sqrt(r["min_dist_sq"])),
                           point_observations=dict(within_2s=point_summary(r["local"]), outside_2s=point_summary(r["far"]),
                                                   all=point_summary(r["far"] + r["local"]))))
    return {
        "source": RUN, "training_frames_checked": len(train), "local_window": "Within 2 s of true native depth device time, inclusive.",
        "selection": "Four training frames selected for diagnostic coverage. In each, a source point was picked after examining the native TSDF, from a 16-pixel-stride grid starting at (35,35), with 0.6<Z<4 m, 7x7 peak-to-peak depth <0.035 m and nearest every-sixth mesh vertex >0.2 m. Failures deliberately selected; no population inference.",
        "voxel_length_m": VOXEL, "sdf_trunc_m": TRUNC, "voxels_per_block_side": SIDE,
        "method": "Apply the native sensor K and delivered corrected OpenGL RGB poses, depth-to-RGB calibration and mm-to-m conversion. Test block activation against all valid native points inside the block expanded by truncation. For every activated block, project every voxel center, sample the nearest native pixel, then apply Open3D's projective signed distance update. All 1478 training frames participate in actual file order. Positive means free space. A TSDF surface requires a sign crossing.",
        "precision_and_scope": "Independent NumPy implementation uses float64 projection/updates and float32 input depths; Open3D uses float32 internally. No complete volume is reconstructed. Counts cover internal edges of each block only. Native pixel locations and optical Z are used. Point observations are evaluated separately at the exact source point. Mesh vertex counts scan all original XYZ records; normals, face intersections and direction of normal are not used. Nearest vertex distance is not a ray depth or nearest-triangle distance.",
        "inputs": {"transforms_sha256": hashlib.sha256(transforms_path.read_bytes()).hexdigest(),
                   "native_calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                   "sensor_mesh_sha256_from_frozen_receipt": receipt["mesh"]["sha256"],
                   "native_sensor_build": "scripts/capture_quality_tsdf.py, sensor() and integrate()",
                   "open3d_block_activation": "https://github.com/isl-org/Open3D/blob/v0.18.0/cpp/open3d/pipelines/integration/ScalableTSDFVolume.cpp#L63-L91",
                   "open3d_voxel_update": "https://github.com/isl-org/Open3D/blob/v0.18.0/cpp/open3d/pipelines/integration/UniformTSDFVolume.cpp#L381-L415"},
        "probes": probes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args()
    result = diagnose(args.capture)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
