"""Check neighboring training observations in the native depth camera, without fusing a mesh."""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def native_observation(packet, frame, selection, depth_to_rgb):
    name = Path(frame["file_path"]).stem
    with np.load(packet / selection[name]["depth"]["file"], allow_pickle=False) as archive:
        depth = archive["depth"].astype(np.float64) / 1000
    depth_to_world = np.asarray(frame["transform_matrix"]) @ np.diag([1, -1, -1, 1]) @ depth_to_rgb
    return depth, depth_to_world


def count_pair(source, target, k, yy, xx):
    depth, source_pose = source
    reference, target_pose = target
    z = depth[yy, xx]
    valid = (z >= 0.3) & (z <= 6)
    z, x, y = z[valid], xx[valid], yy[valid]
    points = np.stack(((x-k["cx"]) / k["fx"] * z, (y-k["cy"]) / k["fy"] * z, z), axis=1)
    relative = np.linalg.solve(target_pose, source_pose)
    points = points @ relative[:3, :3].T + relative[:3, 3]
    points = points[points[:, 2] > 0]
    z = points[:, 2]
    u = np.rint(k["fx"] * points[:, 0] / z + k["cx"]).astype(int)
    v = np.rint(k["fy"] * points[:, 1] / z + k["cy"]).astype(int)
    inside = (u >= 0) & (u < k["width"]) & (v >= 0) & (v < k["height"])
    pixel, z = (v * k["width"] + u)[inside], z[inside]
    # Keep the front sample when more than one projected sample reaches a target pixel.
    index = np.lexsort((z, pixel))
    unique_pixels, first = np.unique(pixel[index], return_index=True)
    reference_z = reference.ravel()[unique_pixels]
    overlap = (reference_z >= 0.3) & (reference_z <= 6)
    errors = z[index[first[overlap]]] - reference_z[overlap]
    return {
        "source_valid_samples": len(x), "target_pixels_with_projection": len(unique_pixels),
        "shared_valid_pixels": len(errors), "within_10cm": int((np.abs(errors) <= 0.1).sum()),
        "in_front_over_10cm": int((errors < -0.1).sum()),
        "behind_over_10cm": int((errors > 0.1).sum()), "absolute_error_sum_m": float(np.abs(errors).sum()),
    }


def summarize(counts):
    totals = {key: sum(row[key] for row in counts) for key in counts[0]}
    n = totals["shared_valid_pixels"]
    return dict(directed_pairs=len(counts), **totals,
                fraction_within_10cm_of_shared=totals["within_10cm"] / n if n else None,
                mae_m_on_shared=totals["absolute_error_sum_m"] / n if n else None)


def revisit_pairs(frames, selection):
    matrices = np.array([f["transform_matrix"] for f in frames])
    centers = matrices[:, :3, 3]
    times = np.array([int(selection[Path(f["file_path"]).stem]["actual_depth_timestamp_ns"])
                      for f in frames], dtype=np.int64)
    for i in range(len(frames) - 1):
        later = np.flatnonzero((times - times[i] >= 10_000_000_000)
                               & (np.linalg.norm(centers - centers[i], axis=1) <= 0.5))
        rotations = matrices[later, :3, :3]
        traces = (rotations * matrices[i, :3, :3]).sum(axis=(1, 2))
        angles = np.degrees(np.arccos(np.clip((traces - 1) / 2, -1, 1)))
        for j in later[angles <= 15]:
            yield i, int(j)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("transforms", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = read_json(args.transforms)
    calibration = read_json(args.packet / "calibration.json")
    k = calibration["cameras"]["depth"]
    if np.any(k["D_sdk_k1_k2_k3_k4_k5_k6_p1_p2"]):
        raise ValueError("This native-grid diagnostic requires zero depth distortion")
    extrinsic = next(e for e in calibration["static_extrinsics"]
                     if (e["source_sensor"], e["destination_sensor"]) == ("depth", "color"))
    depth_to_rgb = np.eye(4)
    depth_to_rgb[:3, :3] = extrinsic["R"]
    depth_to_rgb[:3, 3] = np.asarray(extrinsic["translation_mm"]) / 1000
    selection = {r["id"]: r for r in read_json(args.packet / "selection.json")}
    train = set(meta["train_filenames"])
    frames = sorted((f for f in meta["frames"] if f["file_path"] in train),
                    key=lambda f: int(selection[Path(f["file_path"]).stem]["actual_depth_timestamp_ns"]))
    if {f["file_path"] for f in frames} != train or len(frames) != len(train) or len(frames) < 2:
        raise ValueError("Missing or duplicate training observations")
    yy, xx = np.mgrid[4:k["height"]:8, 4:k["width"]:8]
    yy, xx = yy.ravel(), xx.ravel()
    checks, selected_pairs = [], []
    previous = native_observation(args.packet, frames[0], selection, depth_to_rgb)
    for previous_frame, frame in zip(frames, frames[1:]):
        current = native_observation(args.packet, frame, selection, depth_to_rgb)
        forward = count_pair(previous, current, k, yy, xx)
        backward = count_pair(current, previous, k, yy, xx)
        checks.extend((forward, backward))
        left, right = (int(Path(f["file_path"]).stem[-5:]) for f in (previous_frame, frame))
        if 1290 <= left <= 1307 or 1510 <= left <= 1516:
            relative_rgb = np.linalg.solve(previous_frame["transform_matrix"], frame["transform_matrix"])
            rotation_cosine = (np.trace(relative_rgb[:3, :3]) - 1) / 2
            selected_pairs.append(dict(from_train_id=left, to_train_id=right,
                                       rgb_center_displacement_m=float(np.linalg.norm(relative_rgb[:3, 3])),
                                       rotation_deg=float(np.degrees(np.arccos(np.clip(rotation_cosine, -1, 1)))),
                                       **summarize([forward, backward])))
        previous = current

    @lru_cache(maxsize=8)
    def load_view(i):
        return native_observation(args.packet, frames[i], selection, depth_to_rgb)

    revisits, participating_views = [], set()
    for i, j in revisit_pairs(frames, selection):
        participating_views.update([i, j])
        a, b = load_view(i), load_view(j)
        revisits.extend([count_pair(a, b, k, yy, xx), count_pair(b, a, k, yy, xx)])
    result = dict(train_frames=len(frames), stride_px=8, offset_xy_px=[4, 4],
                  reference="raw native-camera optical Z at nearest target pixel, both measurements restricted to [0.3,6] m",
                  pairing="time-adjacent training observations, both directions; closest sampled source at each target pixel",
                  total=summarize(checks), previously_flagged_neighborhoods=selected_pairs,
                  revisits=dict(criteria="All pairs separated by at least 10 s, RGB centers within 0.5 m, rotations within 15 deg",
                                train_views_with_partner=len(participating_views),
                                stats=summarize(revisits) if revisits else None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
