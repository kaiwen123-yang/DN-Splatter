"""Freeze native-Z build tolerances from reciprocal, interior primary training observations."""
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np

from capture_native_neighbor_check import revisit_pairs
from capture_surface_submaps import B2, FLIP, PACKET, RUN, camera_parameters, check, read, sha, write


BINS = np.array([.3, 1., 2., 4., 6.])


def edge_distance(z):
    valid = np.isfinite(z) & (z >= .3) & (z <= 6)
    edge = ~valid
    for axis in (0, 1):
        a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        discontinuity = valid[a] & valid[b] & (np.abs(z[a] - z[b]) > np.maximum(.05, .03 * np.minimum(z[a], z[b])))
        edge[a] |= discontinuity
        edge[b] |= discontinuity
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return cv2.distanceTransform((~edge).astype("u1"), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def reciprocal(source, target, k):
    z0, d0, pose0 = source
    z1, d1, pose1 = target
    y, x = np.nonzero((d0[4::8, 4::8] > 2))
    y, x = 4 + 8 * y, 4 + 8 * x
    z = z0[y, x]
    p = np.column_stack(((x-k["cx"]) * z/k["fx"], (y-k["cy"]) * z/k["fy"], z))
    rel = np.linalg.solve(pose1, pose0)
    q = p @ rel[:3, :3].T + rel[:3, 3]
    def pixel(q):
        return np.column_stack((k["fx"] * q[:, 0], k["fy"] * q[:, 1])) / q[:, 2, None] + [k["cx"], k["cy"]]
    in_range = (q[:, 2] >= .3) & (q[:, 2] <= 6)
    q, y, x = q[in_range], y[in_range], x[in_range]
    ux, vy = np.rint(pixel(q)).astype(int).T
    inside = (ux >= 0) & (ux < z1.shape[1]) & (vy >= 0) & (vy < z1.shape[0])
    q, x, y, ux, vy = (v[inside] for v in (q, x, y, ux, vy))
    ij = vy * z1.shape[1] + ux
    order = np.lexsort((q[:, 2], ij))
    closest = order[np.unique(ij[order], return_index=True)[1]]
    keep = closest[d1[vy[closest], ux[closest]] > 2]
    q, x, y, ux, vy = (v[keep] for v in (q, x, y, ux, vy))
    measured = z1[vy, ux]
    dest = np.column_stack(((ux-k["cx"]) * measured/k["fx"], (vy-k["cy"]) * measured/k["fy"], measured))
    back = (dest - rel[:3, 3]) @ rel[:3, :3]
    close = (back[:, 2] > 0) & (np.linalg.norm(pixel(back) - np.column_stack((x, y)), axis=1) <= 1.)
    return measured[close], (q[:, 2] - measured)[close]


def derive(label):
    scene = (B2 if label == "B2" else RUN) / "inputs"
    output = RUN / "export" / label
    meta = read(scene / "transforms.json")
    source = {r["id"]: r for r in read(PACKET / "selection.json")}
    frames = {r["file_path"]: r for r in meta["frames"]}
    train = [frames[name] for name in meta["train_filenames"]]
    cal = read(PACKET / "calibration.json")
    k = cal["cameras"]["depth"]
    _, _, rel = camera_parameters(meta, cal)
    cv2.setNumThreads(1)

    @lru_cache(maxsize=16)
    def load(index):
        frame = train[index]
        row = source[Path(frame["file_path"]).stem]
        with np.load(PACKET / row["depth"]["file"]) as packet:
            z = packet["depth"].astype(np.float32) / 1000.
        return z, edge_distance(z), np.asarray(frame["transform_matrix"]) @ FLIP @ rel

    pairs = sorted(set((i, i+1) for i in range(len(train)-1)) | set(revisit_pairs(train, source)))
    arrays = [[] for _ in range(4)]
    pair_counts = np.zeros(4, dtype=int)
    for i, j in pairs:
        has = np.zeros(4, dtype=bool)
        for a, b in ((i, j), (j, i)):
            z, delta = reciprocal(load(a), load(b), k)
            ix = np.minimum(np.searchsorted(BINS[1:], z, side="right"), 3)
            for band in np.unique(ix):
                arrays[band].append(delta[ix == band])
                has[band] = True
        pair_counts += has
    bins = []
    for i, vals in enumerate(arrays):
        residual = np.concatenate(vals) if vals else np.empty(0)
        median = float(np.median(residual)) if residual.size else None
        mad = float(np.median(np.abs(residual-median))) if residual.size else None
        sufficient = len(residual) >= 1000 and pair_counts[i] >= 10
        bins.append(dict(range_m=BINS[i:i+2].tolist(), samples=len(residual), pairs=int(pair_counts[i]),
                         median_signed_m=median, MAD_m=mad, fallback=not sufficient,
                         empirical_term_m=2.5 * 1.4826 * mad if sufficient else 0.))
    check(len(train) == 1478 and len(pairs) >= 1477, "Training repeatability population")
    write(output / "tolerances.json", dict(model=label, camera_sha256=sha(scene / "transforms.json"),
          bins=bins, train_pairs=len(pairs), sample_stride_px=8, sample_offset_px=4, reciprocal_max_px=1.,
          edges="Interior distance >2 native pixels; nearest projected Z per target pixel",
          pair_rule="Both directions of time neighbors and all revisits >=10 s, <=0.5 m, <=15 deg",
          build_tolerance="min(.20,max(.05,.03*reference_Z,empirical_term_m))",
          evaluation_tolerance="max(.05,.03*reference_Z)"))


if __name__ == "__main__":
    if sys.argv[1:] not in (["B2"], ["B3"]):
        raise SystemExit("usage: capture_surface_repeatability.py B2|B3")
    derive(sys.argv[1])
