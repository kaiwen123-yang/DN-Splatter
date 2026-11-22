"""Native-Z visibility votes for B3 surfaces and depth supervision.

Vote counts use a deterministic greedy independent-view set ordered by group ID.
Input points are world coordinates; source_groups excludes self-certification.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np

from capture_surface_submaps import ROOT, PACKET, FLIP, camera_parameters, read, sha
from capture_surface_repeatability import edge_distance


class NativeVotes:
    def __init__(self, identity_path, tolerance_path, cache_frames=8):
        identity = read(Path(identity_path))
        tolerance = read(Path(tolerance_path))
        for key in ("cameras", "calibration", "selection"):
            entry = identity[key]
            if sha(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("identity hash mismatch: " + key)
        if tolerance["camera_sha256"] != identity["cameras"]["sha256"]:
            raise ValueError("tolerances use different cameras")
        meta = read(ROOT / identity["cameras"]["path"])
        calibration = read(ROOT / identity["calibration"]["path"])
        self.k = calibration["cameras"]["depth"]
        _, _, rel = camera_parameters(meta, calibration)
        frames = {f["file_path"]: f for f in meta["frames"]}
        source = {r["id"]: r for r in read(ROOT / identity["selection"]["path"])}
        self.groups = []
        for g in identity["groups"]:
            rows = []
            for name in sorted(g["names"]):
                pose = np.asarray(frames[name]["transform_matrix"], dtype=np.float64) @ FLIP @ rel
                rows.append((name, pose, PACKET / source[Path(name).stem]["depth"]["file"]))
            self.groups.append((g["id"], rows))
        self.groups.sort(key=lambda x: x[0])
        bins = tolerance["bins"]
        self.edges = np.array([b["range_m"][1] for b in bins])
        self.terms = np.array([b["empirical_term_m"] for b in bins])
        self.load = lru_cache(maxsize=cache_frames)(self._load)

    @staticmethod
    def _load(path):
        with np.load(path) as f:
            z = f["depth"].astype(np.float32) / 1000.
        return z, edge_distance(z)

    def observe(self, points, pose, path):
        """Return per-point category, edge distance and absolute Z residual.

        Categories: 0 unknown, 1 support, 2 free-space conflict, 3 occluded.
        Sampling is nearest native pixel, without depth interpolation.
        """
        points = np.asarray(points, dtype=np.float64)
        q = (points - pose[:3, 3]) @ pose[:3, :3]
        n = len(points)
        cat = np.zeros(n, np.uint8)
        distance = np.zeros(n, np.float32)
        residual = np.full(n, np.inf)
        valid = np.isfinite(q).all(axis=1) & (q[:, 2] >= .3) & (q[:, 2] <= 6.)
        idx = np.flatnonzero(valid)
        if not len(idx):
            return cat, distance, residual
        k = self.k
        u = np.rint(k["fx"] * q[idx, 0] / q[idx, 2] + k["cx"]).astype(np.int64)
        v = np.rint(k["fy"] * q[idx, 1] / q[idx, 2] + k["cy"]).astype(np.int64)
        inside = (u >= 0) & (u < k["width"]) & (v >= 0) & (v < k["height"])
        idx, u, v = idx[inside], u[inside], v[inside]
        if not len(idx):
            return cat, distance, residual
        z, edge = self.load(path)
        d = edge[v, u]
        measured = z[v, u]
        good = (d > 2) & np.isfinite(measured) & (measured >= .3) & (measured <= 6.)
        idx, d, measured = idx[good], d[good], measured[good]
        delta = q[idx, 2] - measured
        band = np.minimum(np.searchsorted(self.edges, measured, side="right"), len(self.terms)-1)
        tol = np.minimum(.20, np.maximum(np.maximum(.05, .03 * measured), self.terms[band]))
        cat[idx] = np.where(np.abs(delta) <= tol, 1, np.where(delta < 0, 2, 3))
        distance[idx] = d
        residual[idx] = np.abs(delta)
        return cat, distance, residual

    def vote(self, points, source_groups=None, chunk_size=4096):
        """Count independent S/C/occluded/unknown groups for every point.

        Within each group choose the observation furthest from an edge, then
        smallest absolute residual, then lexicographically smallest frame name.
        Unknown groups are counted separately; they cannot certify free space.
        Chunking is exact and does not change the selected observations.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or chunk_size < 1:
            raise ValueError("points must be N x 3 and chunk_size positive")
        sources = np.full(len(points), -1, dtype=np.int64) if source_groups is None else np.asarray(source_groups)
        if sources.shape != (len(points),):
            raise ValueError("source_groups must contain one group per point")
        result = {key: np.zeros(len(points), dtype=np.int32) for key in ("support", "conflict", "occluded", "unknown")}
        for start in range(0, len(points), chunk_size):
            p = points[start:start+chunk_size]
            src = sources[start:start+chunk_size]
            accepted_centers = [[] for _ in range(len(p))]
            accepted_rays = [[] for _ in range(len(p))]
            for gid, frames in self.groups:
                best_cat = np.zeros(len(p), np.uint8)
                best_dist = np.full(len(p), -1.)
                best_res = np.full(len(p), np.inf)
                centers = np.zeros((len(p), 3))
                for name, pose, path in frames:
                    cat, dist, res = self.observe(p, pose, path)
                    better = (cat > 0) & ((dist > best_dist) | ((dist == best_dist) & (res < best_res)))
                    best_cat[better], best_dist[better], best_res[better] = cat[better], dist[better], res[better]
                    centers[better] = pose[:3, 3]
                for i in np.flatnonzero(src != gid):
                    category = best_cat[i]
                    if category == 0:
                        result["unknown"][start+i] += 1
                        continue
                    ray = centers[i] - p[i]
                    norm = np.linalg.norm(ray)
                    if norm == 0:
                        result["unknown"][start+i] += 1
                        continue
                    ray /= norm
                    if any(np.linalg.norm(centers[i]-c) < .1 or np.dot(ray, r) > np.cos(np.deg2rad(3.))
                           for c, r in zip(accepted_centers[i], accepted_rays[i])):
                        continue
                    accepted_centers[i].append(centers[i].copy())
                    accepted_rays[i].append(ray)
                    result[("support", "conflict", "occluded")[category-1]][start+i] += 1
        return result

