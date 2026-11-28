"""Trace real RGB-D measurements, hide their descendants and project disjoint train views."""
from dataclasses import dataclass
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from interactive_depth_inputs import MASKS
from lingbot_common import require


SOURCE_NAMES = ("nearest", "footprint", "other_view_near", "other_view_far", "MDM")
COS3 = np.cos(np.deg2rad(3.))


def valid(z):
    return np.isfinite(z) & (z > 0)


def primary(z):
    return np.isfinite(z) & (z >= .3) & (z <= 6.)


def tolerance(z):
    return np.maximum(.05, .03 * z)


def atomic_npz(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        with ZipFile(f, "w", compression=ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for key, value in fields.items():
                with archive.open(key+".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)
    os.replace(f.name, path)


def zbuffer(shape, dst, Z, src):
    """Nearest positive target-axis Z, ties resolved by source index as in B3."""
    if len(dst):
        order = np.lexsort((src, Z, dst))
        order = order[np.r_[True, np.diff(dst[order]) != 0]]
    else:
        order = np.empty(0, np.intp)
    image = np.zeros(shape, np.float32)
    winner = np.full(shape, -1, np.int32)
    image.flat[dst[order]] = Z[order]
    winner.flat[dst[order]] = src[order]
    return image, winner


def project_native(hall, raw):
    sources = np.flatnonzero((raw.ravel() > 0) & (raw.ravel() < 65535))
    xyz = hall.rays.reshape(-1, 3)[sources] * (raw.ravel()[sources, None] * .001) + hall.t
    k = hall.k
    h, w = hall.canvas.shape
    u = np.rint(k[0, 0] * xyz[:, 0] / xyz[:, 2] + k[0, 2]).astype(int)
    v = np.rint(k[1, 1] * xyz[:, 1] / xyz[:, 2] + k[1, 2]).astype(int)
    inside = (xyz[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    dst = v[inside] * w + u[inside]
    xyz, sources = xyz[inside], sources[inside]
    keep = hall.canvas.ravel()[dst]
    dst, sources, Z = dst[keep], sources[keep], xyz[keep, 2]
    z, winner = zbuffer((h, w), dst, Z, sources)
    native_target = np.full(raw.size, -1, np.int32)
    native_target[sources] = dst
    occ = np.zeros((h, w), bool)
    behind = Z - z.ravel()[dst] > tolerance(z.ravel()[dst])
    occ.flat[dst[behind]] = True
    return z, winner, native_target, occ


class Footprint:
    """A continuous native cell, with four verified neighbours, projected at its own depth."""
    def __init__(self, hall):
        dep = hall.calibration["cameras"]["depth"]
        ext = next(e for e in hall.calibration["static_extrinsics"]
                   if (e["source_sensor"], e["destination_sensor"]) == ("depth", "color"))
        self.dep = dep
        self.R = np.asarray(ext["R"])
        self.t = hall.t
        self.k = hall.k
        y, x = np.indices(hall.canvas.shape, dtype=np.float64)
        color_rays = np.stack(((x - hall.k[0, 2]) / hall.k[0, 0],
                               (y - hall.k[1, 2]) / hall.k[1, 1], np.ones_like(x)), axis=-1)
        self.rays_in_native = color_rays @ self.R
        self.offset = -self.t @ self.R
        self.canvas = hall.canvas
        self.outside = self._outside_depth_fov()

    def _outside_depth_fov(self):
        masks = []
        for z in (.3, 6.):
            a = self.rays_in_native * z + self.offset
            u = self.dep["fx"] * a[..., 0] / a[..., 2] + self.dep["cx"]
            v = self.dep["fy"] * a[..., 1] / a[..., 2] + self.dep["cy"]
            masks.append(np.stack((u < -.5, u > self.dep["width"]-.5,
                                   v < -.5, v > self.dep["height"]-.5)) & (a[..., 2] > 0))
        # Ratios of affine depths are monotonic between positive endpoints.
        return (masks[0] & masks[1]).any(axis=0)

    def splat(self, raw, rays):
        d = raw.astype(np.float32) * .001
        good = (raw > 0) & (raw < 65535)
        continuous = np.zeros_like(good)
        center = d[1:-1, 1:-1]
        inner = good[1:-1, 1:-1].copy()
        for view in ((slice(None, -2), slice(1,-1)), (slice(2,None), slice(1,-1)),
                     (slice(1,-1), slice(None,-2)), (slice(1,-1), slice(2,None))):
            other = d[view]
            inner &= good[view] & (np.abs(center-other) <= tolerance(np.minimum(center, other)))
        continuous[1:-1, 1:-1] = inner
        sy, sx = np.nonzero(continuous)
        ids = sy * raw.shape[1] + sx
        D = raw[sy, sx].astype(np.float64) * .001
        xyz = rays[sy, sx] * D[:, None] + self.t
        h, w = self.canvas.shape
        # Forward projected quadrilateral bounds, then an exact inverse ray/cell test.
        lo = np.full((len(D), 2), np.inf)
        hi = -lo.copy()
        for dx, dy in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)):
            delta = self.R @ np.array([dx / self.dep["fx"], dy / self.dep["fy"], 0.])
            corner = xyz + D[:, None] * delta
            uv = corner[:, :2] / corner[:, 2:] * np.diag(self.k)[:2] + self.k[:2, 2]
            lo, hi = np.minimum(lo, uv), np.maximum(hi, uv)
        lo = np.maximum(np.ceil(lo), [0, 0]).astype(int)
        hi = np.minimum(np.floor(hi), [w-1, h-1]).astype(int)
        span = hi - lo + 1
        dest, values, parents = [], [], []
        if len(ids):
            for oy in range(max(0, int(span[:, 1].max()))):
                for ox in range(max(0, int(span[:, 0].max()))):
                    use = np.flatnonzero((span[:, 0] > ox) & (span[:, 1] > oy) & (xyz[:, 2] > 0))
                    u, v = lo[use, 0] + ox, lo[use, 1] + oy
                    ray = self.rays_in_native[v, u]
                    z = (D[use] - self.offset[2]) / ray[:, 2]
                    point = ray * z[:, None] + self.offset
                    du = self.dep["fx"] * point[:, 0] / D[use] + self.dep["cx"] - sx[use]
                    dv = self.dep["fy"] * point[:, 1] / D[use] + self.dep["cy"] - sy[use]
                    keep = self.canvas[v, u] & (z > 0) & (np.abs(du) <= .5) & (np.abs(dv) <= .5)
                    dest.append((v * w + u)[keep])
                    values.append(z[keep])
                    parents.append(ids[use[keep]])
        if not dest:
            return np.zeros((h,w), np.float32), np.full((h,w), -1, np.int32)
        return zbuffer((h,w), np.concatenate(dest), np.concatenate(values), np.concatenate(parents))


def edge_pairs(raw):
    """Two measured sides of geometric discontinuities on the registered lattice."""
    h, w = raw.shape
    pairs = []
    for dy, dx in ((0,1),(1,0)):
        for distance in (1,2):
            a = raw[:h-dy*distance, :w-dx*distance]
            b = raw[dy*distance:, dx*distance:]
            edges = primary(a) & primary(b) & (np.abs(a-b) > tolerance(np.minimum(a,b)))
            if distance == 2:
                edges &= ~valid(raw[dy:h-dy, dx:w-dx])
            y, x = np.nonzero(edges)
            first = (y*w+x).astype(np.int32)
            second = first + distance * (dy*w+dx)
            smaller = a[y,x] <= b[y,x]
            pairs.append((np.where(smaller, first, second), np.where(smaller, second, first),
                          np.array([-dy, dx], np.float32)))
    return pairs


def task_mask(fid, kind, full, native, native_target, sides):
    rng = np.random.default_rng(np.random.SeedSequence((42, int(fid.split("_")[-1]), MASKS.index(kind))))
    h, w = full.shape
    remove = np.zeros(full.shape, bool)
    native_hide = None
    if kind == "native_footprint_bundle":
        require(native is not None, "Native mask requires actual native rays")
        ny, nx = native.shape
        native_hide = np.zeros(native.shape, bool)
        # Four contiguous missing packets of adjacent depth scan lines, with fixed source IDs.
        for _ in range(4):
            y, x = int(rng.integers(ny-24+1)), int(rng.integers(nx-96+1))
            native_hide[y:y+24, x:x+96] = True
        projected = native_target[native_hide.ravel()]
        remove.flat[projected[projected >= 0]] = True
    elif kind == "contiguous_block":
        # A full contiguous 128x128 deletion on the original registration canvas.
        y, x = int(rng.integers(h-128+1)), int(rng.integers(w-128+1))
        remove[y:y+128, x:x+128] = True
    elif kind == "geometric_boundary_band":
        # Fixed strip covering both nearer and farther measured sides over the whole image.
        for fg, bg, _ in sides:
            remove.flat[fg] = True
            remove.flat[bg] = True
        remove = cv2.dilate(remove.astype(np.uint8), np.ones((5,5), np.uint8)).astype(bool)
    else:
        raise ValueError(kind)
    if native is not None:
        if native_hide is None:
            native_hide = np.zeros(native.shape, bool)
        within = native_target >= 0
        native_hide.flat[np.flatnonzero(within)] |= remove.ravel()[native_target[within]]
    score = remove & primary(full)
    return remove, native_hide, score


def nearest_map(z):
    if not np.any(valid(z)):
        return np.zeros_like(z), np.full(z.shape, np.inf, np.float32), np.full(z.shape, -1, np.int32)
    dist, index = distance_transform_edt(~valid(z), return_indices=True)
    src = index[0] * z.shape[1] + index[1]
    return z.ravel()[src], dist.astype(np.float32), src.astype(np.int32)


def reproject(d, k, pose, target, shape):
    """True source pixel for each winning reprojection, with B3 per-ray angle exclusion."""
    T = np.linalg.solve(target, pose)
    y, x = np.nonzero(primary(d))
    z = d[y, x].astype(np.float64)
    source_id = (y * d.shape[1] + x).astype(np.int32)
    xyz = np.column_stack(((x-k[0,2])*z/k[0,0], (y-k[1,2])*z/k[1,1], z))
    xyz = xyz @ T[:3,:3].T + T[:3,3]
    forward = xyz[:,2] > 1e-6
    xyz, source_id = xyz[forward], source_id[forward]
    norm = np.linalg.norm(xyz,axis=1)
    from_source = xyz-T[:3,3]
    dot = np.einsum("ij,ij->i", xyz, from_source)
    distinct = dot <= COS3 * norm * np.linalg.norm(from_source,axis=1)
    uv = np.rint(xyz[:,:2]/xyz[:,2:] * np.diag(k)[:2] + k[:2,2]).astype(np.int64)
    keep = distinct & (uv[:,0]>=0) & (uv[:,0]<shape[1]) & (uv[:,1]>=0) & (uv[:,1]<shape[0])
    return zbuffer(shape, uv[keep,1]*shape[1]+uv[keep,0], xyz[keep,2], source_id[keep])


@dataclass
class OtherViews:
    near: np.ndarray
    far: np.ndarray
    near_id: np.ndarray
    far_id: np.ndarray
    near_frame: np.ndarray
    far_frame: np.ndarray
    audit: list
    audit_bits: list
    donor_groups: np.ndarray


def other_views(hall, row):
    shape = hall.canvas.shape
    target = np.array(row["c2w_cv"])
    near = np.full(shape,np.inf,np.float32)
    far = np.zeros(shape,np.float32)
    near_id, far_id = [np.full(shape, -1, np.int32) for _ in range(2)]
    near_frame, far_frame = [np.full(shape, -1, np.int16) for _ in range(2)]
    audit, bit_for, centers = [], [], []
    y, x = np.indices(shape, dtype=np.float32)
    rays = np.stack(((x-hall.k[0,2])/hall.k[0,0], (y-hall.k[1,2])/hall.k[1,1],
                     np.ones_like(x)),axis=-1).astype(np.float32)
    for bit, donor in enumerate(row["donors"]):
        source = hall.by_id[donor["id"]]
        require(source["fold"] == row["fold"] and source["group"] != row["group"] and
                not source["near_other_folds"], "Cross-role donor")
        pose = np.array(source["c2w_cv"])
        z, ids = reproject(hall.depth(source), hall.k, pose, target, shape)
        if donor["role"] == "propose":
            which = int(source["id"].split("_")[-1])
            closer, farther = (z>0) & (z<near), z>far
            near[closer], near_id[closer], near_frame[closer] = z[closer], ids[closer], which
            far[farther], far_id[farther], far_frame[farther] = z[farther], ids[farther], which
        else:
            center = np.linalg.solve(target,pose)[:3,3].astype(np.float32)
            points = rays * z[...,None]
            direction = points - center
            for old_center, old_depth in centers:
                overlap = (z>0) & (old_depth>0)
                if np.linalg.norm(center-old_center)<.1:
                    z[overlap] = 0
                else:
                    u, v = direction[overlap], points[overlap] - old_center
                    duplicate = np.einsum("ij,ij->i",u,v) > COS3*np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
                    z.flat[np.flatnonzero(overlap)[duplicate]] = 0
            centers.append((center, z))
            audit.append(z)
            bit_for.append(bit)
    near[~np.isfinite(near)] = 0
    identical = far==near
    far[identical], far_id[identical], far_frame[identical] = 0,-1,-1
    return OtherViews(near,far,near_id,far_id,near_frame,far_frame,audit,bit_for,
                      np.array([d["group"] for d in row["donors"]],np.int16))
