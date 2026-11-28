"""Consume frozen per-frame treatment packets and add a separate directional DN loss."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_id(name):
    stem = Path(str(name)).stem
    if stem.startswith("frame_"):
        stem = stem[6:]
    if not stem.isdecimal():
        raise ValueError(f"unrecognized training ID {name}")
    return int(stem)


class Packets:
    def __init__(self, manifest, names, shape, cache, boundary=False):
        self.manifest = manifest
        self.root = manifest.parent.resolve()
        self.shape = tuple(shape)
        self.boundary = boundary
        self.cache = cache / digest(manifest)
        self.cache.mkdir(parents=True, exist_ok=True)
        doc = json.loads(manifest.read_bytes())
        if doc.get("scope") != "train" or doc.get("units") != "m" or tuple(doc["shape"]) != self.shape:
            raise ValueError(f"treatment scope, units or resolution differs: {manifest}")
        rows = {frame_id(row["id"]): row for row in doc["frames"]}
        ids = list(map(frame_id, names))
        if len(rows) != len(doc["frames"]) or len(set(ids)) != len(ids) or set(ids) != set(rows):
            raise ValueError("treatment frame membership differs from training set")
        self.rows = [rows[i] for i in ids]
        self.count = 0

    def get(self, index):
        """Transcode only selected pixels once. Packed NPZs retain the full audit data."""
        import fcntl
        row = self.rows[index]
        stem = self.cache / f"{index:05d}"
        destination = stem.with_suffix(".npy")
        if not destination.exists():
            with stem.with_suffix(".lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not destination.exists():
                    self._convert(row, destination)
        return np.load(destination, mmap_mode="c", allow_pickle=False)

    def _convert(self, row, destination):
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root) or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"invalid packet or hash for {row['id']}")
        with np.load(path, allow_pickle=False) as p:
            if self.boundary:
                idx = np.asarray(p["flat_index"])
                other_idx = np.asarray(p["background_flat_index"])
                n = len(idx)
                fields = ("foreground_z_m", "background_z_m", "fg_support_group_bits", "bg_support_group_bits")
                if (idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer)
                        or np.any((idx < 0) | (idx >= np.prod(self.shape)))
                        or other_idx.shape != (n,) or not np.issubdtype(other_idx.dtype,np.integer)
                        or np.any((other_idx<0)|(other_idx>=np.prod(self.shape))|(other_idx==idx))
                        or p["tangent_xy"].shape != (n,2)
                        or any(p[k].shape != (n,) for k in fields)):
                    raise ValueError("invalid sparse boundary geometry")
                fg, bg, fg_bits, bg_bits = (p[k] for k in fields)
                t = p["tangent_xy"]
                if (not np.isfinite(fg).all() or not np.isfinite(bg).all()
                        or not np.all((fg > 0) & (bg > fg)) or not np.isfinite(t).all()
                        or not np.allclose(np.linalg.norm(t, axis=1), 1, atol=.01)):
                    raise ValueError("boundary depth modes or tangents invalid")
                if np.any((fg_bits==0)|(bg_bits==0)) or len(np.unique(p["donor_groups"])) != len(p["donor_groups"]):
                    raise ValueError("one boundary side lacks independent donor support")
                # The two depths belong to different pixels. Ignore corner pixels
                # with incompatible tangents. A collinear tie keeps the larger gap.
                ids=np.r_[idx,other_idx]
                zs,rivals=np.r_[fg,bg],np.r_[bg,fg]
                ts=np.concatenate((t,t))
                order=np.lexsort((-np.abs(rivals-zs),ids))
                starts=np.r_[True,np.diff(ids[order])!=0] if len(order) else np.empty(0,bool)
                first=np.flatnonzero(starts)
                chosen=order[first]
                if len(order):
                    parent=np.cumsum(starts)-1
                    similarity=np.abs((ts[order]*ts[chosen[parent]]).sum(axis=1))
                    chosen=chosen[np.minimum.reduceat(similarity,first)>=.9]
                data = np.empty(len(chosen), dtype=[("index","<i4"),("mode","<f4"),("other","<f4"),("tangent","<f4",2)])
                data["index"],data["mode"],data["other"],data["tangent"]=ids[chosen],zs[chosen],rivals[chosen],ts[chosen]
            else:
                accepted, hole, z = p["accepted"], p["natural_hole"], p["selected_depth"]
                if (any(x.shape != self.shape for x in (accepted,hole,z))
                        or accepted.dtype != np.bool_ or hole.dtype != np.bool_
                        or np.any(accepted & ~hole)):
                    raise ValueError("accepted pixels conflict with full-range measurement lineage")
                idx = np.flatnonzero(accepted)
                target = z.ravel()[idx]
                if (not np.all(np.isfinite(target) & (target > 0))
                        or int(row["accepted_pixels"]) != len(idx)):
                    raise ValueError("invalid accepted depths or manifest count")
                data = np.empty(len(idx), dtype=[("index","<i4"),("z_m","<f4")])
                data["index"], data["z_m"] = idx, target
        from interactive_dn_cache import _save_array
        _save_array(destination, data)


def loss_at_pixels(pred, rgb, sensor, index, target_m, depth_scale, range_m, scene_mask=None):
    """B3 left/top log1p(abs(residual)) with an independent denominator per direction."""
    if pred.ndim != 3 or pred.shape[-1] != 1 or rgb.shape != (*pred.shape[:2],3) or sensor.shape != pred.shape:
        raise ValueError("inconsistent full-resolution hole loss tensors")
    idx = index.long().reshape(-1)
    z = target_m.reshape(-1)
    if idx.shape != z.shape or not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("invalid metric depth packet")
    value = pred.reshape(-1)[idx]
    seen = sensor.reshape(-1)[idx]
    valid = (torch.isfinite(z) & (z > 0) & (z >= range_m[0]) & (z <= range_m[1])
             & torch.isfinite(value) & (value > 0) & ~(torch.isfinite(seen) & (seen > 0)))
    if scene_mask is not None:
        valid &= scene_mask.reshape(-1)[idx].bool()
    idx, z, value = idx[valid], z[valid], value[valid]
    # An empty slice stays differentiable even if rejected pixels contain NaN/Inf.
    result = pred.reshape(-1)[:0].sum()
    residual = torch.log1p(torch.abs(value - z * depth_scale))
    h, w = pred.shape[:2]
    pixels = rgb.reshape(-1,3)
    for available, offset in ((idx % w < w-1, 1), (idx < (h-1)*w, w)):
        source = idx[available]
        weight = torch.exp(-torch.mean(torch.abs(pixels[source] - pixels[source + offset]), dim=-1))
        result = result + (weight * residual[available]).sum() / max(1, source.numel())
    return result, idx.numel()
