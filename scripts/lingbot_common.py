"""Shared camera conventions, frozen input paths and atomic evidence writers."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/lingbot_depth_map_protocol.json"
RUN = ROOT / ".local/runs/lingbot-depth-map-20261128-v1"
SRC = RUN / "source"
P0 = RUN / "prepare"
REPORT = ROOT / "reports/lingbot-depth-map/20261128-v1"
SEQ = ("short_train_256", "full_train_1478", "full_all_1846_exploratory")
HARD = ("frame_01304", "frame_01305", "frame_01514")
FLIP = np.diag([1., -1., -1., 1.])


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf8", delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        f.write("\n")
    os.replace(f.name, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def member(path):
    return dict(path=str(path), bytes=Path(path).stat().st_size, sha256=sha(path))


def masks(z, canvas):
    from capture_rectify import edges
    observed = canvas & np.isfinite(z) & (z > 0)
    primary = observed & (z >= .3) & (z <= 6)
    band = cv2.dilate(edges(z).astype("uint8"), np.ones((5, 5), np.uint8)).astype(bool)
    return observed, primary, band, primary & ~band


def depth(row):
    return np.load(SRC / row["depth"], allow_pickle=False)


def rgb(row):
    image = cv2.imread(str(SRC / row["rgb"]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise IOError(row["rgb"])
    return image


def render_z(z, ks, source_pose, kt, target_pose, target_shape):
    """Forward-project valid optical Z, keeping the nearest positive target-axis Z."""
    y, x = np.nonzero(np.isfinite(z) & (z > 0))
    d = z[y, x]
    points = np.column_stack(((x - ks[0, 2]) * d / ks[0, 0],
                              (y - ks[1, 2]) * d / ks[1, 1], d))
    transform = np.linalg.solve(target_pose, source_pose)
    points = points @ transform[:3, :3].T + transform[:3, 3]
    points = points[points[:, 2] > 1e-6]
    pz = points[:, 2]
    u = np.rint(points[:, 0] * kt[0, 0] / pz + kt[0, 2]).astype(np.int64)
    v = np.rint(points[:, 1] * kt[1, 1] / pz + kt[1, 2]).astype(np.int64)
    h, w = target_shape
    inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    out = np.full(h * w, np.inf, dtype=np.float32)
    np.minimum.at(out, v[inside] * w + u[inside], pz[inside])
    out[~np.isfinite(out)] = 0
    return out.reshape(h, w)


def by_id():
    manifest = read(P0 / "input_manifest.json")
    return manifest, {r["id"]: r for r in manifest["frames"]}


def require(condition, message):
    if not condition:
        raise ValueError(message)
