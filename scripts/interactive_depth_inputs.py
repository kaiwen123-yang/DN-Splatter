"""Freeze the train-only hall observations and disjoint ten-second quality folds."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from capture_rectify import grid_setup
from lingbot_common import ROOT, read, require, save, sha


RUN = ROOT / ".local/runs/interactive-world-20261128-v1/a_depth"
OLD = ROOT / ".local/runs/lingbot-depth-map-20261128-v1"
MANIFEST = ROOT / "reports/lingbot-depth-map/20261128-v1/input_manifest.json"
SOURCE_EVIDENCE = ROOT / "reports/lingbot-depth-map/20261128-v1/sources.json"
PACKET_SHA = "99db85bab5c06622fa6cadbd97fb6298304bde4613fafbe4f4f15fad298b5216"
MODEL_SHA = read(SOURCE_EVIDENCE)["depth"]["weight"]["sha256"]
MODEL_REV = read(SOURCE_EVIDENCE)["depth"]["commit"]
PROTOCOL = ROOT / "configs/interactive_world_protocol.json"
MASKS = ("native_footprint_bundle", "contiguous_block", "geometric_boundary_band")
REPS = ("full_range_sparse", "primary_range_only", "footprint_aware")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def fold_rows(rows, source):
    start = rows[0]["rgb_timestamp_ns"]
    groups = np.array([(r["rgb_timestamp_ns"] - start) // 10_000_000_000 for r in rows])
    names = np.unique(groups)
    parents = {int(g): int(g) for g in names}

    def group_root(g):
        while parents[g] != g:
            g = parents[g]
        return g

    # The same acquisition or pixel payload may never have different fold roles.
    seen = {}
    for row, group in zip(rows, groups):
        meta = source[row["id"]]
        for key in ("actual_depth_timestamp_ns", "depth_pixels_sha256", "rgb_payload_sha256"):
            value = (key, meta[key])
            if value in seen:
                parents[group_root(int(group))] = group_root(seen[value])
            seen[value] = int(group)
    blocks = list(sorted({group_root(int(g)) for g in names}))
    order = np.random.default_rng(42).permutation(blocks)
    which = {int(g): i % 5 for i, g in enumerate(order)}
    folds = np.array([which[group_root(int(g))] for g in groups])
    require(len(set(folds)) == 5, "Ten-second/source groups cannot form five folds")
    poses = np.array([r["c2w_cv"] for r in rows])
    time = np.array([r["rgb_timestamp_ns"] for r in rows]) / 1e9
    centers = poses[:, :3, 3]
    rotations = poses[:, :3, :3].reshape(len(rows), 9)
    cos3 = np.cos(np.deg2rad(3.))
    for i, row in enumerate(rows):
        dist = np.linalg.norm(centers - centers[i], axis=1)
        facing = (rotations @ rotations[i] - 1.) / 2.
        adjacent = (np.abs(time - time[i]) <= 2.) | ((dist < .1) & (facing >= cos3))
        touching = {int(f) for f in folds[adjacent & (folds != folds[i])]}
        fold = int(folds[i])
        row.update(group=int(groups[i]), fold=fold, near_other_folds=sorted(touching),
                   role_for_outer=[
                       "outer" if k == fold else
                       ("excluded_near_outer" if k in touching else "calibration") if fold == (k+1) % 5 else
                       ("excluded_near_calibration_or_outer" if touching & {k, (k+1) % 5} else "fit")
                       for k in range(5)])

    for i, row in enumerate(rows):
        dist = np.linalg.norm(centers - centers[i], axis=1)
        facing = (rotations @ rotations[i] - 1.) / 2.
        # Fixed camera-only ranking. Pixels, errors and held-out labels do not choose sources.
        # Each donor is exclusive to one fold and separated from every other fold.
        eligible = [j for j, donor in enumerate(rows)
                    if donor["fold"] == row["fold"] and donor["group"] != row["group"]
                    and not donor["near_other_folds"] and not donor["suspect_reference_pose"]
                    and dist[j] >= .1 and facing[j] > np.cos(np.deg2rad(65.))]
        eligible.sort(key=lambda j: (abs(dist[j] - .75) + 2 * (1 - facing[j]), rows[j]["id"]))
        donors = []
        for j in eligible:
            if all(rows[j]["group"] != rows[m]["group"] for m in donors):
                donors.append(j)
            if len(donors) == 8:
                break
        # Generator and audit are distinct observations even when few groups are available.
        row["donors"] = [dict(id=rows[j]["id"], group=rows[j]["group"],
                              role="propose" if n % 3 == 2 else "verify") for n, j in enumerate(donors)]


def freeze(out, source_root, packet):
    manifest = read(MANIFEST)
    selection = source_root / "geometry/transfer/selection.json"
    calibration = source_root / "geometry/transfer/calibration.json"
    has_native = packet is not None
    if has_native:
        require(sha(packet / "manifest.json") == PACKET_SHA, "Unrecognized native packet manifest")
        files = {m["path"]: m for m in read(packet / "manifest.json")["files"]}
        for name in ("selection.json", "calibration.json"):
            require(sha(packet / name) == files[name]["sha256"], "Native metadata " + name)
        require(sha(selection) == files["selection.json"]["sha256"] and
                sha(calibration) == files["calibration.json"]["sha256"], "RGB/native pair or calibration identity")
    source = {r["id"]: r for r in read(selection)}
    rows = [dict(r) for r in manifest["frames"] if r["split"] == "train"]
    require(len(rows) == len({r["id"] for r in rows}) == 1478, "Frozen train membership")
    for row in rows:
        require(sha(source_root / row["rgb"]) == row["rgb_sha256"] and
                sha(source_root / row["depth"]) == row["depth_sha256"], "RGB/full-range Z identity " + row["id"])
        pair = source[row["id"]]
        require(int(pair["actual_rgb_timestamp_ns"]) == row["rgb_timestamp_ns"] and
                pair["requested_split"] == "train", "Camera identity " + row["id"])
        if has_native:
            item = pair["depth"]
            p = packet / item["file"]
            require(p.stat().st_size == files[item["file"]]["bytes"] and
                    sha(p) == item["sha256"] == files[item["file"]]["sha256"], "Native Z identity " + row["id"])
            row.update(native_path=item["file"], native_sha256=item["sha256"])
    fold_rows(rows, source)
    result = dict(protocol_sha256=sha(PROTOCOL), source_manifest_sha256=sha(MANIFEST),
                  selection_sha256=sha(selection), calibration_sha256=sha(calibration),
                  packet_manifest_sha256=PACKET_SHA if has_native else None,
                  source_root=str(source_root.resolve()), packet=str(packet.resolve()) if has_native else None,
                  K_cv=manifest["K_cv"], image_wh=manifest["image_wh"], masks=list(MASKS if has_native else MASKS[1:]),
                  representations=list(REPS if has_native else REPS[:2]),
                  blocked={} if has_native else {"native_footprint_bundle": "native observations unavailable",
                                                "footprint_aware": "native footprint unavailable"},
                  rows=rows, code_sha256=sha(Path(__file__)),
                  sample_cap_per_frame_mask=4096, sampling_seed=42,
                  source_rules="10s grouped with identical depth/time/RGB sources joined. Distinct fold near-duplicates"
                  " purged from fit/calibration. Donors stay inside target fold, distinct 10s groups; 0.1m, 3deg"
                  " per-ray baseline enforced before voting. Every third donor proposes, the others verify.")
    target = out / "freeze.json"
    if target.exists():
        require(read(target) == result, "Freeze changed; use a separate attempt")
    else:
        save(target, result)
    print("train frames", len(rows), "native", has_native, "fold role counts",
          [{role: sum(r["role_for_outer"][k] == role for r in rows)
            for role in ("fit", "calibration", "outer")} for k in range(5)], flush=True)


class Hall:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.freeze = read(self.directory / "freeze.json")
        self.rows = self.freeze["rows"]
        self.by_id = {r["id"]: r for r in self.rows}
        self.source = Path(self.freeze["source_root"])
        self.packet = Path(self.freeze["packet"]) if self.freeze["packet"] else None
        self.k = np.asarray(self.freeze["K_cv"])
        self.calibration = read(self.source / "geometry/transfer/calibration.json")
        rays, t, grids = grid_setup(self.calibration)
        self.rays, self.t, self.grid = rays, t, grids["1280"]
        self.canvas = self.grid[3]

    def image(self, row):
        im = cv2.imread(str(self.source / row["rgb"]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        require(im is not None and im.shape[:2] == self.canvas.shape, "RGB " + row["id"])
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def depth(self, row):
        return np.load(self.source / row["depth"], mmap_mode="r", allow_pickle=False)

    def native(self, row):
        if self.packet is None:
            return None
        with np.load(self.packet / row["native_path"], allow_pickle=False) as f:
            raw = f["depth"]
        require(raw.dtype == np.uint16 and raw.shape == self.rays.shape[:2], "Native Z shape/dtype")
        return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=OLD / "source")
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--out", type=Path, default=RUN)
    args = parser.parse_args()
    freeze(args.out, args.source, args.packet)
