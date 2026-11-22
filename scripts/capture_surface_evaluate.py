"""Evaluate frozen B3 mesh candidates against the original sensor-Z denominator.

The JSON manifest specifies scene, candidate meshes and output; see --help. Test
requires a freeze manifest binding each evaluated mesh by SHA256 before any test
metadata or depth is opened. All candidates use the same camera and pixels.
"""
import argparse
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import open3d as o3d

from capture_metric3d_common import B0_REPORT, selection, sha
from capture_metric3d_surface_eval import rays, stats, measures
from capture_quality_paths import check
from capture_s01_mesh import titled
from vr_room_run import CPU

EXPECTED = {"val": 29222080, "test": 29100252}
BANDS = ((.3, 1), (1, 2), (2, 4), (4, 6))
FLIP = np.diag([1., -1., -1., 1.])


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def make_scene(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    check(len(mesh.triangles) > 0, "empty mesh: " + str(path))
    scene = o3d.t.geometry.RaycastingScene(nthreads=CPU)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def evaluate(manifest, split):
    config = load(manifest)
    meshes = {k: Path(v) for k, v in config["meshes"].items()}
    check("supported_poisson" in meshes and len(meshes) >= 2, "baseline and candidate required")
    identities = {k: {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for k, p in meshes.items()}
    if split == "test":
        freeze = load(config["freeze"])
        check(freeze.get("run_id") == "capture-full-quality-v3-20261122-v1" and
              freeze.get("status") == "frozen" and freeze.get("training_decision_frozen") is True and
              freeze.get("checkpoint_selection_frozen") is True and freeze.get("mesh_selection_frozen") is True and
              freeze.get("thresholds_frozen") is True and
              freeze.get("mesh_sha256") == {k: v["sha256"] for k, v in identities.items()},
              "test requires complete selection freeze with matching mesh hashes")
    root = Path(config["scene"])
    meta = load(root / "transforms.json")
    names = meta[split + "_filenames"]
    check(len(names) == len(set(names)) == 184, "184 distinct holdout views required")
    frames = {f["file_path"]: f for f in meta["frames"]}
    times = {r["id"]: r["relative_s"] for r in selection()}
    intr = [meta[k] for k in ("fl_x", "fl_y", "cx", "cy")]
    scenes = {k: make_scene(p) for k, p in meshes.items()}
    output = Path(config["output"]); output.mkdir(parents=True, exist_ok=True)
    rows, totals, errors, transfers = [], {}, {}, {}
    with tempfile.TemporaryDirectory(dir=output, prefix="quantiles_") as temp:
        def add(group, key, ref, z, valid):
            c, err = stats(ref, z, valid)
            entry = totals.setdefault((group, key), {k: 0 for k in c})
            for k, v in c.items(): entry[k] += v
            file = errors.setdefault((group, key), Path(temp) / (str(len(errors)) + ".bin"))
            with file.open("ab") as out: err.tofile(out)
            return measures(c, err)
        for i, name in enumerate(names):
            frame = frames[name]; stem = Path(name).stem
            depth = np.load(root / frame["depth_file_path"], mmap_mode="r")
            y, x = np.where(np.isfinite(depth) & (depth >= .3) & (depth <= 6))
            ref = np.asarray(depth[y, x]); q = rays(x, y, np.asarray(frame["transform_matrix"]) @ FLIP, intr)
            z = {k: s.cast_rays(q, nthreads=CPU)["t_hit"].numpy() for k, s in scenes.items()}
            valid = {k: np.isfinite(v) & (v > 0) for k, v in z.items()}
            row = {"id": stem, "relative_s": times[stem], "reference_pixels": len(ref), "surfaces": {}, "paired": {}}
            base = z["supported_poisson"]; bv = valid["supported_poisson"]
            bc = bv & (np.abs(base-ref) <= .1)
            for k, value in z.items():
                row["surfaces"][k] = add("all", k, ref, value, valid[k])
                add("time_%d" % (int(times[stem] // 30)*30), k, ref, value, valid[k])
                for a, b in BANDS:
                    mask = (ref >= a) & ((ref <= b) if b == 6 else (ref < b))
                    add("range_%s_%s" % (a, b), k, ref[mask], value[mask], valid[k][mask])
                common = bv & valid[k]
                add("common_" + k, k, ref, value, common)
                if k != "supported_poisson":
                    add("common_" + k, "supported_poisson", ref, base, common)
                    correct = valid[k] & (np.abs(value-ref) <= .1)
                    pair = {"reference": len(ref), "common_hits": int(common.sum()),
                            "correct_to_incorrect": int((bc & ~correct).sum()),
                            "incorrect_to_correct": int((~bc & correct).sum())}
                    row["paired"][k] = pair
                    total = transfers.setdefault(k, {a: 0 for a in pair})
                    for a, v in pair.items(): total[a] += v
            rows.append(row)
            if (i+1) % 32 == 0: print(split, i+1, flush=True)
        check(sum(r["reference_pixels"] for r in rows) == EXPECTED[split], "reference denominator mismatch")
        groups = {}
        for (group, key), c in totals.items():
            err = np.fromfile(errors[(group, key)], dtype=np.float64)
            groups.setdefault(group, {})[key] = measures(c, err)
        save(output / ("surface_frames_" + split + ".json"), rows)
        save(output / ("surfaces_" + split + ".json"), {"split": split, "frames": len(rows),
             "meshes": identities, "groups": groups, "paired_10cm": transfers,
             "reference": "all finite sensor optical Z in [0.3,6] m; first mesh intersection"})
    if split == "val":
        local = load(B0_REPORT / "mesh_validation.json")["local_views"]
        crops = [("train_%05d" % n, "images/frame_%05d.png" % n, None) for n in (0, 330, 1420)]
        crops += [(r["object"], "images/frame_%05d.png" % r["train_id"], r["xyxy"]) for r in local]
        report = {}
        for label, name, crop in crops:
            frame = frames[name]; depth = np.load(root / frame["depth_file_path"], mmap_mode="r")
            x0, y0, x1, y1 = crop or (0, 0, meta["w"], meta["h"])
            yy, xx = np.mgrid[y0:y1, x0:x1]; ref = np.asarray(depth[y0:y1, x0:x1])
            mask = np.isfinite(ref) & (ref >= .3) & (ref <= 6)
            query = rays(xx, yy, np.asarray(frame["transform_matrix"]) @ FLIP, intr)
            image = cv2.imread(str(root / name))[y0:y1, x0:x1]
            tiles = [titled(image.copy(), "reference")]; values = {}
            for k, scene in scenes.items():
                hit = scene.cast_rays(query, nthreads=CPU); z = hit["t_hit"].numpy()
                valid = np.isfinite(z) & (z > 0); c, err = stats(ref[mask], z[mask], valid[mask])
                values[k] = measures(c, err)
                diff = z[mask & valid] - ref[mask & valid]
                values[k]["positive_signed"] = int((diff > 0).sum())
                values[k]["negative_signed"] = int((diff < 0).sum())
                image2 = np.full_like(image, 245); normals = hit["primitive_normals"].numpy()[valid]
                shade = .24 + .76*np.abs(normals @ np.array([-.4, -.6, -.69])).clip(0, 1)
                image2[valid] = (shade[:, None]*[204, 212, 219]).astype("u1")
                tiles.append(titled(image2, k))
            cv2.imwrite(str(output / ("local_" + label + ".jpg")), np.hstack(tiles))
            report[label] = {"image": name, "xyxy": [x0, y0, x1, y1], "surfaces": values}
        save(output / "surface_local.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON: scene, output, meshes mapping; test also requires freeze")
    parser.add_argument("split", choices=("val", "test"))
    args = parser.parse_args(); evaluate(args.manifest, args.split)


if __name__ == "__main__":
    main()

