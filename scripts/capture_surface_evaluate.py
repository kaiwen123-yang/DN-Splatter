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

from capture_metric3d_common import B0_REPORT, REPORT as B1_REPORT, selection, sha
from capture_surface_tiles import ROOT, RUN, ply
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
    vertices, faces = ply(path)
    check(len(faces)>0, "empty mesh: " + str(path))
    scene = o3d.t.geometry.RaycastingScene(nthreads=CPU)
    scene.add_triangles(o3d.core.Tensor.from_numpy(np.ascontiguousarray(vertices,dtype='f4')),
                        o3d.core.Tensor.from_numpy(np.ascontiguousarray(faces,dtype='u4')))
    return scene


def native_probes(scenes, frames, meta):
    from capture_surface_submaps import PACKET, camera_parameters
    old=load(ROOT/'reports/capture-quality-review/2026-11-22/tsdf_block_check.json')
    cal=load(PACKET/'calibration.json'); k=cal['cameras']['depth']
    _,_,extrinsic=camera_parameters(meta,cal)
    result=[]
    for probe in old['probes']:
        pose=np.asarray(frames['images/'+probe['frame']+'.png']['transform_matrix'])@FLIP@extrinsic
        u,v=probe['native_xy']
        ray=rays(np.asarray([u]),np.asarray([v]),pose,[k['fx'],k['fy'],k['cx']+.5,k['cy']+.5])
        grid=np.indices((16,16,16)).transpose(1,2,3,0)
        points=np.asarray(probe['block_origin_world_m'])+(grid+.5)*.01
        out={}
        for name,scene in scenes.items():
            hit=float(scene.cast_rays(ray,nthreads=CPU)['t_hit'].numpy()[0])
            ref=probe['native_Z_m']; valid=np.isfinite(hit)
            crossings=0
            for axis in range(3):
                inside=grid[...,axis]<15
                direction=np.zeros(3); direction[axis]=1.
                q=np.concatenate((points[inside],np.broadcast_to(direction,points[inside].shape)),axis=1)
                t=scene.cast_rays(o3d.core.Tensor(q.astype('f4')),nthreads=CPU)['t_hit'].numpy()
                crossings+=int(((t>=0.)&(t<.01)).sum())
            out[name]=dict(hit_Z_m=hit if valid else None, reference_Z_m=ref,
                first_hit_within_10cm=bool(valid and abs(hit-ref)<=.1),
                foreground_conflict=bool(valid and hit<ref-max(.05,.03*ref)),crossed_grid_edges=crossings)
        result.append(dict(frame=probe['frame'],native_xy=[u,v],historical_local_tsdf=probe['local_field_within_2s'],
                        historical_global_tsdf=probe['full_field'],meshes=out))
    return dict(probes=result,edges='First triangle intersection on each positive 1 cm grid edge; open meshes have no global signed TSDF.')


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
    same={r['id'] for r in load(B1_REPORT/'comparison_s01_same_sources.json')[split]['frames']}
    check(len(same)==24,'fixed same-source population')
    times = {r["id"]: r["relative_s"] for r in selection()}
    intr = [meta[k] for k in ("fl_x", "fl_y", "cx", "cy")]
    # Paired development comparisons share the immutable incumbent intersections.
    # Keep one depth vector per camera, not a second BVH or another baseline pass.
    cache = RUN/'evaluation'/'baseline_hits'/split/(sha(root/'transforms.json')+'_'+identities['supported_poisson']['sha256'])
    if split=='test':
        cache.mkdir(parents=True,exist_ok=True)
    scenes = {k: make_scene(p) for k, p in meshes.items() if k!='supported_poisson' or split=='val'}
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
            z = {k: s.cast_rays(q, nthreads=CPU)["t_hit"].numpy() for k, s in scenes.items()
                 if split!='test' or k!='supported_poisson'}
            if split=='test':
                path=cache/(stem+'.npy'); marker=cache/(stem+'.json')
                if marker.is_file():
                    binding=load(marker)
                    check(binding['sha256']==sha(path) and binding['reference_pixels']==len(ref),'baseline reference cache')
                    z['supported_poisson']=np.load(path,allow_pickle=False)
                else:
                    if 'supported_poisson' not in scenes:
                        scenes['supported_poisson']=make_scene(meshes['supported_poisson'])
                    if 'supported_poisson' not in z:
                        z['supported_poisson']=scenes['supported_poisson'].cast_rays(q,nthreads=CPU)['t_hit'].numpy()
                    temp=path.with_suffix('.pending.npy')
                    np.save(temp,z['supported_poisson'],allow_pickle=False);temp.replace(path)
                    save(marker,dict(sha256=sha(path),reference_pixels=len(ref)))
            valid = {k: np.isfinite(v) & (v > 0) for k, v in z.items()}
            row = {"id": stem, "relative_s": times[stem], "reference_pixels": len(ref), "surfaces": {}, "paired": {}}
            base = z["supported_poisson"]; bv = valid["supported_poisson"]
            bc = bv & (np.abs(base.astype('f8')-ref) <= .1)
            for k, value in z.items():
                row["surfaces"][k] = add("all", k, ref, value, valid[k])
                if stem in same: add("same_s01_source",k,ref,value,valid[k])
                add("time_%d" % (int(times[stem] // 30)*30), k, ref, value, valid[k])
                for a, b in BANDS:
                    mask = (ref >= a) & ((ref <= b) if b == 6 else (ref < b))
                    add("range_%s_%s" % (a, b), k, ref[mask], value[mask], valid[k][mask])
                common = bv & valid[k]
                add("common_" + k, k, ref, value, common)
                if k != "supported_poisson":
                    add("common_" + k, "supported_poisson", ref, base, common)
                    correct = valid[k] & (np.abs(value.astype('f8')-ref) <= .1)
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
        save(output/'native_probes.json',native_probes(scenes,frames,meta))
        local = load(B0_REPORT / "mesh_validation.json")["local_views"]
        crops = [("train_%05d" % n, "images/frame_%05d.png" % n, None) for n in (0, 330, 1420)]
        crops += [(r["object"], "images/frame_%05d.png" % r["train_id"], r["xyxy"]) for r in local]
        crops += [('foam_teeth','images/frame_01420.png',[720,224,1104,608])]
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

