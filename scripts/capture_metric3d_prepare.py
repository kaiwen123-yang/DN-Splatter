"""Bind B0 and B1 by actual content and read the frozen data through the DN parser."""
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from capture_metric3d_common import (BASE, B0, B0_REPORT, GEOMETRY, P, PRIOR, REPORT, ROOT, RUN,
                                     SCENE, SOURCE, SPLITS, STATE, demand, member, read, sha, write)
from capture_rectify import files


def curves(label, run):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from vr_room_checkpoint import numpy_scalar_scope

    points = []
    for ck in sorted((run / "training").glob("**/nerfstudio_models/step-*.ckpt")):
        with numpy_scalar_scope():
            state = torch.load(ck, map_location="cpu", mmap=True, weights_only=True)
        means = next(v for k, v in state["pipeline"].items() if k.endswith("gauss_params.means"))
        points.append(dict(step=int(state["step"]), gaussians=int(means.shape[0]), **member(ck)))
        del state, means
    path = REPORT / f"{label}_tensorboard.csv.gz"
    tags = {}
    with gzip.open(path, "wt", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["session", "tag", "step", "value", "wall_time"])
        for group in sorted({p.parent for p in (run / "training").rglob("events.out.tfevents.*")}):
            tb = EventAccumulator(str(group), size_guidance={"scalars": 0})
            tb.Reload()
            for tag in tb.Tags()["scalars"]:
                if not any(s in tag.lower() for s in ("loss", "gauss", "psnr")):
                    continue
                values = tb.Scalars(tag)
                tags[tag] = tags.get(tag, 0) + len(values)
                writer.writerows((str(group.relative_to(run)), tag, v.step, v.value, v.wall_time) for v in values)
    log = run / "logs/training.log"
    # The rich console may wrap a single cull across two lines.
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", log.read_text(errors="replace")) if log.is_file() else ""
    culls = [int(s.replace(",", "")) for s in re.findall(r"Culled\s+([\d,]+)\s+gaussians", text)]
    demand(points and max(p["step"] for p in points) == 29999 and tags, f"{label} training evidence")
    write(REPORT / f"{label}_training_curve.json", dict(checkpoints=points, scalar_samples=tags,
          tensorboard=path.name, logged_cull_union_counts_in_order=culls,
          cull_reason_detail="unavailable for B0 where absent in original log" if label == "B0" else "B1_refinement.json"))


def baseline():
    spec = P["baseline"]
    completed = read(BASE / "training_complete.json")
    frozen = {}
    for key, path in (("checkpoint_sha256", Path(completed["checkpoint"])),
                      ("configuration_sha256", Path(completed["config"])),
                      ("geometry_manifest_sha256", GEOMETRY / "manifest.json"),
                      ("training_input_manifest_sha256", B0 / "training_manifest.json")):
        frozen[key] = sha(path)
        demand(frozen[key] == spec[key], key)
    demand(completed["step"] == 29999 and sha(B0 / "geometry_source_manifest.json") == frozen["geometry_manifest_sha256"],
           "baseline geometry and final update")
    source = subprocess.check_output(["git", "-C", SOURCE, "rev-parse", "HEAD"], text=True).strip()
    dn = ROOT / ".local/upstream/dn-splatter"
    from vr_room_publish import patch_matches
    demand(source == P["prior"]["commit"] and patch_matches(SOURCE, "metric3d-eb5b6fac-*.patch"),
           "Metric3D source and environment-import patch")
    demand(subprocess.check_output(["git", "-C", dn, "rev-parse", "HEAD"], text=True).strip() == P["training"]["dn_commit"]
           and patch_matches(dn, "dn-811f5cd-*.patch") and patch_matches(ROOT / ".local/upstream/gsplat", "gsplat-1.0.0-*.patch"),
           "historical Blackwell source")
    geom = read(B0 / "geometry.json")
    demand(geom["init_points"] == 2001906 and geom["parser_settings"]["scene_scale"] == 21, "initialization")
    meta = read(B0 / "transforms.json")
    sets = {s: meta[s + "_filenames"] for s in SPLITS}
    joined = [name for names in sets.values() for name in names]
    demand(len(meta["frames"]) == len(set(joined)) == len(joined) == 1846 and
           [len(sets[s]) for s in SPLITS] == [1478, 184, 184] and set(joined) == {r["file_path"] for r in meta["frames"]},
           "frozen membership")
    SCENE.mkdir(parents=True, exist_ok=True)
    expected = read(B0 / "geometry_source_manifest.json")["files"]
    frozen_inputs = {r["path"]: r for r in read(B0 / "training_manifest.json")["files"]}
    for top in sorted({Path(r["path"]).parts[0] for r in expected} | {"geometry_source_manifest.json"}):
        # A reflink is copy-on-write. A regular file copy is used on other filesystems.
        subprocess.run(["cp", "-a", "--reflink=auto", "--no-clobber", B0 / top, SCENE], check=True)
    def verify_pair(row):
        rel = row["path"]
        a, b = B0 / rel, SCENE / rel
        demand(not b.samefile(a) and row == frozen_inputs[rel] and b.stat().st_size == row["bytes"] and
               sha(a) == sha(b) == row["sha256"], rel)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(verify_pair, expected):
            pass
    demand(sha(SCENE / "geometry_source_manifest.json") == frozen["geometry_manifest_sha256"], "geometry copy")
    curves("B0", BASE)
    poses = {}
    for label, scene in (("full", GEOMETRY), ("s01", ROOT / ".local/data/custom/capture-s01/20261119-s01-5hz-3982cb26")):
        poses[label] = {Path(f["file_path"]).stem: f["transform_matrix"] for f in read(scene / "transforms.json")["frames"]}
    write(REPORT / "metric_c2w.json", dict(frame="metric OpenGL camera to original world, separate gauge per map", poses=poses))
    write(REPORT / "frozen_inputs.json", dict(**frozen, source=source, unchanged_members=len(expected),
          members_source="geometry_source_manifest.json", point_count=geom["init_points"], scene_scale=21,
          counts={s: len(sets[s]) for s in SPLITS}, sharing="independent inodes with cp --reflink=auto",
          init_ply=member(SCENE / meta["ply_file_path"]), b0=spec["results_commit"]))


def reader():
    from dn_splatter.data.dn_dataset import GDataset
    from capture_s01_train import check_dataset
    from capture_metric3d_train import frozen_config

    info = read(REPORT / "normal_prior.json")
    write(SCENE / "training_inputs.json", dict(model="Metric3D v2 ViT-Large", weight_sha256=info["weight"]["sha256"],
          normal_format="opengl", png_channels="raw OpenCV XYZ", diagnostic_depth_training_use=False))
    write(SCENE / "training_manifest.json", dict(files=files(SCENE, ["training_manifest.json"])))
    cfg = frozen_config()
    parser = cfg.pipeline.datamanager.dataparser.setup()
    counts = {}
    for split in SPLITS:
        ds = GDataset(parser.get_dataparser_outputs(split=split), scale_factor=1.0)
        counts[split] = check_dataset(ds, split, (ds.get_data(i) for i in range(len(ds))))
        del ds
    demand([counts[s]["count"] for s in SPLITS] == [1478, 184, 184], "actual DN reader splits")
    write(REPORT / "reader_check.json", dict(splits=counts, training_manifest_sha256=sha(SCENE / "training_manifest.json"),
          comparison="Historical GDataset.get_data, all RGB/sensor Z/K/c2w and PNG decoded with D=diag(1,-1,-1)",
          absolute_tolerance=1e-6, source_manifest=member(SCENE / "geometry_source_manifest.json")))


if __name__ == "__main__":
    {"baseline": baseline, "reader": reader}[sys.argv[1]]()
