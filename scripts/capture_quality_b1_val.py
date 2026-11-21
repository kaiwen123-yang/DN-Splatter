"""Evaluate the six frozen B1 models on their original 184-camera validation split."""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from capture_metric3d_common import REPORT, ROOT, RUN, SCENE, demand, read, sha, write


PROTOCOL = ROOT / "configs/capture_quality_protocol.json"
PLAN = read(PROTOCOL)
DEST = ROOT / ".local/runs" / PLAN["run_id"] / "b1_val"
CHECKPOINTS = read(REPORT / "B1_training_curve.json")["checkpoints"]


def measure(record, config, names):
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    from capture_s01_eval import depth_stats
    from vr_room_checkpoint import numpy_scalar_scope

    step = record["step"]
    path = ROOT / record["path"]
    demand(path.is_relative_to(RUN / "training") and sha(path) == record["sha256"], "B1 checkpoint")

    def bind(cfg):
        demand(cfg.pipeline.datamanager.dataparser.data == SCENE and
               cfg.get_checkpoint_dir() == path.parent, "original B1 camera data and checkpoint directory")
        cfg.load_step = step
        return cfg

    with numpy_scalar_scope():
        _, pipe, loaded_path, loaded_step = eval_setup(config, test_mode="val", update_config_callback=bind)
    demand(loaded_path == path and loaded_step == step, "loaded B1 checkpoint")
    ds = pipe.datamanager.eval_dataset
    demand([p.relative_to(SCENE).as_posix() for p in ds.image_filenames] == names, "original val ordering")
    rows, diffs, refs = [], [], []
    with torch.no_grad():
        for i, name in enumerate(names):
            stem = Path(name).stem
            batch = ds.get_data(i)
            out = pipe.model.get_outputs_for_camera(camera=ds.cameras[i:i + 1].to(pipe.device))
            pred = out["rgb"].reshape(720, 1280, 3).permute(2, 0, 1)[None]
            gt = batch["image"].to(pred.device).permute(2, 0, 1)[None]
            rgb = {k: float(getattr(pipe.model, k)(gt, pred)) for k in ("psnr", "ssim", "lpips")}
            reference = np.load(SCENE / "depth" / (stem + ".npy"))
            z = out["depth"].detach().cpu().numpy().reshape(reference.shape)
            mask = np.isfinite(reference) & (reference >= .3) & (reference <= 6.)
            valid = mask & np.isfinite(z) & (z > 0)
            target = reference[valid].astype(np.float64)
            error = np.abs(z[valid].astype(np.float64) - target)
            refs.append(target)
            diffs.append(error)
            count = int(mask.sum())
            invalid = count - len(error)
            stats = depth_stats(error, target)
            rows.append(dict(id=stem, rgb=rgb, reference_pixels=count, invalid_prediction_pixels=invalid,
                             depth_full_reference=stats if invalid == 0 else None, depth_conditional=stats))
            if i == 0 or (i + 1) % 32 == 0 or i + 1 == len(names):
                print(f"B1 step={step} val={i+1}/{len(names)}", flush=True)
    invalid = sum(r["invalid_prediction_pixels"] for r in rows)
    pooled = depth_stats(np.concatenate(diffs), np.concatenate(refs))
    supported_rows = [r["depth_conditional"] for r in rows if r["depth_conditional"] is not None]
    return dict(run_id=PLAN["run_id"], checkpoint=record, split="val", frame_count=len(rows),
                config_sha256=sha(config), membership_sha256=sha(SCENE / "transforms.json"),
                reference_pixels=sum(r["reference_pixels"] for r in rows), invalid_prediction_pixels=invalid,
                depth_pixel_pooled=pooled if not invalid else None, depth_conditional_pooled=pooled,
                depth_frame_means={k: float(np.mean([r[k] for r in supported_rows])) for k in pooled}
                if pooled and not invalid else None,
                rgb_frame_means={k: float(np.mean([r["rgb"][k] for r in rows])) for k in rows[0]["rgb"]},
                frames=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    names = read(SCENE / "transforms.json")["val_filenames"]
    frozen = PLAN["baseline"]
    demand(len(names) == len(set(names)) == 184 and
           [r["step"] for r in CHECKPOINTS] == frozen["existing_checkpoint_steps"] and
           sha(SCENE / "geometry_source_manifest.json") == frozen["original_geometry_manifest_sha256"],
           "frozen B1 validation identity")
    config = Path(read(RUN / "training_complete.json")["config"])
    demand(config.is_relative_to(RUN / "training") and sha(config) == sha(ROOT / frozen["configuration"]),
           "B1 config identity")
    DEST.mkdir(parents=True, exist_ok=True)
    summary = []
    for record in CHECKPOINTS:
        step = record["step"]
        path = DEST / f"step-{step:09d}.json"
        if args.step == step:
            write(path, measure(record, config, names))
            return
        if args.step is not None:
            continue
        if not (ROOT / record["path"]).is_file():
            summary.append(dict(step=step, status="missing", expected_sha256=record["sha256"]))
            continue
        if not path.exists():
            subprocess.run([sys.executable, "-u", __file__, "--step", str(step)], check=True)
        result = read(path)
        demand(result["run_id"] == PLAN["run_id"] and result["checkpoint"] == record and
               result["membership_sha256"] == sha(SCENE / "transforms.json") and
               result["config_sha256"] == sha(config) and
               [r["id"] for r in result["frames"]] == [Path(n).stem for n in names], "B1 val receipt")
        summary.append({k: v for k, v in result.items() if k != "frames"} | {"sha256": sha(path)})
    demand(args.step is None, "unexpected B1 step")
    write(DEST / "curve.json", dict(run_id=PLAN["run_id"], split="val", rows=summary))


if __name__ == "__main__":
    main()
