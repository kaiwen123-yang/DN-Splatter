"""Evaluate the final capture model on exact frozen holdouts and retain full renders."""
import importlib.metadata as md
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from nerfstudio.utils.eval_utils import eval_setup
from capture_rectify import imwrite, read, save, sha
from capture_s01_diagnostic import panel
from capture_s01_inputs import FULL, REPORT, ensure
from vr_room_checkpoint import numpy_scalar_scope
from vr_room_run import RUN, SCENE


def final_setup(split):
    completed = read(RUN / "training_complete.json")
    path = Path(completed["checkpoint"])
    ensure(completed["step"] == 29999 and path.is_relative_to(RUN / "training"), "final checkpoint")
    ensure(sha(path) == completed["checkpoint_sha256"], "final checkpoint digest")

    def bind(cfg):
        cfg.load_step = 29999
        ensure(
            cfg.get_checkpoint_dir() == path.parent
            and cfg.pipeline.skip_point_metrics
            and cfg.pipeline.datamanager.dataparser.data == SCENE,
            "evaluation config identity",
        )
        return cfg

    with numpy_scalar_scope():
        setup = eval_setup(Path(completed["config"]), test_mode=split, update_config_callback=bind)
    ensure(setup[2] == path and setup[3] == 29999, "loaded checkpoint mismatch")
    return setup


def depth_stats(diff, ref):
    if not len(diff):
        return None
    return dict(
        MAE_m=float(diff.mean()),
        RMSE_m=float(np.sqrt(np.mean(diff**2))),
        AbsRel=float(np.mean(diff / ref)),
        absolute_error_P50_m=float(np.median(diff)),
        absolute_error_P95_m=float(np.percentile(diff, 95)),
    )


def color(z, low, high, valid):
    clipped = np.where(valid, np.clip((z - low) / (high - low), 0, 1), 0)
    img = cv2.applyColorMap(np.uint8(255 * clipped), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def evaluate(split):
    _, pipe, path, step = final_setup(split)
    ds = pipe.datamanager.eval_dataset
    names = [p.relative_to(SCENE).as_posix() for p in ds.image_filenames]
    ensure(
        names == read(SCENE / "transforms.json")[split + "_filenames"] and (FULL or len(names) == 24),
        "split membership",
    )
    root = RUN / "renders" / split
    for sub in ("rgb", "depth", "normal", "accumulation"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows, diffs, refs = [], [], []
    with torch.no_grad():
        for i, name in enumerate(names):
            stem = Path(name).stem
            batch = ds.get_data(i)
            outputs = pipe.model.get_outputs_for_camera(camera=ds.cameras[i : i + 1].to(pipe.device))
            rgb = outputs["rgb"].reshape(720, 1280, 3)
            gt = batch["image"].to(rgb.device).permute(2, 0, 1)[None]
            pred = rgb.permute(2, 0, 1)[None]
            metrics = {k: float(getattr(pipe.model, k)(gt, pred)) for k in ("psnr", "ssim", "lpips")}
            ref = np.load(SCENE / "depth" / (stem + ".npy"))
            z = outputs["depth"].cpu().numpy().reshape(720, 1280).astype(np.float32)
            a = outputs["accumulation"].cpu().numpy().reshape(z.shape)
            support = np.isfinite(ref) & (ref >= 0.3) & (ref <= 6)
            keep = support & np.isfinite(z) & (z > 0)
            nref, nvalid = int(support.sum()), int(keep.sum())
            target = ref[keep].astype(np.float64)
            diff = np.abs(z[keep].astype(np.float64) - target)
            stats = depth_stats(diff, target)
            row = dict(
                id=stem,
                reference_pixels=nref,
                invalid_prediction_pixels=nref - nvalid,
                valid_prediction_coverage=nvalid / nref if nref else None,
                reference_alpha_zero=int((support & (a == 0)).sum()),
                rgb=metrics,
                depth_full_reference=stats if nvalid == nref else None,
                depth_conditional=stats,
            )
            diffs.append(diff)
            refs.append(target)
            normal = outputs["normal"].detach().cpu().numpy().reshape(720, 1280, 3)
            row["nonfinite_normal_pixels"] = int((~np.isfinite(normal).all(axis=-1)).sum())

            def bgr(arr):
                return np.uint8(np.rint(255 * np.clip(np.nan_to_num(arr), 0, 1)))[..., ::-1]

            rgb_img = bgr(rgb.cpu().numpy())
            normal_img = bgr(normal)
            imwrite(root / "rgb" / (stem + ".png"), rgb_img)
            imwrite(root / "normal" / (stem + ".png"), normal_img)
            np.save(root / "depth" / (stem + ".npy"), z)
            np.save(root / "accumulation" / (stem + ".npy"), a.astype(np.float32))
            if not FULL and (i in {0, len(names) // 2, len(names) - 1} or int(stem[-5:]) in {149, 179, 239}):
                panels = [
                    panel(cv2.imread(str(SCENE / name)), stem + " reference"),
                    panel(rgb_img, "Rendered RGB"),
                    panel(normal_img, "Rendered camera normal"),
                    panel(color(ref, 0.3, 6, support), "Sensor Z 0.3 to 6 m"),
                    panel(color(z, 0.3, 6, np.isfinite(z) & (z > 0)), "Rendered Z 0.3 to 6 m"),
                    panel(color(abs(z - ref), 0, 0.5, keep), "Absolute dZ 0 to 0.5 m"),
                ]
                imwrite(
                    REPORT / "previews" / (split + "_" + stem + ".jpg"),
                    np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])]),
                )
            rows.append(row)
    ensure(
        all(
            {p.stem for p in (root / k).iterdir()} == {r["id"] for r in rows}
            for k in ("rgb", "normal", "depth", "accumulation")
        ),
        "full render coverage",
    )
    conditional = depth_stats(np.concatenate(diffs), np.concatenate(refs)) if rows else None
    nref = sum(r["reference_pixels"] for r in rows)
    invalid = sum(r["invalid_prediction_pixels"] for r in rows)
    all_valid_frames = [
        r["depth_full_reference"] for r in rows if r["depth_full_reference"] is not None
    ]
    avg = (
        lambda items: {k: float(np.mean([row[k] for row in items])) for k in items[0]}
        if items
        else None
    )
    save(
        REPORT / ("metrics_" + split + ".json"),
        dict(
            split=split,
            frames=rows,
            frame_count=len(rows),
            step=step,
            checkpoint=str(path),
            checkpoint_sha256=read(RUN / "training_complete.json")["checkpoint_sha256"],
            metric_implementation=dict(
                torchmetrics=md.version("torchmetrics"),
                psnr="data_range=1",
                ssim="StructuralSimilarityIndexMeasure data_range=1 kernel_size=11",
                lpips="historical model: AlexNet, RGB float [0,1]",
                normalize=bool(pipe.model.lpips.normalize),
            ),
            reference="Same-capture optical-axis sensor Z in [0.3,6] m, holes zero. Frame means weight each frame equally; pooled metrics weight every reference pixel equally.",
            rgb_frame_means=avg([r["rgb"] for r in rows]),
            depth_frame_means=avg(all_valid_frames) if not invalid else None,
            depth_pixel_pooled=conditional if not invalid else None,
            finite_positive_conditional_pooled=conditional,
            reference_pixels=nref,
            invalid_prediction_pixels=invalid,
            valid_prediction_coverage=1 - invalid / nref if nref else None,
            frames_with_reference=sum(r["reference_pixels"]>0 for r in rows),
            reference_alpha_zero=sum(r["reference_alpha_zero"] for r in rows),
            renders=str(root),
        ),
    )


if __name__ == "__main__":
    evaluate(sys.argv[1])
