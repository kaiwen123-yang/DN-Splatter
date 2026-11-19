"""Use the actual historical training entry and inspect the first formal update."""
import copy
from contextlib import nullcontext
import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch
from dn_splatter.data.normal_nerfstudio import NormalNerfstudioConfig
from dn_splatter.dn_config import dn_splatter
from dn_splatter.dn_model import DNSplatterModel
from dn_splatter.losses import DepthLossType
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.scripts.train import main as ns_train

from capture_rectify import files, read, save, sha
from capture_s01_inputs import FULL, P, REPORT, ensure
from vr_room_run import RUN, SCENE, STATE, latest_checkpoint
from vr_room_checkpoint import numpy_scalar_scope


def config():
    cfg = copy.deepcopy(dn_splatter.config)
    cfg.output_dir = RUN / "training"
    cfg.experiment_name = "capture-full" if FULL else "capture-s01"
    cfg.machine.seed = P["training"]["seed"]
    cfg.steps_per_save = 5000
    cfg.save_only_latest_checkpoint = False
    cfg.vis = "tensorboard"
    cfg.viewer.quit_on_train_completion = True
    cfg.pipeline.skip_point_metrics = True
    parser_settings = dict(P["parser"]["settings"])
    if FULL:
        parser_settings["scene_scale"] = read(SCENE / "geometry.json")["parser_settings"]["scene_scale"]
    cfg.pipeline.datamanager.dataparser = NormalNerfstudioConfig(data=SCENE, **parser_settings)
    cfg.pipeline.datamanager.camera_res_scale_factor = 1.0
    for key, value in P["training"]["pipeline"]["model"].items():
        if key == "camera_optimizer":
            cfg.pipeline.model.camera_optimizer.mode = value["mode"]
        elif key == "depth_loss_type":
            cfg.pipeline.model.depth_loss_type = DepthLossType(value)
        else:
            ensure(hasattr(cfg.pipeline.model, key), key)
            setattr(cfg.pipeline.model, key, value)
    ensure(
        cfg.method_name == "dn-splatter"
        and cfg.max_num_iterations == 30000
        and cfg.optimizers["means"]["scheduler"].max_steps == 30000,
        "historical schedule",
    )
    return cfg


def check_dataset(data, split, items):
    meta = read(SCENE / "transforms.json")
    frames = {f["file_path"]: f for f in meta["frames"]}
    names = [p.relative_to(SCENE).as_posix() for p in data.image_filenames]
    ensure(
        names == meta[split + "_filenames"] and (FULL or len(names) == P["input"]["frame_counts"][split]), split
    )
    outputs = data._dataparser_outputs
    normals = outputs.metadata["points3D_normals"]
    lengths = normals.norm(dim=-1)
    geometry = read(SCENE / "geometry.json")
    box = geometry["parser_settings"]["scene_scale"] if FULL else 15
    ensure(
        outputs.metadata["points3D_xyz"].shape == normals.shape == (geometry["init_points"] if FULL else 436361, 3)
        and torch.isfinite(lengths).all()
        and lengths.min() > 0,
        "point normals",
    )
    ensure(
        outputs.dataparser_scale == data.depth_unit_scale_factor == 1
        and np.allclose(outputs.dataparser_transform, np.eye(4)[:3])
        and not torch.any(data.cameras.distortion_params)
        and np.allclose(outputs.scene_box.aabb, [[-box] * 3, [box] * 3]),
        "metric reader",
    )
    for i, (name, item) in enumerate(zip(names, items, strict=True)):
        path = SCENE / name
        nz = data.normal_filenames[i]
        ensure(
            nz == (SCENE / "normals_from_pretrain" / path.name)
            and data.sensor_depth_filenames[i] == SCENE / frames[name]["depth_file_path"],
            "file correspondence",
        )
        rgb = cv2.imread(str(path))[:, :, ::-1]
        expected = np.load(data.sensor_depth_filenames[i])
        normal = cv2.imread(str(nz))[:, :, ::-1].astype(np.float32) / 255
        normal[:, :, 1:] = 1 - normal[:, :, 1:]
        ensure(
            np.array_equal(np.rint(item["image"].cpu().numpy() * 255).astype(np.uint8), rgb)
            and np.array_equal(item["sensor_depth"].squeeze().cpu().numpy(), expected)
            and np.allclose(item["normal"].cpu().numpy(), normal, atol=1e-6),
            "reader pixels " + name,
        )
        ensure(
            np.allclose(
                data.cameras.camera_to_worlds[i], frames[name]["transform_matrix"][:3], atol=2e-6
            )
            and all(
                np.isclose(float(getattr(data.cameras, k)[i]), meta[v])
                for k, v in [("fx", "fl_x"), ("fy", "fl_y"), ("cx", "cx"), ("cy", "cy")]
            ),
            "camera " + name,
        )
    return dict(
        count=len(names),
        ids=[Path(n).stem for n in names],
        grid=[1280, 720],
        init_points=normals.shape[0],
        point_normal_length_min=float(lengths.min()),
        point_normal_length_max=float(lengths.max()),
    )


def reader():
    cfg = config()
    checked = {}
    for mode in ("val", "test"):
        dm = cfg.pipeline.datamanager.setup(device="cuda:0", test_mode=mode)
        if mode == "val":
            checked["train"] = check_dataset(dm.train_dataset, "train", dm.cached_train)
        checked[mode] = check_dataset(dm.eval_dataset, mode, dm.cached_eval)
        del dm
        torch.cuda.empty_cache()
    save(
        REPORT / "reader_check.json",
        dict(
            splits=checked,
            dataparser_scale=1,
            depth_unit_scale_factor=1,
            scene_scale=cfg.pipeline.datamanager.dataparser.scene_scale,
            dataparser_transform=np.eye(4)[:3].tolist(),
            point_normal_radius_m=0.1,
            max_nn=30,
            path="DNSplatterDataManager cached data, GDataset and NormalNerfstudio: exact RGB/depth, PNG decoded with OpenGL Y/Z flip, K and c2w checked for every frame.",
            training_manifest_sha256=sha(SCENE / "training_manifest.json"),
        ),
    )


def train():
    fingerprint = sha(SCENE / "training_manifest.json")
    ensure(
        fingerprint == read(REPORT / "reader_check.json")["training_manifest_sha256"],
        "reader input changed",
    )
    ensure(
        files(SCENE, ["training_manifest.json"]) == read(SCENE / "training_manifest.json")["files"],
        "training member hashes",
    )
    ckpt, step, count = latest_checkpoint()
    cfg = config()
    remaining = 30000 - (step + 1)
    ensure(remaining >= 0, "checkpoint step")
    if remaining:
        cfg.max_num_iterations = remaining
        if ckpt:
            cfg.load_dir = ckpt.parent
            cfg.load_step = step
            cfg.load_scheduler = True
        with (STATE / "training_segments.jsonl").open("a") as f:
            print(
                json.dumps(
                    dict(
                        checkpoint=str(ckpt) if ckpt else None,
                        saved_step=step,
                        requested_updates=remaining,
                        training_manifest_sha256=fingerprint,
                    )
                ),
                file=f,
            )
        first = {}
        observed = False
        original_loss = DNSplatterModel.get_loss_dict
        original_update = Optimizers.optimizer_scaler_step_some
        evidence = STATE / (
            "first_batch_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + ".json"
        )

        def check_loss(model, outputs, batch, *args, **kwargs):
            result = original_loss(model, outputs, batch, *args, **kwargs)
            if not observed:
                normal = outputs["normal"].detach()
                first.update(
                    step=step + 1,
                    image_idx=int(batch["image_idx"]),
                    normal_pixels=normal.numel() // 3,
                    nonfinite_predicted_normal_pixels=int((~normal.isfinite().all(dim=-1)).sum()),
                    zero_accumulation_pixels=int((outputs["accumulation"] == 0).sum()),
                    losses={
                        k: float(v.detach()) if torch.isfinite(v) else str(float(v.detach()))
                        for k, v in result.items()
                    },
                )
                save(evidence, first)
                ensure(
                    all(torch.isfinite(v) for v in result.values())
                    and first["nonfinite_predicted_normal_pixels"] == 0,
                    "nonfinite formal first batch: " + str(first),
                )
            return result

        def check_grad(opts, *args, **kwargs):
            nonlocal observed
            if not observed:
                invalid = {
                    k: sum(int((~p.grad.isfinite()).sum()) for p in group if p.grad is not None)
                    for k, group in opts.parameters.items()
                }
                first["gradient_elements_checked"] = sum(
                    p.grad.numel()
                    for group in opts.parameters.values()
                    for p in group
                    if p.grad is not None
                )
                first["nonfinite_grad_elements"] = invalid
                save(evidence, first)
                ensure(
                    first["gradient_elements_checked"] and not any(invalid.values()),
                    "nonfinite first gradient",
                )
                observed = True
            return original_update(opts, *args, **kwargs)

        with patch.object(DNSplatterModel, "get_loss_dict", check_loss), patch.object(
            Optimizers, "optimizer_scaler_step_some", check_grad
        ), (numpy_scalar_scope() if ckpt else nullcontext()):
            ns_train(cfg)
        ckpt, step, count = latest_checkpoint()
    ensure(step == 29999 and count > 0, "incomplete training")
    save(
        RUN / "training_complete.json",
        dict(
            checkpoint=str(ckpt),
            checkpoint_sha256=sha(ckpt),
            config=str(ckpt.parents[1] / "config.yml"),
            step=step,
            cumulative_updates=step + 1,
            final_gaussians=count,
            training_manifest_sha256=fingerprint,
        ),
    )


if __name__ == "__main__":
    {"reader": reader, "train": train}[sys.argv[1]]()
