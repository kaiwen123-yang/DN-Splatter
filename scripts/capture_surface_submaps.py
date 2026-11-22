"""Build resumable full-resolution B3 local TSDFs from the frozen B2 cameras."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d

import capture_quality_eval as evaluator
from capture_metric3d_common import member, read, sha, write
from capture_quality_paths import GEORUN, ROOT, check
from capture_quality_tsdf import FLIP, integrate, volume


PROTOCOL = ROOT / "configs/capture_surface_quality_protocol.json"
P = read(PROTOCOL)
RUN = ROOT / ".local/runs" / P["run_id"]
B2 = ROOT / ".local/runs" / P["baseline"]["run_id"]
PACKET = GEORUN / "transfer"


def utc():
    return datetime.now(timezone.utc).isoformat()


def immutable(path, data):
    if path.exists():
        check(read(path) == data, f"Changed frozen input: {path}")
    else:
        write(path, data)


def groups(meta, source):
    names = meta["train_filenames"]
    frame = {f["file_path"]: f for f in meta["frames"]}
    start = int(source[Path(names[0]).stem]["actual_rgb_timestamp_ns"])
    previous_window, previous_pose = None, None
    result = []
    last_time = start - 1
    for name in names:
        stem = Path(name).stem
        timestamp = int(source[stem]["actual_rgb_timestamp_ns"])
        check(timestamp > last_time, "Training camera order is not chronological")
        last_time = timestamp
        window = (timestamp - start) // 2_000_000_000
        pose = np.array(frame[name]["transform_matrix"])
        rotation = 0. if previous_pose is None else np.rad2deg(np.arccos(np.clip(
            (np.trace(previous_pose[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1., 1.)))
        if window != previous_window or rotation > 90:
            result.append(dict(id=len(result), time_bin=window, first_rotation_deg=float(rotation), names=[]))
        result[-1]["names"].append(name)
        previous_window, previous_pose = window, pose
    return result


def camera_parameters(meta, calibration):
    d = calibration["cameras"]["depth"]
    check(not np.any(d["D_sdk_k1_k2_k3_k4_k5_k6_p1_p2"]), "Distorted native grid")
    ext = next(r for r in calibration["static_extrinsics"]
               if (r["source_sensor"], r["destination_sensor"]) == ("depth", "color"))
    transform = np.eye(4)
    transform[:3, :3] = ext["R"]
    transform[:3, 3] = np.array(ext["translation_mm"]) / 1000.
    native = o3d.camera.PinholeCameraIntrinsic(d["width"], d["height"], d["fx"], d["fy"], d["cx"], d["cy"])
    rgb = o3d.camera.PinholeCameraIntrinsic(meta["w"], meta["h"], meta["fl_x"], meta["fl_y"],
                                          meta["cx"] - .5, meta["cy"] - .5)
    return native, rgb, transform


def local_maps(label, model_run):
    import torch

    scene = model_run / "inputs"
    output = RUN / "export" / label
    output.mkdir(parents=True, exist_ok=True)
    selected = read(model_run / "state/selection.json")
    step = selected["selected_step"]
    config = Path(selected["config"])
    checkpoint = config.parent / "nerfstudio_models" / f"step-{step:09d}.ckpt"
    meta = read(scene / "transforms.json")
    source = {r["id"]: r for r in read(PACKET / "selection.json")}
    cameras = {f["file_path"]: np.array(f["transform_matrix"]) @ FLIP for f in meta["frames"]}
    split = read(scene / "splits.json")["splits"]
    for key, count in (("train", 1478), ("val", 184), ("test", 184)):
        ids = [Path(n).stem for n in meta[key + "_filenames"]]
        check(len(set(ids)) == len(ids) == count and ids == split[key], "Frozen split " + key)
    check(len(cameras) == len(source) == 1846 and [meta["w"], meta["h"]] == [1280, 720], "Camera population")
    identity = dict(run_id=P["run_id"], model=label, step=step, checkpoint=member(checkpoint),
                    config=member(config), cameras=member(scene / "transforms.json"), splits=member(scene / "splits.json"),
                    calibration=member(PACKET / "calibration.json"), selection=member(PACKET / "selection.json"),
                    protocol=member(PROTOCOL), groups=groups(meta, source))
    if label == "B2":
        old = read(B2 / "meshes/supported/identity.json")
        check(step == P["baseline"]["selected_step"] and identity["checkpoint"]["sha256"] ==
              P["baseline"]["selected_checkpoint_sha256"], "Frozen B2 checkpoint")
        check(identity["cameras"] == old["geometry"], "Frozen B2 camera hash")
    immutable(output / "identity.json", identity)
    identity_hash = sha(output / "identity.json")
    native, rgb, extrinsic = camera_parameters(meta, read(PACKET / "calibration.json"))
    train = meta["train_filenames"]
    indices = {name: i for i, name in enumerate(train)}
    with patch.object(evaluator, "RUN", model_run), patch.object(evaluator, "SCENE", scene):
        _, pipe, loaded, _ = evaluator.load_model(config, step, "val")
    check(Path(loaded).resolve() == checkpoint.resolve(), "Renderer loaded another checkpoint")
    ds = pipe.datamanager.train_dataset
    check([p.relative_to(scene).as_posix() for p in ds.image_filenames] == train, "Renderer camera order")
    status_path = RUN / "state" / (label + "_local_surfaces.json")
    ledger = []
    for group in identity["groups"]:
        group_id = group["id"]
        directory = output / "submaps" / f"{group_id:04d}"
        directory.mkdir(exist_ok=True, parents=True)
        receipt = directory / "receipt.json"
        if receipt.is_file():
            row = read(receipt)
            check(row["identity"] == identity_hash and all(member(ROOT / entry["path"]) == entry
                  for entry in row["meshes"].values()), "Submap resume hash")
        else:
            sensor_vol, model_vol = volume(), volume()
            observations = []
            for name in group["names"]:
                stem = Path(name).stem
                pose = cameras[name]
                with np.load(PACKET / source[stem]["depth"]["file"]) as archive:
                    sensor_z = archive["depth"].astype(np.float32) / 1000.
                check(sensor_z.shape == (480, 848), "Native shape")
                native_count = integrate(sensor_vol, sensor_z, native, np.linalg.inv(pose @ extrinsic))
                with torch.inference_mode():
                    cam = ds.cameras[indices[name]:indices[name]+1].to(pipe.device)
                    pred = pipe.model.get_outputs_for_camera(camera=cam)
                    z = pred["depth"].reshape(720, 1280)
                    alpha = pred["accumulation"].reshape_as(z)
                    normal = 2 * pred["normal"].reshape(720, 1280, 3) - 1
                    valid = torch.isfinite(normal).all(dim=-1) & (normal.square().sum(dim=-1) > 0)
                    depth = torch.where(valid & alpha.isfinite() & (alpha >= .5), z, 0).cpu().numpy()
                count = integrate(model_vol, depth, rgb, np.linalg.inv(pose))
                observations.append(dict(id=stem, sensor_valid=native_count, model_valid=count))
            meshes, counts = {}, {}
            for kind, vol in (("sensor", sensor_vol), ("model", model_vol)):
                mesh = vol.extract_triangle_mesh()
                mesh.compute_vertex_normals()
                target = directory / (kind + ".ply")
                tmp = target.with_suffix(".pending.ply")
                if len(mesh.triangles):
                    check(o3d.io.write_triangle_mesh(str(tmp), mesh), f"Write failed: {target}")
                else:
                    from capture_metric3d_mesh import empty_ply
                    empty_ply(tmp, mesh=True)
                os.replace(tmp, target)
                counts[kind] = dict(vertices=len(mesh.vertices), triangles=len(mesh.triangles))
                meshes[kind] = member(target)
            del sensor_vol, model_vol, vol, mesh
            row = dict(group_id=group_id, identity=identity_hash, observations=observations, counts=counts, meshes=meshes)
            write(receipt, row)
        ledger.append(member(receipt))
        status = dict(run_id=P["run_id"], stage="A1" if label == "B2" else "C_export",
                      pid=os.getpid(), utc=utc(), model=label, submaps=len(ledger), total=len(identity["groups"]),
                      frames=sum(len(x["names"]) for x in identity["groups"][:len(ledger)]),
                      latest_counts=row["counts"], identity=identity_hash)
        write(status_path, status)
        print(json.dumps(status), flush=True)
    write(output / "submaps.json", dict(identity=identity_hash, receipts=ledger, frames=len(train)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("B2", "B3"))
    args = parser.parse_args()
    state = RUN / "state"
    state.mkdir(parents=True, exist_ok=True)
    with (state / (args.model + "_submaps.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        local_maps(args.model, B2 if args.model == "B2" else RUN)
