"""Build and verify the frozen RGB-D copy and the Omnidata input manifest."""
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import cv2
import numpy as np
from capture_rectify import files, read, save, sha, verify
from vr_room_run import ROOT, RUN, SRC, SCENE, STATE, CPU

PROTOCOL = ROOT / "configs/capture_s01_train_protocol.json"
P = read(PROTOCOL)
GEO = ROOT / P["paths"]["geometry_dataset"]
GEO_RUN = ROOT / P["paths"]["geometry_run"]
REPORT = ROOT / P["paths"]["report"]
WEIGHT = SRC / "omnidata_ckpt/omnidata_dpt_normal_v2.ckpt"


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def verify_base():
    expected = read(SCENE / "geometry_source_manifest.json")["files"]
    for row in expected:
        path = SCENE / row["path"]
        ensure(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], str(path))
        ensure(not path.samefile(GEO / row["path"]), "Writable hardlink to geometry")


def prepare():
    REPORT.mkdir(parents=True, exist_ok=True)
    verify(GEO)
    manifest = GEO / "manifest.json"
    ensure(sha(manifest) == P["input"]["geometry_manifest_sha256"], "geometry manifest")
    members = read(manifest)["files"]
    ensure(
        len(members) + 1 == 488
        and sum(f["bytes"] for f in members) + manifest.stat().st_size
        == P["input"]["geometry_bytes_including_manifest"],
        "geometry inventory",
    )
    if not SCENE.exists():
        stage = SCENE.with_name("inputs.building")
        stage.mkdir(exist_ok=True)
        subprocess.run(["cp", "-a", "--reflink=auto", str(GEO) + "/.", str(stage)], check=True)
        verify(stage)
        (stage / "manifest.json").rename(stage / "geometry_source_manifest.json")
        stage.rename(SCENE)
    ensure(sha(SCENE / "geometry_source_manifest.json") == sha(manifest), "copy identity")
    verify_base()
    save(
        REPORT / "inputs.json",
        dict(
            geometry=str(GEO),
            copy=str(SCENE),
            geometry_manifest_sha256=sha(manifest),
            geometry_files=488,
            geometry_bytes=P["input"]["geometry_bytes_including_manifest"],
            train_depth_coverage=read(GEO / "geometry.json")["depth_positive_fraction"],
            geometry_preserved=True,
            shared_inodes=0,
        ),
    )
    versions = {k: md.version(k) for k in ["torch", "nerfstudio", "gsplat", "open3d"]}
    versions["python"] = platform.python_version()
    ensure(versions == P["source"]["expected_versions"], "environment versions")
    save(
        REPORT / "environment.json",
        dict(
            versions=versions,
            python=sys.executable,
            import_paths={
                k: importlib.util.find_spec(k).origin
                for k in ["torch", "gsplat", "nerfstudio", "dn_splatter", "open3d"]
            },
            cpu_budget=CPU,
            affinity=sorted(os.sched_getaffinity(0)),
            memory_limit_bytes=int(Path("/sys/fs/cgroup/memory.max").read_text()),
            gpu=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.free",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
            disk_free_bytes=shutil.disk_usage(RUN).free,
            patches=files(ROOT / "patches"),
        ),
    )


def normal_manifest():
    out = SCENE / "normals_from_pretrain"
    stems = {p.stem for p in (SCENE / "images").glob("*.png")}
    ensure(
        len(stems) == 241
        and all({p.stem for p in out.glob("*." + ext)} == stems for ext in ("png", "npy")),
        "normal frame sets",
    )
    lo, hi, png_lo, png_hi = 1.0, 0.0, 255, 0
    for stem in sorted(stems):
        arr = np.load(out / (stem + ".npy"), allow_pickle=False)
        png = cv2.imread(str(out / (stem + ".png")), cv2.IMREAD_COLOR)
        ensure(
            arr.shape == (3, 720, 1280) and png.shape == (720, 1280, 3) and np.isfinite(arr).all(), stem
        )
        lo, hi = min(lo, float(arr.min())), max(hi, float(arr.max()))
        png_lo, png_hi = min(png_lo, int(png.min())), max(png_hi, int(png.max()))
    ensure(0 <= lo <= hi <= 1, "normal range")
    records = [json.loads(line) for line in (STATE / "stages.jsonl").read_text().splitlines()]
    normal_command = next(
        r["command"] for r in reversed(records) if r["stage"] == "omnidata" and r["exit_code"] == 0
    )
    info = dict(
        model="Omnidata DPT Hybrid low",
        source=P["source"]["commit"],
        source_script_sha256=sha(SRC / "dn_splatter/scripts/normals_from_pretrain.py"),
        environment_sha256=sha(REPORT / "environment.json"),
        command=normal_command,
        internal_hw=[384, 384],
        output_hw=[720, 1280],
        count_per_format=241,
        png_rgb_uint8_range=[png_lo, png_hi],
        npy_chw_float_encoded_01_range=[lo, hi],
        checkpoint=str(WEIGHT.resolve()),
        checkpoint_sha256=sha(WEIGHT),
        protocol_sha256=sha(PROTOCOL),
        geometry_manifest_sha256=sha(SCENE / "geometry_source_manifest.json"),
    )
    save(SCENE / "training_inputs.json", info)
    verify_base()
    save(SCENE / "training_manifest.json", dict(files=files(SCENE, ["training_manifest.json"])))
    save(
        REPORT / "normal_prior.json",
        dict(
            **info,
            training_manifest_sha256=sha(SCENE / "training_manifest.json"),
            training_manifest=str(SCENE / "training_manifest.json")
        ),
    )
    verify(GEO)


if __name__ == "__main__":
    {"copy": prepare, "manifest": normal_manifest}[sys.argv[1]]()
