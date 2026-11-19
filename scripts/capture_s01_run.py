"""Capture-specific persistent workflow. Resume within exactly one input/protocol identity."""
import argparse
import fcntl
import os
import subprocess
import threading

from pathlib import Path
from capture_s01_inputs import P, PROTOCOL, REPORT, WEIGHT, GEO, ensure
from capture_s01_resources import observe
from capture_rectify import read, sha
from vr_room_publish import patch_matches
from vr_room_run import ROOT, RUN, SCENE, SRC, STATE, SCRIPT, CPU, PY, stage, status, monitor, save


def evidence(name):
    paths = {
        "copy": [SCENE / "geometry_source_manifest.json", REPORT / "inputs.json"],
        "cross_view": [REPORT / "cross_view_diagnostic.json"],
        "manifest": [SCENE / "training_manifest.json", REPORT / "normal_prior.json"],
        "reader": [REPORT / "reader_check.json"],
        "training": [RUN / "training_complete.json"],
        "val": [REPORT / "metrics_val.json"],
        "test": [REPORT / "metrics_test.json"],
        "poisson": [
            RUN / "mesh/sampling.json",
            RUN / "mesh/DepthAndNormalMapsPoisson_poisson_mesh.ply",
            RUN / "mesh/DepthAndNormalMapsPoisson_pcd.ply",
        ],
        "mesh_validation": [REPORT / "mesh_validation.json"],
        "publish": [REPORT / "artifact_manifest.json", REPORT / "run_manifest.json"],
    }
    if name == "omnidata":
        for ext in ("png", "npy"):
            ensure(
                {p.stem for p in (SCENE / "normals_from_pretrain").glob("*." + ext)}
                == {p.stem for p in (SCENE / "images").glob("*.png")},
                "normal frame coverage",
            )
    if name == "training":
        completed = read(RUN / "training_complete.json")
        ensure(
            completed["step"] == 29999
            and sha(Path(completed["checkpoint"])) == completed["checkpoint_sha256"]
            and completed["training_manifest_sha256"] == sha(SCENE / "training_manifest.json"),
            "checkpoint binding",
        )
    if name in ("val", "test"):
        m = read(REPORT / ("metrics_" + name + ".json"))
        ensure(
            m["frame_count"] == 24
            and m["checkpoint_sha256"] == read(RUN / "training_complete.json")["checkpoint_sha256"],
            "eval binding",
        )
    return {str(p.relative_to(ROOT)): sha(p) for p in paths.get(name, [])}


def main(until):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU])
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SRC, text=True).strip()
    ensure(source == P["source"]["commit"], "historical source")
    ensure(
        patch_matches(SRC, "dn-811f5cd-*.patch")
        and patch_matches(ROOT / ".local/upstream/gsplat", "gsplat-1.0.0-*.patch"),
        "source patch identity",
    )
    prior = read(ROOT / "reports/vr-room-iphone-30k/run_manifest.json")
    expected_weight = next(item["sha256"] for item in prior["inputs"] if item["name"] == WEIGHT.name)
    identity = dict(
        run=RUN.name,
        protocol_sha256=sha(PROTOCOL),
        source=source,
        geometry_manifest_sha256=sha(GEO / "manifest.json"),
        checkpoint_sha256=sha(WEIGHT),
    )
    ensure(
        identity["geometry_manifest_sha256"] == P["input"]["geometry_manifest_sha256"]
        and identity["checkpoint_sha256"] == expected_weight,
        "input identity",
    )
    identity_path = STATE / "identity.json"
    if identity_path.exists():
        ensure(read(identity_path) == identity, "run identity changed")
    else:
        ensure(not (RUN / "training").exists() and not SCENE.exists(), "unbound existing outputs")
        save(identity_path, identity)
    stages = [
        ("copy", [PY, SCRIPT / "capture_s01_inputs.py", "copy"], 1),
        ("cross_view", [PY, SCRIPT / "capture_s01_diagnostic.py"], 1),
        (
            "omnidata",
            [
                PY,
                "dn_splatter/scripts/normals_from_pretrain.py",
                "--data-dir",
                SCENE,
                "--save-path",
                SCENE / "normals_from_pretrain",
                "--resolution",
                "low",
                "--model-type",
                "omnidata",
            ],
            2,
        ),
        ("manifest", [PY, SCRIPT / "capture_s01_inputs.py", "manifest"], 1),
        ("reader", [PY, SCRIPT / "capture_s01_train.py", "reader"], 2),
        ("training", [PY, SCRIPT / "capture_s01_train.py", "train"], 2),
        ("val", [PY, SCRIPT / "capture_s01_eval.py", "val"], 2),
        ("test", [PY, SCRIPT / "capture_s01_eval.py", "test"], 2),
        ("poisson", [PY, SCRIPT / "capture_s01_mesh.py", "export"], CPU),
        ("mesh_validation", [PY, SCRIPT / "capture_s01_mesh.py", "validate"], CPU),
        ("publish", [PY, SCRIPT / "capture_s01_publish.py"], 1),
    ]
    for name, args, threads in stages:
        stamp = STATE / (name + ".done.json")
        if stamp.is_file():
            recorded = read(stamp)
            ensure(recorded["identity"] == identity, "phase identity")
            actual = evidence(name)
            ensure(recorded.get("evidence", actual) == actual, "phase evidence changed")
        else:
            stage(name, args, threads=threads)
            save(stamp, dict(identity=identity, evidence=evidence(name)))
        if name == until:
            break
    status("complete" if until == "publish" else "stage_complete", last=until)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--until",
        default="publish",
        choices=[
            "copy",
            "cross_view",
            "omnidata",
            "manifest",
            "reader",
            "training",
            "val",
            "test",
            "poisson",
            "mesh_validation",
            "publish",
        ],
    )
    args = parser.parse_args()
    with (STATE / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = threading.Event()
        threading.Thread(target=monitor, args=(stop,), daemon=True).start()
        threading.Thread(target=observe, args=(stop,), daemon=True).start()
        try:
            main(args.until)
        except Exception as exc:
            status("failed", reason=str(exc))
            raise
        finally:
            stop.set()
