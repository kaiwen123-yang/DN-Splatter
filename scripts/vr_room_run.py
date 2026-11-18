"""Single-session runner for the fixed MuSHRoom iPhone protocol."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(os.environ["DN_PROJECT_ROOT"])
SRC = Path(os.environ["DN_SOURCE_DIR"])
SCENE = Path(os.environ["DN_SCENE_DIR"])
RUN = Path(os.environ["DN_RUN_ROOT"])
STATE = RUN / "state"
LOGS = RUN / "logs"
PY = sys.executable
CPU = int(os.environ["DN_CPU_BUDGET"])
SCRIPT = ROOT / "scripts"
START = dt.datetime.now(dt.timezone.utc).isoformat()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, doc):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def status(stage, **facts):
    save(STATE / "current.json", {"stage": stage, "pid": os.getpid(), "updated_utc": now(), **facts})
    print(now(), stage, flush=True)


def monitor(stop):
    while not stop.is_set():
        try:
            stage_name = json.loads((STATE / "current.json").read_text())["stage"]
            raw = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True, timeout=8)
            mem, util = map(int, raw.strip().split(","))
            cg = Path("/sys/fs/cgroup")
            cpu = dict(line.split() for line in (cg/"cpu.stat").read_text().splitlines())
            row = {"utc": now(), "stage": stage_name, "gpu_used_mib_device": mem, "gpu_util_pct_device": util,
                   "cgroup_memory_bytes": int((cg/"memory.current").read_text()),
                   "cgroup_cpu_usage_usec": int(cpu["usage_usec"])}
            with (STATE/"resources.jsonl").open("a") as f:
                f.write(json.dumps(row)+"\n")
        except (ValueError, OSError, KeyError, subprocess.SubprocessError):
            pass
        stop.wait(15)


def stage(name, command, cwd=SRC, threads=1):
    args = list(map(str, command))
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENCV_FOR_THREADS_NUM="1")
    begin = time.monotonic()
    row = {"stage": name, "command": shlex.join(args), "cwd": str(cwd.relative_to(ROOT)) or ".",
           "started_utc": now(), "omp_threads": threads, "blas_threads": 1}
    status(name, started_utc=row["started_utc"])
    with (LOGS / (name + ".log")).open("a") as log:
        log.write("\n" + row["started_utc"] + " " + row["command"] + "\n")
        log.flush()
        proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        status(name, child_pid=proc.pid, started_utc=row["started_utc"])
        row["exit_code"] = proc.wait()
    row["duration_seconds"] = round(time.monotonic() - begin, 2)
    row["ended_utc"] = now()
    with (STATE / "stages.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    if row["exit_code"]:
        raise RuntimeError(f"{name} exited {row['exit_code']}; see logs/{name}.log")


def require_input(name):
    if (RUN / ("input-" + name + ".json")).is_file():
        return
    dest = ROOT / ".local/data/downloads" / name
    while not dest.exists():
        part = dest.with_name(name + ".part")
        status("input_transfer", input=name, received_bytes=part.stat().st_size if part.exists() else 0)
        time.sleep(45)
    stage("extract-" + name, [PY, SCRIPT / "vr_room_inputs.py", "extract", name], ROOT)


def latest_checkpoint():
    import torch
    files = sorted((RUN / "training").glob("**/nerfstudio_models/step-*.ckpt"),
                   key=lambda p: (int(p.stem.removeprefix("step-")), p.stat().st_mtime), reverse=True)
    for path in files:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            step = int(ck["step"])
            if step != int(path.stem.removeprefix("step-")) or not ck.get("optimizers") or not ck.get("schedulers"):
                raise ValueError("incomplete optimizer/scheduler state")
            means = next(v for k,v in ck["pipeline"].items() if k.endswith("gauss_params.means"))
            return path, step, int(means.shape[0])
        except (RuntimeError, EOFError, KeyError, ValueError, StopIteration, OSError) as err:
            print("Unusable checkpoint", path.name, type(err).__name__, flush=True)
    if files:
        raise ValueError("Existing checkpoints could not be read; refusing a duplicate from-scratch run")
    return None, -1, None


def train():
    ckpt, step, count = latest_checkpoint()
    remaining = 30000 - (step + 1)
    if remaining < 0:
        raise ValueError("Saved step exceeds the protocol")
    if remaining:
        args = ["ns-train", "dn-splatter", "--output-dir", RUN / "training",
                "--experiment-name", "vr_room_iphone_dn_official", "--machine.seed", "42",
                "--max-num-iterations", remaining, "--steps-per-save", "5000",
                "--save-only-latest-checkpoint", "False", "--vis", "tensorboard",
                "--viewer.quit-on-train-completion", "True"]
        if ckpt:
            args += ["--load-dir", ckpt.parent, "--load-step", step, "--load-scheduler", "True"]
        args += ["--pipeline.model.use-depth-loss", "True", "--pipeline.model.depth-loss-type", "EdgeAwareLogL1",
                 "--pipeline.model.sensor-depth-lambda", "0.2", "--pipeline.model.use-normal-loss", "True",
                 "--pipeline.model.normal-lambda", "0.1", "--pipeline.model.use-normal-tv-loss", "True",
                 "--pipeline.model.normal-supervision", "mono", "--pipeline.model.two-d-gaussians", "True",
                 "mushroom", "--data", SCENE, "--mode", "iphone", "--normal-format", "opengl",
                 "--eval-mode", "all", "--load-depths", "True", "--depth-mode", "sensor",
                 "--load-normals", "True", "--normals-from", "pretrained",
                 "--use-faro-scanner-depths", "False", "--use-faro-scanner-pd", "False"]
        with (STATE / "training_segments.jsonl").open("a") as f:
            f.write(json.dumps({"started_utc": now(), "saved_step": step if ckpt else None,
                "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt else None, "requested_updates": remaining}) + "\n")
        stage("training", args, threads=max(1, CPU // 2))
        ckpt, step, count = latest_checkpoint()
    if step != 29999:
        raise ValueError(f"Expected checkpoint step 29999, found {step}")
    cfg = ckpt.parents[1] / "config.yml"
    if not cfg.is_file():
        raise ValueError("Missing final config")
    save(RUN / "training_complete.json", {"checkpoint": str(ckpt.relative_to(ROOT)), "step": step,
         "cumulative_updates": step+1, "final_gaussians": count, "config": str(cfg.relative_to(ROOT))})
    return cfg


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU])
    for name in ("vr_room_iphone.tar.gz", "vr_room_mesh_pd.tar.gz"):
        require_input(name)
    while not (STATE / "install.exit").exists():
        status("installing_blackwell_dependencies")
        time.sleep(45)
    if (STATE / "install.exit").read_text().strip() != "0":
        raise RuntimeError("Dependency installation failed; see logs/install.log")
    require_input("omnidata_dpt_normal_v2.ckpt")
    for capture in ("long_capture", "short_capture"):
        folder = SCENE / "iphone" / capture
        marker = STATE / ("normals-" + capture + ".json")
        if not marker.exists():
            stage("normals-" + capture, [PY, SRC / "dn_splatter/scripts/normals_from_pretrain.py",
                "--data-dir", folder, "--transforms-name", "transformations_colmap.json", "--model-type", "omnidata",
                "--resolution", "low", "--save-path", folder / "normals_from_pretrain"], threads=min(CPU, 2))
            doc = json.loads((folder / "transformations_colmap.json").read_text())
            for frame in doc["frames"]:
                if not (folder / "normals_from_pretrain" / (Path(frame["file_path"]).stem + ".png")).is_file():
                    raise ValueError(f"Missing normal {frame['file_path']}")
            save(marker, {"frames": len(doc["frames"])})
    if not (RUN / "frames.json").exists():
        stage("validate-inputs", [PY, SCRIPT / "vr_room_inputs.py", "frames"], ROOT)
    if not (RUN / "blackwell.json").exists():
        stage("verify-blackwell", [PY, SCRIPT / "verify_blackwell.py"], ROOT)
    # Construct the original training-only initialization once before ns-train.
    ply = SCENE / "iphone/long_capture/iphone_pointcloud.ply"
    if not ply.exists():
        code = ("from pathlib import Path; import sys,random,numpy as np; random.seed(42); np.random.seed(42); "
                "from dn_splatter.data.mushroom_utils.pointcloud_utils import generate_iPhone_pointcloud_within_sequence as generate; "
                "generate(Path(sys.argv[1]))")
        stage("initialize", [PY, "-c", code, ply.parent], threads=CPU)
        if not ply.is_file():
            raise ValueError("Initializer did not produce a point cloud")
    cfg = train()
    evaluate = RUN / "evaluation"
    evaluate.mkdir(exist_ok=True)
    raw = evaluate / "rgb_depth_official.json"
    if not raw.is_file():
        stage("rgb-depth-evaluation", ["ns-eval", "--load-config", cfg, "--output-path", raw], threads=max(1, CPU//2))
    mesh = RUN / "mesh/DepthAndNormalMapsPoisson_poisson_mesh.ply"
    if not (STATE / "mesh-export.ok").exists():
        stage("poisson", ["gs-mesh", "dn", "--load-config", cfg, "--output-dir", mesh.parent,
              "--total-points", "5000000", "--poisson-depth", "9"], threads=CPU)
        if not mesh.is_file() or mesh.stat().st_size < 1000:
            raise ValueError("Missing Poisson mesh")
        (STATE / "mesh-export.ok").touch()
    if not (mesh.parent / "mesh_metrics.json").exists():
        p3d = ROOT / ".local/upstream/pytorch3d"
        if subprocess.check_output(["git", "-C", p3d, "rev-parse", "HEAD"], text=True).strip() != "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba":
            raise ValueError("Unexpected PyTorch3D geometry dependency")
        if subprocess.run([PY, "-c", "from pytorch3d import _C"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            stage("install-mesh-rasterizer", [PY, "-m", "pip", "install", "--no-build-isolation", p3d], ROOT)
        stage("geometry-evaluation", [PY, SRC / "dn_splatter/eval/eval_mesh_mushroom_vis_cull.py",
              "--gt-mesh-path", SCENE, "--pred-mesh-path", mesh, "--device", "iphone"], threads=CPU)
    require_input("reference_depth.tar.gz")
    renders = RUN / "renders_official"
    renders.mkdir(exist_ok=True)
    # Historical save_outputs_helper prepends os.getcwd(); its render argument must be relative.
    shutil.copyfile(cfg.parent / "dataparser_transforms.json", renders / "dataparser_transforms.json")
    raw_faro = evaluate / "rgb_depth_faro_official_raw.json"
    if not raw_faro.exists():
        stage("render-and-faro", ["ns-eval", "--load-config", cfg, "--output-path", raw_faro,
              "--render-output-path", os.path.relpath(renders, SRC)], threads=max(1, CPU//2))
    stage("audit-and-previews", [PY, SCRIPT / "vr_room_evaluation.py"], ROOT, threads=CPU)
    stage("publication", [PY, SCRIPT / "vr_room_publish.py", "complete"], ROOT)
    status("complete", cumulative_updates=30000)


if __name__ == "__main__":
    lock = (STATE / "workflow.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("VR Room workflow already active")
    status("starting", started_utc=START)
    stop = threading.Event()
    threading.Thread(target=monitor, args=(stop,), daemon=True).start()
    try:
        main()
    except Exception as exc:
        old = json.loads((STATE / "current.json").read_text())
        status("blocked", failed_stage=old["stage"], error=str(exc))
        subprocess.run([PY, str(SCRIPT / "vr_room_publish.py"), "blocked"], cwd=ROOT)
        (STATE / "workflow.exit").write_text("1\n")
        raise
    finally:
        stop.set()
    (STATE / "workflow.exit").write_text("0\n")
