"""One recoverable Metric3D chain. Every formal stage has a run-bound output receipt."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading

from capture_metric3d_common import (BASE, GEOMETRY, P, PRIOR, PROTOCOL, RAW_PREFIX, REPORT, ROOT, RUN,
                                     SCENE, SOURCE, STATE, WEIGHT, demand, read, sha, write)
from capture_s01_resources import observe
from vr_room_run import CPU, PY, SCRIPT, monitor, now, stage, status


def run():
    from capture_metric3d_publish import deliver
    from capture_metric3d_weight import DIGEST
    demand(Path(os.environ["DN_RUN_ROOT"]) == RUN and Path(os.environ["DN_SCENE_DIR"]) == SCENE, "B1 isolation")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU])
    identity = dict(run=RUN.name, protocol=sha(PROTOCOL), b0=P["baseline"]["checkpoint_sha256"],
                    geometry=P["baseline"]["geometry_manifest_sha256"], weight_sha256=DIGEST,
                    prior_commit=P["prior"]["commit"], training_source=P["training"]["dn_commit"])
    ipath = STATE / "identity.json"
    if ipath.is_file():
        demand(identity == read(ipath), "existing B1 identity")
    else:
        demand(not (RUN / "training").exists() and not (STATE / "training.done.json").exists(), "unbound B1")
        write(ipath, identity)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "previews").mkdir(exist_ok=True)
    fingerprint = sha(ipath)

    def phase(name, script, args, out, threads=1, python=PY):
        mark = STATE / (name + ".done.json")
        if mark.is_file():
            old = read(mark)
            demand(old["identity"] == fingerprint and out.is_file() and old["sha256"] == sha(out), name)
            return
        stage(name, [python, SCRIPT / script, *args], threads=threads)
        demand(out.is_file(), name + " output")
        write(mark, dict(identity=fingerprint, output=out.relative_to(ROOT).as_posix(), sha256=sha(out), utc=now()))

    phase("baseline", "capture_metric3d_prepare.py", ["baseline"], REPORT / "frozen_inputs.json", 2)
    phase("weights", "capture_metric3d_weight.py", [], WEIGHT)
    inf = ROOT / ".local/conda/metric3d-bw/bin/python"
    demand(inf.is_file(), "isolated inference environment")
    phase("prior", "capture_metric3d_prior.py", [], REPORT/"normal_prior.json", 2, inf)
    phase("reader", "capture_metric3d_prepare.py", ["reader"], REPORT/"reader_check.json", 3)
    phase("training", "capture_metric3d_train.py", [], REPORT/"B1_refinement.json", 2)
    for split in ("val", "test"):
        phase(split, "capture_s01_eval.py", [split], REPORT / f"metrics_{split}.json", 2)
    phase("paired", "capture_metric3d_compare.py", [], REPORT/"paired_summary.json", 2)
    phase("video", "capture_metric3d_video.py", [], REPORT/"video_manifest.json", 2)
    phase("poisson", "capture_s01_mesh.py", ["export"], RUN/"mesh"/(RAW_PREFIX+"poisson_mesh.ply"), CPU)
    phase("raw_validation", "capture_s01_mesh.py", ["validate"], REPORT/"mesh_validation.json", CPU)
    for label in ("B0", "B1"):
        for action in ("sample", "fuse", "poisson", "crop"):
            phase(label+"_"+action, "capture_metric3d_mesh.py", [label, action],
                  RUN/"supported"/label/(action+".json"), CPU if action=="poisson" else 3)
    phase("surface_evaluation", "capture_metric3d_surface_eval.py", [], REPORT/"surface_summary.json", CPU)
    phase("report", "capture_metric3d_report.py", [], REPORT/"overview.md", 2)
    status("delivery")
    deliver("complete")
    status("complete", commit=read(STATE/"delivery_ready.json")["commit"])


if __name__ == "__main__":
    with (STATE/"runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("This exact Metric3D run already has an active coordinator")
        if (STATE/"delivery_ready.json").is_file() and read(STATE/"delivery_ready.json")["status"].startswith("completed"):
            sys.exit(0)
        (STATE/"exit_code").unlink(missing_ok=True)
        if (STATE/"delivery_ready.json").is_file():
            (STATE/"delivery_ready.json").rename(STATE/("previous_delivery_"+str(os.getpid())+".json"))
        stop = threading.Event()
        for fn in (monitor, observe):
            threading.Thread(target=fn, args=(stop,), daemon=True).start()
        status("starting", cpu_budget=CPU)
        try:
            run()
        except Exception as exc:
            last = read(STATE/"current.json")
            status("failed", failed_stage=last["stage"], reason=str(exc))
            (STATE/"exit_code").write_text("1\n")
            from capture_metric3d_publish import deliver
            deliver("blocked")
            raise
        else:
            (STATE/"exit_code").write_text("0\n")
        finally:
            stop.set()
