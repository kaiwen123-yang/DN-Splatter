"""Single durable B2 stage chain, using independent processes for memory-intensive phases."""
import fcntl
import os
from pathlib import Path
import sys
import threading
import time

from capture_quality_paths import B1, GEOM, P, ROOT, RUN, SCENE, SPEC, STATE, check, read, sha, write
from capture_s01_resources import observe
from vr_room_run import CPU, PY, SCRIPT, monitor, now, stage, status


def run():
    check(Path(os.environ["DN_RUN_ROOT"])==RUN and Path(os.environ["DN_SCENE_DIR"])==SCENE,"B2 isolation")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU])
    identity=dict(run_id=P["run_id"],protocol_sha256=sha(SPEC),baseline=P["baseline"]["checkpoint_sha256"],
                  geometry=P["baseline"]["original_geometry_manifest_sha256"])
    check(read(B1/"training_complete.json")["checkpoint_sha256"]==identity["baseline"],"B1 final identity")
    ip=STATE/"b2_identity.json"
    if ip.exists(): check(read(ip)==identity,"run identity")
    else: write(ip,identity)
    identity_hash=sha(ip)

    def phase(name,script,args,output,threads=CPU,python=PY):
        marker=STATE/(name+".b2-done.json")
        if marker.exists():
            old=read(marker)
            check(old["identity"]==identity_hash and output.is_file() and sha(output)==old["sha256"],name)
            return
        stage(name,[python,"-u",SCRIPT/script,*args],cwd=ROOT,threads=threads)
        check(output.is_file(),f"{name} receipt missing at {output}")
        write(marker,dict(identity=identity_hash,path=output.relative_to(ROOT).as_posix(),sha256=sha(output),utc=now()))

    status("B1_validation",cpu_budget=CPU)
    external_pid=STATE/"b1_pid"
    if not (RUN/"b1_val/curve.json").is_file():
        if external_pid.is_file() and not (STATE/"b1_exit").exists():
            pid=int(external_pid.read_text())
            while Path(f"/proc/{pid}").exists() and not (STATE/"b1_exit").exists():
                time.sleep(10)
        if not (RUN/"b1_val/curve.json").is_file():
            stage("B1_resume",[PY,"-u",SCRIPT/"capture_quality_b1_val.py"],cwd=ROOT)
    curve=read(RUN/"b1_val/curve.json")
    check(curve["run_id"]==P["run_id"] and len(curve["rows"])==6 and
          all(row.get("status")=="missing" or row["frame_count"]==184 for row in curve["rows"]),"B1 curve")
    phase("inputs","capture_quality_init.py",["copy"],GEOM/"input_copy.json",1)
    phase("pose","capture_quality_pose.py",[],GEOM/"pose.json",1)
    phase("tracks","capture_quality_tracks.py",[],GEOM/"tracks.npz",1,ROOT/".local/conda/capture-geometry/bin/python")
    phase("initialization","capture_quality_init.py",["build"],GEOM/"initialization.json")
    phase("sensor_tsdf","capture_quality_tsdf.py",["sensor"],RUN/"meshes/sensor_tsdf.json")
    phase("training","capture_quality_train.py",[],STATE/"selection.json",2)
    phase("model_tsdf","capture_quality_tsdf.py",["model"],RUN/"meshes/model_tsdf.json")
    for action in ("sample","fuse","poisson","crop"):
        phase("supported_"+action,"capture_quality_tsdf.py",[action],RUN/"meshes/supported"/(action+".json"),
              CPU if action=="poisson" else 2)
    phase("surface_val","capture_quality_surfaces.py",["val"],RUN/"evaluation/surfaces_val.json")
    phase("test","capture_quality_eval.py",["test"],RUN/"evaluation/test_complete.json",2)
    phase("surface_test","capture_quality_surfaces.py",["test"],RUN/"evaluation/surfaces_test.json")
    phase("video","capture_quality_video.py",[],RUN/"video/video_manifest.json",2)
    phase("publication","capture_quality_publish.py",[],STATE/"delivery_ready.json",1)
    status("complete",commit=read(STATE/"delivery_ready.json")["commit"])


if __name__=="__main__":
    with (STATE/"b2.lock").open("a") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: sys.exit("B2 already running")
        stop=threading.Event()
        for fn in (monitor,observe):
            threading.Thread(target=fn,args=(stop,),daemon=True).start()
        (STATE/"b2_exit").unlink(missing_ok=True)
        try:
            run()
        except Exception as exc:
            last=read(STATE/"current.json")
            status("failed",failed_stage=last["stage"],reason=str(exc))
            (STATE/"b2_exit").write_text("1\n")
            raise
        else:
            (STATE/"b2_exit").write_text("0\n")
        finally:
            stop.set()
