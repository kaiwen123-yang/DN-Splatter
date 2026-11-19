"""Run the full packet through one global map, a fresh 30k model, and final delivery."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

from capture_full_bridge import select_map, prepare_aux
from capture_rectify import read, save, sha, verify
from capture_s01_inputs import FULL, GEO, GEO_RUN, REPORT, PROTOCOL, P, WEIGHT, ensure
from capture_s01_resources import observe
from vr_room_publish import patch_matches
from vr_room_run import ROOT, RUN, SCENE, STATE, SRC, SCRIPT, PY, CPU, monitor, now, stage, status

PACKET=GEO_RUN/"transfer"
SFM=ROOT/P["paths"]["geometry_environment"]/"bin/python"
FINAL_MAP=GEO_RUN/"sfm/map_full.json"


def run():
    ensure(FULL and RUN==ROOT/P["paths"]["run"] and SCENE==ROOT/P["paths"]["training_inputs"],"full run paths")
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:CPU])
    verify(PACKET)
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=SRC,text=True).strip()
    ensure(source==P["source"]["upstream_commit"] and patch_matches(SRC,"dn-811f5cd-*.patch")
           and patch_matches(ROOT/".local/upstream/gsplat","gsplat-1.0.0-*.patch"),"historical source/patches")
    identity=dict(run=RUN.name,bag=read(PACKET/"identity.json")["source"]["sha256"],
                  protocol_sha256=sha(PROTOCOL),packet_sha256=sha(PACKET/"manifest.json"),upstream=source,
                  selection_sha256=sha(PACKET/"selection.json"),split_sha256=sha(PACKET/"splits.json"),
                  weight_sha256=sha(WEIGHT),patches={p.name:sha(p) for p in sorted((ROOT/"patches").glob("*.patch"))})
    ensure(identity["bag"]==P["capture"]["source_bag_sha256"] and
           identity["weight_sha256"]==P["normal_prior"]["checkpoint_sha256"],"source input identity")
    binding=STATE/"identity.json"
    if binding.exists():
        ensure(read(binding)==identity,"existing full-run identity")
    else:
        ensure(not (RUN/"training").exists() and not SCENE.exists(),"unbound prior output")
        save(binding,identity)
    REPORT.mkdir(parents=True,exist_ok=True)
    k=sha(binding)

    def phase(name,cmd,output,threads=1):
        mark=STATE/(name+".done.json")
        if mark.exists():
            rec=read(mark)
            ensure(rec["identity"]==k and rec["output_sha256"]==sha(output),"stage identity: "+name)
            return
        stage(name,cmd,threads=threads)
        save(mark,dict(identity=k,output=str(output),output_sha256=sha(output),utc=now()))

    def dn(name,script,args,output,threads=1):
        phase(name,[PY,SCRIPT/script,*args],output,threads)

    def sfm(name,op,output,*flags):
        phase(name,[SFM,SCRIPT/"capture_sfm.py",GEO_RUN,op,"--device","cuda",*flags],output)

    dn("rectify","capture_rectify.py",[PACKET,GEO_RUN/"rect","--all","--workers",str(CPU)],GEO_RUN/"rect/registration.json")
    # Feature and matching phases mutate the same SQLite database. Seal stage-specific summaries.
    def dbstage(name,op,db,*flags):
        import sqlite3
        mark=STATE/(name+".done.json")
        if mark.exists():
            ensure(read(mark)["identity"]==k and db.is_file(),"database stage "+name)
            return
        stage(name,[SFM,SCRIPT/"capture_sfm.py",GEO_RUN,op,"--device","cuda",*flags])
        with sqlite3.connect(db) as conn:
            table="two_view_geometries" if op=="match" else "images"
            count=conn.execute(f"select count(*) from {table}").fetchone()[0]
        save(mark,dict(identity=k,rows=count,table=table,utc=now()))
    db=GEO_RUN/"sfm/train.db"
    dbstage("features","features",db)
    dbstage("match","match",db,"--exhaustive")
    sfm("map","map",GEO_RUN/"sfm/map_exhaustive.json","--exhaustive")
    if select_map(GEO_RUN):
        status("bridge",detail="Awaiting bounded native RGB from local source")
        prepare_aux(GEO_RUN)
        if read(GEO_RUN/"bridge/frames.json"):
            dbstage("bridge_features","features",db,"--bridge")
            dbstage("bridge_match","match",db,"--bridge","--exhaustive")
            sfm("bridge_map","map",GEO_RUN/"sfm/map_bridge.json","--exhaustive","--bridge")
        paths=[GEO_RUN/"sfm/map_exhaustive.json",GEO_RUN/"sfm/map_bridge.json"]
        maps=[read(p) for p in paths if p.exists()]
        def coverage(m):
            return max((len(c["registered_primary"]) for c in m["components"]),default=0)
        save(FINAL_MAP,max(maps,key=coverage))
    ensure(read(FINAL_MAP)["components"],"No usable unified training map after bounded bridge")
    dbstage("queries","queries",GEO_RUN/"sfm/queries.db")
    phase("localize",[SFM,SCRIPT/"capture_localize.py",GEO_RUN,"--map","map_full.json"],GEO_RUN/"sfm/pose_quality.json")
    dn("metric","capture_metric.py",[GEO_RUN],GEO_RUN/"metric/poses.json")
    if not (STATE/"dataset.done.json").exists() and (GEO/"manifest.json").is_file():
        verify(GEO)
        ensure(read(GEO/"geometry.json")["packet_manifest_sha256"]==identity["packet_sha256"],"existing geometry")
        save(STATE/"dataset.done.json",dict(identity=k,output=str(GEO/"manifest.json"),output_sha256=sha(GEO/"manifest.json")))
    if not GEO.exists() and GEO.with_name(GEO.name+".building").exists():
        import datetime
        old=GEO.with_name(GEO.name+".building")
        old.rename(old.with_name(old.name+"."+datetime.datetime.now().strftime("%Y%m%dT%H%M%S")))
    dn("dataset","capture_dataset.py",["export",GEO_RUN,GEO,"--full"],GEO/"manifest.json",4)
    dn("copy","capture_s01_inputs.py",["copy"],REPORT/"inputs.json")
    phase("omnidata",[PY,"dn_splatter/scripts/normals_from_pretrain.py","--data-dir",SCENE,"--save-path",
                      SCENE/"normals_from_pretrain","--resolution","low","--model-type","omnidata"],
          SCENE/"normals_from_pretrain"/(Path(read(SCENE/"transforms.json")["frames"][-1]["file_path"]).stem+".npy"),2)
    dn("manifest","capture_s01_inputs.py",["manifest"],SCENE/"training_manifest.json")
    dn("reader","capture_s01_train.py",["reader"],REPORT/"reader_check.json",2)
    dn("training","capture_full_train.py",[],RUN/"training_complete.json",2)
    for split in ("val","test"):
        dn(split,"capture_s01_eval.py",[split],REPORT/("metrics_"+split+".json"),2)
    dn("video","capture_full_video.py",[],REPORT/"video_manifest.json",2)
    dn("poisson","capture_s01_mesh.py",["export"],RUN/"mesh/DepthAndNormalMapsPoisson_poisson_mesh.ply",CPU)
    dn("mesh_validation","capture_s01_mesh.py",["validate"],REPORT/"mesh_validation.json",CPU)
    dn("publish","capture_full_report.py",[],REPORT/"run_manifest.json")
    status("complete")
    stage("delivery",[PY,SCRIPT/"capture_full_delivery.py","complete"],ROOT)
    status("complete",commit=read(STATE/"github_complete.json")["commit"])


if __name__=="__main__":
    with (STATE/"runner.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (STATE/"exit_code").unlink(missing_ok=True)
        stop=threading.Event()
        for fn in (monitor,observe):
            threading.Thread(target=fn,args=(stop,),daemon=True).start()
        try:
            run()
        except Exception as exc:
            prior=read(STATE/"current.json") if (STATE/"current.json").exists() else {}
            status("failed",failed_stage=prior.get("stage"),reason=str(exc))
            (STATE/"exit_code").write_text("1\n")
            subprocess.run([PY,SCRIPT/"capture_full_delivery.py","blocked"],cwd=ROOT)
            raise
        else:
            (STATE/"exit_code").write_text("0\n")
        finally:
            stop.set()
