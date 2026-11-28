"""Own the experiment queue on AutoDL. Workers keep separate locks and receipts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/".local/runs/interactive-world-20261128-v1"
STATE=RUN/"state"
GIB=1024**3
VARIANTS=("R0C","R1","VR0C","VR1","VR2","R2","R3","R4","R5","R6","R7")
ZONE=ZoneInfo("Asia/Shanghai")


def now():
    return datetime.now(ZONE).isoformat(timespec="seconds")


def read(path, fallback=None):
    if not path.is_file():
        return fallback
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf8")
    os.replace(temp,path)


def event(stage,info):
    with (STATE/"coordinator-events.jsonl").open("a",encoding="utf8") as f:
        f.write(json.dumps(dict(time=now(),stage=stage,details=info),ensure_ascii=False)+"\n")
    print(now(),stage,info,flush=True)


def sample():
    cg=Path("/sys/fs/cgroup")
    limit=int((cg/"memory.max").read_text())
    current=int((cg/"memory.current").read_text())
    counters=dict((k,int(v)) for k,v in (r.split() for r in (cg/"memory.stat").read_text().splitlines()))
    host=int(next(r.split()[1] for r in Path("/proc/meminfo").read_text().splitlines() if r.startswith("MemAvailable:")))*1024
    clean_inactive=max(0,counters["inactive_file"]-counters["file_dirty"]-counters["file_writeback"])
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=memory.total,memory.free,utilization.gpu",
                                 "--format=csv,noheader,nounits"],text=True)
    total,free,busy=map(int,gpu.strip().splitlines()[0].split(","))
    quota,period=map(int,(cg/"cpu.max").read_text().split())
    return dict(time=now(),cpu_quota=quota//period,cgroup_total_bytes=current,cgroup_limit_bytes=limit,
                clean_inactive_file_bytes=clean_inactive,cgroup_working_bytes=current-clean_inactive,
                ram_available_bytes=max(0,min(host,limit-current+clean_inactive)),gpu_total_bytes=total*1024**2,
                gpu_free_bytes=free*1024**2,gpu_util_percent=busy,disk_free_bytes=shutil.disk_usage(RUN).free)


def directory(name):
    return RUN/("f_vr" if name.startswith("VR") else "c_dn")/name


def active_pid(pid,variant):
    try:
        args=Path(f"/proc/{int(pid)}/cmdline").read_bytes().split(b"\0")
        return variant.encode() in args and any(b"interactive_dn_" in arg for arg in args)
    except (FileNotFoundError,ProcessLookupError,ValueError):
        return False


def training_state(name):
    out=directory(name)
    st=out/"state"
    progress=read(st/"progress.json",{})
    ckpt=read(st/"checkpoint.json",{})
    pidfile=st/"worker.pid"
    pid=int(pidfile.read_text().strip()) if pidfile.is_file() and pidfile.read_text().strip().isdigit() else 0
    worker=active_pid(pid,name) if pid else False
    final=read(out/"training_complete.json",{})
    evaluate=out/"evaluation/complete.json"
    first=next(iter(sorted(st.glob("first_batch_*.json"))),None)
    measured=progress.get("resources") or (read(first,{}) if first else {}).get("resources",{})
    lastlog=(st/"current_log").read_text().strip() if (st/"current_log").is_file() else None
    did_exit=(st/"worker.exit").is_file() and (st/"current_log").exists() and (st/"worker.exit").stat().st_mtime>=(st/"current_log").stat().st_mtime
    done=int(ckpt.get("completed_updates",0))
    if final:
        goal=30000 if name.startswith("VR") else 240000
        if done!=goal or final.get("completed_updates")!=goal:
            raise ValueError(f"Training receipt disagrees with durable count: {name}")
    return dict(variant=name,active=worker,pid=pid if worker else None,log=lastlog,
                durable_updates=done,verified_updates_lower_bound=max(done,int(progress.get("step_in_progress",0))),
                phase="evaluate" if final else "train",training_complete=bool(final),evaluation_complete=evaluate.is_file(),
                first_batch=bool(first),measured=measured,
                exit_code=int((st/"worker.exit").read_text()) if did_exit and not worker else None)


def prerequisites(name):
    paths=[RUN/"restore/env.verified.json",RUN/"restore/metric.verified.json",RUN/"restore"/("vr.verified.json" if name.startswith("VR") else "hall.verified.json")]
    if name in ("VR1","VR2"):
        paths.append(RUN/"f_vr/supervision"/name/"manifest.json")
    elif name in ("R1","R2","R3","R4","R5"):
        group=name if name in ("R1","R2") else "R3"
        paths.append(RUN/"a_depth/supervision"/group/"manifest.json")
        if name=="R4":
            paths.append(RUN/"a_depth/boundary/manifest.json")
    if name in ("R6","R7"):
        evidence=read(RUN/"b_map/pose/accepted.json")
        if not evidence:
            return paths,"awaiting independent pose audit"
        if not evidence.get("accepted_frame_ids"):
            return paths,"not_triggered: no accepted pose"
        if not evidence.get("training_manifest_sha256"):
            return paths,"awaiting regenerated pose-specific inputs"
        if name=="R7":
            joint=read(RUN/"c_dn/integration_decision.json")
            if not joint:
                return paths,"awaiting separate boundary and pose results"
            if not (joint["independent_pose_benefit"] and joint["independent_boundary_benefit"]):
                return paths,"not_triggered: independent combination criteria unmet"
    missing=[str(p.relative_to(RUN)) for p in paths if not p.is_file()]
    return paths,("waiting for "+", ".join(missing)) if missing else None


def observe_peaks(hist,row,sample_):
    m=row["measured"]
    if not m:
        return
    for key in ("pss_bytes","uss_bytes","torch_reserved_peak","torch_allocated_peak"):
        hist[key]=max(hist.get(key,0),int(m.get(key,0)))
    if row["active"] and "prelaunch_working" in hist:
        inc=max(0,sample_["cgroup_working_bytes"]-hist["prelaunch_working"])
        hist["incremental_working_peak"]=max(hist.get("incremental_working_peak",0),inc)


def admission(name,row,active,peaks,host):
    if len(active)>=3:
        return False,"three heavy workers already active",{}
    if any((not a["first_batch"] or a["verified_updates_lower_bound"]<100) and not a["training_complete"] for a in active):
        return False,"measuring another worker's first batches",{}
    # Rank shared-image working sets using PSS/USS. The cgroup delta also
    # captures non-Python memory and never sums the shared RSS across workers.
    comparable=[p for k,p in peaks.items() if k.startswith("VR")==name.startswith("VR") and p.get("torch_reserved_peak")]
    fallback_ram=22*GIB if name.startswith("VR") else 79*GIB
    fallback_gpu=12*GIB if name.startswith("VR") else 20*GIB
    if comparable:
        ram=max(max(p.get("uss_bytes",0),p.get("pss_bytes",0),p.get("incremental_working_peak",0)) for p in comparable)
        gpu=max(p["torch_reserved_peak"] for p in comparable)
    else:
        ram,gpu=fallback_ram,fallback_gpu
    if row["phase"]=="evaluate":
        ram=max(ram,22*GIB)
    required_ram=int(1.25*ram+8*GIB)
    required_gpu=int(1.15*gpu+6*GIB)
    data=dict(ram_required_bytes=required_ram,gpu_required_bytes=required_gpu,evidence="measured" if comparable else "conservative initial estimate")
    if host["ram_available_bytes"]<required_ram:
        return False,"RAM admission",data
    if host["gpu_free_bytes"]<required_gpu:
        return False,"GPU admission",data
    return True,"admitted",data


def signature(row,paths):
    evidence=paths+list((ROOT/"scripts").glob("interactive_dn*"))+list((ROOT/"scripts").glob("interactive_vr*"))
    key=[(p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in sorted(evidence) if p.is_file()]
    return hashlib.sha256(json.dumps([row["phase"],row["durable_updates"],key]).encode()).hexdigest()


def cycle():
    host=sample()
    rows={k:training_state(k) for k in VARIANTS}
    ledger=read(STATE/"admission.json",{})
    active=[r for r in rows.values() if r["active"]]
    for k,r in rows.items():
        h=ledger.setdefault(k,{})
        observe_peaks(h,r,host)
        if r["active"] and "prelaunch_working" not in h:
            h["prelaunch_working"]=host["cgroup_working_bytes"]
            h["prelaunch_note"]="attached to already-running worker; pre-attach working set is unknown"
    # Keep formal jobs already admitted. Short external reference runs can use a
    # free slot while the more memory-intensive hall jobs measure their peaks.
    priority=sorted(VARIANTS,key=lambda k:(not rows[k]["training_complete"],VARIANTS.index(k)))
    for k in priority:
        row=rows[k]
        if row["evaluation_complete"]:
            row["queue_reason"]="evaluation receipt present"
            continue
        if row["active"]:
            row["queue_reason"]="running"
            continue
        paths,reason=prerequisites(k)
        if reason:
            row["queue_reason"]=reason
            continue
        h=ledger[k]
        key=signature(row,paths)
        if row["exit_code"] not in (0,None) and row["log"]!=h.get("last_failure_log"):
            h["failure_count"]=(h.get("failure_count",0) if key==h.get("failure_key") else 0)+1
            h.update(last_failure_log=row["log"],failure_key=key,last_failure_time=time.time())
            event("worker_exit",dict(variant=k,code=row["exit_code"],log=row["log"]))
        if h.get("failure_key")==key and h.get("failure_count",0)>=3:
            row["queue_reason"]="failed repeatedly with identical inputs and code; inspect retained log"
            continue
        if h.get("failure_key")==key and time.time()-h.get("last_failure_time",0)<30*h.get("failure_count",0):
            row["queue_reason"]="retry after recorded failure"
            continue
        ok,why,details=admission(k,row,active,ledger,host)
        row.update(queue_reason=why,admission=details)
        if ok:
            threads=8 if not active else 3
            if k in ("R6","R7") and not read(RUN/"b_map/pose/accepted.json",{}).get("scene"):
                row["queue_reason"]="awaiting pose-specific scene path"
                continue
            cmd=["bash",str(ROOT/"scripts/interactive_dn_launch.sh"),k,row["phase"]]
            call=subprocess.run(cmd,cwd=ROOT,env={**os.environ,"IW_DN_THREADS":str(threads)},capture_output=True,text=True)
            event("launch",dict(variant=k,phase=row["phase"],threads=threads,exit=call.returncode,message=call.stdout.strip(),**details))
            if call.returncode==0:
                h.update(prelaunch_working=host["cgroup_working_bytes"],launch_time=now())
                # Sample the new worker before making the next admission decision.
                break
    write(STATE/"admission.json",ledger)
    write(STATE/"queue.json",dict(run=RUN.name,coordinator_pid=os.getpid(),host=host,training=rows))
    with (STATE/"throughput.jsonl").open("a") as f:
        f.write(json.dumps(dict(host=host,active=[r["variant"] for r in active],
              observed_updates={k:r["verified_updates_lower_bound"] for k,r in rows.items()}),separators=(",",":"))+"\n")


def serve():
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/"coordinator.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("A coordinator already owns this run")
            return
        event("coordinator_started",dict(pid=os.getpid(),head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()))
        while True:
            try:
                cycle()
            except Exception as error:
                event("coordinator_error",f"{type(error).__name__}: {error}")
            time.sleep(20)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("serve","status"))
    if p.parse_args().command=="serve":
        serve()
    else:
        print(json.dumps(read(STATE/"queue.json",{}),ensure_ascii=False,indent=2))
