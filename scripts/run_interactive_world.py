"""Own the experiment queue on AutoDL. Workers keep separate locks and receipts."""
import argparse
from functools import lru_cache
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
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
TRAIN_ORDER=("R0C","R1","VR0C","VR1","VR2","R2","R4","R3","R5","R6","R7")
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


@lru_cache(maxsize=1)
def protected_file_bytes(period):
    """Keep the complete live input and shared-cache working set in the estimate."""
    shared=RUN/"shared"
    scene=RUN/"restore/tree/.local/runs/capture-full-quality-v3-20261122-v1"
    paths=[shared/"image-cache",shared/"packet-cache",shared/"initial-geometry",shared/"torch",
           scene/"inputs",scene/"weights"]
    if not all(p.is_dir() for p in paths):
        return None
    result=subprocess.run(["du","--apparent-size","-sb","--",*map(str,paths)],capture_output=True,text=True)
    if result.returncode or len(result.stdout.splitlines())!=len(paths):
        return None
    return sum(int(line.split(None,1)[0]) for line in result.stdout.splitlines())


def sample():
    cg=Path("/sys/fs/cgroup")
    limit=int((cg/"memory.max").read_text())
    current=int((cg/"memory.current").read_text())
    counters=dict((k,int(v)) for k,v in (r.split() for r in (cg/"memory.stat").read_text().splitlines()))
    host=int(next(r.split()[1] for r in Path("/proc/meminfo").read_text().splitlines() if r.startswith("MemAvailable:")))*1024
    clean_inactive=max(0,counters["inactive_file"]-counters["file_dirty"]-counters["file_writeback"])
    # Active clean pages remain in the *availability* working set. For learning
    # launch pressure, distinguish them from memory a new process cannot reclaim.
    clean_file=max(0,counters["file"]-counters["shmem"]-counters["file_dirty"]-counters["file_writeback"])
    pressure=current-clean_file
    cpu_usage=int(next(row.split()[1] for row in (cg/"cpu.stat").read_text().splitlines() if row.startswith("usage_usec ")))
    retained=protected_file_bytes(int(time.monotonic()//600))
    inactive_available=max(0,min(host,limit-current+clean_inactive))
    # Linux can reclaim clean archived file pages under normal pressure. For a
    # large hall mesh, protect every byte in the live datasets in addition to
    # all mapped pages and the usual multipliers, dirty pages and shmem.
    eval_available=inactive_available
    if retained is not None:
        excess=max(0,clean_file-retained-counters["file_mapped"])
        eval_available=max(inactive_available,min(host,limit-current+excess))
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=memory.total,memory.free,utilization.gpu",
                                 "--format=csv,noheader,nounits"],text=True)
    total,free,busy=map(int,gpu.strip().splitlines()[0].split(","))
    quota,period=map(int,(cg/"cpu.max").read_text().split())
    return dict(time=now(),cpu_quota=quota//period,cgroup_total_bytes=current,cgroup_limit_bytes=limit,
                clean_inactive_file_bytes=clean_inactive,cgroup_working_bytes=current-clean_inactive,
                cgroup_nonclean_bytes=pressure,cgroup_anon_bytes=counters["anon"],cgroup_file_bytes=counters["file"],
                cgroup_dirty_bytes=counters["file_dirty"]+counters["file_writeback"],cpu_usage_usec=cpu_usage,
                ram_available_bytes=inactive_available,ram_available_for_mesh_bytes=eval_available,
                retained_dataset_bytes=retained,file_mapped_bytes=counters["file_mapped"],gpu_total_bytes=total*1024**2,
                gpu_free_bytes=free*1024**2,gpu_util_percent=busy,disk_free_bytes=shutil.disk_usage(RUN).free)


def directory(name):
    return RUN/("f_vr" if name.startswith("VR") else "c_dn")/name


def active_pid(pid,variant):
    try:
        args=Path(f"/proc/{int(pid)}/cmdline").read_bytes().split(b"\0")
        return variant.encode() in args and any(b"interactive_dn_" in arg for arg in args)
    except (FileNotFoundError,ProcessLookupError,ValueError):
        return False


def tree_memory(pid,gpu_processes,stage_pid=None):
    """Measure the worker and its helpers without adding their shared RSS."""
    pending=[pid]
    visited=set()
    pss=uss=transient=gpu_usage=0
    while pending:
        n=pending.pop()
        if n in visited:
            continue
        visited.add(n)
        gpu_usage+=gpu_processes.get(n,0)
        proc=Path("/proc")/str(n)
        try:
            pending.extend(map(int,(proc/f"task/{n}/children").read_text().split()))
            mem={r.split(":")[0]:int(r.split()[1])*1024 for r in (proc/"smaps_rollup").read_text().splitlines()
                 if r.startswith(("Pss:","Private_Clean:","Private_Dirty:"))}
            status={r.split(":")[0]:int(r.split()[1])*1024 for r in (proc/"status").read_text().splitlines()
                    if r.startswith(("VmRSS:","VmHWM:"))}
        except (OSError,ProcessLookupError):
            continue
        pss+=mem.get("Pss",0)
        uss+=mem.get("Private_Clean",0)+mem.get("Private_Dirty",0)
        transient+=max(0,status.get("VmHWM",0)-status.get("VmRSS",0))
    return dict(pss_bytes=pss,uss_bytes=uss,transient_allowance_bytes=transient,process_count=len(visited),
                gpu_process_bytes=gpu_usage,contains_stage_pid=stage_pid in visited)


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
    active_phase=None
    if worker:
        try:
            argv=Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            n=argv.index(name.encode())
            active_phase=argv[n+1].decode() if argv[n+1] in (b"train",b"evaluate") else None
        except (OSError,ValueError,IndexError):
            pass
    actual_phase=active_phase or ("evaluate" if final else "train")
    exit_text=(st/"worker.exit").read_text().strip() if did_exit and not worker else ""
    handoff=read(st/"slot_handoff.json",{}) if name=="R3" and exit_text=="76" else {}
    planned=(handoff.get("completed_updates")==done==60000 and handoff.get("checkpoint_sha256")==ckpt.get("sha256")
             and handoff.get("next_variant")=="R5" and handoff.get("exit_code")==76)
    stage=read(st/"evaluation_stage.json",{}) if worker and actual_phase=="evaluate" else {}
    return dict(variant=name,active=worker,pid=pid if worker else None,log=lastlog,active_phase=actual_phase,evaluation_stage=stage,
                durable_updates=done,verified_updates_lower_bound=max(done,int(progress.get("step_in_progress",0))),
                phase="evaluate" if final else "train",training_complete=bool(final),evaluation_complete=evaluate.is_file(),
                first_batch=bool(first),measured=measured,
                exit_code=int(exit_text) if exit_text.lstrip("-").isdigit() else None,checkpoint_handoff=planned)


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


def observe_peaks(hist,row,host,peers):
    m=row["measured"]
    if m:
        for key in ("pss_bytes","uss_bytes","torch_reserved_peak","torch_allocated_peak"):
            hist[key]=max(hist.get(key,0),int(m.get(key,0)))
    if not row["active"]:
        return
    phase=row["active_phase"]
    profile=hist.setdefault("phase_peaks",{}).setdefault(phase,{})
    mem=row["tree_memory"]
    for k in ("pss_bytes","uss_bytes"):
        profile[k]=max(profile.get(k,0),mem[k])
    profile["tree_peak_allowance_bytes"]=max(profile.get("tree_peak_allowance_bytes",0),
                                               mem["pss_bytes"]+mem["transient_allowance_bytes"])
    if row.get("gpu_released"):
        profile["late_cpu_pss_bytes"]=max(profile.get("late_cpu_pss_bytes",0),mem["pss_bytes"])
    profile["gpu_peak_bytes"]=max(profile.get("gpu_peak_bytes",0),row.get("gpu_process_bytes",0),
                                  int(m.get("torch_reserved_peak",0)) if phase=="train" else 0)
    launch=hist.get("launch_probe")
    if not launch or launch.get("closed") or launch["phase"]!=phase:
        return
    # Only the startup interval is attributable. Another worker joining,
    # leaving or changing phase closes the sample before it can contaminate it.
    others={k:v["pid"] for k,v in peers.items() if k!=row["variant"] and v["active"]}
    if others!=launch["peer_pids"]:
        launch["closed"]="cohort changed"
        return
    peer_increase=sum(max(0,peers[k]["tree_memory"]["uss_bytes"]-v) for k,v in launch["peer_uss"].items())
    growth=max(0,host["cgroup_nonclean_bytes"]-launch["nonclean_bytes"]-peer_increase)
    profile["startup_pressure_bytes"]=max(profile.get("startup_pressure_bytes",0),growth)
    launch["samples"]=launch.get("samples",0)+1
    if (phase=="train" and row["verified_updates_lower_bound"]-launch["start_updates"]>=100) or (phase=="evaluate" and launch["samples"]>=4):
        launch["closed"]="startup observed"


def demand(name,phase,peaks):
    comparable=[p for k,p in peaks.items() if k.startswith("VR")==name.startswith("VR")]
    profiles=[p["phase_peaks"][phase] for p in comparable if p.get("phase_peaks",{}).get(phase)]
    # v1's incremental_working_peak spans unrelated jobs over hours. Its
    # independent trainer PSS/USS and allocator peaks remain valid.
    trainers=[p for p in comparable if p.get("torch_reserved_peak")]
    memory=[max(p.get("pss_bytes",0),p.get("uss_bytes",0)) for p in trainers]
    gpu=[p["torch_reserved_peak"] for p in trainers]
    if phase=="evaluate":
        # Real historical hall TSDF HWM was 52.69 GiB. Its growing sparse volume
        # has a different footprint from both a training worker and VR culling.
        memory.append((22 if name.startswith("VR") else 53)*GIB)
    for p in profiles:
        memory.append(max(p.get("pss_bytes",0),p.get("uss_bytes",0),
                          p.get("tree_peak_allowance_bytes",0),p.get("startup_pressure_bytes",0)))
        gpu.append(p.get("gpu_peak_bytes",0))
    fallback_ram=(22 if name.startswith("VR") or phase=="evaluate" else 79)*GIB
    fallback_gpu=(12 if name.startswith("VR") else 20)*GIB
    return max(memory,default=fallback_ram),max(gpu,default=fallback_gpu)


def admission(name,row,active,peaks,host):
    phase=row["phase"]
    training=[a for a in active if a.get("active_phase",a["phase"])=="train"]
    if phase=="train" and len(training)>=3:
        return False,"three training workers already active",{}
    if any(not a["first_batch"] or a["verified_updates_lower_bound"]<100 for a in training):
        return False,"measuring another worker's first batches",{}
    threads=host["cpu_quota"]-1-sum(a.get("thread_budget",3) for a in active)
    if threads<1:
        return False,"CPU thread budget",{}
    ram,gpu=demand(name,phase,peaks)
    # Preserve space for workers that currently use less than their measured
    # peaks, including evaluation workers which will return to GPU rendering.
    extra_ram=extra_gpu=0
    for a in active:
        resident=a.get("tree_memory",{}).get("pss_bytes",0)
        previous_ram,previous_gpu=demand(a["variant"],a.get("active_phase",a["phase"]),peaks)
        if a.get("gpu_released"):
            late=peaks.get(a["variant"],{}).get("phase_peaks",{}).get("evaluate",{})
            previous_ram=max(29*GIB,late.get("late_cpu_pss_bytes",0))
            previous_gpu=GIB  # retained video encoder budget after the last tensor render
        extra_ram+=max(0,previous_ram-resident)
        extra_gpu+=max(0,previous_gpu-a.get("gpu_process_bytes",0))
    required_ram=int(1.25*ram+8*GIB)
    required_gpu=int(1.15*gpu+6*GIB)
    assigned=min(threads,12 if phase=="evaluate" else (8 if not active else 3))
    data=dict(ram_required_bytes=required_ram,gpu_required_bytes=required_gpu,threads=assigned,
              active_ram_growth_reserve_bytes=extra_ram,active_gpu_growth_reserve_bytes=extra_gpu,
              evidence="process-tree PSS/USS, transient high water and cohort-bounded startup pressure")
    hall_eval=(phase=="evaluate" and not name.startswith("VR")) or any(
        a.get("active_phase",a["phase"])=="evaluate" and not a["variant"].startswith("VR") for a in active)
    available=host.get("ram_available_for_mesh_bytes",host["ram_available_bytes"]) if hall_eval else host["ram_available_bytes"]
    data["ram_available_for_decision_bytes"]=available
    if available-extra_ram<required_ram:
        return False,"RAM admission",data
    if host["gpu_free_bytes"]-extra_gpu<required_gpu:
        return False,"GPU admission",data
    return True,"admitted",data


def signature(row,paths):
    evidence=paths+list((ROOT/"scripts").glob("interactive_dn*"))+list((ROOT/"scripts").glob("interactive_vr*"))
    key=[(p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in sorted(evidence) if p.is_file()]
    return hashlib.sha256(json.dumps([row["phase"],row["durable_updates"],key]).encode()).hexdigest()


def scheduled_variants(rows):
    order=list(TRAIN_ORDER)
    if rows["R3"].get("checkpoint_handoff") and not rows["R5"]["log"]:
        order.remove("R5")
        order.insert(order.index("R3"),"R5")
    return sorted(order,key=lambda k:(rows[k]["training_complete"],order.index(k)))


def cycle():
    host=sample()
    rows={k:training_state(k) for k in VARIANTS}
    ledger=read(STATE/"admission.json",{})
    active=[r for r in rows.values() if r["active"]]
    gpu_processes={}
    if active:
        usage=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],text=True)
        for line in usage.splitlines():
            pid,amount=[f.strip() for f in line.split(",")]
            if pid.isdigit() and amount.isdigit():
                gpu_processes[int(pid)]=int(amount)*1024**2
    for r in active:
        stage=r.get("evaluation_stage",{})
        stage_pid=stage.get("pid")
        r["tree_memory"]=tree_memory(r["pid"],gpu_processes,stage_pid)
        r["gpu_process_bytes"]=r["tree_memory"]["gpu_process_bytes"]
        r["gpu_released"]=bool(stage.get("gpu_released") is True and isinstance(stage_pid,int)
                               and active_pid(stage_pid,r["variant"]) and
                               r["tree_memory"]["contains_stage_pid"])
        r["thread_budget"]=ledger.get(r["variant"],{}).get("thread_budget",8 if r["variant"]=="R0C" else 3)
    for k,r in rows.items():
        h=ledger.setdefault(k,{})
        observe_peaks(h,r,host,rows)
    priority=scheduled_variants(rows)
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
        if row["exit_code"] not in (0,None) and not row["checkpoint_handoff"] and row["log"]!=h.get("last_failure_log"):
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
            threads=details["threads"]
            if k in ("R6","R7") and not read(RUN/"b_map/pose/accepted.json",{}).get("scene"):
                row["queue_reason"]="awaiting pose-specific scene path"
                continue
            cmd=["bash",str(ROOT/"scripts/interactive_dn_launch.sh"),k,row["phase"]]
            call=subprocess.run(cmd,cwd=ROOT,env={**os.environ,"IW_DN_THREADS":str(threads),"IW_EVAL_CPU_THREADS":str(threads)},capture_output=True,text=True)
            event("launch",dict(variant=k,phase=row["phase"],exit=call.returncode,message=call.stdout.strip(),**details))
            if call.returncode==0:
                h.update(launch_time=now(),thread_budget=threads,launch_probe=dict(phase=row["phase"],
                    start_updates=row["verified_updates_lower_bound"],nonclean_bytes=host["cgroup_nonclean_bytes"],
                    peer_pids={a["variant"]:a["pid"] for a in active},
                    peer_uss={a["variant"]:a["tree_memory"]["uss_bytes"] for a in active}))
                # Sample the new worker before making the next admission decision.
                break
    write(STATE/"admission.json",ledger)
    write(STATE/"queue.json",dict(run=RUN.name,coordinator_pid=os.getpid(),host=host,training=rows))
    with (STATE/"throughput.jsonl").open("a") as f:
        f.write(json.dumps(dict(host=host,active=[r["variant"] for r in active],
              observed_updates={k:r["verified_updates_lower_bound"] for k,r in rows.items()}),separators=(",",":"))+"\n")


def serve(wait_for_owner=False):
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/"coordinator.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX if wait_for_owner else fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("A coordinator already owns this run")
            return
        event("coordinator_started",dict(pid=os.getpid(),head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()))
        running=True
        def finish_cycle(*_):
            nonlocal running
            running=False
        signal.signal(signal.SIGTERM,finish_cycle)
        signal.signal(signal.SIGINT,finish_cycle)
        while running:
            try:
                cycle()
            except Exception as error:
                event("coordinator_error",f"{type(error).__name__}: {error}")
            if running:
                time.sleep(20)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("serve","status"))
    p.add_argument("--wait-for-owner",action="store_true",help="Take ownership after the current coordinator exits.")
    args=p.parse_args()
    if args.command=="serve":
        serve(args.wait_for_owner)
    else:
        print(json.dumps(read(STATE/"queue.json",{}),ensure_ascii=False,indent=2))
