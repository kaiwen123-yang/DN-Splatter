"""Record sanitized host limits, source identities and verified restore receipts."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/".local/runs/interactive-world-20261128-v1"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def member(path):
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path)}


def git(*args,root=ROOT):
    return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()


def snapshot():
    cg=Path("/sys/fs/cgroup")
    quota,period=(cg/"cpu.max").read_text().split()
    memory={key:int((cg/key).read_text()) for key in ("memory.max","memory.current") if (cg/key).is_file()}
    mem_stat={k:int(v) for k,v in (line.split() for line in (cg/"memory.stat").read_text().splitlines())}
    memory["stat"]={k:mem_stat[k] for k in ("anon","file","inactive_file","active_file","file_dirty","file_writeback","shmem")}
    device=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"],text=True).strip().split(", ")
    disk=shutil.disk_usage(ROOT)
    historic=ROOT/".local/runs/lingbot-depth-map-20261128-v1"
    sources=json.loads((ROOT/"reports/lingbot-depth-map/20261128-v1/sources.json").read_bytes())
    models={}
    for k,source_dir in (("depth","lingbot-depth"),("map","lingbot-map-v2")):
        spec=sources[k]
        weight=member(Path(spec["weight"]["path"]))
        commit=git("rev-parse","HEAD",root=ROOT/".local/upstream"/source_dir)
        if weight["sha256"]!=spec["weight"]["sha256"] or commit!=spec["commit"]:
            raise ValueError(k+" identity changed")
        models[k]=dict(weight=weight,commit=commit)
    return dict(captured=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                run_id=RUN.name,git_head=git("rev-parse","HEAD"),required_base=git("rev-parse","f10e918"),
                correct_origin=git("remote","get-url","origin")=="https://github.com/kaiwen123-yang/DN-Splatter.git",
                cpu_quota=int(quota)//int(period),cpu_affinity=list(sorted(os.sched_getaffinity(0))),
                memory_bytes=memory,gpu={"name":device[0],"total_MiB":int(device[1]),"used_MiB":int(device[2]),
                                         "utilization_percent":int(device[3])},
                disk={"mount":"/root/autodl-tmp","total_bytes":disk.total,"available_bytes":disk.free},
                models=models,frozen_hall_input_manifest=member(historic/"prepare/input_manifest.json"),
                environments={n:(ROOT/".local/conda"/n/"bin/python").exists() for n in ("lingbot-depth-bw","lingbot-map2-bw","dn-splatter-bw")},
                restore={p.name:{"receipt":member(p),"counts":json.loads(p.read_bytes())}
                         for p in sorted((RUN/"restore").glob("*.verified.json"))})


if __name__=="__main__":
    out=RUN/"f_evaluation/inventory"
    out.mkdir(parents=True,exist_ok=True)
    with (out/"inventory.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        doc=snapshot()
        path=out/(datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")+".json")
        path.write_text(json.dumps(doc,indent=2)+"\n")
        print(path)
