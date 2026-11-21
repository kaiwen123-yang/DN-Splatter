"""Prepare private, content-deduplicated transport volumes from the frozen capture results."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

from capture_archive_transport import (ROOT, ID, RESULT, B0, B1, GEO, GEOID, GEORUN, REPORT, BASE_REPORT,
                                      STAGE, VIDEO, atomic, check, checksum_record, digest, document, range_hashes,
                                      safe_rel, utc)

OUT = ROOT / STAGE
RECOVERY = OUT / "recovery"
VOLUMES = OUT / "volumes"
ZSTD = "/root/miniconda3/bin/zstd"
S01 = ".local/data/custom/capture-s01/20261119-s01-5hz-3982cb26"
SHARD = 224 * 1024**2
REUSED = (GEORUN + "/transfer/",)
LOCK_SKIPS = {"runner.lock", "observer.lock", "workflow.lock"}
TERMINAL = {"completed", "completed_with_coverage_gaps"}


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *map(str,args)], cwd=cwd).decode().strip()


def frozen_file(rel):
    raw = subprocess.check_output(["git","show",f"{RESULT}:{safe_rel(rel)}"],cwd=ROOT)
    check(hashlib.sha256(raw).hexdigest() == digest(ROOT/rel), "Report differs from frozen commit: "+rel)
    return raw


def read_frozen(rel):
    return json.loads(frozen_file(rel))


def recovery_files():
    RECOVERY.mkdir(exist_ok=True,parents=True)
    for name in ("dn-splatter","gsplat","Metric3D","pytorch3d"):
        src = ROOT/".local/upstream"/name
        rev = git("rev-parse","HEAD",cwd=src)
        archive = RECOVERY/f"{name}-{rev[:12]}.tar.gz"
        if not archive.is_file():
            tmp = archive.with_suffix(".pending")
            subprocess.run(["git","archive","--format=tar.gz","--prefix="+name+"/","-o",tmp,rev],cwd=src,check=True)
            tmp.replace(archive)
        patch = subprocess.check_output(["git","diff","--binary","HEAD"],cwd=src)
        (RECOVERY/f"{name}.working.patch").write_bytes(patch)
        mods = git("submodule","status","--recursive",cwd=src).splitlines()
        submodules=[]
        for line in mods:
            fields=line.strip().split()
            if len(fields)<2:
                continue
            suffix=fields[1]
            child=src/safe_rel(suffix)
            if not (child/".git").exists():
                submodules.append(dict(path=suffix,reference=fields[0].lstrip("-+"),checkout_present=False))
                continue
            head=git("rev-parse","HEAD",cwd=child)
            pack=RECOVERY/f"{name}-submodule-{suffix.replace('/','_')}-{head[:12]}.tar.gz"
            if not pack.is_file():
                tmp=pack.with_suffix(".pending")
                subprocess.run(["git","archive","--format=tar.gz","--prefix="+suffix+"/","-o",tmp,head],cwd=child,check=True)
                tmp.replace(pack)
            submodules.append(dict(path=suffix,commit=head,archive=pack.name,checkout_present=True))
        atomic(RECOVERY/f"{name}.json",dict(commit=rev,archive=archive.name,patch=name+".working.patch",submodules=submodules))
    for env in ("dn-splatter-bw","metric3d-bw","capture-geometry"):
        prefix=ROOT/".local/conda"/env
        conda=[]
        for p in sorted((prefix/"conda-meta").glob("*.json")):
            d=document(p)
            conda.append({k:d[k] for k in ("name","version","build","subdir","sha256","md5") if k in d})
        code=("from importlib import metadata as m; import json,platform; "
              "print(json.dumps({'python':platform.python_version(),'packages':sorted([(d.metadata['Name'],d.version) "
              "for d in m.distributions() if d.metadata['Name']])}))")
        installed=json.loads(subprocess.check_output([prefix/"bin/python","-I","-c",code]))
        atomic(RECOVERY/(env+".packages.json"),dict(**installed,conda=conda))
    atomic(RECOVERY/"source.json",dict(result_commit=RESULT,baseline_commit="e26ae72a75997dcfd3a907aa26b6c5c0677597a5",
        training_source="811f5cdf342d92d12a9d9a4be185d374c430c184",metric3d_source="eb5b6fac0dc155e4e52f576e304fbf11655ff339",
        device=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip()))


def build_inventory():
    current=document(ROOT/B1/"state/current.json")
    delivery=document(ROOT/B1/"state/delivery_ready.json")
    check(current["stage"]=="complete" and delivery["status"] in TERMINAL and delivery["commit"]==RESULT,
          "Frozen server completion and publication")
    published=read_frozen(REPORT+"/artifact_manifest.json")
    expected={}
    sources={}
    def add(name,member=None,src=None):
        name=safe_rel(name)
        resolved=src if src is not None else ROOT/name
        check(resolved.resolve().is_relative_to(ROOT.resolve()) and resolved.is_file(),name)
        sources[name]=resolved
        if member is not None:
            k=(int(member["bytes"]),member["sha256"])
            check(k==expected.setdefault(name,k),"Conflicting frozen identities: "+name)
    index_bytes=frozen_file(REPORT+"/artifact_files.jsonl.gz")
    check(hashlib.sha256(index_bytes).hexdigest()==published["index"]["sha256"],"B1 index")
    n=total=0
    for line in gzip.decompress(index_bytes).splitlines():
        row=json.loads(line)
        check(row["path"].startswith(B1+"/"),"B1 member root")
        add(row["path"],row)
        n+=1
        total+=row["bytes"]
    check((n,total)==(published["file_count"],published["total_bytes"]),"B1 recorded cardinality")
    for row in published["files"]:
        add(row["path"],row)
    base=read_frozen(BASE_REPORT+"/artifact_manifest.json")
    for row in base["files"]:
        add(B0+"/"+row["path"],row)
    for row in base["external_files"]:
        add(Path(row["path"]).relative_to(ROOT).as_posix(),row)
    for srcdir in (GEO,GEORUN+"/transfer",S01):
        m=ROOT/srcdir/"manifest.json"
        add(srcdir+"/manifest.json")
        for row in document(m)["files"]:
            add(srcdir+"/"+row["path"],row)
    # Include private run evidence and the live full-scene geometry used by the comparison.
    # Authentication helpers, TCP configuration, lock files and unrelated experiments stay outside this selection.
    for folder in (B0+"/state", B0+"/logs", B1+"/state", B1+"/logs", GEORUN+"/rect",
                   GEORUN+"/metric", GEORUN+"/sfm", GEORUN+"/bridge"):
        for p in sorted((ROOT/folder).rglob("*")):
            if p.is_file() and p.name not in LOCK_SKIPS and not p.name.endswith((".lock","-shm")):
                add(p.relative_to(ROOT).as_posix())
    add(B1+"/training_complete.json")
    for p in sorted(RECOVERY.iterdir()):
        if p.is_file() and not p.name.endswith(".pending"):
            add("archive-recovery/"+p.name,src=p)
    def inspect(name):
        p=sources[name]
        actual=checksum_record(p)
        if name in expected:
            check((actual["bytes"],actual["sha256"])==expected[name],"Frozen file changed: "+name)
        return dict(path=name,**actual)
    names=sorted(sources)
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(inspect,names))
    idx=OUT/"inventory.jsonl.gz"
    tmp=idx.with_suffix(".pending")
    with tmp.open("wb") as raw, gzip.GzipFile(fileobj=raw,mode="wb",mtime=0,filename="") as z:
        for row in rows:
            z.write((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode())
    tmp.replace(idx)
    groups=defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)
    candidates={row["sha256"] for row in rows if row["path"].startswith(REUSED)
                or row["path"] in (VIDEO,B0+"/video/reference_render.mp4")}
    # Video and local packet objects get separate volumes so a matching Windows copy can skip the network.
    order=sorted(groups,key=lambda h:(h not in candidates,groups[h][0]["path"]))
    volumes=[]; items=[]; acc=0; kind=None
    for h in order:
        row=groups[h][0]; flag=h in candidates
        if items and (acc+row["bytes"]>SHARD or flag!=kind):
            volumes.append(dict(number=len(volumes),local_candidates=kind,objects=items));items=[];acc=0
        items.append(dict(sha256=h,bytes=row["bytes"],source=row["path"]))
        acc+=row["bytes"];kind=flag
    if items:
        volumes.append(dict(number=len(volumes),local_candidates=kind,objects=items))
    plan=OUT/"plan.json.gz"
    tmp=plan.with_suffix(".pending")
    with tmp.open("wb") as raw,gzip.GzipFile(fileobj=raw,mode="wb",filename="",mtime=0) as z:
        z.write(json.dumps(volumes,separators=(",",":")).encode())
    tmp.replace(plan)
    summary=dict(run_id=ID,result_commit=RESULT,inventory=checksum_record(idx),plan=checksum_record(plan),
                 logical_files=len(rows),logical_bytes=sum(r["bytes"] for r in rows),unique_files=len(groups),
                 unique_bytes=sum(g[0]["bytes"] for g in groups.values()),volume_count=len(volumes),
                 b1_manifest_files=n,b1_manifest_bytes=total,b0_manifest_files=len(base["files"]),
                 b0_manifest_bytes=sum(r["bytes"] for r in base["files"]),
                 source_bag=read_frozen(BASE_REPORT+"/selection.json")["source"],created_utc=utc())
    atomic(OUT/"catalog.json",summary)
    return volumes,summary


def seal_volume(vol, binding):
    number=vol["number"]
    path=VOLUMES/f"part-{number:04d}.tar.zst"
    record=path.with_suffix(".json")
    if record.exists():
        d=document(record)
        check(d["catalog_sha256"]==binding,path)
        if path.is_file():
            check(path.stat().st_size==d["bytes"] and digest(path)==d["sha256"],path)
            return
        retirement=OUT/"transport_retirement.json"
        check(retirement.is_file() and document(retirement)["inventory_sha256"]==binding,
              "Unexplained missing transport volume: "+path.name)
    tmp=path.with_suffix(".pending")
    # Keep every payload byte exact; duplicate original paths are restored from the inventory.
    with tmp.open("wb") as out, subprocess.Popen([ZSTD,"-q","-7","-T4","--no-progress","-c"],
                                                 stdin=subprocess.PIPE,stdout=out) as compressor:
        try:
            with compressor.stdin as sink, tarfile.open(fileobj=sink,mode="w|",bufsize=1024**2,format=tarfile.USTAR_FORMAT) as tar:
                for obj in vol["objects"]:
                    source=obj["source"]
                    p=(RECOVERY/Path(source).name) if source.startswith("archive-recovery/") else (ROOT/source)
                    info=tarfile.TarInfo(obj["sha256"])
                    info.size=obj["bytes"];info.mode=0o644;info.mtime=0
                    with p.open("rb") as f:
                        tar.addfile(info,f)
        finally:
            if not compressor.stdin.closed:
                compressor.stdin.close()
        check(compressor.wait()==0,"zstd "+path.name)
    tmp.replace(path)
    check(path.stat().st_size>0,path)
    atomic(record,dict(filename=path.name,number=number,catalog_sha256=binding,**checksum_record(path),
                       chunks=range_hashes(path)))


def main():
    OUT.mkdir(parents=True,exist_ok=True);VOLUMES.mkdir(exist_ok=True)
    with (OUT/"server.lock").open("a") as lk:
        fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=OUT/"server.json"
        try:
            atomic(state,dict(stage="catalog",pid=os.getpid(),utc=utc()))
            if (OUT/"catalog.json").exists():
                summary=document(OUT/"catalog.json")
                check(summary["result_commit"]==RESULT and summary["inventory"]==checksum_record(OUT/"inventory.jsonl.gz")
                      and summary["plan"]==checksum_record(OUT/"plan.json.gz"),"Existing catalog")
                volumes=json.loads(gzip.decompress((OUT/"plan.json.gz").read_bytes()))
            else:
                recovery_files()
                volumes,summary=build_inventory()
            binding=digest(OUT/"inventory.jsonl.gz")
            atomic(state,dict(stage="packing",volumes=len(volumes),done=0,pid=os.getpid(),utc=utc()))
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs=[pool.submit(seal_volume,v,binding) for v in volumes]
                for done,job in enumerate(as_completed(jobs),1):
                    job.result()
                    atomic(state,dict(stage="packing",volumes=len(volumes),done=done,pid=os.getpid(),utc=utc()))
            descriptors=[document(VOLUMES/f"part-{n:04d}.tar.json") for n in range(len(volumes))]
            atomic(OUT/"sealed.json",dict(run_id=ID,result_commit=RESULT,inventory_sha256=binding,
                volumes=len(descriptors),transport_bytes=sum(x["bytes"] for x in descriptors),finished_utc=utc()))
            atomic(state,dict(stage="sealed",volumes=len(volumes),done=len(volumes),pid=os.getpid(),utc=utc()))
        except Exception as exc:
            atomic(state,dict(stage="failed",reason=str(exc),utc=utc(),pid=os.getpid()))
            raise


if __name__=="__main__":
    main()
