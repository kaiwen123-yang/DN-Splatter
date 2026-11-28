"""Publish only coordinator journal entries and status using the local Git credential."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from world_daily_log import journal


ROOT=Path(__file__).resolve().parents[1]
RUN="interactive-world-20261128-v1"
OUTBOX="/root/autodl-tmp/DN-Splatter/.local/runs/"+RUN+"/f_evaluation/publication/outbox"
LOG="DAILY_RESEARCH_LOG.md"
STATUS="reports/interactive-world/latest.json"
ORIGIN="https://github.com/kaiwen123-yang/DN-Splatter.git"


def git(*args, data=None, env=None):
    return subprocess.run(["git",*args],cwd=ROOT,env=env,input=data,stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,check=True).stdout


def publish(connector):
    if git("remote","get-url","origin").decode().strip()!=ORIGIN:
        raise ValueError("Unexpected publication repository")
    git("fetch","--quiet","--no-tags","origin","main")
    head=git("rev-parse","FETCH_HEAD").decode().strip()
    git("merge-base","--is-ancestor","f10e918",head)
    base=git("show",head+":"+LOG)
    previous=json.loads(git("show",head+":"+STATUS))
    notes=[]
    newest=previous
    with connector.connect() as session, session.open_sftp() as ftp:
        for ent in sorted(ftp.listdir_attr(OUTBOX),key=lambda row:row.filename):
            name=ent.filename
            if not re.fullmatch(r"[0-9T+_-]+-[0-9a-f]{16}\.json",name):
                continue
            ident=name[-21:-5]
            marker=f"<!-- interactive-world-evidence {ident} -->"
            if marker.encode() in base:
                continue
            if ent.st_size>4*1024**2:
                raise ValueError("Oversized status envelope")
            with ftp.open(OUTBOX+"/"+name,"rb") as f:
                doc=json.loads(f.read())
            s=doc["snapshot"]
            if doc["marker"]!=marker or s["run_id"]!=RUN or s["source_baseline"]!="f10e91805877b197d49107347844b675df856d3f":
                raise ValueError("Wrong experiment envelope")
            if not all(item["path"].startswith("/root/autodl-tmp/DN-Splatter/.local/runs/"+RUN+"/") and
                       re.fullmatch("[0-9a-f]{64}",item["sha256"]) for item in s["artifacts"].values()):
                raise ValueError("Invalid artifact reference in receipt")
            # Reconstruct prose locally from the validated structured fields.
            notes.append(("\n\n"+journal(s)+"\n\n"+marker+"\n").encode("utf8"))
            if s["observed_at"]>=newest.get("observed_at",""):
                newest=s
    if not notes:
        return
    files={LOG:base+b"".join(notes),STATUS:(json.dumps(newest,ensure_ascii=False,indent=2)+"\n").encode("utf8")}
    risk=rb"BEGIN [A-Z ]*PRIVATE KEY|https?://[^/\s]+:[^/\s]+@|Bearer [A-Za-z0-9._-]{20,}|gh[pousr]_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9_-]{30,}"
    if any(re.search(risk,d) for d in [*notes,files[STATUS]]):
        raise ValueError("Publication rejected by credential check")
    gitdir=Path(git("rev-parse","--absolute-git-dir").decode().strip())
    with tempfile.TemporaryDirectory(dir=gitdir) as temp:
        env={**os.environ,"GIT_INDEX_FILE":str(Path(temp)/"index")}
        git("read-tree",head,env=env)
        for path,body in files.items():
            digest=git("hash-object","-w","--stdin",data=body).decode().strip()
            git("update-index","--add","--cacheinfo","100644",digest,path,env=env)
        diff=set(git("diff","--cached","--name-only",head,env=env).decode().splitlines())
        if not diff.issubset(files) or LOG not in diff:
            raise ValueError("Publication contains unexpected paths")
        git("diff","--cached","--check",head,env=env)
        tree=git("write-tree",env=env).decode().strip()
        message=f"Record verified interactive-world progress through {newest['observed_at']}\n"
        commit=git("commit-tree",tree,"-p",head,data=message.encode()).decode().strip()
        git("push","--quiet","origin",commit+":refs/heads/main")
        if git("ls-remote","origin","refs/heads/main").decode().split()[0]!=commit:
            raise RuntimeError("Remote readback differs from the published commit")
    # Keep the user's checkout current only while its branch and tracked files
    # still exactly match the previously observed public parent.
    if (git("rev-parse","--abbrev-ref","HEAD").decode().strip()=="main" and
        git("rev-parse","HEAD").decode().strip()==head and
        not git("status","--porcelain","--untracked-files=no")):
        git("merge","--ff-only",commit)
    print("Verified GitHub main",commit,"entries",len(notes),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-module",type=Path,required=True)
    p.add_argument("--watch",action="store_true")
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location("current_project_ssh",a.session_module.resolve())
    connector=importlib.util.module_from_spec(spec);spec.loader.exec_module(connector)
    while True:
        try:
            publish(connector)
        except Exception as error:
            # A connection/credential can go offline without interrupting server jobs.
            print(time.strftime("%Y-%m-%d %H:%M:%S"),type(error).__name__,getattr(error,"returncode",""),flush=True)
            if not a.watch:
                raise
        if not a.watch: break
        time.sleep(120)
