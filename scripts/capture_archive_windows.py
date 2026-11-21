"""Restore the verified full-capture archive onto a private disk using independent, resumable volumes."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time

from capture_archive_transport import (ROOT, B0, B1, GEOID, GEORUN, ID, RESULT, REPORT, BASE_REPORT, STAGE, VIDEO,
                                      atomic, check, checksum_record, digest, document, download, remote_bytes,
                                      safe_rel, ssh_args, utc)


@contextmanager
def exclusive(path):
    import msvcrt
    with path.open("a+b") as f:
        if not f.seek(0,2):
            f.write(b"\0"); f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        yield


def file_from_stream(src, dest, size, expected=None):
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_name(dest.name+f".{os.getpid()}.archive-part")
    h=hashlib.sha256(); todo=size
    with temp.open("wb") as f:
        while todo:
            b=src.read(min(todo,4*1024**2))
            check(b,"Truncated payload for "+str(dest))
            f.write(b);h.update(b);todo-=len(b)
    value=h.hexdigest()
    check(expected is None or value==expected,"Payload hash mismatch: "+str(dest))
    if dest.exists():
        check(dest.stat().st_size==size and digest(dest)==value,"Existing archive member differs: "+str(dest))
        temp.unlink()
    else:
        temp.replace(dest)
    return value


def copy_checked(src,dst,info):
    if dst.is_file():
        check(checksum_record(dst)==info,"Existing member differs: "+str(dst))
        return
    with src.open("rb") as f:
        file_from_stream(f,dst,info["bytes"],info["sha256"])


def git(*args):
    return subprocess.check_output(["git",*map(str,args)],cwd=ROOT,stderr=subprocess.DEVNULL).decode().strip()


class Archive:
    def __init__(self,destination,connections):
        self.dest=destination.resolve()
        self.project=self.dest/"project"; self.ctrl=self.dest/"verification"
        self.ctrl.mkdir(parents=True,exist_ok=True)
        self.ledger=self.ctrl/"volumes"; self.ledger.mkdir(exist_ok=True)
        self.spool=ROOT/".local/archives"/(ID+"-"+RESULT[:8])/"transport"
        self.spool.mkdir(parents=True,exist_ok=True)
        self.connections=connections
        self.state=self.ctrl/"windows.json"; self.ssh=ssh_args()
        self.io_lock=threading.Lock()
        self.metrics_lock=threading.Lock()
        self.active={}; self.done={}; self.last_update=0.
        self.hints={}
        marker=self.ctrl/"identity.json"
        ident=dict(run_id=ID,result_commit=RESULT,format=1)
        if marker.exists():
            check(document(marker)==ident,"Destination identity")
        else:
            check(not self.project.exists(),"Refusing to adopt an unrelated archive tree")
            atomic(marker,ident)
        self.project.mkdir(exist_ok=True)

    def status(self,phase,**kw):
        atomic(self.state,dict(phase=phase,pid=os.getpid(),utc=utc(),**kw))

    def get_catalog(self):
        while True:
            b=remote_bytes(STAGE+"/catalog.json",optional=True)
            if b:break
            s=remote_bytes(STAGE+"/server.json",optional=True)
            if s and json.loads(s).get("stage")=="failed":
                raise RuntimeError("Server catalog failed: "+json.loads(s).get("reason",""))
            time.sleep(7)
        self.catalog=json.loads(b)
        check(self.catalog["result_commit"]==RESULT and self.catalog["run_id"]==ID,"catalog identity")
        catalog_path=self.ctrl/"catalog.json"
        if catalog_path.is_file():
            check(document(catalog_path)["inventory"]==self.catalog["inventory"] and
                  document(catalog_path)["plan"]==self.catalog["plan"],"Archive catalog changed")
        atomic(catalog_path,self.catalog)
        for prop,name in (("inventory","inventory.jsonl.gz"),("plan","plan.json.gz")):
            p=self.ctrl/name
            if not p.exists():
                payload=remote_bytes(STAGE+"/"+name)
                check(hashlib.sha256(payload).hexdigest()==self.catalog[prop]["sha256"],name)
                tmp=p.with_name(name+".pending");tmp.write_bytes(payload);tmp.replace(p)
            check(checksum_record(p)==self.catalog[prop],name)
        self.inventory=[json.loads(x) for x in gzip.decompress((self.ctrl/"inventory.jsonl.gz").read_bytes()).splitlines()]
        self.plan=json.loads(gzip.decompress((self.ctrl/"plan.json.gz").read_bytes()))
        check(len(self.inventory)==self.catalog["logical_files"] and len(self.plan)==self.catalog["volume_count"],"catalog counts")
        self.groups=defaultdict(list)
        paths=set(); lookup={}
        for row in self.inventory:
            name=safe_rel(row["path"])
            check(name.startswith((".local/","archive-recovery/")) and name.casefold() not in paths,"Unsafe or duplicate logical member: "+name)
            paths.add(name.casefold())
            self.groups[row["sha256"]].append(row)
            lookup[name]=(row["bytes"],row["sha256"])
            prior=ROOT/".local/capture-full"/GEOID
            if name.startswith(GEORUN+"/transfer/"):
                self.hints[row["sha256"]]=prior/"transfer"/name.split("/transfer/",1)[1]
            elif name==VIDEO:
                self.hints[row["sha256"]]=prior/"metric3d-20261120-v1/reference_B0_B1.mp4"
            elif name==B0+"/video/reference_render.mp4":
                self.hints[row["sha256"]]=prior/"reference_render.mp4"
        # Establish both frozen scientific manifests independently on the receiving machine.
        def frozen(name):
            return subprocess.check_output(["git","show",RESULT+":"+name],cwd=ROOT)
        asset=json.loads(frozen(REPORT+"/artifact_manifest.json"))
        index=frozen(REPORT+"/artifact_files.jsonl.gz")
        check(hashlib.sha256(index).hexdigest()==asset["index"]["sha256"],"published B1 file index")
        b1=[json.loads(line) for line in gzip.decompress(index).splitlines()]
        b0=json.loads(frozen(BASE_REPORT+"/artifact_manifest.json"))
        for name,row in [(x["path"],x) for x in b1]+[(B0+"/"+x["path"],x) for x in b0["files"]]:
            check(lookup[name]==(row["bytes"],row["sha256"]),"Missing or mismatched frozen member: "+name)
        check(len(b1)==asset["file_count"]==26278 and sum(r["bytes"] for r in b1)==asset["total_bytes"],"complete B1 index")
        canonical=[obj for v in self.plan for obj in v["objects"]]
        check(len(canonical)==len({o["sha256"] for o in canonical})==len(self.groups),"unique volume objects")
        for obj in canonical:
            check(lookup.get(safe_rel(obj["source"]))==(obj["bytes"],obj["sha256"]),obj["source"])

    def frozen_tree(self):
        done=self.ctrl/"git_tree.json"
        if done.is_file():return
        rows=[]
        with subprocess.Popen(["git","archive","--format=tar",RESULT],cwd=ROOT,stdout=subprocess.PIPE) as proc:
            with tarfile.open(fileobj=proc.stdout,mode="r|*",bufsize=1024**2) as archive:
                for m in archive:
                    if m.isdir():continue
                    check(m.isfile(),"Special Git member "+m.name)
                    name=safe_rel(m.name); src=archive.extractfile(m)
                    h=file_from_stream(src,self.project/name,m.size)
                    rows.append(dict(path=name,bytes=m.size,sha256=h))
            check(proc.wait()==0,"git archive")
        atomic(done,dict(commit=RESULT,files=rows))

    def original(self):
        source=self.catalog["source_bag"]
        src=Path(source["path"])
        info={k:source[k] for k in ("bytes","sha256")}
        dst=self.dest/"original"/src.name
        copy_checked(src,dst,info)
        check(not dst.samefile(src),"Original remains independent from archive")
        atomic(self.ctrl/"original.json",dict(source=str(src),member=str(dst.relative_to(self.dest)),**info))

    def record_progress(self,number,byte_count=None,expected=None,force=False):
        with self.metrics_lock:
            if byte_count is not None:self.active[number]=(byte_count,expected)
            if not force and time.monotonic()-self.last_update<4:
                return
            done_bytes=sum(v["download_bytes"] for v in self.done.values())
            self.status("receiving",volumes_verified=len(self.done),volumes_total=len(self.plan),
                unique_bytes_verified=sum(v["payload_bytes"] for v in self.done.values()),unique_bytes_total=self.catalog["unique_bytes"],
                completed_transport_bytes=done_bytes,
                active_transport_bytes=sum(p[0] for p in self.active.values()),
                active_volumes={str(k):{"received":v[0],"total":v[1]} for k,v in self.active.items()})
            self.last_update=time.monotonic()

    def finish_volume(self,vol,mode,download_bytes=0):
        num=vol["number"]
        doc=dict(number=num,catalog_sha256=self.catalog["inventory"]["sha256"],mode=mode,objects=len(vol["objects"]),
                 payload_bytes=sum(o["bytes"] for o in vol["objects"]),download_bytes=download_bytes,utc=utc())
        atomic(self.ledger/f"{num:04d}.json",doc)
        with self.metrics_lock:
            self.done[num]=doc; self.active.pop(num,None)
        self.record_progress(num,force=True)

    def pull_volume(self,vol):
        num=vol["number"]
        marker=self.ledger/f"{num:04d}.json"
        if marker.is_file():
            d=document(marker)
            check(d["catalog_sha256"]==self.catalog["inventory"]["sha256"],"volume receipt identity")
            if all((self.project/o["source"]).is_file() and (self.project/o["source"]).stat().st_size==o["bytes"] for o in vol["objects"]):
                with self.metrics_lock:self.done[num]=d
                self.record_progress(num)
                return
        if vol["local_candidates"] and all(o["sha256"] in self.hints and self.hints[o["sha256"]].is_file() for o in vol["objects"]):
            with self.io_lock:
                for o in vol["objects"]:
                    copy_checked(self.hints[o["sha256"]],self.project/o["source"],dict(bytes=o["bytes"],sha256=o["sha256"]))
            self.finish_volume(vol,"verified local content")
            return
        meta_name=f"part-{num:04d}.tar.json"
        cached=self.ctrl/"transport_manifests"/meta_name
        if cached.is_file():
            desc=document(cached)
        else:
            while True:
                payload=remote_bytes(STAGE+"/volumes/"+meta_name,optional=True)
                if payload:break
                s=json.loads(remote_bytes(STAGE+"/server.json"))
                if s["stage"]=="failed":raise RuntimeError("Server packaging failed: "+s.get("reason",""))
                time.sleep(5)
            desc=json.loads(payload)
            atomic(cached,desc)
        check(desc["catalog_sha256"]==self.catalog["inventory"]["sha256"] and desc["number"]==num and
              desc["filename"]==f"part-{num:04d}.tar.zst","Volume binding")
        file=self.spool/desc["filename"]
        download(self.ssh, STAGE+"/volumes/"+file.name,file,desc,
                 max(1,self.connections//4),progress=lambda b,n,t:self.record_progress(num,b,desc["bytes"]))
        with self.io_lock:
            with subprocess.Popen(["tar","-xOf",str(file)],stdout=subprocess.PIPE,stderr=subprocess.PIPE) as unpack:
                for o in vol["objects"]:
                    file_from_stream(unpack.stdout,self.project/o["source"],o["bytes"],o["sha256"])
                check(not unpack.stdout.read(1),"Unexpected unlisted payload in volume")
                check(unpack.wait()==0,"bsdtar could not decode "+file.name)
        self.finish_volume(vol,"verified server volume",desc["bytes"])
        file.unlink()  # The validated disk files replace this task-owned transport buffer.

    def alias_and_verify(self):
        self.status("restoring_original_paths",logical_files=len(self.inventory))
        for v in self.plan:
            for o in v["objects"]:
                src=self.project/o["source"]
                for r in self.groups[o["sha256"]]:
                    dst=self.project/r["path"]
                    if dst==src:continue
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    if not dst.exists():
                        os.link(src,dst)
                    check(dst.stat().st_size==r["bytes"],"Archive alias size: "+r["path"])
        self.status("reading_back_H_disk",unique_bytes=self.catalog["unique_bytes"])
        observed={}
        bytes_read=0
        for v in self.plan:
            for o in v["objects"]:
                src=self.project/o["source"]
                check(checksum_record(src)==dict(bytes=o["bytes"],sha256=o["sha256"]),"H readback: "+o["source"])
                bytes_read+=o["bytes"]
                st=src.stat();observed[(st.st_dev,st.st_ino)]=o["sha256"]
                for r in self.groups[o["sha256"]]:
                    dst=self.project/r["path"];st=dst.stat();k=(st.st_dev,st.st_ino)
                    if k not in observed:
                        observed[k]=digest(dst);bytes_read+=st.st_size
                    check(st.st_size==r["bytes"] and observed[k]==r["sha256"],"H readback: "+r["path"])
            if v["number"]%12==0:self.status("reading_back_H_disk",last_volume=v["number"],bytes_read=bytes_read,
                                               unique_bytes=self.catalog["unique_bytes"])
        for r in document(self.ctrl/"git_tree.json")["files"]:
            check(digest(self.project/r["path"])==r["sha256"],"Tracked source readback: "+r["path"])
        original=document(self.ctrl/"original.json")
        check(digest(self.dest/original["member"])==original["sha256"],"Original bag readback")
        bundle=self.dest/"DN-Splatter.bundle"
        if not bundle.exists():
            head=git("rev-parse","refs/heads/main")
            git("merge-base","--is-ancestor",RESULT,head)
            tmp=bundle.with_suffix(".pending")
            subprocess.run(["git","bundle","create",tmp,"refs/heads/main"],cwd=ROOT,check=True)
            tmp.replace(bundle)
        subprocess.run(["git","bundle","verify",bundle],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
        bundle_head=git("bundle","list-heads",bundle,"refs/heads/main").split()[0]
        git("merge-base","--is-ancestor",RESULT,bundle_head)
        return dict(storage_readback_bytes=bytes_read,tracked_files=len(document(self.ctrl/"git_tree.json")["files"]),
                    bag_sha256=original["sha256"],source_bundle={**checksum_record(bundle),"head":bundle_head})

    def run(self):
        with exclusive(self.ctrl/"receiver.lock"):
            try:
                self.status("waiting_for_catalog")
                self.get_catalog()
                self.status("frozen_source")
                self.frozen_tree()
                self.status("including_original_bag")
                self.original()
                # Overlap network reads with the single external-disk writer.
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for _ in pool.map(self.pull_volume,self.plan):pass
                sealed=json.loads(remote_bytes(STAGE+"/sealed.json"))
                check(sealed["inventory_sha256"]==self.catalog["inventory"]["sha256"] and
                      sealed["volumes"]==len(self.done)==len(self.plan),"Server/client completeness")
                atomic(self.ctrl/"server_sealed.json",sealed)
                verification=self.alias_and_verify()
                result=dict(status="verified",run_id=ID,result_commit=RESULT,verified_utc=utc(),**verification,
                            logical_files=self.catalog["logical_files"],logical_bytes=self.catalog["logical_bytes"],
                            unique_files=self.catalog["unique_files"],unique_bytes=self.catalog["unique_bytes"],
                            b1_files=self.catalog["b1_manifest_files"],b0_files=self.catalog["b0_manifest_files"],
                            volumes=len(self.plan),archive_root=str(self.dest))
                atomic(self.ctrl/"private_complete.json",result)
                self.status("private_archive_verified",**{k:v for k,v in result.items() if k!="status"})
            except Exception as e:
                self.status("interrupted",error_type=type(e).__name__,reason=str(e)[:650])
                raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",required=True,type=Path)
    parser.add_argument("--connections",type=int,default=16)
    args=parser.parse_args()
    check(4<=args.connections<=32,"Expected 4 to 32 transfer lanes")
    Archive(args.destination,args.connections).run()
