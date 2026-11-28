"""Restore the pinned public MASt3R release in place or via a memory-only relay."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request


ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/".local/runs/interactive-world-20261128-v1/b_map/matching/weights"
REV="06e7259f34c3060f322df5cb0c7b9094f57e41fc"
BASE="https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/"+REV+"/"
FILES={"config.json":(546,"e485617403af6e7900f6157a399a379176b61656"),
       "model.safetensors":(2754661648,"0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb")}


def check(path,name):
    size,expected=FILES[name]
    if path.stat().st_size != size:
        raise IOError("Incomplete public model member: "+name)
    h=hashlib.sha1(f"blob {size}\0".encode()) if name=="config.json" else hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(4<<20),b""):
            h.update(data)
    if h.hexdigest()!=expected:
        raise IOError("Public model member hash mismatch: "+name)


def offset(name):
    final=OUTPUT/name
    if final.exists():
        check(final,name)
        return FILES[name][0]
    part=OUTPUT/(name+".part")
    return part.stat().st_size if part.exists() else 0


def receive(name,start,stream):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    current=offset(name)
    if current!=start:
        raise IOError("Partial download changed concurrently")
    if (OUTPUT/name).exists():
        return
    part=OUTPUT/(name+".part")
    with part.open("ab") as dst:
        shutil.copyfileobj(stream,dst,1<<20)
    check(part,name)
    os.replace(part,OUTPUT/name)


def transfer(name,start,dst):
    size=FILES[name][0]
    step=8<<20
    def part(left):
        right=min(left+step,size)-1
        for attempt in range(5):
            url=BASE+name+f"?slice={left}-{right}&attempt={attempt}"
            headers={"Accept-Encoding":"identity","Range":f"bytes={left}-{right}"}
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as response:
                    single=(left==0 and right+1==size and response.status==200)
                    if not single and (response.status!=206 or response.headers.get("Content-Range")!=f"bytes {left}-{right}/{size}"):
                        raise IOError("Origin returned a different byte range")
                    data=response.read(right-left+2)
                    if len(data)!=right-left+1:
                        raise IOError("Incomplete public download range")
                    return data
            except OSError:
                if attempt==4:
                    raise
                time.sleep(1+attempt)
    positions=iter(range(start,size,step))
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending=deque()
        for _ in range(5):
            pos=next(positions,None)
            if pos is not None:
                pending.append(pool.submit(part,pos))
        while pending:
            dst.write(pending.popleft().result())
            pos=next(positions,None)
            if pos is not None:
                pending.append(pool.submit(part,pos))


def restore(args):
    for name,(size,_) in FILES.items():
        cmd=["/root/miniconda3/bin/python",args.remote_root+"/scripts/world_match_sources.py"]
        if args.transport:
            out=subprocess.check_output([*args.transport,"run",shlex.join([*cmd,"offset",name])])
            start=int(out)
        else:
            start=offset(name)
        if start>=size:
            if start>size:
                raise IOError("Unexpected partial download size")
            if args.transport:
                subprocess.run([*args.transport,"stream",shlex.join([*cmd,"receive",name,str(start)])],input=b"",check=True)
            else:
                import io
                receive(name,start,io.BytesIO())
            continue
        if args.transport:
            with subprocess.Popen([*args.transport,"stream",shlex.join([*cmd,"receive",name,str(start)])],stdin=subprocess.PIPE) as proc:
                transfer(name,start,proc.stdin)
                proc.stdin.close()
                if proc.wait():
                    raise IOError("Remote public release verification failed")
        else:
            OUTPUT.mkdir(parents=True,exist_ok=True)
            tmp=OUTPUT/(name+".part")
            with tmp.open("ab") as dst:
                transfer(name,start,dst)
            check(tmp,name)
            os.replace(tmp,OUTPUT/name)
        print(json.dumps(dict(name=name,total_bytes=size,transmitted_bytes=size-start,revision=REV)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["restore","offset","receive"])
    p.add_argument("name",nargs="?",choices=FILES)
    p.add_argument("start",nargs="?",type=int)
    p.add_argument("--remote-root",default="/root/autodl-tmp/DN-Splatter")
    p.add_argument("--transport",nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.action=="restore":
        restore(a)
    elif a.action=="offset":
        print(offset(a.name))
    else:
        receive(a.name,a.start,sys.stdin.buffer)
