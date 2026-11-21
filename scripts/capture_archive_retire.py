"""Retire only this run's registered transport volumes after the private and public receipts verify."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from datetime import datetime, timezone

ID="capture-full-metric3d-30k-20261120-v1"
RESULT="7d4a8a1d7d47379102cc5a17d2bbb7e07231ac42"


def need(ok,msg):
    if not ok:raise ValueError(msg)


def get(path):
    return json.loads(path.read_bytes())


def seal(path,doc):
    tmp=path.with_suffix(path.suffix+".pending")
    tmp.write_text(json.dumps(doc,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)


def retire(project,bundle,apply):
    root=project.resolve(strict=True)
    stage=root/".local"/"archives"/(ID+"-"+RESULT[:8])
    vols=stage/"volumes"
    need(stage.resolve(strict=True)==stage and vols.resolve(strict=True)==vols,"canonical, nonsymlink archive volume directory")
    src=get(stage/"catalog.json");sealed=get(stage/"sealed.json")
    delivered=json.loads(base64.b64decode(bundle["archive_complete"],validate=True))
    priv_bytes=base64.b64decode(bundle["private_complete"],validate=True)
    pub_bytes=base64.b64decode(bundle["public_complete"],validate=True)
    need(hashlib.sha256(priv_bytes).hexdigest()==delivered["private_receipt_sha256"] and
         hashlib.sha256(pub_bytes).hexdigest()==delivered["public_receipt_sha256"],"disk receipt digests")
    private=json.loads(priv_bytes);public=json.loads(pub_bytes)
    need(delivered["status"]=="complete_verified" and private["status"]=="verified" and
         public["status"]=="published_verified" and
         all(d["result_commit"]==RESULT for d in (src,sealed,delivered,private,public)) and
         src["run_id"]==sealed["run_id"]==delivered["run_id"]==private["run_id"]==ID,"frozen job identity")
    binding=src["inventory"]["sha256"]
    priv_seal=json.loads(base64.b64decode(bundle["server_sealed"],validate=True))
    need(priv_seal==sealed and binding==sealed["inventory_sha256"] and
         private["logical_files"]==src["logical_files"] and private["unique_files"]==src["unique_files"] and
         private["storage_readback_bytes"]>=src["unique_bytes"] and private["volumes"]==sealed["volumes"]==src["volume_count"],
         "H disk readback and server seal")
    originals=[]
    for run in ("capture-full-dn-30k-20261119-v1",ID):
        d=get(root/".local"/"runs"/run/"training_complete.json");p=Path(d["checkpoint"])
        need(d["step"]==29999 and p.is_file() and p.resolve().is_relative_to((root/".local"/"runs"/run/"training").resolve()),"preserved checkpoint")
        originals.append(dict(run=run,bytes=p.stat().st_size,sha256_at_freeze=d["checkpoint_sha256"]))
    activity=stage/"transport_retirement.json"
    targets=[];expected=set();total=0
    prior=get(activity) if activity.is_file() else None
    if prior:
        need(prior["inventory_sha256"]==binding and prior["private_receipt_sha256"]==delivered["private_receipt_sha256"],"cleanup retry identity")
    with (stage/"server.lock").open("a") as lk:
        fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for i in range(src["volume_count"]):
            name=f"part-{i:04d}.tar.zst";p=vols/name
            record=vols/f"part-{i:04d}.tar.json"
            need(record.is_file() and not record.is_symlink(),"registered volume descriptor")
            meta=get(record)
            need(meta["filename"]==name and meta["number"]==i and meta["catalog_sha256"]==binding,"registered volume binding")
            expected.add(name);total+=meta["bytes"]
            if not p.exists():
                need(prior is not None and not p.is_symlink(),"unexplained missing volume")
                continue
            s=p.lstat()
            need(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and p.resolve(strict=True).parent==vols and s.st_size==meta["bytes"],
                 "guarded volume target: "+name)
            if s.st_mtime_ns>record.stat().st_mtime_ns:
                h=hashlib.sha256()
                with p.open("rb") as f:
                    for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
                need(h.hexdigest()==meta["sha256"],"changed volume: "+name)
            targets.append((p,s.st_dev,s.st_ino,s.st_size))
        need(total==sealed["transport_bytes"] and {p.name for p in vols.glob("*.tar.zst")}<=expected,"only registered transport volumes")
        if not apply:
            return dict(stage="checked",remaining=len(targets),transport_bytes=total)
        if prior and prior["stage"]=="released" and not targets:
            return prior
        now=lambda:datetime.now(timezone.utc).isoformat()
        receipt=dict(stage="releasing",run_id=ID,result_commit=RESULT,inventory_sha256=binding,
                     private_receipt_sha256=delivered["private_receipt_sha256"],scope="registered part-NNNN.tar.zst only",
                     volume_count=len(expected),transport_bytes=total,files_released_this_invocation=len(targets),
                     free_before=(prior if prior and prior["stage"]=="releasing" else {}).get("free_before",shutil.disk_usage(root).free),
                     original_final_checkpoints_present=originals,utc=now())
        seal(activity,receipt)
        for p,dev,ino,size in targets:
            s=p.lstat()
            need((s.st_dev,s.st_ino,s.st_size)==(dev,ino,size) and stat.S_ISREG(s.st_mode),"volume changed after preflight")
            p.unlink()
        need(not list(vols.glob("*.tar.zst")),"volume removal incomplete")
        receipt.update(stage="released",free_after=shutil.disk_usage(root).free,utc=now())
        seal(activity,receipt)
        seal(stage/"server.json",dict(stage="temporary_volumes_released",archived_volumes=len(expected),
                                      inventory_sha256=binding,utc=receipt["utc"]))
        return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args()
    print(json.dumps(retire(args.project,json.load(sys.stdin),args.apply)))
