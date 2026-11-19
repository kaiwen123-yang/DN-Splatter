"""Coordinate the one bounded bridge on the source host and the mapping instance."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import time

from capture_rectify import files, read, save, sha, verify


def select_map(geo):
    initial=read(geo/"sfm/map_exhaustive.json")
    rows=read(geo/"transfer/selection.json")
    train=[r for r in rows if r["requested_split"]=="train" and not r["drop_reason"]]
    best=max(initial["components"],key=lambda c:len(c["registered_primary"]),default={"registered_primary":[]})
    registered=set(best["registered_primary"])
    missing=[r["id"]+".png" not in registered for r in train]
    bounds=[int(r["actual_rgb_timestamp_ns"]) for i,r in enumerate(train)
            if missing[i] and (i==0 or i==len(train)-1 or not missing[i-1] or not missing[i+1])]
    bridge=geo/"bridge"
    bridge.mkdir(exist_ok=True)
    save(bridge/"request.json",dict(bag_sha256=read(geo/"transfer/identity.json")["source"]["sha256"],
         packet_sha256=sha(geo/"transfer/manifest.json"),windows_ns=[[t-2_000_000_000,t+2_000_000_000] for t in bounds],
         initial_missing=[r["id"] for r,m in zip(train,missing) if m]))
    if not bounds:
        save(geo/"sfm/map_full.json",initial)
    return bool(bounds)


def serve(host):
    from capture_bag_samples import fetch_from_chunk
    from capture_bag_scan import RGB, stamp
    from capture_full_transfer import transfer
    from rosbags.rosbag1 import Reader
    p=read(Path("configs/capture_full_test_protocol.json"))
    local=Path(p["paths"]["local_extraction"])
    remote=p["paths"]["server_project_root"]+"/"+p["paths"]["geometry_run"]+"/bridge"
    run=p["paths"]["server_project_root"]+"/"+p["paths"]["run"]
    command=f"if test -s {remote}/request.json; then cat {remote}/request.json; elif test -e {run}/state/exit_code; then echo null; fi"
    while True:
        probe=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10",host,command],capture_output=True)
        if probe.returncode==0 and probe.stdout.strip():
            request=json.loads(probe.stdout)
            break
        time.sleep(45)
    if not request or not request["windows_ns"]:
        return
    assert request["packet_sha256"]==sha(local/"transfer/manifest.json")
    primary=read(local/"transfer/selection.json")
    deny={r["rgb_zero_based_index"] for r in primary}
    times={int(r["actual_rgb_timestamp_ns"]) for r in primary}
    identity=read(local/"transfer/identity.json")
    bag=Path(p["paths"]["windows_raw_bag"])
    assert request["bag_sha256"]==identity["source"]["sha256"] and str(stamp(bag)["mtime_ns"])==str(identity["source"]["mtime_ns"])
    audit=Path(".local/bag-audit")/identity["index_run"]
    idx=audit/"times.sqlite"
    assert sha(idx)==identity["index_sha256"]
    output=local/"bridge"
    output.mkdir(exist_ok=True)
    if not (output/"manifest.json").exists():
        with sqlite3.connect(idx.resolve().as_uri()+"?mode=ro",uri=True) as db:
            db.row_factory=sqlite3.Row
            cid=read(audit/"scan.json")["catalog"][RGB]["connection_id"]
            chunks=defaultdict(list)
            for r in db.execute("select * from times where cid=? order by idx",(cid,)):
                if r["idx"] not in deny and r["device_ns"] not in times and any(a<=r["device_ns"]<=b for a,b in request["windows_ns"]):
                    chunks[r["chunk"]].append(r)
        frames=[]
        with Reader(bag) as reader:
            for pos,entries in sorted(chunks.items()):
                c=reader.chunks[pos]
                reader.bio.seek(c.datapos)
                block=io.BytesIO(c.decompressor(reader.bio.read(c.datasize)))
                for r in entries:
                    data,m=fetch_from_chunk(block,r["offset"],cid,r["idx"],r["record_ns"])
                    assert int(m["device_ns"])==r["device_ns"] and m["format"]["encoding"]=="mjpg"
                    stem=f"aux_{r['idx']:05d}"
                    (output/(stem+".jpg")).write_bytes(data)
                    frames.append(dict(id=stem,source_index=r["idx"],device_ns=r["device_ns"],role="pose_auxiliary_train"))
        save(output/"frames.json",frames)
        save(output/"request_copy.json",request)
        save(output/"manifest.json",dict(files=files(output)))
    assert read(output/"request_copy.json")==request
    while True:
        try:
            transfer(host,output,remote+"/packet")
            return
        except (OSError,subprocess.CalledProcessError,RuntimeError):
            time.sleep(45)


def prepare_aux(geo):
    import cv2
    from capture_rectify import grid_setup,imwrite
    bridge=geo/"bridge"
    packet=bridge/"packet"
    while not (packet/"manifest.json").is_file():
        time.sleep(15)
    verify(packet)
    assert read(packet/"request_copy.json")==read(bridge/"request.json")
    rows=read(geo/"transfer/selection.json")
    deny={r["rgb_zero_based_index"] for r in rows}
    times={int(r["actual_rgb_timestamp_ns"]) for r in rows}
    aux=read(packet/"frames.json")
    _,_,grids=grid_setup(read(geo/"transfer/calibration.json"))
    _,mx,my,valid,_=grids["1280"]
    for row in aux:
        assert row["source_index"] not in deny and row["device_ns"] not in times
        rgb=cv2.imread(str(packet/(row["id"]+".jpg")),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        im=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR)
        im[~valid]=0
        imwrite(geo/"rect/1280/images"/(row["id"]+".png"),im)
    save(bridge/"frames.json",aux)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("host")
    serve(ap.parse_args().host)
