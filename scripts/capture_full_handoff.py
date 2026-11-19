"""Record the sealed packet and a verified snapshot from the live server runner."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw
from capture_rectify import read, save, sha

def main(host):
    p=read(Path("configs/capture_full_test_protocol.json"))
    paths=p["paths"]
    local=Path(paths["local_transfer"])
    report=Path(paths["report"])
    report.mkdir(parents=True,exist_ok=True)
    run=paths["run"]
    geo=paths["geometry_run"]
    now=datetime.now(timezone.utc).isoformat()
    snippet=f"""import json,hashlib,pathlib
p=pathlib.Path('{run}')/'state'
out={{k:json.loads((p/k).read_text()) for k in ('identity.json','current.json','features.done.json','rectify.done.json')}}
g=pathlib.Path('{geo}')
out['packet_sha256']=hashlib.sha256((g/'transfer/manifest.json').read_bytes()).hexdigest()
out['rectified_pairs']=json.loads((g/'rect/registration.json').read_text())['frame_count']
out['tmux_session']='dn-capture-full-30k'
out['stage_records']=[json.loads(l) for l in (p/'stages.jsonl').read_text().splitlines()]
print(json.dumps(out))"""
    command=f"cd {paths['server_project_root']} && tmux has-session -t '=dn-capture-full-30k' && {paths['environment']}/bin/python -"
    evidence=json.loads(subprocess.check_output(["ssh","-o","BatchMode=yes",host,command],input=snippet.encode()))
    assert evidence["packet_sha256"]==sha(local/"manifest.json")==evidence["identity.json"]["packet_sha256"]
    rows=read(local/"selection.json")
    selection_keys=("id","candidate_index_zero_based","target_device_timestamp_ns","actual_rgb_timestamp_ns","actual_depth_timestamp_ns",
                    "rgb_zero_based_index","depth_zero_based_index","requested_split","relative_s","depth_minus_rgb_ns",
                    "rgb_payload_sha256","depth_pixels_sha256","drop_reason","annotation","laplacian_variance")
    save(report/"selection.json",dict(source=read(local/"identity.json")["source"],packet_selection_sha256=sha(local/"selection.json"),
         rows=[{k:r.get(k) for k in selection_keys} for r in rows]))
    split=read(local/"splits.json")
    save(report/"split_manifest.json",dict(requested={s:[r["id"] for r in rows if r["requested_split"]==s] for s in split["splits"]},
         selected=split["splits"],packet_split_sha256=sha(local/"splits.json"),identity_check=split["review"]))
    members=read(local/"manifest.json")["files"]
    sizes=sum(f["bytes"] for f in members)+ (local/"manifest.json").stat().st_size
    evidence.update(observed_utc=now,server_run=paths["server_project_root"]+"/"+run,
                    server_packet=paths["server_project_root"]+"/"+geo+"/transfer",packet_files_including_manifest=len(members)+1,
                    packet_bytes_including_manifest=sizes,rgb_span_s=(int(rows[-1]["actual_rgb_timestamp_ns"])-int(rows[0]["actual_rgb_timestamp_ns"]))/1e9)
    save(report/"handoff_running.json",evidence)
    js=Path(paths["latest"])
    latest=read(js)
    assert latest["status"] in ("running","planned_not_started_here")
    latest.update(status="running",kind="full_capture_execution",server_accessed_this_task=True,windows_bag_accessed_this_task=True,
        actual_requested=dict(Counter(r["requested_split"] for r in rows),total=len(rows)),
        actual_selected={s:len(m) for s,m in split["splits"].items()}|{"total":sum(map(len,split["splits"].values()))},
        actual_updates=0,observed_stage=evidence["current.json"]["stage"],observed_utc=now,
        note="Live handoff snapshot. Geometry/30k/evaluation/Poisson completion is recorded in the final verified report.")
    save(js,latest)
    subset=[rows[0],rows[238],rows[len(rows)//2],min(rows,key=lambda r:abs(r["relative_s"]-353.25)),rows[-1]]
    canvas=Image.new("RGB",(320*len(subset),210),(32,33,36))
    pen=ImageDraw.Draw(canvas)
    for i,row in enumerate(subset):
        with Image.open(local/row["rgb"]["file"]) as im:
            tile=im.convert("RGB").resize((320,180),Image.Resampling.LANCZOS)
        if 349<=row["relative_s"]<=353.7:
            ImageDraw.Draw(tile).rectangle((0,0,170,125),fill=(45,48,50))
        canvas.paste(tile,(320*i,0))
        pen.text((320*i+7,187),f"{row['id']}   {row['relative_s']:.3f} s   {row['requested_split']}",fill="white")
    previews=report/"previews"
    previews.mkdir(exist_ok=True)
    canvas.save(previews/"source_timeline.jpg",quality=88)
    print(json.dumps(dict(stage=evidence["current.json"]["stage"],bytes=sizes,files=len(members)+1,observed=now)))

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("host")
    main(ap.parse_args().host)
