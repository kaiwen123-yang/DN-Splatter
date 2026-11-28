"""Paired open-loop MuJoCo rollouts with streamed RGB8/metric-depth16 shards."""

import argparse
import concurrent.futures
import json
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from interactive_physics import RigidWorld
from interactive_push_assets import file_hash,write_json
from interactive_push_replay import episode,read


def bank(root,out,track):
    path=out/"initial_states_actions.npz"
    identity=file_hash(root/"prepare/split.json")[0]+file_hash(track/"tracker_freeze.json")[0]
    if (out/"bank.json").is_file():
        r=read(out/"bank.json")
        if identity!=r["input_identity"] or r["sha256"]!=file_hash(path)[0]:
            raise ValueError("Existing rollout bank identity changed")
        return
    ids=read(root/"prepare/split.json")["roles"]["fit"]
    records={i:episode(track,i) for i in ids}
    rng=np.random.default_rng(42)
    selected=rng.choice(ids,512,replace=True)
    obj=[];pusher=[];actions=[];source=[]
    for e in selected:
        d=records[e]
        valid=np.flatnonzero(d["object_valid"][:len(d["object_valid"])-150])
        if len(valid)==0:raise ValueError(f"No eligible window in fit episode {e}")
        i=int(rng.choice(valid))
        obj.append(d["object_xytheta"][i]);pusher.append(d["executed_pusher_xy"][i])
        actions.append(d["action_command"][i:i+150]);source.append([e,i])
    visual_ids=np.sort(rng.choice(np.arange(512),size=32,replace=False))
    np.savez_compressed(path,object_xytheta=obj,pusher_xy=pusher,commands=actions,
                        source_episode_frame=source,full_visual_ids=visual_ids)
    write_json(out/"bank.json",{"episodes":512,"control_steps":150,"input_identity":identity,
        "sources":"frozen fit-only action windows, identical initial poses/commands for all world versions",
        "sha256":file_hash(path)[0],"visual_ids":visual_ids.tolist(),"free_bytes_before":shutil.disk_usage(out).free,
        "uncompressed_full_rgbd_bytes":32*3*150*2*240*240*5,"uncompressed_preview_rgbd_bytes":512*3*25*2*96*96*5})


def png(stream,path,img):
    ok,b=cv2.imencode(".png",img,[cv2.IMWRITE_PNG_COMPRESSION,1])
    if not ok:raise OSError("PNG encode failed")
    stream.writestr(path,b.tobytes())


def visual(stream,world,step,full):
    for cam in world.scene["cameras"]:
        observation=world.observe(cam)
        rgb=observation["rgb"]
        z=observation["depth_m"]
        # 0 is unknown, 65535 means an overflow. The code is an explicit overflow mask.
        code=np.zeros(z.shape,np.uint16)
        v=np.isfinite(z)&(z>0)
        raw=np.rint(z[v]*1000)
        code[v]=np.clip(raw,1,65535).astype(np.uint16)
        if full:
            prefix=f"full/{cam}/{step:03d}"
            png(stream,prefix+"_rgb.png",rgb[:,:,::-1]);png(stream,prefix+"_depth.png",code)
        if (step-1)%6==0:
            prefix=f"preview/{cam}/{step:03d}"
            png(stream,prefix+"_rgb.png",cv2.resize(rgb[:,:,::-1],(96,96),interpolation=cv2.INTER_AREA))
            png(stream,prefix+"_depth.png",cv2.resize(code,(96,96),interpolation=cv2.INTER_NEAREST))


def one(task):
    root,out,tag,i=task
    cv2.setNumThreads(1)
    world_path=root/"calibration"/(tag+".json");bank_path=out/"initial_states_actions.npz"
    dst=out/tag/f"episode_{i:06d}";dst.mkdir(parents=True,exist_ok=True)
    receipt=dst/"receipt.json"
    identity=[file_hash(world_path)[0],file_hash(bank_path)[0]]
    if receipt.is_file():
        r=read(receipt)
        if r["identity"]!=identity or any(file_hash(dst/k)[0]!=v["sha256"] for k,v in r["files"].items()):
            raise ValueError(f"Existing rollout identity changed: {tag}/{i}")
        return r
    with np.load(bank_path,allow_pickle=False) as a:
        start=a["object_xytheta"][i];pusher=a["pusher_xy"][i];cmd=a["commands"][i]
        full=i in a["full_visual_ids"]
    world=RigidWorld(read(world_path));state=world.reset({"object_xytheta":start,"pusher_xy":pusher})
    positions=[state["object_xytheta"]];pp=[state["pusher_xy"]];quat=[state["quaternion_wxyz"]];xyz=[state["object_xyz"]]
    pen=[];ct=[];versions=[world.version]
    tmp=dst/"visuals.pending.zip"
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as stream:
        for j in range(150):
            lag=world.scene["physics"].get("command_delay_frames",0)
            s=world.step(cmd[max(0,j-lag)])
            positions.append(s["object_xytheta"]);pp.append(s["pusher_xy"]);quat.append(s["quaternion_wxyz"])
            xyz.append(s["object_xyz"]);pen.append(s["penetration_m"]);ct.append(s["contact"]);versions.append(s["version"])
            if full or j%6==0:visual(stream,world,j+1,full)
        metadata={"episode":i,"version":tag,"cameras":world.scene["cameras"],
            "full_visual":bool(full),"full_wh":[240,240],"preview_wh":[96,96],"preview_intrinsics_scale":.4,
            "preview_principal_point_rule":"(c + 0.5)*0.4 - 0.5","action":"planar base-frame command in meters",
            "depth":"uint16 optical Z in millimeters; zero unknown, code 65535 overflow", "rgb":"RGB8 in PNG channel order",
            "start_time_s":0.,"frame_index":"j+1 over simulated interval [j/30,(j+1)/30]", "contact_source":"MuJoCo"}
        stream.writestr("metadata.json",json.dumps(metadata))
    os.replace(tmp,dst/"visuals.zip")
    np.savez_compressed(dst/"states.npz",time_s=np.arange(151)/30,action_command=cmd,executed_pusher_xy=pp,
                        object_xytheta=positions,object_xyz=xyz,quaternion_wxyz=quat,sim_contact=ct,penetration_m=pen,
                        scene_version=versions,observed_present=np.zeros(151,bool),
                        state_finite=np.isfinite(positions).all(axis=1))
    z=np.array(xyz)[:,2]; failures=[]
    if not np.isfinite(xyz).all():failures.append("nonfinite")
    if np.any(z<world.scene["background"]["top_z"]-.015):failures.append("fell_below_support")
    if max(pen)>.005:failures.append("penetration_over_5mm")
    r={"episode":i,"version":tag,"steps":150,"full_visual":bool(full),"preview_frames_per_camera":25,
       "contact_steps":int(sum(ct)),"failure_reasons":failures,"identity":identity,"observations_during_generation":0,
       "files":{f:{"sha256":file_hash(dst/f)[0],"bytes":(dst/f).stat().st_size} for f in ("states.npz","visuals.zip")}}
    write_json(receipt,r)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("--track",default="track_v2")
    p.add_argument("--workers",type=int,default=2);a=p.parse_args()
    out=a.root/"generated";out.mkdir(exist_ok=True)
    bank(a.root,out,a.root/a.track)
    tasks=[(a.root,out,t,i) for i in range(512) for t in ("C0","C2","C3")]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=[]
        for r in pool.map(one,tasks,chunksize=1):
            results.append(r)
            if len(results)%24==0:print(f"rollouts {len(results)}/1536",flush=True)
    write_json(out/"manifest.json",{"episodes":len(results),"control_steps":sum(x["steps"] for x in results),
          "full_visual_episodes":sum(x["full_visual"] for x in results),"all_failures_retained":True,
          "inputs":read(out/"bank.json"),"episodes_index":results})


if __name__=="__main__":
    main()
