"""Fit the causal actuator response and one effective friction, then replay complete episodes."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import ss2tf,lfilter

from interactive_physics import HZ,SUBSTEPS,RigidWorld,look_at
from interactive_physics_memory import EvidenceMemory
from interactive_push_assets import file_hash,write_json


def read(path):
    return json.loads(path.read_text())


def episode(track,e):
    root=track/"labels"/f"episode_{e:06d}"
    r=read(root.with_suffix(".json"))
    if r["labels_sha256"]!=file_hash(root.with_suffix(".npz"))[0]:
        raise ValueError("Object label identity mismatch")
    with np.load(root.with_suffix(".npz"),allow_pickle=False) as f:
        return dict(f)


def wrapped(x):
    return np.arctan2(np.sin(x),np.cos(x))


def filtered_robot(data,kp,b,lag):
    dt=1/(HZ*SUBSTEPS);m=.08
    a=1/(1+dt*b/m);g=dt*kp*a/m
    micro=np.array([[1-dt*g,dt*a],[-g,a]])
    force=np.array([[dt*g],[g]])
    trans=np.linalg.matrix_power(micro,SUBSTEPS)
    target=sum((np.linalg.matrix_power(micro,j)@force for j in range(SUBSTEPS)))
    num,den=ss2tf(trans,target,np.array([[1.,0.]]),np.zeros((1,1)))
    scores=[]
    for d in data:
        cmd=d["action_command"];base=d["executed_pusher_xy"][0]
        delayed=cmd[np.maximum(np.arange(len(cmd))-lag,0)]-base
        pos=lfilter(num[0],den,delayed,axis=0)+base
        scores.append(float(np.linalg.norm(pos-d["executed_pusher_xy"],axis=-1).mean()))
    return float(np.mean(scores))


def replay(world,d,initial_fallback,mode="open_loop",memory_mode="M1"):
    pose=d["object_xytheta"];valid=d["object_valid"]
    initial=pose[0] if valid[0] else initial_fallback
    st=world.reset({"object_xytheta":initial,"pusher_xy":d["executed_pusher_xy"][0]})
    n=len(pose)
    pred=np.zeros((n,3));post=np.zeros_like(pred);pusher=np.zeros((n,2))
    contact=np.zeros(n,bool);penetration=np.zeros(n);vers=np.zeros(n,int)
    mem=EvidenceMemory(memory_mode)
    for i in range(n):
        pred[i]=st["object_xytheta"];pusher[i]=st["pusher_xy"]
        contact[i]=st["contact"];penetration[i]=st["penetration_m"]
        if mode=="causal_online" and i:
            record=mem.pose_observation(i,pose[i],i,bool(valid[i]))
            if record is not None:
                world.correct_from_evidence(record["pose"],record["version"],record["source_group"])
                st=world.state()
        post[i]=st["object_xytheta"];vers[i]=st["version"]
        if i+1<n:
            lag=world.scene["physics"].get("command_delay_frames",0)
            st=world.step(d["action_command"][max(0,i-lag)])
    return {"predicted_pre_observation":pred,"posterior":post,"executed_pusher_xy":pusher,
            "contact_simulated":contact,"penetration_m":penetration,"world_version":vers,
            "initial_from_observation":bool(valid[0])}


def errors(result,d):
    y=d["object_xytheta"]; mask=d["object_valid"]
    pred=result["predicted_pre_observation"]
    dist=np.linalg.norm(pred[:,:2]-y[:,:2],axis=1)
    angle=np.abs(wrapped(pred[:,2]-y[:,2]))
    take=lambda x,keep: float(np.mean(x[keep])) if keep.any() else None
    return {"frames":len(y),"valid":int(mask.sum()),"mean_position_m":take(dist,mask),"mean_angle_rad":take(angle,mask),
            "all_frame_invalid":int((~mask).sum()),"open_initial_valid":bool(mask[0]),
            "pusher_mean_m":float(np.linalg.norm(result["executed_pusher_xy"]-d["executed_pusher_xy"],axis=-1).mean()),
            "simulated_contact_steps":int(result["contact_simulated"].sum()),"max_penetration_m":float(result["penetration_m"].max()),
            "at_seconds":{str(h):{"position_m":float(dist[h*HZ]) if mask[h*HZ] else None,
                                 "angle_rad":float(angle[h*HZ]) if mask[h*HZ] else None} for h in (1,2) if h*HZ<len(y)}|
                         {"0.5":{"position_m":float(dist[15]) if mask[15] else None,"angle_rad":float(angle[15]) if mask[15] else None}},
            "endpoint_position_m":float(dist[-1]) if mask[-1] else None,"endpoint_angle_rad":float(angle[-1]) if mask[-1] else None}


def fit(root,track,output):
    roles=read(root/"prepare/split.json")["roles"]
    data={e:episode(track,e) for e in roles["fit"]+roles["calibration"]}
    ff=[data[i] for i in roles["fit"]];cc=[data[i] for i in roles["calibration"]]
    starts=np.array([d["object_xytheta"][0] for d in ff if d["object_valid"][0]])
    if len(starts)<3:raise RuntimeError("Too few independent starting poses")
    fallback=np.mean(starts,axis=0);fallback[2]=np.arctan2(np.sin(starts[:,2]).mean(),np.cos(starts[:,2]).mean())
    c0=read(root/"prepare/scene_C0.json")
    tracker=read(track/"tracker_freeze.json")
    c0["cameras"]["front"]={"K":tracker["K"],"T_world_camera_cv":tracker["T_world_camera_cv"],
                            "size_wh":[240,240],"source":"shared frozen RGB-to-table camera"}
    c0["cameras"]["second"]=look_at([.2,-.4,.4],[.46,0.,0.])
    c0["physics"]["command_delay_frames"]=0
    trials=[]
    for lag in range(7):
        candidates=[(filtered_robot(ff,k,b,lag),k,b) for k in (60.,120.,240.,420.,750.,1200.) for b in (5.,10.,15.,25.,40.)]
        f,k,b=min(candidates)
        trials.append({"lag":lag,"kp":k,"damping":b,"fit_m":f,"calibration_m":filtered_robot(cc,k,b,lag)})
    chosen=min(trials,key=lambda row:row["calibration_m"])
    c1=copy.deepcopy(c0)
    c1["physics"].update(servo_kp=chosen["kp"],servo_damping=chosen["damping"],command_delay_frames=chosen["lag"])
    qualities=[]
    for mu in (.20,.35,.55,.80,1.10):
        scene=copy.deepcopy(c1);scene["objects"][0]["friction"]={"value":mu,"status":"candidate"}
        world=RigidWorld(scene);scores={}
        for key in ("fit","calibration"):
            qs=[errors(replay(world,data[i],fallback),data[i])["mean_position_m"] for i in roles[key]]
            scores[key]=float(np.mean([v for v in qs if v is not None]))
        qualities.append({"mu":mu,**scores})
        print(f"friction {mu}: {scores}",flush=True)
    # Calibrate on allowed episodes; only the one effective contact coefficient can change.
    eligible=sorted(qualities,key=lambda row:row["fit"])[:2]
    pick=min(eligible,key=lambda row:row["calibration"])
    c2=copy.deepcopy(c1);c2["objects"][0]["friction"]={"value":pick["mu"],"status":"fit effective sliding coefficient",
        "fit_cases":len(ff),"calibration_cases":len(cc),"range_considered":[.20,1.10]}
    c3=copy.deepcopy(c2)
    c3["evidence"]["generation_rule"]="freeze accepted static shape and parameters; start each paired episode from shared initial pose; no observation updates"
    for tag,s in zip(("C0","C1","C2","C3"),(c0,c1,c2,c3)):
        s["versions"]["calibration"]=tag
        write_json(output/f"{tag}.json",s)
    write_json(output/"freeze.json",{"fit_episodes":roles["fit"],"calibration_episodes":roles["calibration"],
        "tracker_sha256":file_hash(track/"tracker_freeze.json")[0],"default_initial_from_fit":fallback.tolist(),
        "control_response_profile":trials,"selected_control":chosen,"friction_profile":qualities,
        "selected_friction":pick,"mass_status":"assumed; no measured contact force",
        "C3_frozen_geometry":"same known official shape and physical coefficients as C2. RGB pose evidence is episode-local; no persistent geometry correction established",
        "versions":{k:file_hash(output/f"{k}.json")[0] for k in ("C0","C1","C2","C3")}})


def evaluate(root,track,output):
    roles=read(root/"prepare/split.json")["roles"];freeze=read(output/"freeze.json")
    fallback=np.asarray(freeze["default_initial_from_fit"])
    for mode in ("open_loop","causal_online"):
        result={}
        for tag in ("C0","C1","C2","C3"):
            world=RigidWorld(read(output/f"{tag}.json"))
            for role,items in roles.items():
                rows=[]
                for e in items:
                    d=episode(track,e);dest=output/"replay"/mode/tag/f"episode_{e:06d}"
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.with_suffix(".json").is_file():
                        rec=read(dest.with_suffix(".json"))
                        if rec["sha256"]!=file_hash(dest.with_suffix(".npz"))[0] or rec["world_sha256"]!=freeze["versions"][tag]:
                            raise ValueError("Replay identity changed")
                    else:
                        arr=replay(world,d,fallback,mode,"M2" if tag=="C3" else "M1")
                        np.savez_compressed(dest.with_suffix(".npz"),**arr)
                        rec={"episode":e,**errors(arr,d),"sha256":file_hash(dest.with_suffix(".npz"))[0],
                             "world_sha256":freeze["versions"][tag]}
                        write_json(dest.with_suffix(".json"),rec)
                    rows.append(rec)
                result.setdefault(tag,{})[role]=rows
        write_json(output/f"{mode}.json",{"mode":mode,"forecast_label":"command-conditioned pre-observation state",
                   "test_state_reused_across_episodes":False,"physical_force_reference":None,"results":result})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["fit","evaluate"])
    p.add_argument("root",type=Path);p.add_argument("--track",default="track_v2")
    a=p.parse_args();out=a.root/"calibration";out.mkdir(parents=True,exist_ok=True)
    (fit if a.mode=="fit" else evaluate)(a.root,a.root/a.track,out)
