"""Paired state/action forecasting on real held-out complete PushT episodes."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from interactive_push_assets import file_hash,write_json


STEPS=np.array([15,30,60]);HIST=15
CONDITIONS=("RealOnly","Real+resampledReal","Real+C0","Real+C2","Real+C3")


def windows(path,e,real):
    with np.load(path,allow_pickle=False) as d:
        obj=d["object_xytheta"]; robot=d["executed_pusher_xy"]; cmd=d["action_command"]
        valid=d["object_valid"] if real else d["state_finite"]
    n=len(obj);valid=np.asarray(valid)&np.isfinite(obj).all(axis=-1)
    results=[]
    for h in STEPS:
        t=np.arange(HIST,n-h)
        if not len(t):continue
        used=np.array([valid[j-HIST:j+1].all() for j in t])&valid[t+h]
        t=t[used]
        if not len(t):continue
        hist=t[:,None]+np.arange(-HIST,1)
        cur=obj[t];prev=obj[hist]
        theta=prev[:,:,2]-cur[:,None,2]
        histx=np.concatenate((prev[:,:,:2]-cur[:,None,:2],np.sin(theta)[:,:,None],np.cos(theta)[:,:,None],
                              robot[hist]-cur[:,None,:2]),axis=-1).reshape(len(t),-1)
        future=np.zeros((len(t),60,2),np.float32)
        future[:,:h]=cmd[t[:,None]+np.arange(h)]-cur[:,None,:2]
        x=np.column_stack((histx, np.sin(cur[:,2]),np.cos(cur[:,2]),future.reshape(len(t),-1),np.full(len(t),h/30)))
        delta=obj[t+h]-cur
        y=np.column_stack((delta[:,:2],np.sin(delta[:,2]),np.cos(delta[:,2])-1))
        velocity=(obj[t]-obj[t-3])/3
        velocity[:,2]=np.arctan2(np.sin(obj[t,2]-obj[t-3,2]),np.cos(obj[t,2]-obj[t-3,2]))/3
        nominal=velocity*h
        # Distance to the expanded observed T region gives an RGB/waypoint contact proxy, not force.
        rel=robot[t]-cur[:,:2];c=np.cos(cur[:,2]);s=np.sin(cur[:,2])
        loc=np.column_stack((c*rel[:,0]+s*rel[:,1],-s*rel[:,0]+c*rel[:,1]))
        moving=(np.linalg.norm(delta[:,:2],axis=1)>.005)|(np.abs(np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2])))>np.deg2rad(3))
        contact=(np.abs(loc[:,0])<.087)&(loc[:,1]>-.113)&(loc[:,1]<.061)&((loc[:,1]>.003)|(np.abs(loc[:,0])<.037))
        results.append((x.astype("f4"),y.astype("f4"),np.column_stack((np.full(len(t),e),t,np.full(len(t),h),moving,contact)).astype("i4"),
                        nominal.astype("f4")))
    return results


def dataset(paths,real):
    batch=[]
    for e,path in paths:
        batch.extend(windows(path,e,real))
    if not batch:raise RuntimeError("No usable independent-label forecasting windows")
    return tuple(np.concatenate(group) for group in zip(*batch))


def error(pred,y):
    pos=np.linalg.norm(pred[:,:2]-y[:,:2],axis=1)
    a=np.arctan2(pred[:,2],pred[:,3]+1)-np.arctan2(y[:,2],y[:,3]+1)
    return pos,np.abs(np.arctan2(np.sin(a),np.cos(a)))


def metrics(pred,data):
    _,y,meta,_=data
    p,a=error(pred,y);result={}
    for h in STEPS:
        for name,sel in (("all",np.ones(len(meta),bool)),("moving",meta[:,3]==1),("stationary",meta[:,3]==0),("contact_proxy",meta[:,4]==1)):
            valid=(meta[:,2]==h)&sel
            eps=np.unique(meta[valid,0])
            key=f"{h/30:g}s/{name}"
            if not len(eps):result[key]={"windows":0,"episodes":0,"position_m":None,"angle_rad":None};continue
            per=np.array([[v[(meta[:,0]==e)&valid].mean() for v in (p,a)] for e in eps])
            boot=np.random.default_rng(42).integers(0,len(eps),(1500,len(eps)))
            ci=np.quantile(per[boot].mean(1),[.025,.975],axis=0)
            result[key]={"windows":int(valid.sum()),"episodes":len(eps),"mean_episode_position_m":float(per[:,0].mean()),
                "mean_episode_angle_rad":float(per[:,1].mean()),"position_window_p95_m":float(np.quantile(p[valid],.95)),
                "position_ci95_m":ci[:,0].tolist(),"angle_ci95_rad":ci[:,1].tolist()}
    return result


def eval_pred(model,x,device,means):
    xp,sp,yp,sy=means
    result=[]
    with torch.no_grad():
        for start in range(0,len(x),4096):
            tensor=torch.from_numpy((x[start:start+4096]-xp)/sp).to(device)
            result.append(model(tensor).cpu().numpy()*sy+yp)
    return np.concatenate(result)


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("root",type=Path)
    cli.add_argument("--track",default="track_v2");cli.add_argument("--device",default="cpu");cli.add_argument("--threads",type=int,default=2)
    arg=cli.parse_args();root=arg.root;torch.set_num_threads(arg.threads)
    out=root/"probe";out.mkdir(exist_ok=True)
    roles=json.loads((root/"prepare/split.json").read_text())["roles"]
    track=root/arg.track
    sets={k:dataset([(i,track/"labels"/f"episode_{i:06d}.npz") for i in ids],True) for k,ids in roles.items()}
    x,y,_,_=sets["fit"]
    means=[x.mean(0),np.maximum(x.std(0),.001),y.mean(0),np.maximum(y.std(0),.001)]
    dev=arg.device
    def normalize(d):
        return (torch.from_numpy((d[0]-means[0])/means[1]).to(dev),torch.from_numpy((d[1]-means[2])/means[3]).to(dev))
    real=normalize(sets["fit"])
    test=sets["test"];nom=test[3]
    vel=np.column_stack((nom[:,:2],np.sin(nom[:,2]),np.cos(nom[:,2])-1))
    write_json(out/"baselines.json",{"test_windows":len(test[0]),"persistence":metrics(np.zeros_like(test[1]),test),
                                    "constant_velocity":metrics(vel,test)})
    np.savez(out/"normalization_real_fit.npz",mean_x=means[0],std_x=means[1],mean_y=means[2],std_y=means[3])
    common=None
    for cond in CONDITIONS:
        synth=cond[5:] if cond.startswith("Real+C") else None
        if synth:
            source=dataset([(e,root/"generated"/synth/f"episode_{e:06d}"/"states.npz") for e in range(512)],False)
            keys=source[2][:,:3]
            if common is not None and not np.array_equal(keys,common):
                raise ValueError("Paired generated sample IDs differ; freeze a common support before training")
            common=keys
            aux=normalize(source)
            del source
        else:aux=real
        for seed in (42,43,44):
            dst=out/cond/f"seed_{seed}";dst.mkdir(parents=True,exist_ok=True)
            if (dst/"result.json").is_file():
                r=json.loads((dst/"result.json").read_text())
                if r["updates"]!=20000 or r["checkpoint_sha256"]!=file_hash(dst/"selected.pt")[0]:
                    raise ValueError("Probe receipt mismatch")
                continue
            torch.manual_seed(seed)
            model=nn.Sequential(nn.Linear(x.shape[1],128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,4)).to(dev)
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            rg=np.random.default_rng(seed);ag=np.random.default_rng(seed+5000)
            best=float("inf");best_step=0;start=0;history=[]
            resumep=dst/"resume.pt"
            if resumep.exists():
                ck=torch.load(resumep,map_location=dev,weights_only=False)
                model.load_state_dict(ck["model"]);opt.load_state_dict(ck["opt"]);start=ck["step"];best=ck["best"]
                best_step=ck["best_step"];history=ck["history"];rg.bit_generator.state=ck["rg"];ag.bit_generator.state=ck["ag"]
            for step in range(start+1,20001):
                i=rg.integers(0,len(real[0]),64)
                loss=torch.nn.functional.mse_loss(model(real[0][i]),real[1][i])
                if cond!="RealOnly":
                    j=ag.integers(0,len(aux[0]),64)
                    loss=loss+torch.nn.functional.mse_loss(model(aux[0][j]),aux[1][j])
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if step%1000==0:
                    pred=eval_pred(model,sets["calibration"][0],dev,means)
                    value=float(np.square((pred-sets["calibration"][1])/means[3]).mean())
                    history.append([step,value])
                    if value<best:
                        best_step,best=step,value
                        torch.save(model.state_dict(),dst/"selected.pending.pt");os.replace(dst/"selected.pending.pt",dst/"selected.pt")
                    torch.save({"step":step,"model":model.state_dict(),"opt":opt.state_dict(),"rg":rg.bit_generator.state,"ag":ag.bit_generator.state,
                                "best":best,"best_step":best_step,"history":history},dst/"resume.pending.pt")
                    os.replace(dst/"resume.pending.pt",resumep)
            model.load_state_dict(torch.load(dst/"selected.pt",map_location=dev,weights_only=True))
            pred=eval_pred(model,test[0],dev,means)
            write_json(dst/"result.json",{"condition":cond,"seed":seed,"updates":20000,"shared_real_examples":1280000,
                "aux_examples":(0 if cond=="RealOnly" else 1280000),"selected_on_real_calibration_step":best_step,
                "calibration_curve":history,"test":metrics(pred,test),"real_fit_windows":len(real[0]),"test_windows":len(test[0]),
                "checkpoint_sha256":file_hash(dst/"selected.pt")[0]})
            print(f"{cond} seed {seed}: 20000 updates; selected {best_step}",flush=True)
    paths=sorted(out.glob("*/seed_*/result.json"))
    if len(paths)!=15:raise RuntimeError("Incomplete probe matrix")
    write_json(out/"manifest.json",{"results":[{"path":str(p),"sha256":file_hash(p)[0]} for p in paths],"training_updates":300000})


if __name__=="__main__":
    main()
