"""Continuous B2 training with exact cumulative checkpoints and in-process full frozen validation."""
from contextlib import ExitStack
import copy
import inspect
import json
import os
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch
import yaml
from dn_splatter.dn_datamanager import DNSplatterDataManager
from dn_splatter.dn_model import DNSplatterModel as Model
from nerfstudio.engine.trainer import Trainer
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.scripts.train import _set_random_seed

import capture_full_train as legacy
from capture_quality_eval import evaluate_fixed, NODES, select
from capture_quality_paths import B1, GEOM, P, ROOT, RUN, SCENE, STATE, check, read, record, sha, write
from vr_room_checkpoint import numpy_scalar_scope

TOTAL=P["training"]["total_updates"]
AUDIT=STATE/"trainer"
GROWTH_FIELDS=("xys_grad_norm","vis_counts","max_2Dsize")


def rng_state():
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy=[state[0],state[1].tolist(),*state[2:]],
                torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state["python"])
    kind,keys,*tail=state["numpy"]
    np.random.set_state((kind,np.asarray(keys,dtype=np.uint32),*tail))
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def config():
    path=Path(read(B1/"training_complete.json")["config"])
    check(sha(path)==sha(ROOT/P["baseline"]["configuration"]),"frozen B1 config")
    before=yaml.load(path.read_text(),Loader=yaml.Loader)
    cfg=copy.deepcopy(before)
    cfg.output_dir=RUN/"training"
    cfg.experiment_name="capture-full-b2"; cfg.timestamp="quality-v2"
    cfg.pipeline.datamanager.dataparser.data=SCENE
    cfg.steps_per_save=10000; cfg.max_num_iterations=TOTAL
    cfg.optimizers["means"]["scheduler"].max_steps=TOTAL
    m=cfg.pipeline.model
    for key in ("cull_alpha_thresh","warmup_length","refine_every","reset_alpha_every","stop_split_at","continue_cull_post_densification"):
        setattr(m,key,P["training"][key])
    check(cfg.machine.seed==42 and m.cull_alpha_thresh*2==.01 and m.camera_optimizer.mode=="off"
          and m.densify_grad_thresh==.0008 and m.stop_screen_size_at==4000,"fixed method")
    import difflib
    (AUDIT/"configuration.diff").write_text("".join(difflib.unified_diff(yaml.dump(before).splitlines(True),
             yaml.dump(cfg).splitlines(True),fromfile="B1",tofile="B2")))
    cfg.save_config()
    return cfg


def cap_added(model,sensor_count):
    scales=model.gauss_params["scales"]
    n=len(scales)
    check(n==read(SCENE/"geometry.json")["init_points"],"seed order")
    with torch.no_grad():
        tail=scales[sensor_count:]
        original=tail[:,0].exp().clone()
        delta=(np.log(.25)-tail[:,0:1]).clamp(max=0)
        tail.add_(delta)
        def quantiles(a):
            return np.percentile(a,[0,10,50,90,95,100]).tolist() if len(a) else None
        evidence=dict(sensor_count=sensor_count,added_count=n-sensor_count,cap_m=.25,
                      capped=int((original>.25).sum()),actual_before_m=quantiles(original.cpu().numpy()),
                      actual_after_m=quantiles(tail[:,0].exp().cpu().numpy()),applied_once=True)
        if len(tail):
            with np.load(GEOM/"added_tracks.npz") as a:
                ids=a["id"]; xyz=a["xyz_m"]
            with np.load(GEOM/"tracks.npz") as t:
                obs=t["obs"]; pi=np.searchsorted(t["point_ids"],ids)
            _,first=np.unique(obs[:,1].astype(int),return_index=True)
            seen=obs[first,1].astype(int); fids=obs[first[np.searchsorted(seen,pi)],0].astype(int)
            meta=read(SCENE/"transforms.json")
            frames={int(Path(f["file_path"]).stem[-5:]):np.asarray(f["transform_matrix"]) for f in meta["frames"]}
            Z=np.empty(len(xyz))
            for fid in np.unique(fids):
                g=fids==fid; T=frames[fid]
                Z[g]=-(xyz[g]-T[:3,3])@T[:3,2]
            evidence["first_train_view_tangential_radius_fx_over_Z_px"]=quantiles(meta["fl_x"]*tail[:,0].exp().cpu().numpy()/Z)
        write(AUDIT/"seed_scale.json",evidence)


def main():
    AUDIT.mkdir(parents=True,exist_ok=True)
    cfg=config(); canonical=cfg.get_base_dir()/"config.yml"; directory=cfg.get_checkpoint_dir()
    meta=read(SCENE/"transforms.json"); names=meta["train_filenames"]
    check(len(names)==1478 and cfg.pipeline.datamanager.dataparser.scene_scale==21,"input identities")
    manifest=sha(SCENE/"training_manifest.json")
    sensor=read(SCENE/"geometry.json")["sensor_points"]
    perm=random.Random(42)
    sampler=dict(epoch=0,order=[])
    def reset_order(dm):
        sampler["order"]=list(range(len(names))); perm.shuffle(sampler["order"])
        dm.train_unseen_cameras=list(sampler["order"])
    checkpoints=sorted(directory.glob("step-*.ckpt"))
    latest=checkpoints[-1] if checkpoints else None
    saved=int(latest.stem.split("-")[1]) if latest else -1
    check(saved<TOTAL and (saved+1)%10000==0,"B2 resume step")
    for node in NODES:
        if node<=saved and not (RUN/"evaluation/val"/f"step-{node:09d}.json").exists():
            from capture_quality_eval import evaluate
            evaluate(canonical,node,"val")
    if saved==TOTAL-1:
        select(canonical)
        return
    cfg.max_num_iterations=TOTAL-(saved+1)
    population=Model.populate_modules
    def populate(model):
        population(model)
        if not latest: cap_added(model,sensor)
    next_train=DNSplatterDataManager.next_train
    def sample(dm,step):
        if not sampler["order"]: reset_order(dm)
        last=len(dm.train_unseen_cameras)==1
        pair=next_train(dm,step)
        if last:
            sampler["epoch"]+=1
            reset_order(dm)
        return pair
    recovered={}
    def restore(trainer):
        if latest is None: return
        with numpy_scalar_scope():
            state=torch.load(latest,map_location="cpu",weights_only=True,mmap=True)
        r=state["b2_recovery"]
        check(r["manifest"]==manifest and state["step"]==saved,"checkpoint and input")
        trainer.pipeline.load_pipeline(state["pipeline"],saved)
        trainer.optimizers=trainer.setup_optimizers()  # Gaussian parameters are resized by load_pipeline.
        trainer.optimizers.load_optimizers(state["optimizers"]); trainer.optimizers.load_schedulers(state["schedulers"])
        trainer.grad_scaler.load_state_dict(state["scalers"]); trainer._start_step=saved+1
        dm=trainer.pipeline.datamanager
        sampler.update(r["sampler"]); perm.setstate(r["perm"])
        dm.train_unseen_cameras=r["remaining"]
        for k in GROWTH_FIELDS:
            v=r["growth"][k]; setattr(trainer.pipeline.model,k,v.to(trainer.device) if v is not None else None)
        recovered["rng"]=r["rng"]
        if r.get("eval_remaining") is not None: dm.eval_unseen_cameras=r["eval_remaining"]
    diff=[]; before="> self.num_train_data + self.config.refine_every"
    refine_impl=legacy.replacement(Model.refinement_after,before,"> 3 * self.num_train_data + self.config.refine_every",diff)
    (AUDIT/"refinement.patch").write_text("".join(diff))
    event={}
    def refine(model,opts,step):
        eligible=(step>4000 and step<120000 and step%24000>3*1478+800)
        event.clear(); event.update(step=step,before=int(model.num_points),eligible=eligible,split_children=0,duplicates=0,cull_union=0)
        ans=refine_impl(model,opts,step)
        event.update(after=int(model.num_points),opacity_reset=4000<step<120000 and step%24000==800)
        check(("masks" in event)==eligible and event["after"]==event["before"]+event["split_children"]+event["duplicates"]-event["cull_union"],"refinement accounting")
        print(json.dumps(event),file=events)
        if step==5600: write(AUDIT/"first_refinement.json",dict(event))
        return ans
    cull_method=Model.cull_gaussians
    def cull(model,extra_cull_mask=None):
        opacity=(model.opacities.sigmoid()<model.config.cull_alpha_thresh).reshape(-1)
        empty=torch.zeros_like(opacity)
        world=empty; screen=empty
        if model.step>model.config.refine_every*model.config.reset_alpha_every:
            world=model.scales.exp().max(dim=-1).values>model.config.cull_scale_thresh
            if model.step<model.config.stop_screen_size_at and model.max_2Dsize is not None:
                screen=model.max_2Dsize.reshape(-1)>model.config.cull_screen_size
        masks={"opacity":opacity,"world":world,"screen":screen,"parent":extra_cull_mask.reshape(-1) if extra_cull_mask is not None else empty}
        actual=cull_method(model,extra_cull_mask).reshape(-1)
        check(torch.equal(actual,opacity|world|screen|masks["parent"]),"actual cull union")
        event["masks"]={k:int(m.sum()) for k,m in masks.items()}
        signature=sum(mask.to(torch.uint8)*(1<<i) for i,mask in enumerate(masks.values()))
        event["exclusive_overlap_bits_opacity_world_screen_parent"]=torch.bincount(signature.long(),minlength=16).tolist()
        event["cull_union"]=int(actual.sum())
        return actual
    def observe_growth(fn,field):
        def wrapped(model,*args):
            result=fn(model,*args); event[field]=len(result["means"])
            return result
        return wrapped
    def save(trainer,step):
        if (step+1)%10000: return
        path=directory/f"step-{step:09d}.ckpt"
        if path.exists(): return
        directory.mkdir(exist_ok=True,parents=True)
        dm=trainer.pipeline.datamanager; model=trainer.pipeline.model
        state=dict(step=step,pipeline=trainer.pipeline.state_dict(),optimizers={k:v.state_dict() for k,v in trainer.optimizers.optimizers.items()},
                   schedulers={k:v.state_dict() for k,v in trainer.optimizers.schedulers.items()},scalers=trainer.grad_scaler.state_dict(),
                   b2_recovery=dict(manifest=manifest,sampler=dict(sampler),remaining=list(dm.train_unseen_cameras),perm=perm.getstate(),
                                    rng=rng_state(),eval_remaining=getattr(dm,"eval_unseen_cameras",None),
                                    growth={k:getattr(model,k,None) for k in GROWTH_FIELDS}))
        tmp=path.with_suffix(".pending"); torch.save(state,tmp); os.replace(tmp,path)
        write(STATE/"latest_checkpoint.json",dict(step=step,updates=step+1,gaussians=int(model.num_points),**record(path)))
        # Retain all validation nodes and one latest intervening recovery point.
        for prev in directory.glob("step-*.ckpt"):
            num=int(prev.stem[5:])+1
            if num<step+1 and num%30000 and num%10000==0: prev.unlink()
    evaluation=Trainer.eval_iteration
    def evaluate(trainer,step):
        evaluation(trainer,step)
        if (step+1)%10000: return
        save(trainer,step)
        if step not in NODES: return
        state=rng_state(); current=trainer.pipeline.training
        try:
            trainer.pipeline.eval()
            evaluate_fixed(canonical,step,"val",trainer.pipeline,directory/f"step-{step:09d}.ckpt")
        finally:
            trainer.pipeline.train(current); restore_rng(state)
    first={}
    original_loss=Model.get_loss_dict
    def loss(model,outputs,batch,*args,**kwargs):
        result=original_loss(model,outputs,batch,*args,**kwargs)
        if model.training:
            idx=int(batch["image_idx"]); step=int(model.step)
            print(f"{step},{idx},{Path(names[idx]).stem},{int(model.num_points)}",file=samples)
            if step%100==0:
                write(STATE/"training_progress.json",dict(updates=step,step_in_progress=step,frame=names[idx],points=int(model.num_points)))
            if not first:
                first.update(step=step,frame=names[idx],loss={k:float(v.detach()) for k,v in result.items()},
                             normal_nonfinite=int((~outputs["normal"].isfinite()).sum()))
                check(all(np.isfinite(list(first["loss"].values()))) and not first["normal_nonfinite"],"first formal batch")
        return result
    opt_step=Optimizers.optimizer_scaler_step_some
    def grad(opt,*args,**kwargs):
        if "grad_elements" not in first:
            grads=[p.grad for group in opt.parameters.values() for p in group if p.grad is not None]
            first.update(grad_elements=sum(g.numel() for g in grads),nonfinite_grad=sum(int((~g.isfinite()).sum()) for g in grads))
            write(AUDIT/f"first_batch_{saved+1:09d}.json",first)
            check(first["grad_elements"] and first["nonfinite_grad"]==0,"first formal gradient")
        return opt_step(opt,*args,**kwargs)
    def train():
        _set_random_seed(cfg.machine.seed)
        t=cfg.setup(local_rank=0,world_size=1)
        t.setup()
        if recovered:
            # Materialize lazy caches before restoring the exact next-update RNG.
            _=t.pipeline.datamanager.cached_train,t.pipeline.datamanager.cached_eval
            restore_rng(recovered.pop("rng"))
        write(AUDIT/"segment.json",dict(saved_step=saved,remaining=cfg.max_num_iterations,canonical_steps=TOTAL))
        t.train()
    with ExitStack() as scope:
        events=scope.enter_context((AUDIT/"refinement.jsonl").open("a",buffering=1))
        samples=scope.enter_context((AUDIT/f"samples_{saved+1:09d}.csv").open("a",buffering=1))
        for obj,attr,fn in ((Model,"populate_modules",populate),(DNSplatterDataManager,"next_train",sample),
                 (Trainer,"_load_checkpoint",restore),(Model,"refinement_after",refine),(Model,"cull_gaussians",cull),
                 (Model,"dup_gaussians",observe_growth(Model.dup_gaussians,"duplicates")),
                 (Model,"split_gaussians",observe_growth(Model.split_gaussians,"split_children")),
                 (Trainer,"save_checkpoint",save),(Trainer,"eval_iteration",evaluate),(Model,"get_loss_dict",loss),
                 (Optimizers,"optimizer_scaler_step_some",grad),(legacy,"config",lambda:cfg),(legacy,"train",train),(legacy,"REPORT",AUDIT)):
            scope.enter_context(patch.object(obj,attr,fn))
        write(AUDIT/"loaded_sources.json",{k:dict(path=str(Path(inspect.getfile(obj))),sha256=sha(Path(inspect.getfile(obj))))
              for k,obj in (("model",Model),("datamanager",DNSplatterDataManager),("trainer",Trainer),("optimizers",Optimizers))})
        legacy.main(recovery_epochs=3)
    final=read(STATE/"latest_checkpoint.json")
    check(final["updates"]==TOTAL,"formal 240k completion")
    all_events={e["step"]:e for e in map(json.loads,(AUDIT/"refinement.jsonl").read_text().splitlines())}
    check(sum(e["eligible"] for e in all_events.values())==115,"complete refinement events")
    write(RUN/"training_complete.json",dict(final,config=str(canonical)))
    select(canonical)


if __name__=="__main__":
    main()
