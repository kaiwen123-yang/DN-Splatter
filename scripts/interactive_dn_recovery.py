"""Read back the restored tensors, optimizer clocks and sampler of a formal DN segment."""
import torch


def assert_exact(left,right,name):
    if torch.is_tensor(right):
        if not torch.is_tensor(left) or left.dtype!=right.dtype or left.shape!=right.shape:
            raise ValueError(f"restored tensor layout differs: {name}")
        a=left.detach().contiguous().reshape(-1).view(torch.uint8)
        b=right.detach().contiguous().reshape(-1).view(torch.uint8)
        block=16*1024**2
        for off in range(0,len(b),block):
            if not torch.equal(a[off:off+block].cpu(),b[off:off+block].cpu()):
                raise ValueError(f"restored tensor bytes differ: {name}")
    elif isinstance(right,dict):
        if not isinstance(left,dict) or set(left)!=set(right):
            raise ValueError(f"restored keys differ: {name}")
        for k,v in right.items():
            assert_exact(left[k],v,f"{name}.{k}")
    elif isinstance(right,(tuple,list)):
        if type(left)!=type(right) or len(left)!=len(right):
            raise ValueError(f"restored sequence differs: {name}")
        for i,(a,b) in enumerate(zip(left,right)):
            assert_exact(a,b,f"{name}.{i}")
    elif left!=right:
        raise ValueError(f"restored scalar differs: {name}")


def readback(trainer,ck,sampler,order,epoch,growth,keys,names):
    r=ck["world_recovery"]
    state=trainer.pipeline.state_dict()
    assert_exact(state,ck["pipeline"],"pipeline")
    opts={k:o.state_dict() for k,o in trainer.optimizers.optimizers.items()}
    sched={k:s.state_dict() for k,s in trainer.optimizers.schedulers.items()}
    assert_exact(opts,ck["optimizers"],"optimizers")
    assert_exact(sched,ck["schedulers"],"schedulers")
    assert_exact(trainer.grad_scaler.state_dict(),ck["scalers"],"scalers")
    model=trainer.pipeline.model;dm=trainer.pipeline.datamanager
    assert_exact({k:getattr(model,k) for k in keys},r["growth"],"refinement")
    assert_exact(None if growth is None else growth.snapshot(),r["boundary"],"boundary")
    assert_exact(sampler.getstate(),r["sampler_rng"],"sampler_rng")
    assert_exact(order,r["order"],"shuffled_epoch")
    assert_exact(epoch,r["epoch"],"epoch")
    assert_exact(dm.train_unseen_cameras,r["remaining"],"train_queue")
    assert_exact(dm.eval_unseen_cameras,r["eval_remaining"],"eval_queue")
    active_partial={k:trainer.gradient_accumulation_steps[k] for k in opts if trainer.gradient_accumulation_steps[k]!=1}
    if active_partial or trainer._start_step!=ck["step"]+1 or not r["remaining"]:
        raise ValueError("cannot prove optimizer boundary or next sampler update")
    next_id=r["remaining"][0]
    return dict(saved_step=ck["step"],next_step=trainer._start_step,identity=r["identity"],
                persistent_tensor_count=len(state),
                persistent_numel=sum(t.numel() for t in state.values() if torch.is_tensor(t)),
                optimizer_groups=sorted(opts),scheduler_epochs={k:v.get("last_epoch") for k,v in sched.items()},
                optimizer_lr={k:[g["lr"] for g in v["param_groups"]] for k,v in opts.items()},
                active_gradient_accumulation=active_partial,grad_scaler_exact=True,
                refinement_fields={k:(None if r["growth"][k] is None else list(r["growth"][k].shape)) for k in keys},
                boundary_state_exact=True,sampler_rng_exact=True,epoch=epoch,remaining_images=len(r["remaining"]),
                eval_remaining=len(r["eval_remaining"]),next_image_idx=next_id,next_frame=names[next_id])
