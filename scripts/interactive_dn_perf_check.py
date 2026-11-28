"""Compare native and scheduled diagnostic calls on actual frozen checkpoint renders; no updates."""
import argparse
from copy import deepcopy
import hashlib
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from interactive_dn_cache import MMapViews, atomic_json, shared_seed_geometry, shared_views
from interactive_dn_supervision import FrozenHoleTerms, Packets, digest, loss_at_pixels
from interactive_dn_metrics import acceleration_policy,implementation,metrics_at_writer_steps
from interactive_dn_train import ROOT, RUN_ID


def check_policy():
    from datetime import datetime
    run=ROOT/".local/runs"/RUN_ID
    out=run/"f_evaluation/acceleration"
    info=json.loads((run/"c_dn/R0C/state/inputs.json").read_bytes())
    info["supervision_sha256"]=digest(run/"a_depth/supervision/R3/manifest.json")
    verified={v:acceleration_policy(run,v,info) for v in ("R3","R4","R5")}
    if not all(x["enabled"] for v in verified.values() for x in v.values()):
        print(json.dumps(verified),flush=True)
        raise ValueError("current runtime does not match every expected numerical verification")
    mismatch=acceleration_policy(run,"R3",dict(info,historical_config_sha256="different"))
    if any(x["enabled"] for x in mismatch.values()):
        raise ValueError("unverified configuration selected an optimized path")
    baseline_start={}
    for v in ("R0C","R1","R2"):
        f=run/"c_dn"/v/"state/first_batch_000000000.json"
        baseline_start[v]=datetime.fromtimestamp(f.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    doc=dict(future_segment_policy=verified,mismatched_config_uses_native=True,first_formal_batch=baseline_start,
             source_files={n:digest(ROOT/"scripts"/n) for n in ("interactive_dn_train.py","interactive_dn_metrics.py",
             "interactive_dn_cache.py","interactive_dn_supervision.py","interactive_dn_recovery.py")})
    atomic_json(out/"future_segment_policy.json",doc)
    trainer=doc["source_files"]["interactive_dn_train.py"]
    if not all(x["enabled"] for x in acceleration_policy(run,"R3",info,integration_sha=trainer).values()):
        raise ValueError("verified integration did not enable the three expected paths")
    if any(x["enabled"] for x in acceleration_policy(run,"R3",info,integration_sha="changed").values()):
        raise ValueError("modified integration did not fall back to native")
    print(json.dumps({v:{k:x["enabled"] for k,x in rows.items()} for v,rows in verified.items()}),flush=True)


def publish(root,kind,doc):
    out=root/"f_evaluation/acceleration"
    if not (out/(kind+".json")).exists():
        atomic_json(out/(kind+".json"),doc)
    doc=dict(doc,verified=True,implementation=implementation(kind))
    atomic_json(out/(kind+"_verified.json"),doc)


def check_images():
    root=ROOT/".local/runs"/RUN_ID
    cache=next(x for x in (root/"shared/image-cache").glob("*/*") if len(list(x.glob("*.json")))==1478)
    class Dataset:
        cameras=SimpleNamespace(distortion_params=None)
        def __len__(self):
            return 1478
    cpu=MMapViews(Dataset(),"float32",cache)
    gpu=MMapViews(Dataset(),"float32",cache,device="cuda")
    ids=np.random.default_rng(42).choice(1478,32,replace=False).tolist()
    fields={}
    elapsed=[[],[]]
    try:
        for n,i in enumerate(ids):
            batches=[None,None]
            # Alternate ordering on the identical view to avoid favouring the second page-cache read.
            for j in (n%2,1-n%2):
                reader=(cpu,gpu)[j]
                torch.cuda.synchronize()
                start=time.perf_counter()
                batch=deepcopy(reader[i])
                for k,x in batch.items():
                    if torch.is_tensor(x):
                        batch[k]=x.to("cuda")
                torch.cuda.synchronize()
                elapsed[j].append(time.perf_counter()-start)
                batches[j]=batch
            # Also exercise a real executor-to-consumer CUDA event handoff.
            gpu.prefetch(i)
            ahead=gpu[i]
            for k,x in batches[0].items():
                if torch.is_tensor(x):
                    if not torch.equal(x,batches[1][k]) or not torch.equal(x,ahead[k]):
                        raise ValueError(f"frame {i} field {k} changed through compressed transfer")
                    fields[k]=fields.get(k,0)+x.numel()
                elif x!=ahead[k]:
                    raise ValueError("literal sample data changed")
    finally:
        cpu.close()
        gpu.close()
    doc=dict(train_frame_indices=ids,equal_tensor_values=fields,async_handoffs=gpu.hits,
             native_mean_ms=1000*float(np.mean(elapsed[0][2:])),gpu_decode_mean_ms=1000*float(np.mean(elapsed[1][2:])),
             native_median_ms=1000*float(np.median(elapsed[0][2:])),gpu_decode_median_ms=1000*float(np.median(elapsed[1][2:])),
             timing_scope="on-demand view read, native deepcopy and H2D, excluding the first two calls; not complete training steps")
    publish(root,"image_transfer",doc)
    print(json.dumps(doc),flush=True)


def check_holes():
    run=ROOT/".local/runs"/RUN_ID
    scene=run/"restore/tree/.local/runs/capture-full-quality-v3-20261122-v1/inputs"
    ids=json.loads((scene/"transforms.json").read_bytes())["train_filenames"]
    packets=Packets(run/"a_depth/supervision/R3/manifest.json",ids,(720,1280),run/"shared/packet-cache")
    fixed=FrozenHoleTerms(packets,1.,(.3,6.))
    hist=ROOT/".local/runs/lingbot-depth-map-20261128-v1/depth/unmasked"
    from PIL import Image
    eligible=[i for i,n in enumerate(ids) if (hist/Path(n).stem/"depth_refined_m.npy").is_file()]
    selected=np.random.default_rng(42).choice(eligible,min(12,len(eligible)),replace=False).tolist()
    if len(selected)<4:
        raise ValueError("Insufficient real frozen depth predictions")
    timing=[[],[]]; total=0; mx=0.; gradients=0; extremes=0
    for i in selected:
        fid=Path(ids[i]).stem
        rgb=torch.from_numpy(np.asarray(Image.open(scene/ids[i]).convert("RGB")).copy()).cuda().float().div_(255).clamp_min_(10/255)
        sensor=torch.from_numpy(np.load(scene/"depth"/(fid+".npy"))).cuda().reshape(720,1280,1)
        values=torch.from_numpy(np.load(hist/fid/"depth_refined_m.npy")).cuda().reshape_as(sensor)
        pts=packets.get(i);index=torch.as_tensor(np.array(pts["index"],copy=True),device="cuda")
        z=torch.as_tensor(np.array(pts["z_m"],copy=True),device="cuda")
        total+=len(pts)
        for mode in ("positive","raw","nonfinite"):
            pred=values.detach().clone()
            if mode=="positive":
                pred[~torch.isfinite(pred)|(pred<=0)]=1.
            elif mode=="nonfinite" and len(index):
                pred.view(-1)[index[::3]]=float("inf")
                pred.view(-1)[index[1::3]]=float("nan")
                extremes+=len(index[::3])+len(index[1::3])
            pred.requires_grad_()
            def original():
                l,n=loss_at_pixels(pred,rgb,sensor,index,z,1.,(.3,6.))
                return l,n
            def cached():
                l,seen,n=fixed.loss(i,pred,rgb,sensor)
                if seen!=len(pts):raise ValueError("raw packet count changed")
                return l,n
            a,na=original();b,nb=cached()
            ga=torch.autograd.grad(a,pred)[0];gb=torch.autograd.grad(b,pred)[0]
            if na!=nb or not torch.equal(a,b) or not torch.equal(ga,gb):
                raise ValueError(f"R3 cached operands changed difference/count/gradient: {fid} {mode}")
            gradients+=ga.numel()
            mx=max(mx,float((ga-gb).abs().max()))
            if mode=="positive":
                for rep in range(8):
                    for j in (rep%2,1-rep%2):
                        torch.cuda.synchronize();start=time.perf_counter()
                        l,_=(original,cached)[j]();torch.autograd.grad(l,pred)
                        torch.cuda.synchronize();timing[j].append(time.perf_counter()-start)
    doc=dict(frames=[ids[i] for i in selected],real_R3_packet_pixels=total,compared_gradient_elements=gradients,
             supervision_sha256=digest(packets.manifest),
             deliberate_nonfinite_positions=extremes,scalar_and_gradient_bit_exact=True,gradient_max_abs=mx,
             cached_bytes=fixed.cached_bytes,maximum_budget_bytes=fixed.budget,calls=fixed.calls,fallbacks=fixed.fallbacks,
             native_ms=1000*float(np.median(timing[0])),cached_ms=1000*float(np.median(timing[1])),
             timing_scope="warm sparse hole loss and its full prediction gradient, identical positive version of frozen LingBot estimates")
    publish(run,"hole_terms",doc)
    print(json.dumps(doc),flush=True)


def check_boundary():
    from vr_room_checkpoint import numpy_scalar_scope
    def pixel_rows(table,pix,sparse):
        table[sparse]=torch.arange(sparse.numel(),device=pix.device,dtype=table.dtype)
        where=table[pix]
        table[sparse]=-1
        return where>=0,where
    run=ROOT/".local/runs"/RUN_ID
    meta=json.loads((run/"restore/tree/.local/runs/capture-full-quality-v3-20261122-v1/inputs/transforms.json").read_bytes())
    ids=meta["train_filenames"];frames={f["file_path"]:f for f in meta["frames"]}
    packets=Packets(run/"a_depth/boundary/manifest.json",ids,(720,1280),run/"shared/packet-cache",boundary=True)
    ck=next((run/"c_dn/R0C/training").rglob("step-000119999.ckpt"))
    with numpy_scalar_scope():
        weights=torch.load(ck,map_location="cpu",weights_only=True,mmap=True)
    xyz=weights["pipeline"]["_model.gauss_params.means"].cuda()
    choose=np.random.default_rng(42).choice([i for i,r in enumerate(packets.rows) if r["accepted_pixels"]],16,replace=False)
    table=torch.full((720*1280,),-1,dtype=torch.int32,device="cuda")
    pixels=torch.arange(720*1280,device="cuda",dtype=torch.long)
    timings=[[],[]];projected=0;matched=0;exhaustive=0
    for i in choose:
        rec=packets.get(i)
        if not len(rec):continue
        sp=torch.tensor(np.array(rec["index"]),device="cuda").long()
        # Include every possible camera pixel, then time the real projected Gaussian population.
        cam=torch.tensor(frames[ids[i]]["transform_matrix"],device="cuda",dtype=xyz.dtype)
        view=(xyz-cam[:3,3])@cam[:3,:3]
        z=-view[:,2]
        x=(meta["fl_x"]*view[:,0]/z+meta["cx"]).floor().long()
        y=(-meta["fl_y"]*view[:,1]/z+meta["cy"]).floor().long()
        ok=(z>0)&(x>=0)&(x<1280)&(y>=0)&(y<720)
        pix=y[ok]*1280+x[ok];projected+=pix.numel()
        def native(query):
            row=torch.searchsorted(sp,query).clamp_max(len(sp)-1)
            return sp[row]==query,row
        for query in (pixels,pix):
            ma,wa=native(query);mb,wb=pixel_rows(table,query,sp)
            if not torch.equal(ma,mb) or not torch.equal(wa[ma],wb[mb].long()):
                raise ValueError("boundary record identity differs at a native pixel")
            if query is pixels:
                exhaustive+=query.numel()
            else:
                matched+=int(ma.sum())
        for rep in range(10):
            for j in (rep%2,1-rep%2):
                torch.cuda.synchronize();start=time.perf_counter()
                if j==0: native(pix)
                else: pixel_rows(table,pix,sp)
                torch.cuda.synchronize();timings[j].append(time.perf_counter()-start)
    doc=dict(boundary_sha256=digest(packets.manifest),complete_pixel_comparisons=exhaustive,
             projected_gaussians=projected,matched_projected_ids=matched,identical_pixel_and_record_ids=True,
             workspace_bytes=table.numel()*table.element_size(),
             native_median_ms=float(np.median(timings[0]))*1000,lookup_median_ms=float(np.median(timings[1]))*1000,
             scope="identical ordered IDs preserve both depth modes, tangent and evidence; matching subroutine only")
    atomic_json(run/"f_evaluation/acceleration/boundary_timing.json",doc)
    print(json.dumps(doc),flush=True)


def run():
    import capture_quality_eval as ev
    from capture_surface_train import weighted_edge_aware_log_l1
    from dn_splatter.dn_datamanager import DNSplatterDataManager as Manager
    from dn_splatter.dn_model import DNSplatterModel as Model
    from dn_splatter.losses import EdgeAwareLogL1
    run_root=ROOT/".local/runs"/RUN_ID
    out=run_root/"f_evaluation/acceleration"
    ev.RUN=run_root/"c_dn/R0C"
    ev.SCENE=run_root/"restore/tree/.local/runs/capture-full-quality-v3-20261122-v1/inputs"
    meta=json.loads((ev.SCENE/"transforms.json").read_bytes())
    config=next((ev.RUN/"training").glob("*/dn-splatter/*/config.yml"))
    init=run_root/"shared/initial-geometry"
    packet=Packets(run_root/"a_depth/supervision/R3/manifest.json",meta["train_filenames"],
                   (meta["h"],meta["w"]),run_root/"shared/packet-cache")
    cache=run_root/"shared/image-cache"
    view_cache=next(path for path in cache.iterdir() if any(len(list(x.glob("*.json")))==1478 for x in path.iterdir()))
    with ExitStack() as scope:
        scope.enter_context(shared_seed_geometry(Model,init))
        scope.enter_context(shared_views(Manager,view_cache))
        cfg,pipe,ck,_=ev.load_model(config,119999,"val")
        model=pipe.model
        dataset=pipe.datamanager.train_dataset
        image_ids=[17,317]
        samples=[]
        for i in image_ids:
            batch={k:(v.to(pipe.device) if torch.is_tensor(v) else v)
                   for k,v in dataset.get_data(i).items()}
            with torch.no_grad():
                outputs=model.get_outputs_for_camera(camera=dataset.cameras[i:i+1].to(pipe.device))
            fid=Path(meta["train_filenames"][i]).stem
            weights=ev.SCENE.parent/"weights/B2"/(fid+".npy")
            wm=torch.as_tensor(np.load(weights),device=pipe.device).reshape_as(outputs["depth"])
            points=packet.get(i)
            idx=torch.as_tensor(np.array(points["index"],copy=True),device=pipe.device)
            z=torch.as_tensor(np.array(points["z_m"],copy=True),device=pipe.device)
            samples.append((batch,outputs,wm,idx,z))
        model.train()
        def scalar(v):
            return float(v.detach()) if torch.is_tensor(v) else float(v)

        def benchmark(fast,steps=20,collect_graph=False):
            cpu_rng=torch.random.get_rng_state()
            gpu_rng=torch.cuda.get_rng_state()
            log={}
            grads=[]
            losses=[]
            elapsed=[]
            fingerprints=[]
            # All branches feed the same renders through DN's real loss and R3
            # auxiliary supervision. The trainer and optimizer are never called.
            with (metrics_at_writer_steps(Model,cfg.logging.steps_per_log) if fast else nullcontext()) as n:
                for j in range(steps):
                    step=130000+j
                    model.step=step
                    batch,render,w,idx,z=samples[j%len(samples)]
                    inp=dict(batch)
                    out=dict(render)
                    for k in ("rgb","normal","depth"):
                        out[k]=render[k].detach().requires_grad_()
                    torch.cuda.synchronize()
                    t=time.perf_counter()
                    metrics=model.get_metrics_dict(out,inp)
                    torch.cuda.synchronize()
                    elapsed.append(time.perf_counter()-t)
                    graph=hashlib.sha256()
                    def pack(t):
                        graph.update(str((t.dtype,tuple(t.shape),tuple(t.stride()))).encode())
                        raw=t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
                        graph.update(memoryview(raw).cast("B"))
                        return t
                    hooks=torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t) if collect_graph else nullcontext()
                    with hooks,patch.object(EdgeAwareLogL1,"forward",lambda _,p,g,i,m:weighted_edge_aware_log_l1(p,g,i,m,w)):
                        loss=model.get_loss_dict(out,inp,metrics)
                        aux,count=loss_at_pixels(out["depth"],model.get_gt_img(inp["image"]).clamp(min=10/255),
                                                inp["sensor_depth"],idx,z,1.,(.3,6.))
                    if collect_graph:
                        fingerprints.append(graph.hexdigest())
                    loss["hole_depth"]=.1*model.config.sensor_depth_lambda*aux
                    losses.append({k:scalar(v) for k,v in loss.items()})
                    if j in (0,1,10):
                        g=torch.autograd.grad(sum(loss.values()),[out[k] for k in ("rgb","normal","depth")])
                        grads.append([v.detach().cpu().contiguous() for v in g])
                    if step%cfg.logging.steps_per_log==0:
                        log[step]={k:scalar(v) for k,v in metrics.items()}
                model.eval()
                batch,render,*_=samples[0]
                val={k:scalar(v) for k,v in model.get_metrics_dict(render,batch).items()}
                model.train()
            if not (torch.equal(cpu_rng,torch.random.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())):
                raise ValueError("unreported metric path consumed randomness")
            return losses,grads,log,val,elapsed[2:],n,fingerprints
        old=benchmark(False)
        new=benchmark(True)
        def different(a,b):
            return {k:[v,b[k]] for k,v in a.items() if v!=b[k] and not(np.isnan(v) and np.isnan(b[k]))}
        def compare_grads(a,b):
            rows=[]
            for aa,bb in zip(a,b):
                rows.append([dict(exact=torch.equal(x,y),max_abs=float((x-y).abs().max()),
                                  rms_relative=float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x).clamp_min(1e-30)))
                             for x,y in zip(aa,bb)])
            return rows
        raw_grads=compare_grads(old[1],new[1])
        # Independently measure native repeat sensitivity on retained checkpoints.
        native_replay=benchmark(False,2,collect_graph=True)
        new_replay=benchmark(True,2,collect_graph=True)
        native_spread=compare_grads(old[1][:2],native_replay[1])
        # The production RGB SSIM GPU backward has native self-noise. For a
        # held input, require the exact same saved backward operands as well.
        graphs_match=native_replay[6]==new_replay[6]
        failures={"losses":[different(a,b) for a,b in zip(old[0],new[0]) if different(a,b)],
                  "identical_backward_inputs":graphs_match,
                  "train_logs":{k:different(v,new[2][k]) for k,v in old[2].items() if different(v,new[2][k])},
                  "evaluation":different(old[3],new[3])}
        # Report native spread and enforce a numerical bound at the scale of the
        # native RGB kernel. This check is in addition to bitwise graph inputs,
        # loss scalars, emitted metrics, and depth and normal gradients.
        native_max=max(row[0]["max_abs"] for row in native_spread)
        native_rms=max(row[0]["rms_relative"] for row in native_spread)
        default_ok=all(row[1]["exact"] and row[2]["exact"] and
                       (row[0]["exact"] or (native_max and row[0]["max_abs"]<=2*native_max and
                        row[0]["rms_relative"]<=2*native_rms)) for row in raw_grads)
        if failures["losses"] or failures["train_logs"] or failures["evaluation"] or not default_ok or not graphs_match:
            print(json.dumps(failures),flush=True)
            print(json.dumps({"default":raw_grads,"native_native":native_spread}),flush=True)
            raise ValueError("native loss, rendered-output gradients or emitted diagnostics differ")
        doc=dict(checkpoint=str(ck),checkpoint_sha256=digest(ck),frame_ids=[meta["train_filenames"][i] for i in image_ids],
                 full_resolution=[meta["h"],meta["w"]],reduction_steps=20,matched_loss_components=list(old[0][0]),
                 identical_gradient_comparisons=sum(v["exact"] for row in raw_grads for v in row),production_gradients=raw_grads,
                 native_native_gradient_spread=native_spread,backward_operand_sha256=native_replay[6],
                 identical_emitted_train_metrics=old[2],
                 identical_evaluation_metrics=old[3],cpu_and_cuda_rng_unchanged=True,calls=new[5],
                 mean_native_metric_ms=1e3*float(np.mean(old[4])),mean_scheduled_metric_ms=1e3*float(np.mean(new[4])),
                 timing_scope="metric dispatch only with live GPU competition; first two calls excluded; no optimizer update")
        publish(run_root,"metric_dispatch",doc)
        print(json.dumps({k:doc[k] for k in ("identical_gradient_comparisons","calls","mean_native_metric_ms","mean_scheduled_metric_ms")}),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--images-only",action="store_true")
    cli.add_argument("--holes-only",action="store_true")
    cli.add_argument("--boundary-only",action="store_true")
    cli.add_argument("--check-policy",action="store_true")
    args=cli.parse_args()
    torch.set_num_threads(1)
    if args.images_only:
        check_images()
    elif args.holes_only:
        check_holes()
    elif args.boundary_only:
        check_boundary()
    elif args.check_policy:
        check_policy()
    else:
        run()
