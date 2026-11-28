"""Run the fixed hall and VR comparisons on the project-owned AutoDL instance."""
import argparse
from contextlib import ExitStack
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from interactive_dn_cache import atomic_json, shared_seed_geometry, shared_views
from interactive_dn_growth import Growth, freeze_cap
from interactive_dn_supervision import Packets, digest, loss_at_pixels

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"configs/interactive_world_protocol.json"
RUN_ID = "interactive-world-20261128-v1"
B3_ID = "capture-full-quality-v3-20261122-v1"
B3_REPORT = ROOT/"reports/capture-full-quality-v3/20261122-v1"
VR_REPORT = ROOT/"reports/vr-room-iphone-30k"
HALL_CONFIG = ROOT/"reports/capture-full-quality-v2/20261121-v1/config.yml"
RUNS = ("R0C","R1","R2","R3","R4","R5","R6","R7","VR0C","VR1","VR2")


def read(path):
    return json.loads(path.read_bytes())


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,separators=(",",":")).encode()).hexdigest()


def freeze(path, doc):
    if path.exists():
        if read(path) != doc:
            raise ValueError(f"run input or treatment changed: {path}")
    else:
        atomic_json(path,doc)


def validate_files(rows, cache):
    """Reuse verified hashes only while path, inode, size, mtime and ctime are identical."""
    cache.parent.mkdir(parents=True,exist_ok=True)
    with cache.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        memo = read(cache) if cache.exists() else {}
        for path, expected in rows:
            stat = path.stat()
            record = [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,expected]
            if memo.get(str(path)) == record:
                continue
            if digest(path) != expected:
                raise ValueError(f"source SHA-256 mismatch: {path}")
            memo[str(path)] = record
        atomic_json(cache,memo)


def prepare(args):
    run = args.run_root.resolve()
    if run.name != RUN_ID:
        raise ValueError(f"expected dedicated run directory {RUN_ID}")
    vr = args.variant.startswith("VR")
    dest = run/("f_vr" if vr else "c_dn")/args.variant
    audit = dest/"state"
    audit.mkdir(parents=True,exist_ok=True)
    restored = run/"restore/tree/.local"
    b3 = args.b3_root or restored/"runs"/B3_ID
    positioned = args.variant in ("R6","R7")
    if positioned:
        proof = read(args.pose_receipt or run/"b_map/pose/accepted.json")
        if not proof.get("accepted_frame_ids") or not proof.get("audit_independent") or not proof.get("scene"):
            raise ValueError("pose-specific inputs require a nonzero independently accepted pose and a regenerated scene")
        scene = (run / proof["scene"]).resolve()
        if args.scene is not None and args.scene.resolve() != scene:
            raise ValueError("requested scene differs from the independently verified pose input")
    else:
        scene = (args.scene or (restored/"data/room_datasets/vr_room" if vr else b3/"inputs")).resolve()
    config_file = VR_REPORT/"config.yml" if vr else HALL_CONFIG
    info = {"run_id":RUN_ID, "variant":args.variant,"budget":30000 if vr else 240000,"seed":42,
            "protocol_sha256":digest(PROTOCOL), "historical_config_sha256":digest(config_file),
            "scene":str(scene),"save_every":10000,"validation_every":30000}
    if vr:
        registry = read(VR_REPORT/"frames.json")["frames"]
        if [len(registry[k]) for k in ("train","within","with")] != [376,42,186]:
            raise ValueError("unexpected VR splits")
        names = [r["rgb"] for r in registry["train"]]
        required = set()
        for rows in registry.values():
            for r in rows:
                required.update(r[k] for k in ("rgb","depth","normal"))
        for cap in ("short_capture","long_capture"):
            required.update(f"iphone/{cap}/{name}" for name in ("transformations.json","transformations_colmap.json"))
        required.update(("iphone/long_capture/test.txt","iphone/long_capture/iphone_pointcloud.ply"))
        asset_receipt = run/"shared/vr_source_hashes.json"
        asset_receipt.parent.mkdir(parents=True,exist_ok=True)
        with asset_receipt.with_suffix(".lock").open("a+b") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if asset_receipt.exists():
                source = read(asset_receipt)
                if set(source) != required:
                    raise ValueError("VR source membership changed")
                validate_files(((scene/k,v) for k,v in source.items()),run/"shared/vr_verified.json")
            else:
                source = {k:digest(scene/k) for k in sorted(required)}
                atomic_json(asset_receipt,source)
        info.update(scene_sha256=signature(source),initialization_sha256=source["iphone/long_capture/iphone_pointcloud.ply"],
                    initialization_count=None, depth_unit="m after sensor millimetre conversion", historical_sampling=False,
                    baseline_reason="new common epoch-shuffled sampler and explicit recovery state")
        shape = tuple(registry["train"][0]["rgb_size"][::-1])
        if any(tuple(r["rgb_size"][::-1]) != shape for rows in registry.values() for r in rows):
            raise ValueError("heterogeneous VR image sizes require per-frame packet shapes")
        weights = []
    else:
        historical = {r["path"].split(B3_ID+"/",1)[1]:r for r in read(B3_REPORT/"artifacts.json")["artifacts"] if B3_ID+"/" in r["path"]}
        curve = freeze_cap(b3/"state/trainer/refinement.jsonl",historical["state/trainer/refinement.jsonl"]["sha256"])
        manifest = scene/"training_manifest.json"
        original = historical["inputs/training_manifest.json"]["sha256"]
        if positioned:
            if proof["training_manifest_sha256"] != digest(manifest) or digest(manifest) == original:
                raise ValueError("no independent accepted pose and regenerated training manifest")
            if args.variant == "R7":
                joint = read(args.joint_receipt or run/"c_dn/integration_decision.json")
                if not (joint["independent_pose_benefit"] and joint["independent_boundary_benefit"]):
                    raise ValueError("combined comparison conditions not met")
                info["joint_evidence_sha256"] = signature(joint)
            info["pose_evidence_sha256"] = signature(proof)
        elif digest(manifest) != original:
            raise ValueError("B3 source manifest differs from the frozen historical identity")
        doc = read(manifest)
        files = doc["files"]
        meta = read(scene/"transforms.json")
        names = meta["train_filenames"]
        all_names = [n for split in ("train","val","test") for n in meta[split+"_filenames"]]
        if [len(meta[k+"_filenames"]) for k in ("train","val","test")] != [1478,184,184] or len(set(all_names))!=1846:
            raise ValueError("hall split differs")
        validate_files(((scene/k,v) for k,v in files.items()),run/"shared"/("scene_"+digest(manifest)+".json"))
        with gzip.open(B3_REPORT/"weights.json.gz","rt",encoding="utf8") as f:
            wrows = json.load(f)["frames"]
        # Accepted pose updates must also carry newly computed pose-dependent weights.
        wdir = b3/"weights/B2"
        if positioned:
            wrows = doc["weights"]["frames"]
            wdir = run / proof["weight_directory"]
        if [r["frame"] for r in wrows] != names:
            raise ValueError("B3 sensor weights not aligned to training frames")
        weights = [wdir/Path(r["path"]).name for r in wrows]
        validate_files(zip(weights,(r["sha256"] for r in wrows)),run/"shared"/("weights_"+signature(wrows)+".json"))
        geom = read(scene/"geometry.json")
        info.update(scene_sha256=digest(manifest), initialization_sha256=files[meta["ply_file_path"]],
                    initialization_count=geom["init_points"],sensor_points=geom["sensor_points"],
                    weight_sha256=signature(wrows), common_cap=curve,
                    baseline_reason="historical B3 had no explicit population cap")
        shape = (int(meta["h"]),int(meta["w"]))
        if shape != (720,1280):
            raise ValueError("hall image size differs")
    packets = boundary = None
    if args.variant not in ("VR0C","R0C"):
        treatment = {"R4":"R3","R5":"R3","R6":"R3","R7":"R3"}.get(args.variant,args.variant)
        if vr:
            default = run/"f_vr/supervision"/treatment/"manifest.json"
        elif positioned:
            default = run / proof["q1_supervision_manifest"]
        else:
            default = run/"a_depth/supervision"/treatment/"manifest.json"
        path = args.supervision or default
        packets = Packets(path,names,shape,run/"shared/packet-cache")
        info.update(supervision_sha256=digest(path),treatment=treatment)
        if args.variant in ("R4","R7"):
            default = run / proof["boundary_manifest"] if args.variant=="R7" else run/"a_depth/boundary/manifest.json"
            path = args.boundary or default
            boundary = Packets(path,names,shape,run/"shared/packet-cache",boundary=True)
            info.update(boundary_sha256=digest(path), split_replacement_fraction=.2)
    info["image_hw"] = list(shape)
    freeze(audit/"inputs.json",info)
    return run,dest,scene,config_file,names,weights,packets,boundary,info


def resources():
    row = {"utc":dt.datetime.now(dt.timezone.utc).isoformat(),"pid":os.getpid(),
           "torch_reserved_peak":torch.cuda.max_memory_reserved(),"torch_allocated_peak":torch.cuda.max_memory_allocated(),
           "torch_threads":torch.get_num_threads()}
    cg = Path("/sys/fs/cgroup/memory.current")
    if cg.exists():
        row["cgroup_current_bytes"] = int(cg.read_text())
        stat_path=cg.with_name("memory.stat")
        if stat_path.is_file():
            mem=dict(x.split() for x in stat_path.read_text().splitlines())
            row["cgroup_anon_bytes"]=int(mem.get("anon",0))
            row["cgroup_file_bytes"]=int(mem.get("file",0))
    mem = Path("/proc/self/smaps_rollup")
    if mem.exists():
        fields = {s.split()[0].rstrip(":"):int(s.split()[1])*1024 for s in mem.read_text().splitlines()
                  if s.startswith(("Pss:","Private_Clean:","Private_Dirty:","Rss:"))}
        row.update(pss_bytes=fields["Pss"],rss_bytes=fields["Rss"],uss_bytes=fields["Private_Clean"]+fields["Private_Dirty"])
    return row


def train(args, prepared):
    from nerfstudio.scripts.train import _set_random_seed
    from nerfstudio.engine.trainer import Trainer
    from nerfstudio.engine.optimizers import Optimizers
    from dn_splatter.dn_datamanager import DNSplatterDataManager as Manager
    from dn_splatter.dn_model import DNSplatterModel as Model
    from dn_splatter.losses import EdgeAwareLogL1
    import yaml
    import capture_full_train as legacy
    import capture_quality_eval as evaluation
    from capture_quality_train import restore_rng,rng_state,GROWTH_FIELDS
    from capture_surface_train import weighted_edge_aware_log_l1
    from capture_surface_select import choose_checkpoint
    from vr_room_checkpoint import numpy_scalar_scope

    run,dest,scene,config_file,names,weights,packets,boundary,info = prepared
    audit = dest/"state"
    vr = args.variant.startswith("VR")
    budget = info["budget"]
    cfg = yaml.load(config_file.read_text(encoding="utf8"), Loader=yaml.Loader)
    cfg.output_dir = dest/"training"
    cfg.experiment_name = args.variant
    cfg.timestamp = "interactive-world-v1"
    cfg.pipeline.datamanager.dataparser.data = scene
    cfg.pipeline.datamanager.cache_images = "cpu"
    cfg.max_num_iterations = budget
    cfg.steps_per_save = 10000
    cfg.save_only_latest_checkpoint = False
    if cfg.machine.seed != 42 or cfg.pipeline.model.num_downscales or not cfg.pipeline.model.use_normal_tv_loss:
        raise ValueError("historical model configuration differs")
    if args.variant == "R5":
        cfg.pipeline.model.use_normal_tv_loss = False
    cfg.save_config()
    canonical = cfg.get_base_dir()/"config.yml"
    directory = cfg.get_checkpoint_dir()
    identity = signature(info)
    evaluation.SCENE, evaluation.RUN, evaluation.EVAL = scene,dest,dest/"evaluation"
    latest = max(directory.glob("step-*.ckpt"),default=None)
    model_state = None
    saved = -1
    if latest is not None:
        with numpy_scalar_scope():
            model_state = torch.load(latest,map_location="cpu",weights_only=True,mmap=True)
        saved = model_state["step"]
        if (type(saved) is not int or saved != int(latest.stem[5:]) or (saved+1)%10000
                or not 0 <= saved < budget or model_state["world_recovery"]["identity"] != identity):
            raise ValueError("checkpoint does not belong to this treatment or budget")
    cfg.max_num_iterations = budget-(saved+1)
    growth = None if vr else Growth(info["common_cap"]["cap"],boundary)
    sampler = random.Random(42)
    epoch = 0
    order = []
    scale = None
    pending_rng = None
    use_weight = None
    weight_records = {}
    tick = time.monotonic()

    def reorder(dm):
        nonlocal order
        order = list(range(len(names)))
        sampler.shuffle(order)
        dm.train_unseen_cameras = list(order)

    next_train = Manager.next_train
    def next_view(dm,step):
        nonlocal epoch
        if not order:
            reorder(dm)
        last = len(dm.train_unseen_cameras) == 1
        camera,batch = next_train(dm,step)
        if last:
            epoch += 1
            reorder(dm)
        dm.cached_train.prefetch(dm.train_unseen_cameras[0])
        if growth is not None and boundary is not None:
            growth.frame = int(batch["image_idx"])
            growth.camera = camera
        return camera,batch

    populate = Model.populate_modules
    def initialize(model):
        populate(model)
        if saved < 0:
            if not vr:
                if model.num_points != info["initialization_count"]:
                    raise ValueError("initial point population differs")
                with torch.no_grad():
                    tail = model.gauss_params["scales"][info["sensor_points"]:]
                    tail.add_((np.log(.25)-tail[:,:1]).clamp(max=0))
            meas = {"gaussians":int(model.num_points)}
            for key in ("means","scales","quats","features_dc"):
                ar = model.gauss_params[key].detach().cpu().contiguous().numpy()
                meas[key+"_sha256"] = hashlib.sha256(ar).hexdigest()
            freeze(audit/"initial_parameters.json",meas)
            base = "VR0C" if vr else "R0C"
            baseline = dest.parent/base/"state/initial_parameters.json"
            if args.variant not in (base,"R6","R7") and baseline.is_file() and read(baseline)!=meas:
                raise ValueError("common initial parameters differ from the paired baseline")

    def restore(trainer):
        nonlocal model_state,pending_rng,order,epoch,scale
        dm = trainer.pipeline.datamanager
        actual = [p.relative_to(scene).as_posix() for p in dm.train_dataset.image_filenames]
        if actual != names:
            raise ValueError("dataparser order disagrees with frozen treatment IDs")
        scale = float(dm.train_dataparser_outputs.dataparser_scale)
        if not np.isfinite(scale) or scale<=0 or (not vr and scale!=1):
            raise ValueError("unexpected DN metre conversion")
        if model_state is None:
            return
        ck = model_state
        r = ck["world_recovery"]
        trainer.pipeline.load_pipeline(ck["pipeline"],saved)
        trainer.optimizers = trainer.setup_optimizers()
        trainer.optimizers.load_optimizers(ck["optimizers"])
        trainer.optimizers.load_schedulers(ck["schedulers"])
        trainer.grad_scaler.load_state_dict(ck["scalers"])
        trainer._start_step = saved+1
        for key in GROWTH_FIELDS:
            value = r["growth"][key]
            setattr(trainer.pipeline.model,key,value.to(trainer.device) if value is not None else None)
        if growth is not None:
            growth.restore(r["boundary"],trainer.device)
        sampler.setstate(r["sampler_rng"])
        order,epoch = r["order"],r["epoch"]
        dm.train_unseen_cameras,dm.eval_unseen_cameras = r["remaining"],r["eval_remaining"]
        pending_rng = r["rng"]
        model_state = None

    def checkpoint(trainer,step):
        if (step+1)%10000:
            return
        directory.mkdir(parents=True,exist_ok=True)
        path = directory/f"step-{step:09d}.ckpt"
        if path.exists():
            return
        dm = trainer.pipeline.datamanager
        m = trainer.pipeline.model
        r = {"identity":identity,"sampler_rng":sampler.getstate(),"order":order,"epoch":epoch,
             "remaining":list(dm.train_unseen_cameras),"eval_remaining":list(dm.eval_unseen_cameras),
             "rng":rng_state(),"growth":{k:getattr(m,k,None) for k in GROWTH_FIELDS},
             "boundary":None if growth is None else growth.snapshot()}
        ck = {"step":step,"pipeline":trainer.pipeline.state_dict(),"world_recovery":r,
              "optimizers":{k:v.state_dict() for k,v in trainer.optimizers.optimizers.items()},
              "schedulers":{k:v.state_dict() for k,v in trainer.optimizers.schedulers.items()},
              "scalers":trainer.grad_scaler.state_dict()}
        part = path.with_suffix(".pending")
        torch.save(ck,part)
        part.replace(path)
        atomic_json(audit/"checkpoint.json",{"saved_step":step,"completed_updates":step+1,"path":str(path),
                    "bytes":path.stat().st_size,"sha256":digest(path),"gaussians":int(m.num_points)})

    original_eval = Trainer.eval_iteration
    def evaluate(trainer,step):
        if not vr:
            original_eval(trainer,step)
        if (step+1)%10000:
            return
        checkpoint(trainer,step)
        if vr or (step+1)%30000:
            return
        rng,training = rng_state(),trainer.pipeline.training
        try:
            trainer.pipeline.eval()
            evaluation.evaluate_fixed(canonical,step,"val",trainer.pipeline,directory/f"step-{step:09d}.ckpt")
        finally:
            trainer.pipeline.train(training)
            restore_rng(rng)

    base_weighted = EdgeAwareLogL1.forward
    def sensor_loss(module,pred,gt,rgb,mask):
        if use_weight is None:
            return base_weighted(module,pred,gt,rgb,mask)
        return weighted_edge_aware_log_l1(pred,gt,rgb,mask,use_weight)

    native_loss = Model.get_loss_dict
    first = {}
    def loss(model,outputs,batch,*pos,**kw):
        nonlocal use_weight
        if not model.training:
            return native_loss(model,outputs,batch,*pos,**kw)
        idx = int(batch["image_idx"])
        pred = outputs["depth"]
        if weights:
            wm = np.load(weights[idx],mmap_mode="c",allow_pickle=False)
            use_weight = torch.as_tensor(wm,device=pred.device,dtype=pred.dtype).reshape(pred.shape)
        try:
            result = native_loss(model,outputs,batch,*pos,**kw)
        finally:
            use_weight = None
        if packets is not None:
            points = packets.get(idx)
            index = torch.as_tensor(np.array(points["index"],copy=True),device=pred.device)
            z = torch.as_tensor(np.array(points["z_m"],copy=True),device=pred.device)
            rgb = model.get_gt_img(batch["image"]).clamp(min=10/255)
            term,accepted = loss_at_pixels(pred,rgb,batch["sensor_depth"],index,z,scale,
                                           (0,float("inf")) if vr else (.3,6.),batch.get("mask"))
            result["hole_depth"] = .1*model.config.sensor_depth_lambda*term
            weight_records[idx] = [len(points),accepted]
        if not first:
            first.update(frame=names[idx],step=int(model.step),losses={k:float(v.detach()) for k,v in result.items()},
                         sensor_weighted=bool(weights),packet_pixels=weight_records.get(idx),
                         resources=resources(),metres_to_dn_scale=scale)
        if model.step%100 == 0:
            atomic_json(audit/"progress.json",{"durable_checkpoint_updates":(int(model.step)//10000)*10000,
                         "step_in_progress":int(model.step),"verified_completed_this_segment":int(model.step)-(saved+1),
                         "frame":names[idx],"gaussians":int(model.num_points),"wall_seconds":time.monotonic()-tick,
                         "resources":resources()})
        return result

    optimizer = Optimizers.optimizer_scaler_step_some
    def gradients(opt,*pos,**kw):
        if "gradient_elements" not in first:
            grads = [p.grad for group in opt.parameters.values() for p in group if p.grad is not None]
            first.update(gradient_elements=sum(g.numel() for g in grads),
                         nonfinite_gradients=sum(int((~g.isfinite()).sum()) for g in grads))
            atomic_json(audit/f"first_batch_{saved+1:09d}.json",first)
            if not first["gradient_elements"] or first["nonfinite_gradients"] or not all(np.isfinite(list(first["losses"].values()))):
                raise ValueError("first formal update has invalid gradients or loss")
        return optimizer(opt,*pos,**kw)

    def train_segment():
        nonlocal pending_rng
        _set_random_seed(42)
        trainer = cfg.setup(local_rank=0,world_size=1)
        trainer.setup()
        if pending_rng is not None:
            _ = trainer.pipeline.datamanager.cached_train,trainer.pipeline.datamanager.cached_eval
            restore_rng(pending_rng)
            pending_rng = None
        atomic_json(audit/"segment.json",{"saved_step":saved,"remaining":cfg.max_num_iterations,"canonical_steps":budget,
                                         "started":resources(),"command_variant":args.variant})
        trainer.train()

    # Bind to the serialization implementation of image tensors without invalidating
    # R0C's already-produced identical pages for an unrelated seed-cache extension.
    image_format_hash = "33e79d0225f1402be7a151bd139035c2928d04956b7be44333494195403f13ee"
    cache_identity = signature([info["scene_sha256"],info["historical_config_sha256"],image_format_hash])
    with ExitStack() as stack:
        stack.enter_context(shared_seed_geometry(Model,run/"shared/initial-geometry"))
        views = stack.enter_context(shared_views(Manager,run/"shared/image-cache"/cache_identity))
        # Refill validation nodes from already-saved immutable models before a resumed segment.
        if not vr:
            for n in evaluation.NODES:
                if n <= saved and not (dest/"evaluation/val"/f"step-{n:09d}.json").exists():
                    evaluation.evaluate(canonical,n,"val")
        if saved+1 < budget:
            for obj,attr,impl in ((Model,"populate_modules",initialize),(Manager,"next_train",next_view),
                    (Trainer,"_load_checkpoint",restore),(Trainer,"save_checkpoint",checkpoint),
                    (Trainer,"eval_iteration",evaluate),(Model,"get_loss_dict",loss),
                    (Optimizers,"optimizer_scaler_step_some",gradients)):
                stack.enter_context(patch.object(obj,attr,impl))
            if vr:
                train_segment()
            else:
                stream = stack.enter_context((audit/"refinement.jsonl").open("a",encoding="utf8"))
                stack.enter_context(growth.hooks(Model,stream,audit,resources))
                # Preserve historical empty sensor reductions and metric guards. Install
                # the B3 weighted depth hook inside legacy.main's patch scope.
                def with_weights():
                    nonlocal base_weighted
                    base_weighted=EdgeAwareLogL1.forward
                    with patch.object(EdgeAwareLogL1,"forward",sensor_loss):
                        train_segment()
                for key,val in (("config",lambda:cfg),("train",with_weights),("SCENE",scene),("STATE",audit),("REPORT",audit)):
                    stack.enter_context(patch.object(legacy,key,val))
                legacy.main(recovery_epochs=3)
            atomic_json(audit/"shared_cache.json",[{"reads":v.reads,"prefetch_hits":v.hits,"directory":str(v.directory)} for v in views])
    last = read(audit/"checkpoint.json")
    if last["completed_updates"] != budget or digest(Path(last["path"])) != last["sha256"]:
        raise ValueError("terminal checkpoint is absent, incomplete or corrupt")
    if vr:
        chosen = {"selected_step":budget-1,"final_step":budget-1,"reason":"fixed 30k without Faro model selection"}
    else:
        chosen = choose_checkpoint([dest/"evaluation/val"/f"step-{n:09d}.json" for n in evaluation.NODES])
        chosen.pop("recommended_rendering_model",None)
        chosen["candidate"] = args.variant
    freeze(audit/"selection.json",chosen)
    atomic_json(dest/"training_complete.json",dict(last,variant=args.variant,config=str(canonical),selection=chosen))
    if args.phase == "all":
        evaluate_selected(args,dest,scene)


def evaluate_selected(args,dest,scene):
    completed = read(dest/"training_complete.json")
    if args.variant.startswith("VR"):
        from interactive_vr_evaluate import evaluate
        evaluate(dest,scene,Path(completed["config"]),Path(completed["path"]))
        return
    import capture_quality_eval as ev
    ev.SCENE,ev.RUN,ev.EVAL = scene,dest,dest/"evaluation"
    chosen = read(dest/"state/selection.json")
    nodes = sorted({chosen["selected_step"],chosen["final_step"]})
    for n in nodes:
        path = ev.EVAL/"test"/f"step-{n:09d}.json"
        if not path.exists():
            ev.evaluate(Path(completed["config"]),n,"test")
        row = read(path)
        if row["frame_count"]!=184 or row["step"]!=n or row["split"]!="test":
            raise ValueError("development-test receipt mismatch")
    atomic_json(ev.EVAL/"test_complete.json",{"steps":nodes,"frames_per_step":184})
    from interactive_dn_evaluate import run as evaluate_geometry_and_media
    evaluate_geometry_and_media(dest,scene,Path(completed["config"]),chosen["selected_step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant",choices=RUNS)
    parser.add_argument("--phase",choices=("prepare","train","evaluate","all"),default="all")
    parser.add_argument("--run-root",type=Path,default=ROOT/".local/runs"/RUN_ID)
    for k in ("b3-root","scene","supervision","boundary","pose-receipt","joint-receipt"):
        parser.add_argument("--"+k,type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if root.name != RUN_ID:
        parser.error("different run identity")
    dest = root/("f_vr" if args.variant.startswith("VR") else "c_dn")/args.variant
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/"training.lock").open("a+b") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepared = prepare(args)
        if args.phase=="prepare":
            print(json.dumps(prepared[-1]),flush=True)
        elif args.phase=="evaluate":
            evaluate_selected(args,dest,prepared[2])
        else:
            train(args,prepared)


if __name__ == "__main__":
    main()
