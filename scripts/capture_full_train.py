"""Run the historical trainer, applying empty-mask reductions only for measured empty support."""
from contextlib import ExitStack
import difflib
import inspect
import json
from pathlib import Path
import textwrap
from unittest.mock import patch

import numpy as np
from dn_splatter.dn_model import DNSplatterModel
from dn_splatter.losses import EdgeAwareLogL1
from capture_rectify import read, save, sha
from capture_s01_inputs import REPORT, ensure
from capture_s01_train import config, train
from vr_room_run import SCENE, STATE


def replacement(fn, before, after, diffs):
    original=textwrap.dedent(inspect.getsource(fn))
    ensure(original.count(before)==1,"unexpected historical implementation: "+fn.__qualname__)
    edited=original.replace(before,after)
    diffs.extend(difflib.unified_diff(original.splitlines(True),edited.splitlines(True),
                                      fromfile=fn.__qualname__,tofile=fn.__qualname__+".full"))
    namespace={}
    exec(compile(edited,inspect.getfile(fn),"exec"),fn.__globals__,namespace)
    return namespace[fn.__name__]


def main(recovery_epochs=1):
    cfg=config()
    m=cfg.pipeline.model
    meta=read(SCENE/"transforms.json")
    bad={}
    for row in meta["frames"]:
        mask=np.load(SCENE/row["depth_file_path"],mmap_mode="r")>m.depth_tolerance
        axes=[axis for axis,v in (("x",mask[:,:-1]),("y",mask[:-1])) if not v.any()]
        if axes:
            bad[Path(row["file_path"]).stem]=dict(empty_axes=axes,zero_reference=not bool(mask.any()))
    info=dict(frames=bad,applied=bool(bad),weight=m.sensor_depth_lambda,depth_tolerance=m.depth_tolerance)
    events=STATE/"empty_sensor_events.jsonl"
    names=[Path(n).stem for n in meta["train_filenames"]]
    interval=m.reset_alpha_every*m.refine_every
    info["densification_candidate_steps"]=[s for s in range(0,m.stop_split_at,m.refine_every)
                                          if s>m.warmup_length and s%interval>recovery_epochs*len(names)+m.refine_every]
    diff=[]
    with ExitStack() as stack:
        if bad:
            fn=EdgeAwareLogL1.forward
            term="return loss_x.mean() + loss_y.mean()"
            new=replacement(fn,term,"return (loss_x.mean() if loss_x.numel() else loss_x.sum()) + (loss_y.mean() if loss_y.numel() else loss_y.sum())",diff)
            stack.enter_context(patch.object(EdgeAwareLogL1,"forward",new))
            # Keep the original visualization even when the depth scalar has no support.
            for name,target in (("get_metrics_dict","sensor_depth_gt"),("get_image_metrics_and_images","gt_depth")):
                fn=getattr(DNSplatterModel,name)
                body=textwrap.dedent(inspect.getsource(fn))
                lines=body.splitlines(True)
                start=next(i for i,line in enumerate(lines) if "(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = self.depth_metrics(" in line)
                end=next(i for i in range(start,len(lines)) if "metrics_dict.update(depth_metrics)" in lines[i])
                old="".join(lines[start:end+1])
                indent=lines[start][:len(lines[start])-len(lines[start].lstrip())]
                guard=indent+"if ("+target+" > 0).any():\n"+textwrap.indent(old,"    ")
                stack.enter_context(patch.object(DNSplatterModel,name,replacement(fn,old,guard,diff)))
            loss=DNSplatterModel.get_loss_dict
            def observed(model,outputs,batch,*args,**kwargs):
                # Only the training split has a training update identity.
                if model.training:
                    stem=names[int(batch["image_idx"])]
                    if stem in bad:
                        with events.open("a") as out:
                            print(json.dumps(dict(id=stem,step=int(model.step),**bad[stem])),file=out)
                return loss(model,outputs,batch,*args,**kwargs)
            stack.enter_context(patch.object(DNSplatterModel,"get_loss_dict",observed))
            path=REPORT/"empty_sensor.patch"
            path.write_text("".join(diff))
            info["runtime_patch_sha256"]=sha(path)
        save(REPORT/"conditional_adaptations.json",info)
        train()
    rows=[json.loads(x) for x in events.read_text().splitlines()] if events.exists() else []
    info["training_events"]=rows
    info["empty_reduction_counts"]={axis:sum(axis in r["empty_axes"] for r in rows) for axis in ("x","y")}
    save(REPORT/"conditional_adaptations.json",info)


if __name__=="__main__":
    main()
