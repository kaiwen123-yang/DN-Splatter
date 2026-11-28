"""Schedule full-resolution diagnostics at the exact steps consumed by the frozen trainer."""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch


def implementation(kind):
    """Bind each approved fast path to its relevant implementations and numerical runtime."""
    import torch
    from dn_splatter.dn_datamanager import DNSplatterDataManager
    from dn_splatter.dn_model import DNSplatterModel
    from dn_splatter.metrics import RGBMetrics,DepthMetrics
    from nerfstudio.engine.trainer import Trainer
    from nerfstudio.models.splatfacto import SplatfactoModel
    from nerfstudio.utils.misc import step_check
    from interactive_dn_cache import MMapViews,shared_views,_encode_image
    from interactive_dn_supervision import FrozenHoleTerms,loss_at_pixels,Packets
    funcs={
        "image_transfer":(MMapViews,shared_views,_encode_image,DNSplatterDataManager.next_train),
        "hole_terms":(FrozenHoleTerms,loss_at_pixels,Packets._convert,DNSplatterModel.get_gt_img),
        "metric_dispatch":(metrics_at_writer_steps,step_check,Trainer.train,Trainer.train_iteration,
                           DNSplatterModel.get_metrics_dict,DNSplatterModel.get_loss_dict,
                           SplatfactoModel.get_loss_dict,RGBMetrics.forward,DepthMetrics.forward),
    }[kind]
    source={f.__module__+"."+f.__qualname__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in funcs}
    config=Path(__file__).resolve().parents[1]/"reports/capture-full-quality-v2/20261121-v1/config.yml"
    return dict(sources=source,torch=str(torch.__version__),cuda=str(torch.version.cuda),
                capability=list(torch.cuda.get_device_capability()),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())


def acceleration_policy(run,variant,info,integration_sha=None):
    folder=run/"f_evaluation/acceleration"
    policy={}
    for kind in ("image_transfer","hole_terms","metric_dispatch"):
        path=folder/(kind+"_verified.json")
        enabled=False
        reason="no verification receipt"
        sha=None
        if kind=="hole_terms" and variant not in ("R3","R4","R5","R6","R7"):
            reason="different treatment"
        elif path.is_file():
            try:
                raw=path.read_bytes();doc=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
                expected=doc.get("implementation",{})
                enabled=(doc.get("verified") is True and info["image_hw"]==[720,1280]
                         and expected.get("config_sha256")==info["historical_config_sha256"]
                         and expected==implementation(kind))
                if kind=="hole_terms":
                    enabled=enabled and doc.get("supervision_sha256")==info.get("supervision_sha256")
                reason="verified" if enabled else "input, runtime, source or configuration differs"
            except (OSError,KeyError,TypeError,ValueError):
                reason="unreadable or incompatible verification"
        policy[kind]=dict(enabled=enabled,reason=reason,evidence=str(path),sha256=sha)
    if integration_sha is not None:
        attestation=folder/"future_segment_policy.json"
        try:
            raw=attestation.read_bytes()
            release=json.loads(raw)
            current=release["source_files"]["interactive_dn_train.py"]==integration_sha
            known=release["future_segment_policy"]["R3"]
            bound=hashlib.sha256(raw).hexdigest()
        except (OSError,ValueError,TypeError,KeyError):
            current,known,bound=False,{},None
        for kind,row in policy.items():
            row["integration_receipt_sha256"]=bound
            if row["enabled"] and not (current and known.get(kind,{}).get("sha256")==row["sha256"]):
                row.update(enabled=False,reason="unverified trainer integration or revised benchmark receipt")
    return policy


@contextmanager
def metrics_at_writer_steps(model_class, interval):
    from nerfstudio.utils.misc import step_check
    original = model_class.get_metrics_dict
    counts = {"writer_interval":int(interval),"train_computed":0,"train_unconsumed":0,"evaluation_computed":0}
    def get_metrics(model,outputs,batch):
        if not model.training:
            counts["evaluation_computed"] += 1
        elif step_check(model.step,interval,run_at_zero=True):
            counts["train_computed"] += 1
        else:
            # In the pinned DN and Splatfacto losses the supplied metrics dict is
            # never read. Trainer discards it at these steps. LPIPS stays in eval
            # mode throughout training. TorchMetrics' running sums are nonpersistent.
            counts["train_unconsumed"] += 1
            return {}
        return original(model,outputs,batch)
    with patch.object(model_class,"get_metrics_dict",get_metrics):
        yield counts
