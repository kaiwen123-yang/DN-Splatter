"""Use the xFormers varlen dispatcher with Hopper FA3 disabled on SM120."""
import json
from pathlib import Path
import runpy
import sys


def configure():
    import torch
    from xformers.ops.fmha import dispatch
    if torch.cuda.get_device_capability()==(12,0):
        dispatch._set_use_fa3(False)


def example(path,receipt):
    import torch
    from mdm.model.dinov2_rgbd.layers import attention
    from xformers.ops.fmha import Inputs,dispatch
    original=attention.memory_efficient_attention
    def measured(q,k,v,attn_bias=None,**kwargs):
        output=original(q,k,v,attn_bias=attn_bias,**kwargs)
        if attn_bias is None or not hasattr(attn_bias,"q_seqinfo"):
            return output
        starts=attn_bias.q_seqinfo.seqstart_py
        differences=[]
        for a,b in zip(starts,starts[1:]):
            if a==b: continue
            query=q[:,a:min(a+8,b)].float().transpose(1,2)
            keys=k[:,a:b].float().transpose(1,2)
            values=v[:,a:b].float().transpose(1,2)
            with torch.autocast("cuda",enabled=False):
                ref=((query@keys.transpose(-1,-2))/q.shape[-1]**.5).softmax(-1)@values
            differences.append((ref.transpose(1,2)-output[:,a:min(a+8,b)].float()).abs().flatten())
        diff=torch.cat(differences)
        op=dispatch._dispatch_fw(Inputs(query=q,key=k,value=v,attn_bias=attn_bias),needs_gradient=False)
        receipt.write_text(json.dumps(dict(operator=op.NAME,mask=type(attn_bias).__name__,sequence_offsets=starts,
             compared_scalar_values=diff.numel(),max_abs=float(diff.max()),mean_abs=float(diff.mean()),
             rms=float(diff.square().mean().sqrt()),uncorrected_FA3="invalid argument on SM120",
             comparison="first eight real queries per nested sequence vs FP32 softmax over all its keys; example 0"))+"\n")
        attention.memory_efficient_attention=original
        if only:
            raise SystemExit(0)
        return output
    configure()
    attention.memory_efficient_attention=measured
    only="--kernel-reference-only" in sys.argv
    sys.argv=[str(path)]+[s for s in sys.argv[3:] if s!="--kernel-reference-only"]
    runpy.run_path(str(path),run_name="__main__")


if __name__=="__main__":
    example(Path(sys.argv[1]),Path(sys.argv[2]))
