"""Check new DN numeric paths before formal updates, without starting a training run."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from interactive_dn_cache import _encode_image,atomic_json
from interactive_dn_supervision import loss_at_pixels


def verify_packets(run):
    from interactive_dn_supervision import Packets,digest
    rows=[]
    dirs=[(run/"a_depth/supervision/R1",False),(run/"a_depth/boundary",True)]
    for directory,boundary in dirs:
        path=directory/"manifest.json"
        doc=json.loads(path.read_bytes())
        names=[r["id"] for r in doc["frames"]]
        if len(names)!=1478:
            raise ValueError("complete hall train membership required")
        packets=Packets(path,names,(720,1280),run/"shared/packet-cache",boundary=boundary)
        start=time.monotonic()
        count=0
        for i in range(len(names)):
            data=packets.get(i)
            if len(data) and (np.any(data["index"][1:]<=data["index"][:-1]) or
                              not np.isfinite(data["mode" if boundary else "z_m"]).all()):
                raise ValueError(f"packet conversion lost pixel identities: {names[i]}")
            count+=len(data)
        expected=sum(r["accepted_pixels"] for r in doc["frames"])
        if not boundary and count!=expected:
            raise ValueError("treatment population changed after conversion")
        rows.append({"scope":directory.name,"manifest_sha256":digest(path),"frames":len(names),
                     "accepted_source_rows":expected,"model_pixels_after_per_side_corner_filter":count,
                     "elapsed_seconds":time.monotonic()-start})
    return rows


def check(device):
    rng = torch.Generator().manual_seed(42)
    pred = torch.rand((7,9,1),generator=rng,dtype=torch.float64).add(.5).to(device).requires_grad_()
    gt = torch.rand(pred.shape,generator=rng,dtype=pred.dtype).add(.5).to(device)
    rgb = torch.rand((7,9,3),generator=rng,dtype=pred.dtype).to(device)
    mask = (torch.rand(pred.shape,generator=rng)>.4).to(device)
    idx = torch.nonzero(mask.flatten()).flatten()
    sensor = torch.where(mask,torch.zeros_like(gt),gt)
    def compute(p,ix,z):
        return loss_at_pixels(p,rgb,sensor,ix,z,1,(.3,6))
    actual, n = compute(pred,idx,gt.flatten()[idx])
    dx = torch.exp(-(rgb[:,:-1]-rgb[:,1:]).abs().mean(dim=-1,keepdim=True))
    dy = torch.exp(-(rgb[:-1]-rgb[1:]).abs().mean(dim=-1,keepdim=True))
    residual = (pred-gt).abs().log1p()
    ref = (dx*residual[:,:-1])[mask[:,:-1]].mean() + (dy*residual[:-1])[mask[:-1]].mean()
    g0, = torch.autograd.grad(ref,pred,retain_graph=True)
    g1, = torch.autograd.grad(actual,pred)
    torch.testing.assert_close(actual,ref,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(g0,g1,rtol=1e-12,atol=1e-12)
    mixed = torch.full_like(pred,float("nan")).requires_grad_()
    empty,count = compute(mixed,idx,torch.full_like(idx,float("inf"),dtype=mixed.dtype))
    zero, = torch.autograd.grad(empty,mixed)
    assert count==0 and float(empty.detach())==0 and torch.isfinite(zero).all() and not zero.count_nonzero()
    original = (pred*2).sum()
    empty,n0 = compute(pred,idx[:0],gt.flatten()[:0])
    grad0, = torch.autograd.grad(original,pred,retain_graph=True)
    grad1, = torch.autograd.grad(original+.02*empty,pred)
    torch.testing.assert_close(grad0,grad1,rtol=0,atol=0)
    assert n0==0
    codes = np.broadcast_to(np.arange(256,dtype=np.uint8).reshape(16,16,1),(16,16,3)).copy()
    x = codes.astype(np.float32)/255
    normal = (1-(x*2-1))/2
    for value in (x,normal):
        enc = _encode_image(value)
        assert enc is not None
        code,lut=enc
        decoded=(np.stack([row[code[...,c]] for c,row in enumerate(lut)],axis=-1)
                 if lut is not None else code.astype(np.float32)/255)
        assert np.array_equal(decoded.view(np.uint32),value.view(np.uint32))
    return {"device":str(device),"torch":str(torch.__version__),"accepted":n,"directional_loss_abs_error":float((actual-ref).abs().detach()),
            "gradient_max_abs_error":float((g1-g0).abs().max()),"empty_set_gradients_unchanged":True,
            "nonfinite_rejected_gradient_elements":int(zero.numel()),"bit_exact_rgb_and_normal_values":int(x.size+normal.size)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--packets",type=Path,help="validate and transcode full R1 and supported edge packets")
    args=parser.parse_args()
    if args.packets:
        rows=verify_packets(args.packets)
    else:
        rows=[check("cpu")]
        if torch.cuda.is_available():
            rows.append(check("cuda"))
    if args.output is not None:
        atomic_json(args.output,rows)
    print(json.dumps(rows,allow_nan=False))
