"""Verify real hidden-source caches, spatial indexing, gradients and batched epoch scoring."""
from pathlib import Path

import numpy as np
import torch

from interactive_depth_candidates import BASE_CHANNELS, feature_table
from interactive_depth_geometry import valid
from interactive_depth_inputs import MASKS, REPS, RUN
from interactive_depth_mdm import patch_ids,unpack
from interactive_depth_quality import HEADS, input_for,load_npz,network,stacked_logits
from lingbot_common import read,require,save,sha


def run(root):
    d=read(root/"freeze.json")
    first=d["rows"][0]
    by_id={r["id"]:r for r in d["rows"]}
    require(all(all(by_id[v["id"]]["fold"]==r["fold"] and by_id[v["id"]]["group"]!=r["group"]
                and not by_id[v["id"]]["near_other_folds"] for v in r["donors"]) for r in d["rows"]),
            "Ineligible donor")
    rows=[]
    for kind in d["masks"]:
        folder=root/"cases"/first["id"]/kind
        r=read(folder/"receipt.json")
        require(all(sha(folder/x["path"])==x["sha256"] for x in r["outputs"]),"First official cache integrity")
        b=load_npz(folder/"source_and_score.npz")
        hidden=np.unpackbits(b["hidden_input"],count=int(np.prod(d["image_wh"]))).astype(bool)
        if len(b["hidden_native"]):
            src=np.unpackbits(b["hidden_native"],count=848*480).astype(bool)
            for col in (0,1):
                parent=b["parent"][:,col]
                require(not src[parent[parent>=0]].any(),"NN or footprint reused a hidden native measurement")
        require(hidden[b["flat"]].all(),"Scored visible reference")
        for rep in d["representations"]:
            s=load_npz(folder/(rep+".npz"))
            grid=s["token_hw"]
            mapped=patch_ids(b["flat"],tuple(reversed(d["image_wh"])),grid)
            np.testing.assert_array_equal(mapped,s["token_grid_ids"][s["token_for_pixel"]])
        rows.append(dict(kind=kind,receipt_sha256=sha(folder/"receipt.json"),scored=len(b["flat"]),
                         hidden_samples=r["hidden_native"],separate_forward_pairs=len(d["representations"])))
    # Exercise the downstream gradient and all-epoch logit path on the first real boundary condition.
    ix=np.arange(min(64,len(b["flat"])))
    depths=np.column_stack((b["depths"][ix],s["mdm_depth"][ix]))
    x=feature_table(depths,b["observations"][ix],b["local"][ix],b["audit_bits"],rep)
    i,j=np.nonzero(valid(depths))
    geom=torch.from_numpy(x[i,j]).to("cuda")
    actual=unpack(s["token_values"][s["token_for_pixel"][ix[i]]])
    # Here coordinates 0:32 are used only to test tensor/gradient plumbing; experimental PCA is fit-only.
    m=torch.from_numpy(actual[:,:32].copy()).to("cuda")
    checks={}
    for n in HEADS:
        torch.manual_seed(42)
        head=network(n).cuda()
        inputs=input_for(n,geom,m)
        z=head(inputs)
        weights={k:torch.stack((v.detach(),v.detach())) for k,v in head.state_dict().items()}
        stack=stacked_logits(n,inputs,weights)
        torch.testing.assert_close(z,stack[0],rtol=3e-5,atol=5e-5)
        z.square().mean().backward()
        require(all(t.grad is not None and torch.isfinite(t.grad).all() for t in head.parameters()),"Head gradient")
        checks[n]=dict(parameters=sum(t.numel() for t in head.parameters()),
                       stacked_max_abs=float((z-stack[0]).abs().max().detach()),finite_gradients=True)
    save(root/"interface_check.json",dict(first_frame=first["id"],masked=rows,refs_verified=True,
         validated_donor_edges=sum(len(r["donors"]) for r in d["rows"]),checks=checks,optimizer_updates=0,
         shape=[len(i),len(BASE_CHANNELS)],scope="interface verification of formal caches; fitted scores pending"))


if __name__=="__main__":
    run(RUN)
