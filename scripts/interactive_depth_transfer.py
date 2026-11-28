"""Apply the frozen hall depth and Q1 models to a single calibrated RGB-D camera."""
import argparse
from pathlib import Path

import numpy as np

from interactive_depth_candidates import CaseGeometry
from interactive_depth_geometry import OtherViews, atomic_npz, valid
from interactive_depth_inputs import OLD, RUN
from interactive_depth_mdm import FrozenMDM, patch_ids
from interactive_depth_quality import FrozenQuality, project_tokens
from lingbot_common import require, save, sha


def empty_views(shape):
    zero=np.zeros(shape,np.float32)
    index=np.full(shape,-1,np.int32)
    return OtherViews(zero,zero,index,index,index,index,[],[],np.empty(0,np.int16))


def transfer_image(model, quality, rgb, visible_z_m, K_cv, query_mask):
    """Return NN/direct/Q1 arrays for the same query rays. Caller supplies already-hidden depth."""
    z=np.asarray(visible_z_m,np.float32)
    mask=np.asarray(query_mask,bool)
    require(z.shape==mask.shape and not valid(z[mask]).any(),"Pass missing query rays, never their reference depths")
    require(np.isfinite(z).all() and (z>=0).all(),"Use zero for missing input Z")
    ge=CaseGeometry(rgb,z,np.zeros_like(z),empty_views(z.shape))
    dep,spatial,receipt=model.predict(rgb,z,K_cv)
    flat=np.flatnonzero(mask)
    grid=spatial.reshape(spatial.shape[0],-1).T.copy()
    projected=project_tokens({"token_values":grid},quality.prep)
    where=patch_ids(flat,z.shape,spatial.shape[-2:])
    selected=np.zeros_like(z)
    selected_source=np.zeros(z.shape,np.uint8)
    risk=np.full(z.shape,np.nan,np.float32)
    for start in range(0,len(flat),16384):
        sub=flat[start:start+16384]
        ds,obs,local=ge.table(sub,dep)
        choice,p=quality.select(ds,obs,local,[],projected[where[start:start+len(sub)]],"full_range_sparse")
        ok=choice>=0
        selected.flat[sub[ok]]=ds[np.flatnonzero(ok),choice[ok]]
        selected_source.flat[sub[ok]]=choice[ok]+1
        risk.flat[sub]=np.where(np.isfinite(p),1.-p,np.nan)
    out=dict(G_NN=np.where(mask,ge.nn,0).astype(np.float32),
             G_LingBot=np.where(mask & valid(dep),dep,0).astype(np.float32),G_Q1=selected,
             Q1_accepted=(selected_source>0),Q1_source=selected_source,Q1_risk=risk,query_mask=mask)
    return out,dict(camera_count=1,independent_views_supplied=0,queried=int(mask.sum()),
                    accepted=int((selected_source>0).sum()),geometry_feature="no native footprint or auxiliary views",
                    Q1_calibration=quality.cal["full_range_sparse"]["Q1"],**receipt)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path,help="NPZ: rgb uint8 HWC, visible_z_m float32, K_cv, query_mask (already hidden)")
    parser.add_argument("output",type=Path)
    parser.add_argument("--fold-dir",type=Path,default=RUN/"quality/fold_0")
    parser.add_argument("--weight",type=Path,default=OLD/"weights/depth/model.pt")
    args=parser.parse_args()
    q=FrozenQuality(args.fold_dir)
    m=FrozenMDM(args.weight)
    with np.load(args.input,allow_pickle=False) as f:
        result,meta=transfer_image(m,q,f["rgb"],f["visible_z_m"],f["K_cv"],f["query_mask"])
    atomic_npz(args.output,**result)
    save(args.output.with_suffix(".json"),dict(input_sha256=sha(args.input),output_sha256=sha(args.output),
         source_fold=str(args.fold_dir),frozen_calibration_sha256=sha(args.fold_dir/"calibration.json"),**meta))
