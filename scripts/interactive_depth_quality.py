"""Five-fold fit-only PCA, Q0/Q1 and multinomial logistic training for metric depth risk."""
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import numpy as np
import torch
from torch import nn

from interactive_depth_candidates import BASE_CHANNELS, feature_table
from interactive_depth_geometry import atomic_npz, primary, valid
from interactive_depth_inputs import RUN, REPS, digest_json
from interactive_depth_mdm import unpack
from lingbot_common import read, require, save, sha


HEADS = ("logistic", "Q0", "Q1")
BINS = np.array([.05,.10,.20],np.float32)
TEMPERATURES = (.5,.67,.8,1.,1.25,1.5,2.,3.)
RESOLUTION = 1000
COVERAGES = (.1,.25,.5,.75)


def load_npz(path, keys=None):
    with np.load(path,allow_pickle=False) as a:
        return {key:a[key] for key in (a.files if keys is None else keys)}


def epoch_file(folder, epoch):
    return folder / "epochs" / f"epoch_{epoch:02d}.pt"


def save_pt(path, state):
    path.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".pt",delete=False) as f:
        torch.save(state,f)
    os.replace(f.name,path)


def copy_weights(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def network(kind):
    d=len(BASE_CHANNELS)
    if kind=="logistic":
        return nn.Sequential(nn.Linear(d,4))
    return nn.Sequential(nn.Linear(d+32,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,4))


def input_for(kind, geom, mdm):
    if kind=="logistic":
        return geom
    return torch.cat((geom,mdm if kind=="Q1" else torch.zeros_like(mdm)),dim=1)


def iterate(root,dataset,fold,role,selected=False):
    for row in dataset["rows"]:
        if row["role_for_outer"][fold]!=role:
            continue
        for kind in dataset["masks"]:
            d=root/"cases"/row["id"]/kind
            base=load_npz(d/"source_and_score.npz",("reference","edge_side","sample_index","sample_rep","depths","local",
                                                    "observations","audit_bits","flat"))
            for ri,rep in enumerate(dataset["representations"]):
                pick=(base["sample_index"][base["sample_rep"]==ri] if selected else np.arange(len(base["reference"])))
                if not len(pick):
                    continue
                source=load_npz(d/(rep+".npz"),("mdm_depth","token_values","token_for_pixel"))
                yield row,kind,rep,base,source,pick


def make_data(base,source,rows,rep):
    depths=np.column_stack((base["depths"][rows],source["mdm_depth"][rows]))
    features=feature_table(depths,base["observations"][rows],base["local"][rows],base["audit_bits"],rep)
    err=np.abs(depths-base["reference"][rows,None])
    labels=np.searchsorted(BINS,err,side="left").astype(np.int64)
    return depths,features,labels


def preprocessing(root,dataset,fold,folder):
    path=folder/"preprocess.npz"
    record=folder/"preprocess.json"
    if record.exists():
        r=read(record)
        require(sha(path)==r["sha256"],"Fit-only PCA/normalization changed")
        return load_npz(path)
    dim=len(BASE_CHANNELS)
    count=0
    sumg=np.zeros(dim,np.float64)
    gramg=np.zeros(dim,np.float64)
    n_token=0
    token_sum,gram=None,None
    all_tokens=[]
    fit_ids=set()
    rng=np.random.default_rng(42)

    def consume():
        nonlocal n_token,token_sum,gram
        if not all_tokens:
            return
        x=torch.from_numpy(np.concatenate(all_tokens)).to("cuda",torch.float64)
        if gram is None:
            gram=torch.zeros((x.shape[1],x.shape[1]),device="cuda",dtype=torch.float64)
            token_sum=torch.zeros(x.shape[1],device="cuda",dtype=torch.float64)
        gram.addmm_(x.T,x)
        token_sum+=x.sum(0)
        n_token+=len(x)
        all_tokens.clear()

    for row,_,rep,base,src,pick in iterate(root,dataset,fold,"fit",True):
        d,g,_=make_data(base,src,pick,rep)
        v=g[valid(d)].astype(np.float64)
        sumg+=v.sum(axis=0)
        gramg+=np.square(v).sum(axis=0)
        count+=len(v)
        ids=np.unique(src["token_for_pixel"][pick])
        # Eight distinct local patches per fit frame/mask/representation, reproducibly sampled.
        ids=rng.choice(ids,min(len(ids),8),replace=False)
        all_tokens.append(unpack(src["token_values"][ids]))
        fit_ids.add(row["id"])
        if len(all_tokens)==128:
            consume()
    consume()
    require(count>0 and n_token>=32,"Not enough fit evidence for PCA/normalization")
    gm=sumg/count
    gs=np.sqrt(np.maximum(gramg/count-gm**2,1e-12))
    gs[gs<1e-5]=1.
    mean=(token_sum/n_token).cpu().numpy()
    cov=gram.cpu().numpy()/n_token-np.outer(mean,mean)
    vals,vec=np.linalg.eigh((cov+cov.T)*.5)
    basis=vec[:,-32:][:,::-1].copy()
    # Canonicalize the arbitrary sign to keep exported projections portable.
    basis*=np.sign(basis[np.abs(basis).argmax(axis=0),np.arange(32)])
    fields=dict(geometry_mean=gm.astype(np.float32),geometry_scale=gs.astype(np.float32),
                token_mean=mean.astype(np.float32),pca_basis=basis.astype(np.float32),
                pca_scale=np.sqrt(np.maximum(vals[-32:][::-1],1e-12)).astype(np.float32))
    atomic_npz(path,**fields)
    save(record,dict(sha256=sha(path),fit_frames=sorted(fit_ids),normalization_candidate_rows=count,
                     pca_fit_local_tokens=n_token,pca_dim=32,fit_role_only=True,
                     source_channels=list(BASE_CHANNELS),explained_variance_ratio=float(vals[-32:].sum()/max(vals.sum(),1e-12))))
    return fields


def project_tokens(data,prep):
    values=unpack(data["token_values"])
    return ((values-prep["token_mean"])@prep["pca_basis"])/prep["pca_scale"]


def normalize(g,prep):
    return (g-prep["geometry_mean"])/prep["geometry_scale"]


def fit_arrays(root,dataset,fold,prep):
    # Keep one fold's bounded training pixels in process memory. Full scoring masks stay on disk.
    # Features used by several candidate types share the same pixel's projected MDM descriptor.
    result=[]
    group_x,group_y,group_p,group_i=[],[],[],[]
    pixels=0
    def flush():
        nonlocal pixels
        if group_x:
            result.append(tuple(np.concatenate(v) for v in (group_x,group_y,group_p,group_i)))
            group_x.clear(); group_y.clear(); group_p.clear(); group_i.clear()
        pixels=0
    for _,_,rep,base,src,pick in iterate(root,dataset,fold,"fit",True):
        d,g,y=make_data(base,src,pick,rep)
        ii,jj=np.nonzero(valid(d))
        if not len(ii):
            continue
        ix,inv=np.unique(src["token_for_pixel"][pick],return_inverse=True)
        projected=project_tokens({"token_values":src["token_values"][ix]},prep)
        group_x.append(normalize(g[ii,jj],prep))
        group_y.append(y[ii,jj].astype(np.uint8))
        group_p.append(projected)
        group_i.append((pixels+inv[ii]).astype(np.int32))
        pixels+=len(projected)
        if sum(len(x) for x in group_y)>=150000:
            flush()
    flush()
    require(result,"No finite candidates in fit")
    return result


def train(root,dataset,fold,folder,prep):
    last=next((k for k in range(30,0,-1) if epoch_file(folder,k).is_file()),0)
    if last==30:
        return
    arrays=fit_arrays(root,dataset,fold,prep)
    models={}
    for name in HEADS:
        torch.manual_seed(42)
        models[name]=network(name).cuda()
    optim={n:torch.optim.AdamW(m.parameters(),lr=1e-3) for n,m in models.items()}
    if last:
        checkpoint=torch.load(epoch_file(folder,last),map_location="cpu",weights_only=True)
        for n in HEADS:
            models[n].load_state_dict(checkpoint["models"][n])
            optim[n].load_state_dict(checkpoint["optimizers"][n])
    total=sum(len(a[1]) for a in arrays)
    print("training fold",fold,"candidate rows/epoch",total,"RAM fit array bytes",
          sum(a.nbytes for shard in arrays for a in shard),flush=True)
    for epoch in range(last+1,31):
        rng=np.random.default_rng(np.random.SeedSequence((42,fold,epoch)))
        cumulative={n:torch.zeros((),device="cuda",dtype=torch.float64) for n in HEADS}
        steps=0
        for j in rng.permutation(len(arrays)):
            geometry,labels,patches,lookup=arrays[j]
            perm=rng.permutation(len(labels))
            for off in range(0,len(perm),8192):
                batch=perm[off:off+8192]
                geom=torch.from_numpy(geometry[batch]).cuda()
                mdm=torch.from_numpy(patches[lookup[batch]]).cuda()
                y=torch.from_numpy(labels[batch].astype(np.int64)).cuda()
                for n in HEADS:
                    optim[n].zero_grad(set_to_none=True)
                    logits=models[n](input_for(n,geom,mdm))
                    loss=nn.functional.cross_entropy(logits,y)
                    loss.backward()
                    optim[n].step()
                    cumulative[n]+=loss.detach()*len(batch)
                steps+=1
        loss={n:float(s/total) for n,s in cumulative.items()}
        require(all(np.isfinite(v) for v in loss.values()),"Nonfinite fit loss")
        save_pt(epoch_file(folder,epoch),dict(fold=fold,epoch=epoch,models={n:copy_weights(m) for n,m in models.items()},
                optimizers={n:o.state_dict() for n,o in optim.items()},fit_candidate_rows=total,updates_this_epoch=steps,
                fit_loss=loss,trainable_parameters={n:sum(p.numel() for p in m.parameters()) for n,m in models.items()}))
        print("fold",fold,"epoch",epoch,"fit losses",loss,flush=True)


def stacked_checkpoints(folder):
    epochs=[torch.load(epoch_file(folder,j),map_location="cpu",weights_only=True)["models"] for j in range(1,31)]
    result={}
    for n in HEADS:
        keys=epochs[0][n].keys()
        result[n]={k:torch.stack([e[n][k] for e in epochs]).cuda() for k in keys}
    return result


def stacked_logits(n,x,weights):
    # First batched matmul broadcasts the common inputs; one streamed calibration read scores all epochs.
    h=torch.matmul(x,weights["0.weight"].transpose(-2,-1))+weights["0.bias"][:,None,:]
    if n=="logistic":
        return h
    h=torch.bmm(h.relu(),weights["2.weight"].transpose(1,2))+weights["2.bias"][:,None,:]
    return torch.bmm(h.relu(),weights["4.weight"].transpose(1,2))+weights["4.bias"][:,None,:]


def complete_batches(root,dataset,fold,role,prep):
    for row,kind,rep,base,src,pick in iterate(root,dataset,fold,role):
        grid=project_tokens(src,prep)
        for off in range(0,len(pick),8192):
            idx=pick[off:off+8192]
            d,g,y=make_data(base,src,idx,rep)
            ii,jj=np.nonzero(valid(d))
            geom=torch.from_numpy(normalize(g[ii,jj],prep)).cuda()
            mdm=torch.from_numpy(grid[src["token_for_pixel"][idx[ii]]]).cuda()
            labels=torch.from_numpy(y[ii,jj]).cuda()
            yield row,kind,rep,base,idx,d,ii,jj,geom,mdm,labels


@torch.no_grad()
def choose_epochs(root,dataset,fold,folder,prep):
    path=folder/"epoch_selection.json"
    if path.exists():
        return read(path)["epoch"]
    weights=stacked_checkpoints(folder)
    sums={n:torch.zeros(30,device="cuda",dtype=torch.float64) for n in HEADS}
    total=0
    for *_,g,m,y in complete_batches(root,dataset,fold,"calibration",prep):
        total+=len(y)
        for start in range(0,len(y),4096):
            sl=slice(start,start+4096)
            target=y[sl][None,:,None].expand(30,-1,1)
            for n in HEADS:
                z=stacked_logits(n,input_for(n,g[sl],m[sl]),weights[n])
                sums[n]-=z.log_softmax(-1).gather(-1,target).sum((1,2),dtype=torch.float64)
    require(total>0,"No full-mask calibration labels")
    values={n:(s/total).cpu().numpy() for n,s in sums.items()}
    best={n:1+int(v.argmin()) for n,v in values.items()}
    save(path,dict(epoch=best,full_calibration_candidate_rows=total,loss_by_epoch={n:v.tolist() for n,v in values.items()}))
    return best


def best_networks(folder,epochs,device="cuda"):
    nets={}
    for n,e in epochs.items():
        net=network(n).to(device)
        d=torch.load(epoch_file(folder,e),map_location="cpu",weights_only=True)
        net.load_state_dict(d["models"][n])
        nets[n]=net.eval()
    return nets


def threshold_from(hist,bad,total):
    acc=hist[::-1].cumsum()[::-1]
    failures=bad[::-1].cumsum()[::-1]
    choices=np.flatnonzero((acc>0)&(failures <= .05*acc))
    target=int(choices[0]) if len(choices) else RESOLUTION+1
    operating={}
    for c in COVERAGES:
        if acc[0]<c*total or total==0:
            operating[str(c)]=None
            continue
        valid_cut=np.flatnonzero(acc>0)
        operating[str(c)]=int(valid_cut[np.argmin(abs(acc[valid_cut]/total-c))])
    return dict(risk5_cut_bin=target,coverage_cut_bins=operating,calibration_denominator=int(total),
                accepted_at_risk5=int(acc[target]) if target<=RESOLUTION else 0,
                errors_at_risk5=int(failures[target]) if target<=RESOLUTION else 0)


def choose_pixel_prob(probs, admissible):
    p=np.where(admissible,probs,-np.inf)
    choice=p.argmax(axis=1)
    selected=p[np.arange(len(p)),choice]
    choice[~np.isfinite(selected)]=-1
    return choice,selected


@torch.no_grad()
def calibrate(root,dataset,fold,folder,prep,epochs):
    path=folder/"calibration.json"
    if path.exists():
        return read(path)
    nets=best_networks(folder,epochs)
    stats={}
    for rep in dataset["representations"]:
        for n in HEADS:
            stats[rep,n]=dict(total=0, nonempty=0, score=np.zeros(len(TEMPERATURES),np.float64),
                       hist=np.zeros((len(TEMPERATURES),RESOLUTION+1),np.int64),
                       bad=np.zeros((len(TEMPERATURES),RESOLUTION+1),np.int64))
    for _,_,rep,base,idx,d,ii,jj,g,m,y in complete_batches(root,dataset,fold,"calibration",prep):
        in_range=primary(d)
        for n,net in nets.items():
            st=stats[rep,n]
            st["total"]+=len(d)
            st["nonempty"]+=len(y)
            if not len(y):
                continue
            logits=net(input_for(n,g,m))
            for j,T in enumerate(TEMPERATURES):
                z=logits/T
                st["score"][j]+=float(nn.functional.cross_entropy(z,y,reduction="sum"))
                p=np.zeros(d.shape,np.float32)
                p[ii,jj]=z.softmax(1)[:,:2].sum(1).cpu().numpy()
                choice,p=choose_pixel_prob(p,in_range)
                accepted=choice>=0
                bins=np.floor(np.clip(p[accepted],0,1)*RESOLUTION).astype(int)
                err=abs(d[np.nonzero(accepted)[0],choice[accepted]]-base["reference"][idx[accepted]])>.10
                st["hist"][j]+=np.bincount(bins,minlength=RESOLUTION+1)
                st["bad"][j]+=np.bincount(bins[err],minlength=RESOLUTION+1)
    result={}
    for rep in dataset["representations"]:
        result[rep]={}
        for n in HEADS:
            s=stats[rep,n]
            require(s["nonempty"]>0,"No valid calibration candidates "+rep)
            j=int(s["score"].argmin())
            result[rep][n]=dict(temperature=TEMPERATURES[j],epoch=epochs[n],nll_grid=(s["score"]/s["nonempty"]).tolist(),
                               temperature_grid=TEMPERATURES,**threshold_from(s["hist"][j],s["bad"][j],s["total"]))
    save(path,result)
    return result


class FrozenQuality:
    """Read one frozen hall fold; external scenes use fold 0 without external tuning."""
    def __init__(self,folder,device="cuda"):
        self.folder=Path(folder)
        self.prep=load_npz(self.folder/"preprocess.npz")
        self.cal=read(self.folder/"calibration.json")
        self.device=device
        self.nets=best_networks(self.folder,read(self.folder/"epoch_selection.json")["epoch"],device)

    @torch.no_grad()
    def probabilities(self,depths,observations,local,audit_bits,projected,rep,name="Q1"):
        geom=feature_table(depths,observations,local,audit_bits,rep)
        i,j=np.nonzero(valid(depths))
        result=np.zeros((*depths.shape,3),np.float32)
        if len(i):
            g=torch.from_numpy(normalize(geom[i,j],self.prep)).to(self.device)
            m=torch.from_numpy(projected[i]).to(self.device)
            z=self.nets[name](input_for(name,g,m))/self.cal[rep][name]["temperature"]
            result[i,j]=z.softmax(1).cumsum(1)[:,:3].cpu().numpy()
        return result

    def select(self,depths,obs,local,audit_bits,projected,rep,name="Q1"):
        probs=self.probabilities(depths,obs,local,audit_bits,projected,rep,name)[:,:,1]
        choice,p=choose_pixel_prob(probs,primary(depths))
        bins=np.floor(np.clip(np.where(np.isfinite(p),p,0),0,1)*RESOLUTION).astype(int)
        choice[bins<self.cal[rep][name]["risk5_cut_bin"]]=-1
        return choice,p


def train_fold(root,fold):
    data=read(root/"freeze.json")
    masked=read(root/"masked_manifest.json")
    require(masked["cases"]==len(data["rows"])*len(data["masks"]),"Incomplete masked forwards")
    folder=root/"quality"/f"fold_{fold}"
    binding=dict(masked_manifest=sha(root/"masked_manifest.json"),freeze=sha(root/"freeze.json"),
                 quality_code=sha(Path(__file__)),fold=fold,seed=42,epochs=30,optimizer="AdamW",lr=.001,
                 hidden=[64,64],representation_slot=32,train_pixel_limit=4096)
    identity=folder/"identity.json"
    if identity.exists():
        require(read(identity)==binding,"Changed quality fit identity")
    else:
        save(identity,binding)
    prep=preprocessing(root,data,fold,folder)
    train(root,data,fold,folder,prep)
    epochs=choose_epochs(root,data,fold,folder,prep)
    calibrate(root,data,fold,folder,prep,epochs)
    print("fold",fold,"calibrated checkpoints",epochs,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=RUN)
    p.add_argument("--fold",type=int,nargs="+",default=list(range(5)),choices=range(5))
    a=p.parse_args()
    torch.set_num_threads(1)
    for k in a.fold:
        train_fold(a.out,k)
