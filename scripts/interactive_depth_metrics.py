"""Score every frozen mask ray, bootstrap by time group, and publish train-only R3 supervision."""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from interactive_depth_candidates import HallFrame, export_selected, finalize, hard_select
from interactive_depth_geometry import Footprint, SOURCE_NAMES, primary, valid
from interactive_depth_inputs import Hall, RUN, digest_json
from interactive_depth_quality import (BINS,COVERAGES,HEADS,RESOLUTION,FrozenQuality,choose_pixel_prob,
        complete_batches,input_for,iterate,load_npz,project_tokens,threshold_from,train_fold)
from lingbot_common import read,require,save,sha


def safe(a,b):
    return float(a/b) if b else None


def baseline_predictions(d,obs,bits,local):
    # No hidden reference enters confidence or selection. Unattainable target coverage is reported.
    nn=np.where(primary(d[:,0]),0,-1)
    yield "NN_distance", nn, np.where(nn>=0,1./(1.+local[:,0]),-np.inf)
    for name,end in (("hard_sensor_R1",4),("hard_all_R2",5)):
        c,group_bits,_=hard_select(d[:,:end],obs,bits)
        s=np.array([int(i).bit_count() for i in range(1 << 8)],np.uint8)[group_bits]
        yield name,c,np.where(c>=0,1.-1./(1.+s),-np.inf)


def confidence_bins(p):
    return np.floor(np.clip(np.where(np.isfinite(p),p,0),0,1)*RESOLUTION).astype(np.int32)


def hist_choice(choice,p,d,z):
    ok=choice>=0
    bins=confidence_bins(p)[ok]
    wrong=np.abs(d[np.flatnonzero(ok),choice[ok]]-z[ok])>.10
    return np.bincount(bins,minlength=RESOLUTION+1),np.bincount(bins[wrong],minlength=RESOLUTION+1)


def baseline_cuts(root,data,fold):
    stats={}
    for _,_,rep,b,s,idx in iterate(root,data,fold,"calibration"):
        for off in range(0,len(idx),16384):
            j=idx[off:off+16384]
            d=np.column_stack((b["depths"][j],s["mdm_depth"][j]))
            for n,c,p in baseline_predictions(d,b["observations"][j],b["audit_bits"],b["local"][j]):
                key=rep,n
                h,w=hist_choice(c,p,d,b["reference"][j])
                if key not in stats:
                    stats[key]=[0,np.zeros_like(h),np.zeros_like(w)]
                stats[key][0]+=len(j)
                stats[key][1]+=h
                stats[key][2]+=w
    return {rep:{n:threshold_from(*stats[rep,n][1:],stats[rep,n][0])
                 for r,n in stats if r==rep} for rep in data["representations"]}


class Scores:
    def __init__(self):
        self.raw=defaultdict(lambda:np.zeros(8,np.float64))
        self.cal=defaultdict(lambda:np.zeros((3,15,4),np.float64))
        self.ops=defaultdict(lambda:np.zeros(7,np.int64))
        self.curves=defaultdict(lambda:np.zeros((2,RESOLUTION+1),np.int64))
        self.count=defaultdict(lambda:np.zeros(5,np.int64))

    def raw_scores(self,prefix,d,z,side):
        self.count[prefix]+=np.asarray([len(z),int((side==1).sum()),int((side==2).sum()),
                                        int((side==0).sum()),int(primary(d).any(axis=1).sum())])
        for c,n in enumerate(SOURCE_NAMES):
            pred=d[:,c]
            v=valid(pred)
            e=np.abs(pred[v]-z[v])
            self.raw[prefix,n]+=np.array([len(z),v.sum(),primary(pred).sum(),e.sum(),(e<=.05).sum(),
                                        (e<=.1).sum(),(e<=.2).sum(),(e>.2).sum()])

    def calibration(self,prefix,name,probs,d,z):
        dist=np.searchsorted([1.,2.,4.],z)
        for c,typ in enumerate(SOURCE_NAMES):
            v=valid(d[:,c])
            e=np.abs(d[:,c]-z)
            for k in range(4):
                sub=v&(dist==k)
                if not sub.any():
                    continue
                arr=self.cal[prefix,name,typ,k]
                for b,threshold in enumerate(BINS):
                    p=probs[sub,c,b].astype(np.float64)
                    y=e[sub]<=threshold
                    i=np.minimum((p*15).astype(int),14)
                    for col,values in enumerate((None,p,y.astype(np.float64),(p-y)**2)):
                        arr[b,:,col]+=np.bincount(i,weights=values,minlength=15)

    def chosen(self,prefix,group,method,c,p,d,z,cut):
        bins=confidence_bins(p)
        h,w=hist_choice(c,p,d,z)
        self.curves[prefix,method]+=np.stack((h,w))
        possible=(primary(d)&(abs(d-z[:,None])<=.1)).any(axis=1)
        error=np.full(len(c),np.inf)
        selected=c>=0
        error[selected]=np.abs(d[np.flatnonzero(selected),c[selected]]-z[selected])
        points={"risk_5pct":cut["risk5_cut_bin"],**cut["coverage_cut_bins"]}
        for point,threshold in points.items():
            if threshold is None:
                # Expected coverage can be unattainable on one fold and attainable on another.
                self.ops[prefix,group,method,point]+=np.asarray([len(c),0,0,0,int(possible.sum()),len(c),0])
                continue
            accept=selected & (bins>=threshold)
            correct=accept & (error<=.1)
            self.ops[prefix,group,method,point]+=np.array([len(c),accept.sum(),correct.sum(),
                      (accept & (error>.1)).sum(),(~accept & possible).sum(),0,
                      (selected & ~accept & (error<=.1)).sum()])

    def report(self):
        raw=[dict(mask=k[0][0],rep=k[0][1],candidate=k[1],reference=int(v[0]),finite_positive=int(v[1]),
                  primary_range=int(v[2]),mae_m=safe(v[3],v[1]),correct_coverage_5_10_20cm=[safe(x,v[0]) for x in v[4:7]],
                  over20cm=int(v[7])) for k,v in self.raw.items()]
        strata=[]
        for (p,n,typ,k),v in self.cal.items():
            bins=[]
            for arr in v:
                N=arr[:,0].sum()
                bins.append(dict(n=int(N),Brier=safe(arr[:,3].sum(),N),ECE=safe(abs(arr[:,1]-arr[:,2]).sum(),N),
                                 reliability_bins=[[int(x[0]),float(x[1]),int(x[2])] for x in arr]))
            strata.append(dict(mask=p[0],rep=p[1],head=n,candidate=typ,depth_bin=k,metrics_5_10_20cm=bins))
        grouped=[dict(mask=p[0],rep=p[1],group=int(group),method=n,point=op,counts=v.tolist())
                 for (p,group,n,op),v in self.ops.items()]
        curves=[dict(mask=p[0],rep=p[1],method=n,total=int(self.count[p][0]),hist=v[0].tolist(),wrong=v[1].tolist())
                for (p,n),v in self.curves.items()]
        return dict(mask_denominators=[dict(mask=k[0],rep=k[1],counts=v.tolist()) for k,v in self.count.items()],
                    denominator_fields=["score","foreground_side","background_side","other","has_primary_range_candidate"],
                    raw=raw,calibration=strata,group_operations=grouped,curves=curves,
                    group_count_fields=["all","accepted","correct_at_10cm","incorrect_at_10cm",
                                        "rejected_with_correct_candidate","unreachable_calibration_denominator",
                                        "rejected_with_correct_prethreshold_choice"],
                    depth_bins_m=["[0.3,1]","(1,2]","(2,4]","(4,6]"])


@torch.no_grad()
def score_fold(root,data,fold,folder):
    out=folder/"outer_scores.json"
    if out.exists():
        return
    q=FrozenQuality(folder)
    scores=Scores()
    baseline=baseline_cuts(root,data,fold)
    save(folder/"baseline_calibration.json",baseline)
    for row,kind,rep,b,idx,d,ii,jj,g,m,_ in complete_batches(root,data,fold,"outer",q.prep):
        prefix=(kind,rep)
        z=b["reference"][idx]
        scores.raw_scores(prefix,d,z,b["edge_side"][idx])
        for n in HEADS:
            probs=np.zeros((*d.shape,3),np.float32)
            if len(ii):
                logits=q.nets[n](input_for(n,g,m))/q.cal[rep][n]["temperature"]
                probs[ii,jj]=logits.softmax(1).cumsum(1)[:,:3].cpu().numpy()
            scores.calibration(prefix,n,probs,d,z)
            c,p=choose_pixel_prob(probs[:,:,1],primary(d))
            scores.chosen(prefix,row["group"],n,c,p,d,z,q.cal[rep][n])
        for n,c,p in baseline_predictions(d,b["observations"][idx],b["audit_bits"],b["local"][idx]):
            scores.chosen(prefix,row["group"],n,c,p,d,z,baseline[rep][n])
    save(out,dict(fold=fold,calibration_sha256=sha(folder/"calibration.json"),**scores.report()))


def export_fold(root,hall,fold,folder):
    q=FrozenQuality(folder)
    fp=Footprint(hall)
    rep="footprint_aware" if "footprint_aware" in hall.freeze["representations"] else "full_range_sparse"
    binding=digest_json(dict(head=sha(folder/"calibration.json"),preprocessing=sha(folder/"preprocess.npz"),
                             selected_checkpoint=sha(folder/"epochs"/f"epoch_{q.cal[rep]['Q1']['epoch']:02d}.pt"),
                             natural=sha(root/"natural_manifest.json"),code=sha(Path(__file__))))
    records=[]
    for row in hall.rows:
        if row["fold"]!=fold:
            continue
        fid=row["id"]
        rec=root/"supervision/R3/receipts"/(fid+".json")
        if rec.exists():
            r=read(rec)
            require(r["identity"]==binding and sha(rec.parent.parent/r["file"]["path"])==r["file"]["sha256"],
                    "R3 resume identity")
            records.append(r["file"])
            continue
        frame=HallFrame(hall,fp,row)
        data=frame.condition()
        flat=np.flatnonzero(frame.natural)
        src=load_npz(root/"cases"/fid/"natural"/(rep+".npz"))
        require(len(src["mdm_depth"])==len(flat),"Natural mask identity")
        mdm=np.zeros(frame.full.shape,np.float32)
        mdm.flat[flat]=src["mdm_depth"]
        patches=project_tokens(src,q.prep)
        def choose(sub,ds,obs,local):
            idx=np.searchsorted(flat,sub)
            loc=patches[src["token_for_pixel"][idx]]
            return q.select(ds,obs,local,frame.other.audit_bits,loc,rep)
        file=export_selected(root/"supervision/R3",frame,data,mdm,choose)
        save(rec,dict(identity=binding,fold=fold,file=file))
        records.append(file)
    save(folder/"R3_frames.json",dict(identity=binding,frames=records))


def aggregate(root,hall):
    units=defaultdict(lambda:defaultdict(lambda:np.zeros(7,np.int64)))
    combined={}
    rows=[]
    fold_refs=[]
    for k in range(5):
        f=root/"quality"/f"fold_{k}"
        r=read(f/"outer_scores.json")
        fold_refs.append(dict(fold=k,score_sha256=sha(f/"outer_scores.json"),calibration_sha256=sha(f/"calibration.json"),
                              preprocessing_sha256=sha(f/"preprocess.npz"),epochs=read(f/"epoch_selection.json")["epoch"]))
        for e in r["group_operations"]:
            key=e["mask"],e["rep"],e["method"],e["point"]
            units[key][e["group"]]+=np.asarray(e["counts"])
        for e in r["curves"]:
            key=e["mask"],e["rep"],e["method"]
            if key not in combined:
                combined[key]=[0,np.zeros(RESOLUTION+1,np.int64),np.zeros(RESOLUTION+1,np.int64)]
            combined[key][0]+=e["total"]
            combined[key][1]+=e["hist"]
            combined[key][2]+=e["wrong"]
        rows.extend(read(f/"R3_frames.json")["frames"])
    ops=[]
    for key,g in units.items():
        mat=np.stack([v for _,v in sorted(g.items())])
        total,acc,correct,bad,reject,unreachable,lost=mat.sum(axis=0)
        rng=np.random.default_rng(42)
        res=mat[rng.integers(0,len(mat),(1000,len(mat)))].sum(axis=1)
        def interval(a,b):
            good=res[:,b]>0
            return np.quantile(res[good,a]/res[good,b],[.025,.975]).tolist() if good.any() else None
        ops.append(dict(mask=key[0],rep=key[1],method=key[2],point=key[3],n=int(total),accepted=int(acc),
                   correct=int(correct),wrong=int(bad),coverage=safe(acc,total),risk=safe(bad,acc),correct_coverage=safe(correct,total),
                   coverage_ci95=interval(1,0),risk_ci95=interval(3,1),correct_coverage_ci95=interval(2,0),
                   groups=len(mat),rejected_but_correct_available=int(reject),unreachable_denominator=int(unreachable),
                   rejected_chosen_correct=int(lost)))
    curves=[]
    for key,(total,h,w) in combined.items():
        accepted=h[::-1].cumsum()[::-1]
        wrong=w[::-1].cumsum()[::-1]
        curves.append(dict(mask=key[0],rep=key[1],method=key[2],n=int(total),
                    by_threshold=[dict(prob=t/RESOLUTION,accepted=int(accepted[t]),coverage=safe(accepted[t],total),
                                       risk=safe(wrong[t],accepted[t])) for t in range(0,RESOLUTION+1,25)]))
    save(root/"quality/summary.json",dict(folds=fold_refs,operations=ops,risk_coverage=curves,
          heads=list(HEADS),bootstrap="1000 resamples with replacement of complete ten-second acquisition groups, seed 42",
          zero_acceptance_risk=None,role="OOF held-out train measurements; hall sensor reference"))
    lookup={r["id"]:r for r in rows}
    require(len(lookup)==len(rows)==1478,"Full OOF natural export")
    finalize(root,[dict(identity="fold-specific Q1; see receipts and quality summary",
             files={"supervision/R3":lookup[r["id"]]}) for r in hall.rows],hall.freeze)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=RUN)
    p.add_argument("--fold",type=int,nargs="+",default=list(range(5)),choices=range(5))
    a=p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    hall=Hall(a.out)
    for k in a.fold:
        train_fold(a.out,k)
        folder=a.out/"quality"/f"fold_{k}"
        score_fold(a.out,hall.freeze,k,folder)
        export_fold(a.out,hall,k,folder)
    if all((a.out/"quality"/f"fold_{k}"/"R3_frames.json").exists() for k in range(5)):
        aggregate(a.out,hall)
