"""Depth errors and coverage on a prediction-independent sensor denominator."""
from collections import defaultdict
import csv
import gzip

import numpy as np


def ratio(n,d):
    return float(n/d) if d else None


def summary(values):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    if not a.size: return dict(n=0,p05=None,p50=None,p95=None,mean=None)
    return dict(n=len(a),p05=float(np.percentile(a,5)),p50=float(np.median(a)),
                p95=float(np.percentile(a,95)),mean=float(a.mean()))


def parts(reference,prediction,fixed,valid):
    support=fixed & valid & np.isfinite(prediction) & (prediction>0)
    delta=(prediction[support]-reference[support]).astype(np.float32)
    rel=float(np.sum(np.abs(delta)/reference[support],dtype=np.float64))
    return int(np.count_nonzero(fixed)),delta,rel


def measures(total,delta,relative_sum):
    d=np.abs(delta)
    n=len(d)
    out=dict(reference_pixels=total,predicted_pixels=n,invalid_pixels=total-n,
             coverage=ratio(n,total),invalid_fraction=ratio(total-n,total),
             mae_m=None,rmse_m=None,abs_rel=None,p50_m=None,p95_m=None,mean_signed_m=None,
             median_signed_m=None)
    if n:
        out.update(mae_m=float(d.mean(dtype=np.float64)),rmse_m=float(np.sqrt(np.mean(np.square(delta,dtype=np.float64)))),
                   abs_rel=relative_sum/n,p50_m=float(np.median(d)),p95_m=float(np.percentile(d,95)),
                   mean_signed_m=float(delta.mean(dtype=np.float64)),median_signed_m=float(np.median(delta)))
    for cm in (5,10,20):
        hits=int(np.count_nonzero(d<=cm/100))
        out[f"hits_{cm}cm"]=hits
        out[f"correct_{cm}cm"]=ratio(hits,total)
    return out


class Accumulator:
    """Pool pixel errors exactly, and keep a separate equal-frame aggregation."""
    def __init__(self):
        self.groups=defaultdict(list)

    def add(self,keys,component):
        total,delta,rel=component
        row=measures(total,delta,rel)
        for key in keys:
            self.groups[key].append((total,delta,rel,row))
        return row

    def finish(self):
        out={}
        for key in list(self.groups):
            records=self.groups.pop(key)
            pooled=measures(sum(v[0] for v in records),np.concatenate([v[1] for v in records]),sum(v[2] for v in records))
            rows=[v[3] for v in records]
            eq={k:summary([r[k] for r in rows])["mean"] for k in
                ("coverage","invalid_fraction","mae_m","rmse_m","abs_rel","p50_m","p95_m","mean_signed_m","median_signed_m",
                 "correct_5cm","correct_10cm","correct_20cm")}
            out[key]=dict(pooled=pooled,equal_frame=eq,frames=len(rows),empty_reference_frames=sum(r["reference_pixels"]==0 for r in rows),
                          failed_frames=sum(r["reference_pixels"]>0 and r["predicted_pixels"]==0 for r in rows),
                          frames_with_error=sum(r["predicted_pixels"]>0 for r in rows))
        return out


def distance_masks(z):
    for lo,hi in ((.3,1),(1,2),(2,4),(4,6)):
        yield f"{lo:g}-{hi:g}m",(z>=lo)&((z<=hi) if hi==6 else (z<hi))


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with gzip.open(path,"wt",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def visibility(projected,target,reference):
    overlap=reference & np.isfinite(projected)&(projected>0)
    tau=np.maximum(.05,.03*target)
    delta=projected[overlap]-target[overlap]
    tol=tau[overlap]
    front=delta < -tol
    behind=delta > tol
    visible=~behind
    return dict(target_reference_pixels=int(reference.sum()),projected_overlap=len(delta),
                agrees=int(np.sum(np.abs(delta)<=tol)),foreground_conflicts=int(front.sum()),
                potentially_occluded=int(behind.sum()),visible_pixels=int(visible.sum()),
                visible_mae_m=float(np.abs(delta[visible]).mean()) if visible.any() else None)
