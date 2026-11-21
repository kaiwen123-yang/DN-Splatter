"""Local post-archive depth stratification and paired sensitivity analysis; no model or server access."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from capture_metric3d_scorecard import ROOT, RESULT, B0, B1, OUTPUT, RGB, ERROR, close, require

BANDS=((.3,1.),(1.,2.),(2.,4.),(4.,6.))
A_RUN="capture-full-dn-30k-20261119-v1"
B_RUN="capture-full-metric3d-30k-20261120-v1"


def frozen(folder,name):
    return subprocess.check_output(["git","show",RESULT+":"+folder+"/"+name],cwd=ROOT)


def stats(ref,z):
    err=np.asarray(z,dtype=np.float64)-np.asarray(ref,dtype=np.float64)
    abs_e=np.abs(err)
    return dict(hits=len(err),abs_sum=float(abs_e.sum()),sq_sum=float(np.dot(err,err)),
                rel_sum=float(np.sum(abs_e/ref)),signed_sum=float(err.sum()),correct_05=int((abs_e<=.05).sum()),
                correct_10=int((abs_e<=.10).sum()),correct_20=int((abs_e<=.20).sum()),
                over_1m=int((abs_e>1).sum()),absolute_error_sum_over_1m=float(abs_e[abs_e>1].sum())),abs_e


def depth_bands(root,split,pairs,reference):
    run=root/"project/.local/runs"
    labels=("B0","B1")
    buffers=[[[] for _ in labels] for _ in BANDS]; totals=[[{} for _ in labels] for _ in BANDS]
    refs=[0]*len(BANDS)
    for r in pairs:
        name=r["id"]+".npy"
        ref=np.load(run/B_RUN/"inputs/depth"/name,allow_pickle=False)
        good=np.isfinite(ref)&(ref>=.3)&(ref<=6.)
        ref=ref[good]
        require(len(ref)==r["reference_pixels"],"archived reference "+name)
        masks=[(ref>=lo)&((ref<hi) if j<3 else (ref<=hi)) for j,(lo,hi) in enumerate(BANDS)]
        require(sum(int(m.sum()) for m in masks)==len(ref),name+" band coverage")
        for j,mask in enumerate(masks):refs[j]+=int(mask.sum())
        for li,(label,run_id) in enumerate(zip(labels,(A_RUN,B_RUN))):
            z=np.load(run/run_id/"renders"/split/"depth"/name,allow_pickle=False)[good]
            require(np.all(np.isfinite(z)&(z>0)),label+" unexpected invalid prediction")
            for j,mask in enumerate(masks):
                agg,errors=stats(ref[mask],z[mask])
                buf=totals[j][li]
                for k,v in agg.items():buf[k]=buf.get(k,0)+v
                buffers[j][li].append(errors)
    require(sum(refs)==reference["models"]["B1"]["reference"],split+" complete depth range denominator")
    n_all=sum(refs);rows=[]
    for j,((lo,hi),n) in enumerate(zip(BANDS,refs)):
        require(n>0,"no depth support in analysis band")
        row=dict(low_m=lo,high_m=hi,high_inclusive=(j==len(BANDS)-1),reference=n,share=n/n_all)
        for li,label in enumerate(labels):
            v=totals[j][li]
            err=np.concatenate(buffers[j][li]); buffers[j][li].clear()
            p50,p95,p99=np.percentile(err,[50,95,99],overwrite_input=True)
            require(len(err)==n==v["hits"],split+" "+label)
            row[label]=dict(**v,MAE_m=v["abs_sum"]/n,RMSE_m=float(np.sqrt(v["sq_sum"]/n)),AbsRel=v["rel_sum"]/n,
                signed_bias_m=v["signed_sum"]/n,P50_m=float(p50),P95_m=float(p95),P99_m=float(p99),
                fraction_over_1m=v["over_1m"]/n,mean_error_contribution_over_1m=v["absolute_error_sum_over_1m"]/n,
                coverage_05=v["correct_05"]/n,coverage_10=v["correct_10"]/n,coverage_20=v["correct_20"]/n)
            del err
        row["absolute_error_sum_saved_m"]=row["B0"]["abs_sum"]-row["B1"]["abs_sum"]
        rows.append(row)
    net=sum(r["absolute_error_sum_saved_m"] for r in rows)
    for r in rows:r["share_of_net_absolute_error_saved"]=r["absolute_error_sum_saved_m"]/net
    for label in labels:
        model=reference["models"][label]["depth"]
        for k,num in (("MAE_m","abs_sum"),("AbsRel","rel_sum"),("RMSE_m","sq_sum")):
            value=sum(r[label][num] for r in rows)/n_all
            if k=="RMSE_m":value=np.sqrt(value)
            require(close(value,model[k]),split+" stratification does not reconcile: "+k)
        surf=reference["surfaces"][label+"_model"]
        for k in ("correct_05","correct_10","correct_20"):
            require(sum(r[label][k] for r in rows)==surf[k],split+" coverage discrepancy")
    return rows


def excluding(windows,omit):
    group=[w for w in windows if w["start_s"] not in omit]
    nf=sum(w["frames"] for w in group); nr=sum(w["reference"] for w in group)
    row=dict(omit_window_starts_s=sorted(omit),frames=nf,reference=nr)
    for label in ("B0","B1"):
        row[label]=dict(MAE_m=sum(w[label]["MAE_m"]*w["reference"] for w in group)/nr,
                        PSNR_dB=sum(w[label]["PSNR_dB"]*w["frames"] for w in group)/nf)
    row["MAE_B1_minus_B0_m"]=row["B1"]["MAE_m"]-row["B0"]["MAE_m"]
    row["MAE_relative_drop"]=1-row["B1"]["MAE_m"]/row["B0"]["MAE_m"]
    row["PSNR_B1_minus_B0_dB"]=row["B1"]["PSNR_dB"]-row["B0"]["PSNR_dB"]
    return row


def further(archive):
    receipt=json.loads((archive/"archive_complete.json").read_bytes())
    require(receipt["status"]=="complete_verified" and receipt["result_commit"]==RESULT,"verified H archive required")
    baseline=json.loads((OUTPUT/"numbers.json").read_bytes())
    require(baseline["result_commit"]==RESULT,"scorecard identity")
    out=dict(result_commit=RESULT,reference_depth_bins_m=[[lo,hi] for lo,hi in BANDS],
             protocol="Secondary descriptive analysis on measured model Z and frozen sensor reference. No retraining or scale fit.",
             frozen_inventory_sha256=json.loads((archive/"verification/catalog.json").read_bytes())["inventory"]["sha256"],splits={})
    for split in ("val","test"):
        paired=json.loads(frozen(B1,f"paired_{split}.json"))
        frames=paired["frames"];s=baseline["splits"][split]; windows=s["windows"]
        depth=depth_bands(archive,split,frames,s)
        q={}
        for k in (*RGB,*ERROR):
            values=[r["delta"][k] for r in frames]
            q[k]=dict(zip(("min","p25","p50","p75","max"),map(float,np.percentile(values,[0,25,50,75,100]))))
        counts=dict(both=0,PSNR_only=0,depth_MAE_only=0,neither=0)
        for r in frames:
            a=r["delta"]["psnr"]>0;b=r["delta"]["MAE_m"]<0
            counts["both" if a and b else "PSNR_only" if a else "depth_MAE_only" if b else "neither"]+=1
        extremes={}
        for metric,where in (("psnr","rgb"),("MAE_m","depth_full_reference")):
            def vals(x):
                return dict(id=x["id"],relative_s=x["relative_s"],reference=x["reference_pixels"],
                            B0=x["B0"][where][metric],B1=x["B1"][where][metric],delta=x["delta"][metric])
            extremes[metric]=dict(min_delta=vals(min(frames,key=lambda r:r["delta"][metric])),
                                   max_delta=vals(max(frames,key=lambda r:r["delta"][metric])),
                                   worst_B1=vals((min if metric=="psnr" else max)(frames,key=lambda r:r["B1"][where][metric])))
        out["splits"][split]=dict(depth_bands=depth,paired_delta_quantiles=q,joint_frame_counts=counts,
            leave_one_window_out=[excluding(windows,{w["start_s"]}) for w in windows],
            without_180_210_and_270_300=excluding(windows,{180,270}),extreme_frames=extremes)
    rows=list(csv.DictReader(io.StringIO(gzip.decompress(frozen(B1,"B1_samples.csv.gz")).decode())))
    require(len(rows)==30000 and [int(r["step"]) for r in rows]==list(range(30000)),"training sampling identity")
    from collections import Counter
    count=Counter(r["frame"] for r in rows);before=Counter(r["frame"] for r in rows[:15000])
    out["training_exposure"]=dict(distinct_views=len(count),updates=len(rows),distribution={str(k):v for k,v in sorted(Counter(count.values()).items())},
            before_stop_split_distribution={str(k):v for k,v in sorted(Counter(before.values()).items())})
    (OUTPUT/"detailed_numbers.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive",type=Path)
    further(ap.parse_args().archive)
