"""Recompute the capture-full quantitative interpretation from frozen local Git objects."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULT="7d4a8a1d7d47379102cc5a17d2bbb7e07231ac42"
B0="reports/capture-full-dn-30k/20261119-v1"
B1="reports/capture-full-metric3d-30k/20261120-v1"
OUTPUT=ROOT/"reports/capture-full-quality/2026-11-21"
RGB=("psnr","ssim","lpips")
ERROR=("MAE_m","RMSE_m","AbsRel","absolute_error_P50_m","absolute_error_P95_m")
COUNTS=("reference","hits","misses","correct_05","correct_10","correct_20","correct_tau","foreground","nearer_than_03")
METHODS=("model","raw_pcd","fused_pcd","raw_mesh","pre_crop_mesh","supported_mesh")


def require(ok,what):
    if not ok:raise ValueError(what)


def close(a,b):
    return np.isclose(a,b,rtol=2e-8,atol=2e-9)


def calculate():
    fingerprints={}
    def source(parent,name):
        rel=parent+"/"+name
        blob=subprocess.check_output(["git","show",RESULT+":"+rel],cwd=ROOT)
        fingerprints[rel]=hashlib.sha256(blob).hexdigest()
        return json.loads(gzip.decompress(blob) if name.endswith(".gz") else blob)
    summary=dict(result_commit=RESULT,splits={})
    paired_ref=source(B1,"paired_summary.json")
    cameras=source(B1,"frozen_inputs.json")
    srcs=set()
    for split in ("val","test"):
        labels=("B0","B1")
        hold={k:source(repo,f"metrics_{split}.json") for k,repo in zip(labels,(B0,B1))}
        paired=source(B1,f"paired_{split}.json")
        frames=paired["frames"]
        ids=[r["id"] for r in frames]
        require(len(ids)==len(set(ids))==184 and not (srcs & set(ids)),split+" paired identities")
        srcs.update(ids)
        images=184*720*1280
        by_surface=source(B1,f"surface_{split}_metrics.json")
        per_frame=source(B1,f"surface_{split}_frames.json.gz")
        require([r["id"] for r in per_frame]==ids,split+" mesh/model source frames")
        surfaces={}
        for label in labels:
            for key in [label+"_"+m for m in METHODS]+["common_supported_models__"+label+"_supported_mesh"]:
                sums=by_surface[key]["pixel_pooled"]
                rows=[r["metrics"][key] for r in per_frame]
                for k in COUNTS:require(sum(row[k] for row in rows)==sums[k],key+" "+k)
                require(sums["hits"]+sums["misses"]==sums["reference"] and
                        0<=sums["correct_05"]<=sums["correct_10"]<=sums["correct_20"]<=sums["hits"] and
                        sums["correct_05"]<=sums["correct_tau"]<=sums["correct_20"] and
                        sums["correct_tau"]+sums["foreground"]<=sums["hits"],key+" counts")
                fit={k:sums[k] for k in COUNTS}
                for k in COUNTS[1:]:
                    fit[k+"_pct_reference"]=100*sums[k]/sums["reference"]
                for k in ("MAE_hit_m","RMSE_hit_m","P50_hit_m","P95_hit_m","signed_mean_hit_m"):
                    fit[k]=sums[k]
                require(close(sum(r["abs_sum"] for r in rows)/sums["hits"],sums["MAE_hit_m"]),key+" hit accuracy")
                surfaces[key]=fit
        models={}
        for label in labels:
            src=hold[label]; r=src["frames"]
            require([x["id"] for x in r]==ids,label+" scale")
            require(all(x==peer[label] for x,peer in zip(r,frames)),label+" paired evidence")
            n=sum(x["reference_pixels"] for x in r)
            invalid=sum(x["invalid_prediction_pixels"] for x in r)
            require(n==src["reference_pixels"] and invalid==src["invalid_prediction_pixels"]==0,split+" denominator")
            means={k:float(np.mean([x["rgb"][k] for x in r])) for k in RGB}
            for k in RGB:require(close(means[k],src["rgb_frame_means"][k]),label+" "+k)
            pooled=src["depth_pixel_pooled"]
            for k in ("MAE_m","RMSE_m","AbsRel"):
                vals=np.asarray([x["depth_full_reference"][k] for x in r])
                w=[x["reference_pixels"] for x in r]
                observed=np.sqrt(np.average(vals**2,weights=w)) if k=="RMSE_m" else np.average(vals,weights=w)
                require(close(observed,pooled[k]),label+" pooled "+k)
            for k,other in (("MAE_m","MAE_hit_m"),("RMSE_m","RMSE_hit_m"),
                            ("absolute_error_P50_m","P50_hit_m"),("absolute_error_P95_m","P95_hit_m")):
                require(close(pooled[k],surfaces[label+"_model"][other]),label+" independent depth summary "+k)
            models[label]=dict(rgb=means,depth=pooled,frame_weighted_depth=src["depth_frame_means"],invalid=invalid,
                         reference=n,reference_share_of_all_rgb=n/images,reference_alpha_zero=src["reference_alpha_zero"])
        require(models["B0"]["reference"]==models["B1"]["reference"],"same reference")
        changes={}
        for k in (*RGB,*ERROR):
            delta=np.array([r["delta"][k] for r in frames])
            sign=1 if k in ("psnr","ssim") else -1
            direct=[r["B1"]["rgb"][k]-r["B0"]["rgb"][k] if k in RGB else
                    r["B1"]["depth_full_reference"][k]-r["B0"]["depth_full_reference"][k] for r in frames]
            require(np.allclose(delta,direct,rtol=1e-10,atol=1e-10),k+" per-frame delta")
            wins=int((sign*delta>0).sum()); losses=int((sign*delta<0).sum())
            require((wins,losses)==(paired_ref[split]["changes"][k]["improved"],paired_ref[split]["changes"][k]["degraded"]),k+" direction")
            changes[k]=dict(improved=wins,degraded=losses,tied=int((delta==0).sum()),median_B1_minus_B0=float(np.median(delta)),
                            mean_B1_minus_B0=float(delta.mean()))
        joint=sum(r["delta"]["psnr"]>0 and r["delta"]["MAE_m"]<0 for r in frames)
        windows=[]
        for window in paired["windows"]:
            lo,hi=window["start_s"],window["end_exclusive_s"]
            group=[r for r in frames if lo<=r["relative_s"]<hi]
            n=sum(r["reference_pixels"] for r in group)
            obs={}
            for label in labels:
                z=sum(r[label]["depth_full_reference"]["MAE_m"]*r["reference_pixels"] for r in group)/n
                psnr=np.mean([r[label]["rgb"]["psnr"] for r in group])
                require(window[label]["evaluated"]==len(group) and n==window[label]["reference_pixels"] and
                        close(z,window[label]["depth_pixel_pooled"]["MAE_m"]) and close(psnr,window[label]["rgb_frame_means"]["psnr"]),f"{split} time {lo}")
                obs[label]=dict(MAE_m=float(z),PSNR_dB=float(psnr))
            windows.append(dict(start_s=lo,end_exclusive_s=hi,frames=len(group),reference=n,**obs,
                                absolute_error_sum_saved_m=(obs["B0"]["MAE_m"]-obs["B1"]["MAE_m"])*n))
        net=sum(w["absolute_error_sum_saved_m"] for w in windows)
        require(close(net,(models["B0"]["depth"]["MAE_m"]-models["B1"]["depth"]["MAE_m"])*models["B1"]["reference"]),
                "net pixel error improvement")
        early=paired["s01_same_sources"]
        require(len(early)==len({r["id"] for r in early})==24,"s01 subset")
        common={l:float(np.mean([r["s01"]["psnr"] if l=="s01" else r[l]["rgb"]["psnr"] for r in early])) for l in ("s01","B0","B1")}
        common.update(improved=sum(r["delta"]["psnr"]>0 for r in early),degraded=sum(r["delta"]["psnr"]<0 for r in early))
        prior=source(B1,"prior_diagnostic_"+split+".json")["summary"]
        summary["splits"][split]=dict(frames=len(frames),rgb_pixels=images,models=models,surfaces=surfaces,changes=changes,
            jointly_improved_PSNR_and_MAE=joint,windows=windows,
            better_MAE_windows=sum(w["B1"]["MAE_m"]<w["B0"]["MAE_m"] for w in windows),
            better_PSNR_windows=sum(w["B1"]["PSNR_dB"]>w["B0"]["PSNR_dB"] for w in windows),
            fraction_of_net_depth_absolute_error_reduction_in_180_210_and_270_300=sum(w["absolute_error_sum_saved_m"] for w in windows if w["start_s"] in (180,270))/net,
            old_s01_24=common,prior_depth=prior["conditional_pooled"],prior_sensor_normal_samples=prior["normal_sensor_samples"])
    local={}
    for target,r in source(B1,"surface_local.json").items():
        a,b=[r["metrics"]["common_supported_models__"+l+"_supported_mesh"] for l in ("B0","B1")]
        require((a["reference"],a["hits"])==(b["reference"],b["hits"]),target)
        local[target]=dict(train_frame=r["stem"],reference=a["reference"],common_hits=a["hits"],
                           B0_MAE_m=a["MAE_hit_m"],B1_MAE_m=b["MAE_hit_m"])
    summary["fixed_training_crops"]=local
    summary["gaussians"]={k:[(p["step"],p["gaussians"]) for p in source(B1,k+"_training_curve.json")["checkpoints"]] for k in ("B0","B1")}
    summary["initial_gaussians"]=cameras["point_count"]
    summary["largest_B1_net_drop"]=min(source(B1,"B1_refinement.json")["events"],key=lambda x:x["after"]-x["before"])
    summary["source_sha256"]=fingerprints
    return summary


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for family in ("Microsoft YaHei","Noto Sans CJK SC","Source Han Sans CN","SimHei"):
        try:
            fm.findfont(family,fallback_to_default=False)
        except ValueError:
            continue
        break
    else:raise RuntimeError("Install a CJK font or supply a matching Matplotlib font family.")
    plt.rcParams.update({"font.family":family,"font.size":10,"pdf.fonttype":42,"axes.spines.top":False,
        "axes.spines.right":False,"axes.unicode_minus":False,"axes.edgecolor":"#526374","text.color":"#1d2f40",
        "axes.labelcolor":"#1d2f40","xtick.color":"#425567","ytick.color":"#425567"})
    fig,(a,b)=plt.subplots(1,2,figsize=(10.4,4.25),layout="constrained",gridspec_kw={"width_ratios":[1.08,1]})
    d=summary["splits"]["test"];wins=d["windows"]
    times=[(w["start_s"]+min(w["end_exclusive_s"],369.34))*.5 for w in wins]
    for label,col,ls,m,legend in (("B0","#646a79","--","o","B0  Omnidata"),("B1","#007777","-","s","B1  Metric3D")):
        a.plot(times,[w[label]["MAE_m"] for w in wins],ls,marker=m,color=col,markersize=4.3,linewidth=1.7,label=legend)
    a.set(xlim=(0,375),ylim=(0,1.2),xticks=list(range(0,361,60)),yticks=np.arange(0,1.21,.2),
          xlabel="采集时间（秒）",ylabel="传感器深度 MAE（米）")
    a.set_title("a   模型深度误差随时间的变化",loc="left",fontweight="bold",fontsize=12,pad=10)
    a.grid(axis="y",color="#dfe5e9",linewidth=.65);a.set_axisbelow(True)
    a.legend(loc="upper left",frameon=False,fontsize=8.5,handlelength=2.25)
    series=[("B0_model","B0 模型"),("B1_model","B1 模型"),("B0_raw_mesh","B0 原始网格"),
            ("B1_raw_mesh","B1 原始网格"),("B0_supported_mesh","B0 支持网格"),("B1_supported_mesh","B1 支持网格")]
    ys=np.asarray([5.7,4.7,3.3,2.3,.9,-.1])
    for yy,(key,_) in zip(ys,series):
        row=d["surfaces"][key]
        xs=[row["correct_"+k+"_pct_reference"] for k in ("05","10","20")]
        b.plot(xs,[yy+.17,yy,yy-.17],color="#bec8cd",linewidth=.8,zorder=1)
    for k,off,c,m,text in (("05",.17,"#007777","o","5 cm"),("10",0,"#c06b23","s","10 cm"),("20",-.17,"#464a7b","^","20 cm")):
        b.scatter([d["surfaces"][key]["correct_"+k+"_pct_reference"] for key,_ in series],ys+off,
                  color=c,marker=m,s=30,zorder=3,label=text)
    b.set(ylim=(-.65,6.35),xlim=(0,100),xticks=np.arange(0,101,20),yticks=ys,yticklabels=[s[1] for s in series],
          xlabel="全部有效参考像素中的正确覆盖率（%）")
    b.set_title("b   不同误差容限下的正确覆盖率",loc="left",fontweight="bold",fontsize=12,pad=10)
    b.grid(axis="x",color="#dfe5e9",linewidth=.65);b.set_axisbelow(True)
    b.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(.56,.98),fontsize=8,
             handletextpad=.2,columnspacing=.7,borderaxespad=0.)
    return fig


if __name__=="__main__":
    OUTPUT.mkdir(parents=True,exist_ok=True)
    summary=calculate()
    (OUTPUT/"numbers.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    fig=plot(summary)
    fig.savefig(OUTPUT/"quality.png",dpi=300)
    fig.savefig(OUTPUT/"quality.pdf")
