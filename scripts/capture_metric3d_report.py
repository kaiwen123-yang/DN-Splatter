"""Publish measured comparison results in Chinese, with exact denominators and bulk checksums."""
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from capture_metric3d_common import (BASE,B0,B0_REPORT,GEOMETRY,HOLDOUT,P,PRIOR,RAW_PREFIX,REPORT,
                                     ROOT,RUN,SCENE,SOURCE,STATE,WEIGHT,demand,member,read,sha,write)
from capture_s01_publish import timings, train_stats


def curves():
    fig, axes = plt.subplots(2,1,figsize=(8,6),sharex=True,layout="constrained")
    for label,col in (("B0","#3468a1"),("B1","#bd5a24")):
        values = {}
        with gzip.open(REPORT/f"{label}_tensorboard.csv.gz","rt") as f:
            for row in csv.DictReader(f):
                if row["tag"] in ("Train Loss","Train Metrics Dict/gaussian_count"):
                    values.setdefault(row["tag"],{})[int(row["step"])]=float(row["value"])
        for ax,key in zip(axes,("Train Metrics Dict/gaussian_count","Train Loss")):
            records=sorted((s,v) for s,v in values.get(key,{}).items() if np.isfinite(v))
            if not records:
                continue
            x,y=np.asarray(records).T
            ax.plot(x,y,color=col,alpha=.7,linewidth=.75,linestyle="-" if label=="B0" else "--",label=label)
        cs=read(REPORT/f"{label}_training_curve.json")["checkpoints"]
        axes[0].scatter([r["step"] for r in cs],[r["gaussians"] for r in cs],color=col,s=12,marker="o" if label=="B0" else "s")
    for ax in axes:
        ax.grid(axis="y",alpha=.2)
        ax.legend()
    axes[0].set_ylabel("Gaussians")
    axes[1].set_ylabel("Logged training loss")
    axes[1].set_xlabel("Training step")
    fig.savefig(REPORT/"previews/training_curves.png",dpi=160)
    plt.close(fig)


def manifest():
    inventory=REPORT/"artifact_files.jsonl.gz"
    count=size=0
    with gzip.open(inventory,"wt",encoding="utf-8",compresslevel=4) as f:
        for folder in ("inputs","metric3d","training","renders","video","mesh","supported","surface_evaluation"):
            for p in sorted((RUN/folder).rglob("*")):
                if p.is_file():
                    row=member(p)
                    f.write(json.dumps(row,separators=(",",":"))+"\n")
                    count+=1
                    size+=row["bytes"]
    key=[WEIGHT,PRIOR/"manifest.json",BASE/"training_complete.json",Path(read(BASE/"training_complete.json")["checkpoint"]),
         B0/"training_manifest.json",GEOMETRY/"manifest.json",SCENE/"training_manifest.json",RUN/"training_complete.json",
         Path(read(RUN/"training_complete.json")["checkpoint"]),Path(read(REPORT/"video_manifest.json")["path"])]
    for label,run in (("B0",BASE),("B1",RUN)):
        key.extend([run/"mesh"/(RAW_PREFIX+"pcd.ply"),run/"mesh"/(RAW_PREFIX+"poisson_mesh.ply")])
        key.extend(RUN/"supported"/label/n for n in ("fused.ply","before_crop.ply","supported.ply","face_votes.npz"))
    write(REPORT/"artifact_manifest.json",dict(index=member(inventory),file_count=count,total_bytes=size,
          files=[member(p) for p in key],root=str(ROOT),run=str(RUN)))


def fmt(x,digits=4):
    return "无有效分母" if x is None else f"{x:.{digits}f}"


def main():
    trained=read(RUN/"training_complete.json")
    prior=read(REPORT/"normal_prior.json")
    video=read(REPORT/"video_manifest.json")
    paired=read(REPORT/"paired_summary.json")
    surface=read(REPORT/"surface_summary.json")["splits"]
    demand(trained["step"]==29999 and prior["completed"]==video["frames"]==1846 and
           sha(Path(trained["checkpoint"]))==video["B1_checkpoint_sha256"]==trained["checkpoint_sha256"],"final identity")
    train_stats(trained)
    curves()
    by_model={}
    for label in ("B0","B1"):
        folder=RUN/"supported"/label
        fusion,poisson,crop=(read(folder/(s+".json")) for s in ("fuse","poisson","crop"))
        demand(crop["source_training_views"]==1478 and crop["before"]==crop["retained"]+crop["insufficient"]+crop["conflict_dominant"],label)
        by_model[label]=dict(fusion=fusion,poisson=poisson,crop=crop)
        shutil.copyfile(folder/"sample.json",REPORT/f"{label}_sampling.json")
    write(REPORT/"supported_meshes.json",by_model)
    manifest()
    resource=timings()
    git_head=subprocess.check_output(["git","-C",ROOT,"rev-parse","HEAD"],text=True).strip()
    write(REPORT/"run_manifest.json",dict(fingerprint=read(STATE/"identity.json"),training=trained,resources=resource,
          run=str(RUN),implementation_commit_at_report=git_head,stage_commands="bash scripts/launch_capture_metric3d.sh",
          baseline=read(REPORT/"frozen_inputs.json"),normal_adapter_patch=member(ROOT/"patches/metric3d-eb5b6fac-mmengine-env.patch")))
    events=read(REPORT/"B1_refinement.json")["events"]
    worst=min(events,key=lambda r:r["after"]-r["before"])
    steps=read(REPORT/"B0_training_curve.json")["checkpoints"]
    lines=["# capture-full 全程 Metric3D v2 法线对照","",
        f"从相同的 2,001,906 个点初始化，B1 完成 30,000 次更新，最终 step 29,999，共 {trained['final_gaussians']:,} 个高斯。"
        f"B0 最终有 {steps[-1]['gaussians']:,} 个高斯。两组使用同一 1,478 张训练图和相同配置，B1 训练使用 Metric3D v2 ViT-Large 法线与原传感器 Z。","",
        "## 全部留出视角","","| 划分 | 模型 | 帧数 | PSNR (dB) | SSIM | LPIPS | 合并像素 MAE (m) | 合并像素 RMSE (m) | 有效参考像素 | 无效预测 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    signs=[]
    for split in HOLDOUT:
        data=paired[split]
        for label,model in data["models"].items():
            rgb=model["rgb_frame_means"]
            z=model["depth_pixel_pooled"] or {}
            lines.append(f"| {split} | {label} | {data['frames']} | {fmt(rgb['psnr'],3)} | {fmt(rgb['ssim'])} | {fmt(rgb['lpips'])} | "
                         f"{fmt(z.get('MAE_m'))} | {fmt(z.get('RMSE_m'))} | {model['reference_pixels']:,} | {model['invalid_prediction_pixels']:,} |")
        a,b=data["models"]["B0"],data["models"]["B1"]
        for k,direction in (("psnr",1),("ssim",1),("lpips",-1)):
            signs.append((b["rgb_frame_means"][k]-a["rgb_frame_means"][k])*direction)
        if a["depth_pixel_pooled"] and b["depth_pixel_pooled"]:
            signs.append(a["depth_pixel_pooled"]["MAE_m"]-b["depth_pixel_pooled"]["MAE_m"])
    conclusion=("本次留出集上的各项均值均向改善方向变化。" if all(v>0 for v in signs) else
                "本次留出集上的各项均值均未改善。" if all(v<=0 for v in signs) else "法线替换带来取舍，各项留出均值的变化方向不同。")
    lines.extend(["",conclusion+" 这些数值来自本次采集的完整划分。",""])
    for split in HOLDOUT:
        data=paired[split]
        c=data["changes"]["psnr"]
        m=data["same_source_24"]["psnr"]
        prior_z=data["prior_diagnostic"]
        lines.append(f"- {split} 的 PSNR 有 {c['improved']} 帧上升、{c['degraded']} 帧下降。原 s01 同源 24 帧中有 {m['improved']} 帧上升。"
                     f"独立保存的 Metric3D 单帧深度与传感器有 {prior_z['finite_positive']:,}/{prior_z['reference_pixels']:,} 个有效配对，"
                     f"条件 MAE 为 {fmt((prior_z['conditional_pooled'] or {}).get('MAE_m'))} m。"
                     f"逐帧差值和 30 秒窗见 [配对明细](paired_{split}.json)。")
    lines.extend(["","## 高斯数量与训练曲线","",
        f"B1 单次净减少最大出现在 step {worst['step']}，由 {worst['before']:,} 到 {worst['after']:,}。"
        f"该次剔除并集为 {worst['cull_union']:,}，父点替换为 {worst['parent_replacement']:,}。B0 的历史记录提供了各保存步数量和日志中的剔除数。"
        "详细原因及其重叠以 B1 实际回调记录为准。[B0 曲线](B0_training_curve.json) · [B1 剔除记录](B1_refinement.json)。","",
        "![训练过程中观测到的高斯数与损失](previews/training_curves.png)","",
        "## 网格及其有效覆盖","","覆盖率使用 0.3–6 m 的全部有效传感器像素作分母。"
        "正确覆盖要求误差不超过 max(0.05 m, 参考深度的 3%)。近于参考且超过同一容差的首交点计入前方冲突。","",
        "| 划分 | 模型 | 导出 | 命中 / 参考像素 | 命中率 | 正确覆盖率 | 命中 MAE (m) | 前方冲突 / 全部参考 |",
        "|---|---|---|---:|---:|---:|---:|---:|"])
    for split in HOLDOUT:
        for label in ("B0","B1"):
            for kind,title in (("raw","原始 depth 9"),("pre_crop","depth 11 裁前"),("supported","支持裁剪后")):
                z=surface[split][f"{label}_{kind}_mesh"]["pixel_pooled"]
                lines.append(f"| {split} | {label} | {title} | {z['hits']:,} / {z['reference']:,} | {fmt(z['hits_fraction_all_reference'])} | "
                             f"{fmt(z['correct_tau_fraction_all_reference'])} | {fmt(z['MAE_hit_m'])} | {fmt(z['foreground_fraction_all_reference'])} |")
    lines.extend([""])
    for label in ("B0","B1"):
        export=by_model[label]
        z=surface["test"][label+"_raw_mesh"]["pixel_pooled"]
        b=surface["test"][label+"_supported_mesh"]["pixel_pooled"]
        lines.append(f"- {label} 的训练深度筛选得到 {export['fusion']['samples']:,} 个样点，1 cm 融合后为 {export['fusion']['fused']:,} 点。"
                     f"支持裁剪保留 {export['crop']['retained']:,}/{export['crop']['before']:,} 个面。"
                     f"同一模型更换导出时，test 命中覆盖变化 {100*(b['hits_fraction_all_reference']-z['hits_fraction_all_reference']):+.2f} 个百分点，"
                     f"正确覆盖变化 {100*(b['correct_tau_fraction_all_reference']-z['correct_tau_fraction_all_reference']):+.2f} 个百分点。")
    lines.extend(["","法线的作用可在相同导出下比较 B0 与 B1，筛选与裁剪的作用可在同一模型内比较。"
        "这里只评价同次采集传感器的一致性。逐帧、等权帧、合并像素、5/10/20 cm 正确覆盖、共同命中和三个固定局部见"
        " [test 网格指标](surface_test_metrics.json)、[val 网格指标](surface_val_metrics.json)和[局部结果](surface_local.json)。","",
        "![两模型原始、裁前与裁后网格](previews/surfaces_global.jpg)","",
        "## 视频与产物","",
        f"全程视频按采集时间播放，共 {video['frames']:,} 帧、{video['duration_s']:.3f} 秒，从左到右为参考 RGB、B0 与 B1。"
        "源权重、训练输入、checkpoint、完整视频、点云和网格的路径与哈希见[产物清单](artifact_manifest.json)。"
        "精度与坐标转换见[先验记录](normal_prior.json)，B1 与 B0 的配置逐项核对见[配置差异](config_diff.json)。","",
        "本轮中学习率、首遍以后的顺序采样和 15k 停止增密均与 B0 一致。判断训练预算或采样顺序的作用还需单独的完整对照。"])
    (REPORT/"overview.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    pointer=REPORT.parent/"latest.json"
    latest=read(pointer)
    latest.update(status="completed",note=conclusion,observed_stage="report_complete",
          actual_inferred_frames=prior["completed"],actual_weights_sha256=prior["weight"]["sha256"],
          actual_updates=trained["cumulative_updates"],actual_final_step=trained["step"],actual_gaussians=trained["final_gaussians"],
          actual_checkpoint_sha256=trained["checkpoint_sha256"],actual_metrics={s:paired[s]["models"] for s in HOLDOUT},
          actual_meshes={m:data["crop"]["mesh"] for m,data in by_model.items()},video=video,
          completed_overview=(REPORT/"overview.md").relative_to(ROOT).as_posix(),
          updated_utc=datetime.now(timezone.utc).isoformat())
    write(pointer,latest)


if __name__=="__main__":
    main()
