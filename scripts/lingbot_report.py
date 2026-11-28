"""Publish compact evidence from verified hall outputs. Large deliverables stay on the instance."""
import gzip
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from lingbot_common import P0, PROTOCOL, REPORT, ROOT, RUN, SEQ, by_id, member, read, require, save, sha
from lingbot_metrics import summary


def cmd(*argv):
    return subprocess.check_output(list(map(str,argv)),text=True).strip()


def package_inventory(python):
    code="""import sys,json,importlib.metadata as m,torch
print(json.dumps(dict(python=sys.version,prefix=sys.prefix,packages={r.metadata['Name']:r.version for r in m.distributions() if r.metadata['Name']},torch_cuda=torch.version.cuda,arch=torch.cuda.get_arch_list(),bf16=torch.cuda.is_bf16_supported())))"""
    return json.loads(cmd(python,"-c",code))


def jsonlines(path,rows):
    with gzip.open(path,"wt",encoding="utf8",compresslevel=6) as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")


def latency(rows,keys):
    return {k:summary([r.get(k) for r in rows]) for k in keys}


def main():
    import torch
    REPORT.mkdir(exist_ok=True,parents=True)
    manifest,frames=by_id()
    selection=read(P0/"selection.json")
    dep=read(RUN/"depth/manifest.json")
    d=read(RUN/"depth/evaluation/metrics.json")
    maps={seq:read(RUN/"map"/seq/"evaluation/metrics.json") for seq in SEQ}
    require(dep["unmasked"]==1478 and dep["masked"]==60 and [maps[s]["predicted_frames"] for s in SEQ]==[256,1478,1846],"Incomplete results")
    for p in (P0/"input_manifest.json",P0/"selection.json",RUN/"media/depth/fixed_roi_metrics.json"):
        shutil.copyfile(p,REPORT/p.name)
    for p in (RUN/"depth/evaluation").glob("*.csv.gz"): shutil.copyfile(p,REPORT/p.name)
    records=[]
    for r in dep["records"]:
        for v in r["outputs"]:
            records.append(dict(id=r["id"],mode=r["mode"],path=str(RUN/"depth"/r["mode"]/r["id"]/v["name"]),
                                bytes=v["bytes"],sha256=v["sha256"]))
    jsonlines(REPORT/"depth_outputs.jsonl.gz",records)
    dt=dep["records"]
    timing={"depth":dict(load_s=dep["model_load_s"],first_formal_call=dt[0],num_tokens=dep["num_tokens"],batch_size=1,
                  stable_seconds=latency(dt[1:],["loading_s","h2d_s","gpu_s","d2h_s","save_hash_s","end_to_end_s"]),
                  peak_vram_bytes=dep["peak_vram_bytes"],peak_rss_bytes=dep["peak_rss_bytes"])}
    artifacts=[]
    for seq in SEQ:
        root=RUN/"map"/seq; status=read(root/"complete.json")
        a=status["audit"]; t=read(status["timing_file"]); chunks={Path(c["path"]).name:c for c in a["chunks"]}
        jsonlines(REPORT/(seq+"_outputs.jsonl.gz"),[dict(**r,timestamp_ns=frames[r["id"]]["rgb_timestamp_ns"],
                  split=frames[r["id"]]["split"],artifact=chunks[r["chunk"]]) for r in a["frames"]])
        folder=REPORT/seq;folder.mkdir(exist_ok=True)
        for p in (root/"evaluation").iterdir():
            if p.name in ("per_frame.csv.gz","overlap_pairs.csv.gz","alignment.json","trajectory.json"):
                shutil.copyfile(p,folder/p.name)
        rows=t["per_call"][8:]
        timing[seq]=dict(model_load_s=t["load_s"],compile_warmup_s=t["warmup_compile_s"],total_s=t["elapsed_s"],
                         stable_seconds=latency(rows,["input_s","h2d_s","model_step_s","d2h_s","online_s","save_s"]),
                         peak_vram_bytes=t["peak_vram_bytes"],peak_rss_bytes=t["peak_rss_bytes"])
        artifacts.extend([member(root/"complete.json"),member(Path(status["timing_file"]))])
    save(REPORT/"timing.json",timing)
    save(REPORT/"metrics.json",dict(depth=d,map=maps,definition=dict(primary="original finite optical Z, 0.3–6 m",
      masked_coverage="correct predictions / all frozen scoring pixels; invalid predictions count as misses",
      conditional_errors="MAE/RMSE/AbsRel/P95 on positive finite method-valid intersection, with support",
      map_geometry="per independent stream: median of up to 32 train-frame scale medians, one rigid fit on those camera centres",
      clocks="Seconds. Map model_step_s is synchronized model execution including CPU dispatch, excluding measured decode and device copies.")))
    save(REPORT/"environment.json",dict(gpu=cmd("nvidia-smi","--query-gpu=name,driver_version,compute_cap,memory.total","--format=csv,noheader"),
       startup_free_vram_MiB=97252,startup_free_data_disk_GiB_rounded=550,cpu_affinity=len(os.sched_getaffinity(0)),
       cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),memory_limit_bytes=int(Path("/sys/fs/cgroup/memory.max").read_text()),
       cuda_toolkit=cmd("/usr/local/cuda-12.8/bin/nvcc","--version"),platform=cmd("uname","-srm"),
       depth=package_inventory(ROOT/".local/conda/lingbot-depth-bw/bin/python"),
       map=package_inventory(ROOT/".local/conda/lingbot-map2-bw/bin/python"),
       depth_attention=read(RUN/"state/depth_kernel_fp32.json"),inference_concurrency="one Depth process, one Map stream, overlap where stages allowed",
       video_encoders={seq:read(RUN/"media/map"/seq/"display.json").get("encoder",["libx264"]) for seq in SEQ}))
    weights=RUN/"weights"
    ckpt=torch.load(weights/"depth/model.pt",map_location="cpu",mmap=True,weights_only=True)
    config_path=RUN/"identity/depth_model_config.json"
    save(config_path,ckpt["model_config"]);del ckpt
    source_info=dict(run_id=RUN.name,task_commit="aa979aea18d4e7ac492ef60b6c8449b4b5516f49",protocol=member(PROTOCOL),
        freeze=member(P0/"freeze.json"),B3_selection=member(ROOT/"reports/capture-full-quality-v3/20261122-v1/selection_freeze.json"),
        original_split="1478 train, 184 val, 184 development-test",frozen_baselines=["B2 210k rendering","B3 210k supported Poisson"],
        depth=dict(repo="https://github.com/Robbyant/lingbot-depth",commit=cmd("git","-C",ROOT/".local/upstream/lingbot-depth","rev-parse","HEAD"),
          model="robbyant/lingbot-depth-pretrain-vitl-14-v0.5",provider="ModelScope official release",
          provider_revision="8d8942390f40f5c4e2e0c3b3e0964d94d9fe5afc",identical_hf_revision="79204ed6b837f4fdd192cf563e59481fecfa0295",
          weight=member(weights/"depth/model.pt"),model_config=member(config_path)),
        map=dict(repo="https://github.com/Robbyant/lingbot-map-v2",commit=cmd("git","-C",ROOT/".local/upstream/lingbot-map-v2","rev-parse","HEAD"),
          model="robbyant/lingbot-map-v2-base-streaming",provider="official download_model.py --source modelscope",
          provider_revision="984dcffb664bcd6deabb61645ab9cbb4b0ef7b8d",identical_hf_revision="e6a22da8d071200b25dd3bfde2bd499a230819ee",
          weight=member(weights/"map2/model_weights.pt"),model_config=member(weights/"map2/config.json"),GCR=False),
        adapters={p.name:sha(p) for p in sorted((ROOT/"scripts").glob("*lingbot*")) if p.is_file()})
    save(REPORT/"sources.json",source_info)
    for p in sorted((RUN/"media").rglob("*")):
        if p.is_file(): artifacts.append(member(p))
    for p in (P0/"freeze.json",RUN/"depth/manifest.json",weights/"depth/model.pt",weights/"map2/config.json",weights/"map2/model_weights.pt"):
        artifacts.append(member(p))
    for p in (RUN/"depth-example0").glob("*"):
        if p.is_file():artifacts.append(member(p))
    jsonlines(REPORT/"media_artifacts.jsonl.gz",artifacts)
    save(REPORT/"artifacts.json",dict(server_run=str(RUN),input_catalog="input_manifest.json",
        full_prediction_catalogs=["depth_outputs.jsonl.gz"]+[s+"_outputs.jsonl.gz" for s in SEQ],
        complete_media_catalog="media_artifacts.jsonl.gz",files=[r for r in artifacts if r["path"].endswith((".mp4","model_weights.pt","model.pt","stride12.ply"))],
        three_dimensional_viewer="bash scripts/view_lingbot_map.sh full_train_1478 18788"))
    dm=d["masked_recovery"]; obs=d["measurement_change"]["all"]["pooled"]; holes=d["raw_hole_completion"]; independent=d["two_view_holes"]
    full=maps[SEQ[1]]; short=maps[SEQ[0]]; allseq=maps[SEQ[2]]
    depth_conclusion="全图替换会改变原有测量，散点和连续块恢复均落后于最近邻。仅在其他视角能验证的孔洞保留为局部候选。"
    map_conclusion="可用于 RGB 轨迹回放和场景查看。各序列需独立校准，长程差异与困难帧仍需独立定位证据。"
    decision=dict(depth=dict(label="local_use_only",recommendation=depth_conclusion),
                  map=dict(label="local_use_only",recommendation=map_conclusion),
                  dn_single_factor="暂缓全图换深度和全程换位姿。下一项可检验保留传感器原值、仅对多视角支持孔洞启用补值，且使用同一冻结划分。")
    save(REPORT/"decision.json",decision)
    lines=["# Gemini 335 大厅 · LingBot 离线结果","",
      "Depth 完成 1,478 张独立训练图的全画布推理，另完成 30 张固定样例各两种人工遮挡。Map 分别输出 256、1,478 和 1,846 个原始时间位置。","",
      "先看[桌沿、椅腿与泡沫齿边的同视角对照](previews/depth_structures.png)和[深度视频节选](previews/depth_preview.mp4)。查看 Map 可打开[训练轨迹](previews/full_train_1478_trajectory.png)和[困难段视频节选](previews/map_preview.mp4)。","",
      "## Depth","",depth_conclusion,"","| 固定遮挡 | 方法 | 计分像素 | 预测覆盖 | 条件 MAE | 10 cm 正确恢复覆盖 |","|---|---|---:|---:|---:|---:|"]
    for mode,label in (("random_points","10% 散点"),("blocks","64×64 连续块")):
        for method,name in (("nearest","二维最近邻"),("lingbot","LingBot")):
            v=dm[f"{mode}/{method}/all"]["pooled"]
            lines.append(f"| {label} | {name} | {v['reference_pixels']:,} | {v['coverage']:.2%} | {v['mae_m']*100:.2f} cm | {v['correct_10cm']:.2%} |")
    lines+=["",f"在原有效区的 {obs['reference_pixels']:,} 个参考像素上，修改幅度中位数 {obs['p50_m']*100:.2f} cm，均值 {obs['mae_m']*100:.2f} cm，P95 为 {obs['p95_m']*100:.2f} cm。原生孔洞补全覆盖 {holes['coverage']:.2%}。64 个固定诊断视角里，仅 {independent['support_fraction']:.2%} 的孔洞有两个其他训练视角的相符传感器观测，其中有有效预测的位置有 {independent['supported_agreement']:.2%} 落在传感器支持带内。该支持依赖 B2 位姿。","",
      "深度图共用 0.3–6 m 色标，紫色表示区间外观测，黑色表示无值。桌面和椅架的轮廓在补全图中可辨。泡沫齿边在 RGB 中清楚可见，修复深度仍然接近平滑表面。固定椅腿框中有 872 个孔洞像素获得双视角传感器支持，其中 474 个与预测相符。桌沿框对应数字为 1,089 和 557。区域内不同表面的采样密度不同，这些像素数不能验证细杆的物理直径。[固定 ROI 统计](fixed_roi_metrics.json) · [逐帧配对误差图](previews/depth_recovery.png)","",
      "## Map 2 公开 base-streaming","",map_conclusion,"","| 独立 RGB 序列 | 预测 / 输入 | 持久帧 | 米/模型单位 | 与 B2 相机中心差异中位数 | 传感器 Z 的固定尺度 MAE | 默认置信筛选覆盖 |","|---|---:|---:|---:|---:|---:|---:|"]
    for seq in SEQ:
        j=maps[seq]; lines.append(f"| {seq} | {j['predicted_frames']}/{j['input_frames']} | {j['persistent_frames']} | {j['model_scale']:.4f} | {j['centre_difference_m']['p50']:.3f} m | {j['pooled_depth']['all']['pooled']['mae_m']:.3f} m | {j['default_filter_coverage']:.2%} |")
    lines+=["",f"完整 train 的 Map 路程为 {full['map_path_m']:.2f} m，B2 参考路程为 {full['reference_path_m']:.2f} m。最大相邻平移出现在 {full['top_translation_steps'][0]['id']}，为 {full['top_translation_steps'][0]['map_step_m']:.2f} m。困难区域可结合[相同时间的 RGB、传感器、Map 深度与点云投影](previews/map_difficult.png)查看。","",
      "每个序列在同一组 32 张预选训练图上独立估计一个尺度和一个刚体变换。1846 帧探索使用了验证和开发测试的 RGB，其结果仅用于完整采集回放。轨迹差异参照历史 B2，相机和传感器均缺独立几何真值。缺少经独立确认的闭环对，闭环误差留空。","",
      "## 下一轮 DN","",decision["dn_single_factor"],"","[完整产物、字节数与 SHA-256](artifacts.json) · [指标与分母](metrics.json) · [计时与峰值资源](timing.json) · [来源](sources.json) · [执行、故障与恢复](execution.md)"]
    (REPORT/"README.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    save(REPORT.parent/"latest.json",dict(run_id=RUN.name,status="completed",depth_unmasked=1478,depth_masked=60,
        map_sequences={s:maps[s]["predicted_frames"] for s in SEQ},report=str((REPORT/"README.md").relative_to(ROOT)),
        decisions={k:decision[k]["label"] for k in ("depth","map")}))
    print("Report complete",REPORT,flush=True)


if __name__=="__main__":
    main()
