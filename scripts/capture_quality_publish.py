"""Build the B2 comparison from measured artifacts and publish the private transfer manifest."""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

from capture_metric3d_common import REPORT as OLD
from capture_quality_paths import GEOM, P, REPORT, ROOT, RUN, SCENE, STATE, check, read, record, write


def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()


def transport(*args):
    return subprocess.run(["bash",ROOT/"scripts/github_transport.sh","git",*args],cwd=ROOT,
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0


def zipped_report(path,dest):
    with path.open("rb") as src,gzip.open(dest,"wb") as out:
        shutil.copyfileobj(src,out)


def comparison(split,step):
    prev=read(OLD/f"metrics_{split}.json")
    current=read(RUN/"evaluation"/split/f"step-{step:09d}.json")
    before={r["id"]:r for r in prev["frames"]}
    s01={r["id"]:r for r in read(OLD/"comparison_s01_same_sources.json")[split]["frames"]}
    paired=[]
    for row in current["frames"]:
        a=before[row["id"]]
        check(row["depth"]["reference_pixels"]==a["reference_pixels"],"same reference denominator")
        pair=dict(id=row["id"],time_s=row["relative_s"],reference_pixels=a["reference_pixels"],
                  B1_rgb=a["rgb"],B2_rgb=row["rgb"],rgb_delta={k:row["rgb"][k]-v for k,v in a["rgb"].items()},
                  B1_depth=a["depth_full_reference"],B2_depth=row["depth"],s01=s01.get(row["id"]))
        paired.append(pair)
    check(len(paired)==184 and len(s01)==24,"paired comparison population")
    write(REPORT/f"paired_{split}_{step}.json",dict(frames=paired))
    ma=prev["depth_pixel_pooled"]["MAE_m"]; mb=current["depth_pixel_pooled"]["MAE_m"]
    return dict(B1_rgb=prev["rgb_frame_means"],B2_rgb=current["rgb_frame_means"],
                delta_dB=current["rgb_frame_means"]["psnr"]-prev["rgb_frame_means"]["psnr"],B1_MAE_m=ma,
                B2_MAE_m=mb,relative_MAE_reduction=(ma-mb)/ma if mb is not None else None,
                reference_pixels=current["depth_pixel_pooled"]["reference_pixels"],same_source=current["same_s01_source"])


def main():
    REPORT.mkdir(exist_ok=True,parents=True)
    ev=RUN/"evaluation"; meshes=RUN/"meshes"
    sel=read(STATE/"selection.json"); chosen=sel["selected_step"]; final=sel["final_step"]
    check(final==239999 and read(RUN/"training_complete.json")["updates"]==240000,"actual final update")
    export=read(ev/"export_selection.json")["selected"]
    results={s:{str(step):comparison(s,step) for step in sorted({chosen,final})} for s in ("val","test")}
    surfaces={s:read(ev/f"surfaces_{s}.json")["groups"]["all"] for s in ("val","test")}
    base=surfaces["test"]["B1_supported"]; new=surfaces["test"][export]
    def percent(x): return "无全分母值" if x is None else f"{100*x:.2f}%"
    def metres(x): return "无全分母值" if x is None else f"{x:.4f} m"
    test=results["test"][str(chosen)]; val=results["val"][str(chosen)]
    cover="correct_10_fraction_all_reference"; hit="hits_fraction_all_reference"; conflict="foreground_fraction_all_reference"
    video=read(RUN/"video/video_manifest.json")
    poses=read(GEOM/"pose.json")["results"]
    unresolved=[k for k,v in poses.items() if not v["accepted"]]
    init=read(GEOM/"initialization.json"); seed=read(STATE/"trainer/seed_scale.json")
    vmesh=surfaces["val"][export]
    thresholds=dict(val_psnr_plus_2dB=val["delta_dB"]>=2,
                    val_depth_minus_25pct=val["relative_MAE_reduction"] is not None and val["relative_MAE_reduction"]>=.25,
                    val_mesh_correct_10cm=vmesh[cover]>=.70,val_mesh_coverage=vmesh[hit]>=.90,val_mesh_conflict=vmesh[conflict]<=.15)
    for path in [STATE/"selection.json",STATE/"trainer/seed_scale.json",STATE/"trainer/first_refinement.json",
                 STATE/"trainer/loaded_sources.json",STATE/"trainer/configuration.diff",Path(sel["config"]),GEOM/"pose.json",
                 GEOM/"initialization.json",GEOM/"depth_coverage.json",ev/"export_selection.json",
                 ev/"surfaces_val.json",ev/"surfaces_test.json",RUN/"video/video_manifest.json",RUN/"b1_val/curve.json"]:
        shutil.copyfile(path,REPORT/path.name)
    for path in sorted((STATE/"trainer").glob("first_batch_*.json")): shutil.copyfile(path,REPORT/path.name)
    zipped_report(STATE/"trainer/refinement.jsonl",REPORT/"refinement.jsonl.gz")
    with (STATE/"stages.jsonl").open() as src,gzip.open(REPORT/"stage_timing.jsonl.gz","wt",encoding="utf-8") as dst:
        for line in src:
            stage=json.loads(line)
            print(json.dumps({k:v for k,v in stage.items() if k not in ("command","cwd")}),file=dst)
    for path in [*ev.glob("*/*.json"),*ev.glob("surface_frames*.json"),*(RUN/"b1_val").glob("step*.json")]:
        label=path.parent.name+"_"+path.name+".gz"
        zipped_report(path,REPORT/label)
    preview=REPORT/"previews"; preview.mkdir(exist_ok=True)
    for path in sorted((RUN/"previews").glob("*.jpg")): shutil.copyfile(path,preview/path.name)
    summary=dict(selection=sel,selected_export=export,comparison=results,surfaces=surfaces,goals=thresholds,
                 corrected=len(poses)-len(unresolved),unresolved_frames=unresolved)
    write(REPORT/"quality.json",summary)
    lines=["# Capture-full B2 模型与网格质量","",
           f"主模型采用累计 {chosen+1:,} 次更新的检查点。184 张固定开发测试图上，PSNR 从 {test['B1_rgb']['psnr']:.3f} 到 {test['B2_rgb']['psnr']:.3f} dB（{test['delta_dB']:+.3f} dB），合并参考像素深度 MAE 从 {metres(test['B1_MAE_m'])} 到 {metres(test['B2_MAE_m'])}，相对降幅 {percent(test['relative_MAE_reduction'])}。","",
           f"验证集选出的网格为 `{export}`。测试集全部有效参考像素中，10 cm 正确覆盖从 B1 的 {percent(base[cover])} 到 {percent(new[cover])}（{100*(new[cover]-base[cover]):+.2f} 个百分点），命中率 {percent(new[hit])}，前方冲突 {percent(new[conflict])}。","",
           "| 划分 | 累计更新 | PSNR (dB) | SSIM | LPIPS | 合并参考 MAE |","|---|---:|---:|---:|---:|---:|"]
    for split,points in results.items():
        for k,v in points.items():
            rgb=v["B2_rgb"];lines.append(f"| {split} | {int(k)+1} | {rgb['psnr']:.3f} | {rgb['ssim']:.4f} | {rgb['lpips']:.4f} | {metres(v['B2_MAE_m'])} |")
    lines+=["","| 网格 | 划分 | 命中率 | 5 cm 正确覆盖 | 10 cm 正确覆盖 | 20 cm 正确覆盖 | 前方冲突 | 命中 MAE |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split,g in surfaces.items():
        for key,m in g.items():
            lines.append(f"| {key} | {split} | {percent(m[hit])} | {percent(m['correct_05_fraction_all_reference'])} | {percent(m[cover])} | {percent(m['correct_20_fraction_all_reference'])} | {percent(m[conflict])} | {metres(m['MAE_hit_m'])} |")
    lines+=["",f"14 个候选位姿中，{len(poses)-len(unresolved)} 个通过外部训练观测纠正。保留原估计的帧为 {', '.join(unresolved) or '无'}。传感器初始化 {init['sensor_points']:,} 点，新增合格 SfM 初始化 {init['sfm_points']:,} 点，其中 {seed['capped']:,} 点触发一次性 0.25 m 尺度上限。",
            "",f"训练完成 240,000 次更新，验证集含 8 个完整节点。{video['frames']} 个时间位置均进入遮挡版视频，时长 {video['duration_s']:.2f} s。几何评价使用同次采集的有效传感器 Z。原生传感器 TSDF 同时依赖估计位姿。","",
            "验证集工程目标结果为 "+ "，".join(f"{k} {'达到' if v else '未达到'}" for k,v in thresholds.items())+"。",
            "","逐帧配对、30 s 时间窗、距离分层、原 s01 同源帧、固定局部预览和各曲面共同命中结果见同目录文件。B1 与 B2 各自使用对应的相机估计，变化量衡量联合场景适配的端到端效果。"]
    (REPORT/"overview.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    logs=RUN/"delivery"/"experiment_logs.zip"
    logs.parent.mkdir(exist_ok=True)
    sources=[*sorted((RUN/"logs").glob("*.log")),STATE/"resources.jsonl",STATE/"process_resources.jsonl",
             STATE/"stages.jsonl",*sorted((STATE/"trainer").glob("*.json*"))]
    with zipfile.ZipFile(logs.with_suffix(".tmp"),"w",compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in sources:
            if not p.is_file(): continue
            n=p.stat().st_size
            with p.open("rb") as src,z.open(p.relative_to(RUN).as_posix(),"w",force_zip64=True) as dst:
                while n:
                    block=src.read(min(n,1024*1024));check(block,"log truncated during snapshot")
                    dst.write(block); n-=len(block)
    logs.with_suffix(".tmp").replace(logs)
    private=[logs,RUN/"video/reference_B1_B2_public.mp4",meshes/"sensor_tsdf.ply",meshes/"model_tsdf.ply",meshes/"supported/supported.ply",
             GEOM/"added_tracks.npz",SCENE/"init_train.ply",SCENE/"transforms.json",SCENE/"training_manifest.json",
             *sorted((RUN/"training").glob("**/step-*.ckpt")),*sorted((RUN/"training").glob("**/events.out.tfevents.*")),
             *sorted((STATE/"trainer").glob("samples_*.csv")),*sorted((GEOM).glob("pose_*.jpg")),
             *[meshes/"supported"/(name+".json") for name in ("sample","fuse","poisson","crop","protocol")]]
    artifacts=[record(p) for p in private if p.is_file()]
    write(REPORT/"artifacts.json",dict(run_id=P["run_id"],artifacts=artifacts))
    write(REPORT.parent/"latest.json",dict(run_id=P["run_id"],status="completed",updates=240000,selected_step=chosen,
          report=REPORT.relative_to(ROOT).as_posix()+"/overview.md",utc=datetime.now(timezone.utc).isoformat()))
    tracked=[REPORT.relative_to(ROOT).as_posix(),(REPORT.parent/"latest.json").relative_to(ROOT).as_posix()]
    check(re.fullmatch(r"(https://github\.com/|git@github\.com:)kaiwen123-yang/DN-Splatter(?:\.git)?/?",git("remote","get-url","origin"),re.I),"publish destination")
    check(all(p.stat().st_size<25*1024**2 for p in REPORT.rglob("*") if p.is_file()),"small git artifacts")
    # No private camera images or console/network transcripts are staged.
    git("add","--",*tracked); git("diff","--cached","--check","--",*tracked)
    staged=git("diff","--cached","--name-only","--",*tracked).splitlines()
    check(all(p.startswith("reports/capture-full-quality-v2/") for p in staged),"staged report scope")
    for p in staged:
        if p.endswith((".md",".json",".yml",".diff")):
            check(not re.search(rb"(?:BEGIN [A-Z ]*PRIVATE KEY|Bearer [A-Za-z0-9._-]{20,}|sk-[A-Za-z0-9_-]{30,})",
                                (ROOT/p).read_bytes()),"sensitive staged material in "+p)
    if staged: git("commit","--only","-m","Publish measured capture-full B2 model and surface comparisons","--",*tracked)
    commit=git("rev-parse","HEAD")
    receipt=dict(run_id=P["run_id"],status="completed",commit=commit,artifacts=artifacts)
    if not transport("push","origin","HEAD:main"):
        write(STATE/"delivery_candidate.json",receipt)
        while not (transport("fetch","-q","origin","main") and
                   subprocess.run(["git","merge-base","--is-ancestor",commit,"origin/main"],cwd=ROOT).returncode==0):
            time.sleep(60)
    check(transport("fetch","-q","origin","main"),"GitHub receipt verification")
    git("merge-base","--is-ancestor",commit,"origin/main")
    write(STATE/"delivery_ready.json",receipt)


if __name__=="__main__":
    main()
