"""Assemble measured coverage, time slices, matched-source comparisons and bulk inventories."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import subprocess
import numpy as np

from capture_rectify import read, save, sha, verify
from capture_s01_diagnostic import main as cross_view
from capture_s01_eval import depth_stats
from capture_s01_inputs import P, GEO, GEO_RUN, REPORT, ensure
from capture_s01_publish import inventory, timings, train_stats
from vr_room_publish import records
from vr_room_run import ROOT, RUN, SCENE, STATE, CPU, now


def average(items):
    return {k:float(np.mean([r[k] for r in items])) for k in items[0]} if items else None


def temporal(group, scored, split):
    rows=[scored[r["id"]] for r in group if r["id"] in scored]
    diffs,refs=[],[]
    for r in rows:
        ref=np.load(SCENE/"depth"/(r["id"]+".npy"))
        z=np.load(RUN/"renders"/split/"depth"/(r["id"]+".npy"))
        mask=(ref>=.3)&(ref<=6)&np.isfinite(z)&(z>0)
        v=ref[mask].astype(float)
        refs.append(v)
        diffs.append(np.abs(z[mask]-v))
    invalid=sum(r["invalid_prediction_pixels"] for r in rows)
    return dict(evaluated=len(rows),with_depth=sum(r["reference_pixels"]>0 for r in rows),
                reference_pixels=sum(r["reference_pixels"] for r in rows),invalid_predictions=invalid,
                zero_alpha_reference_pixels=sum(r["reference_alpha_zero"] for r in rows),
                rgb_frame_means=average([r["rgb"] for r in rows]),
                depth_frame_means=average([r["depth_full_reference"] for r in rows if r["depth_full_reference"]]) if not invalid else None,
                depth_pixel_pooled=depth_stats(np.concatenate(diffs),np.concatenate(refs)) if rows and not invalid else None)


def matched(metrics, selection):
    prev=ROOT/"reports/capture-s01-dn-30k/20261119-v1"
    old_geo=ROOT/".local/data/custom/capture-s01/20261119-s01-5hz-3982cb26"
    prior={r["id"]:r for r in read(ROOT/P["selection"]["prior_selection"])["rows"]}
    current={r["id"]:r for r in selection}
    result={}
    for s,data in metrics.items():
        old=read(prev/f"metrics_{s}.json")["frames"]
        new={r["id"]:r for r in data["frames"]}
        pairs=[]
        for r in old:
            stem=r["id"]
            a,b=prior[stem],current[stem]
            same=all(a[k]==b[k] for k in ("rgb_zero_based_index","actual_rgb_timestamp_ns","rgb_payload_sha256","depth_pixels_sha256"))
            ensure(same and b["requested_split"]==s,"prior source/split identity "+stem)
            if stem not in new:
                continue
            if any(sha(old_geo/f)!=sha(SCENE/f) for f in (Path("images")/(stem+".png"),Path("depth")/(stem+".npy"))):
                continue
            pairs.append(dict(id=stem,original_rgb_index=b["rgb_zero_based_index"],timestamp=b["actual_rgb_timestamp_ns"],
                              s01_rgb=r["rgb"],full_rgb=new[stem]["rgb"],s01_depth=r["depth_full_reference"],
                              full_depth=new[stem]["depth_full_reference"]))
        result[s]=dict(requested=len(old),compared=len(pairs),missing=[r["id"] for r in old if r["id"] not in {p["id"] for p in pairs}],
                       frames=pairs,s01_rgb=average([p["s01_rgb"] for p in pairs]),full_rgb=average([p["full_rgb"] for p in pairs]))
    save(REPORT/"comparison_s01_same_sources.json",result)


def main():
    verify(GEO)
    trained=read(RUN/"training_complete.json")
    ensure(trained["step"]==29999 and sha(Path(trained["checkpoint"]))==trained["checkpoint_sha256"],"final model")
    train_stats(trained)
    cross_view()
    packet=GEO_RUN/"transfer"
    selection=read(packet/"selection.json")
    splits=read(packet/"splits.json")
    meta=read(SCENE/"transforms.json")
    metrics={s:read(REPORT/f"metrics_{s}.json") for s in ("val","test")}
    posed={Path(r["file_path"]).stem for r in meta["frames"]}
    scored={}
    for s,m in metrics.items():
        ensure(m["checkpoint_sha256"]==trained["checkpoint_sha256"] and
               {r["id"] for r in m["frames"]}=={Path(n).stem for n in meta[s+"_filenames"]},s)
        scored.update({r["id"]:r for r in m["frames"]})
    details=[]
    for r in selection:
        stem=r["id"]
        reason=r["drop_reason"]
        if not reason and stem not in posed:
            reason="outside selected train map" if r["requested_split"]=="train" else "fixed-map PnP failed"
        details.append(dict(id=stem,source_index=r["rgb_zero_based_index"],device_ns=r["actual_rgb_timestamp_ns"],
                            relative_s=r["relative_s"],split=r["requested_split"],selected=not r["drop_reason"],
                            posed=stem in posed,evaluated=stem in scored,missing_reason=reason))
    def counts(rows):
        return dict(requested=len(rows),selected=sum(r["selected"] for r in rows),posed=sum(r["posed"] for r in rows),
                    evaluated=sum(r["evaluated"] for r in rows))
    counts_by={s:counts([r for r in details if r["split"]==s]) for s in ("train","val","test")}
    gaps=[r for r in details if r["missing_reason"]]
    state="completed_with_coverage_gaps" if gaps else "completed"
    save(REPORT/"coverage.json",dict(counts=counts_by,gaps=gaps,timeline=details))
    windows=[]
    for start in range(0,int(details[-1]["relative_s"])+1,30):
        block={}
        for s in counts_by:
            group=[r for r in details if start<=r["relative_s"]<start+30 and r["split"]==s]
            block[s]=counts(group)
            if s!="train":
                block[s].update(temporal(group,scored,s))
        windows.append(dict(start_s=start,end_exclusive_s=start+30,splits=block))
    save(REPORT/"metrics_by_time.json",dict(timebase="Seconds from first depth device timestamp",windows=windows))
    matched(metrics,selection)
    pose=read(GEO_RUN/"sfm/pose_quality.json")
    save(REPORT/"pose_summary.json",{k:v for k,v in pose.items() if k!="poses_w2c_sfm_units"} |
         {"all_map_attempts":[read(p) for p in sorted((GEO_RUN/"sfm").glob("map_*.json")) if p.name!="map_full.json"],
          "bridge":read(GEO_RUN/"bridge/request.json"),"poses_path":str(GEO_RUN/"metric/poses.json")})
    shutil.copyfile(GEO_RUN/"metric/scale_alignment.json",REPORT/"scale_fit.json")
    sampling=read(RUN/"mesh/sampling.json")
    mesh=read(REPORT/"mesh_validation.json")
    video=read(REPORT/"video_manifest.json")
    ensure(mesh["views"]==counts_by["train"]["posed"] and sampling["checkpoint_sha256"]==video["checkpoint_sha256"]==trained["checkpoint_sha256"]
           and video["frames"]==len(selection) and set(video["gaps"])=={r["id"] for r in gaps},"export coverage")
    commands=records(STATE/"stages.jsonl")
    (REPORT/"commands.sh").write_text("#!/usr/bin/env bash\nset -eo pipefail\ncd /root/autodl-tmp/DN-Splatter\nbash scripts/launch_capture_full.sh\n\n"
        +"\n".join("# "+r["cwd"]+" $ "+r["command"] for r in commands)+"\n")
    geometry=read(GEO/"geometry.json")
    stages=timings()
    bulk=inventory()
    manifest=read(REPORT/"artifact_manifest.json")
    for path in (GEO_RUN/"transfer/manifest.json",GEO_RUN/"sfm/map_full.json",GEO_RUN/"metric/poses.json",GEO_RUN/"metric/scale_alignment.json"):
        manifest["external_files"].append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    save(REPORT/"artifact_manifest.json",manifest)
    manifest_data=dict(status=state,identity=read(STATE/"identity.json"),run=str(RUN),geometry=str(GEO),
                       observed_utc=now(),stages=stages,cpu_budget=CPU,training=trained,counts=counts_by,
                       code_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                       init_points=geometry["init_points"],scene_scale=geometry["parser_settings"]["scene_scale"],
                       checkpoint_sha256=trained["checkpoint_sha256"],video=video["path"],mesh=mesh["mesh_path"])
    save(REPORT/"run_manifest.json",manifest_data)
    geom_scale=read(REPORT/"scale_fit.json")["scale_m_per_sfm_unit"]
    lines=["# 全程 capture 首次 30k 测试","",f"状态 {state}。共请求 {len(selection)} 帧，取得位姿 {len(posed)} 帧，缺口 {len(gaps)} 帧。",
           f"米制尺度为 {geom_scale:.8g} m/SfM 单位。以 {geometry['init_points']:,} 个训练点初始化，最终 {trained['final_gaussians']:,} 个高斯，累计 30,000 次更新。",
           "", "| 划分 | 请求 | 位姿 | 评估 | PSNR | SSIM | LPIPS | 合并像素深度 MAE (m) |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s,c in counts_by.items():
        m=metrics.get(s,{})
        rgb=m.get("rgb_frame_means") or {}
        z=m.get("depth_pixel_pooled") or {}
        vals=[rgb.get(k) for k in ("psnr","ssim","lpips")]+[z.get("MAE_m")]
        lines.append(f"| {s} | {c['requested']} | {c['posed']} | {c['evaluated']} | "+" | ".join("—" if v is None else f"{v:.4f}" for v in vals)+" |")
    worst=min(scored.values(),key=lambda r:r["rgb"]["psnr"]) if scored else None
    lines+=["",f"全轨迹视频包含 {video['frames']} 个时间位置，回放 {video['duration_s']:.3f} 秒。视频与检查点身份见 [视频清单](video_manifest.json)。",
            f"原始 Poisson 使用 {mesh['views']} 个训练视角和 {mesh['samples']:,} 个样点，得到 {mesh['vertices']:,} 个顶点、{mesh['faces']:,} 个面；最大连通分量占全部面的 {mesh['largest_component_face_fraction']:.1%}。",
            "", "同采集传感器深度用于一致性比较。每个时间窗的覆盖、误差和有效像素数见 [分时段结果](metrics_by_time.json)。相同源帧的旧基线和本轮结果见 [同帧比较](comparison_s01_same_sources.json)。",""]
    if worst:
        lines.append(f"留出图最低 PSNR 为 {worst['rgb']['psnr']:.2f} dB，对应 {worst['id']}。见 [该帧参考与渲染](previews/{worst['id']}_timeline.jpg)。")
    if gaps:
        lines.append(f"下一步优先核查 {len(gaps)} 个覆盖缺口的采集时刻与全局地图连接。缺口的源索引和原因见 [覆盖记录](coverage.json)。")
    else:
        regions=[r for r in mesh["local_views"] if r.get("depth_comparison",{}).get("Mesh Z",{}).get("MAE_m") is not None]
        if regions:
            worst_mesh=max(regions,key=lambda r:r["depth_comparison"]["Mesh Z"]["MAE_m"])
            d=worst_mesh["depth_comparison"]
            lines.append(f"三个固定局部中，{worst_mesh['object']} 的网格与传感器 Z 偏差最大（MAE {d['Mesh Z']['MAE_m']:.3f} m）。下一步优先在这个区域核查模型、样点和网格各自的深度覆盖与误差。各层匹配像素分母见 [局部对照](mesh_validation.json)。")
    lines+=["","![原始网格全景](previews/mesh_global.jpg)","","[大产物路径、字节数与 SHA256](artifact_manifest.json) · [阶段耗时和资源](run_manifest.json) · [测试集逐帧指标](metrics_test.json)"]
    (REPORT/"overview.md").write_text("\n".join(lines)+"\n")
    latest=read(REPORT.parent/"latest.json")
    latest.update(kind="full_capture_execution",status=state,observed_stage="report_complete",observed_utc=now(),
                  computation_finished_at_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  actual_requested={s:c["requested"] for s,c in counts_by.items()}|{"total":len(selection)},
                  actual_pose_available={s:c["posed"] for s,c in counts_by.items()}|{"total":len(posed)},
                  actual_updates=30000,actual_final_step=29999,actual_checkpoint_sha256=trained["checkpoint_sha256"],
                  actual_metrics={s:m["rgb_frame_means"] for s,m in metrics.items()},actual_mesh_sha256=sha(Path(mesh["mesh_path"])),
                  completed_overview=(REPORT/"overview.md").relative_to(ROOT).as_posix(),note="Final available-view evaluations and raw Poisson bound to the same checkpoint.")
    save(REPORT.parent/"latest.json",latest)


if __name__=="__main__":
    main()
