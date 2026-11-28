"""Collect actual lane evidence and append a single coordinator-owned Beijing journal."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/".local/runs/interactive-world-20261128-v1"
REPORT=ROOT/"reports/interactive-world"
PUBLIC=RUN/"f_evaluation/publication"
TZ=ZoneInfo("Asia/Shanghai")


def read(p):
    return json.loads(p.read_bytes()) if p.is_file() else {}


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    os.replace(temp,path)


def ref(path):
    with path.open("rb") as f:
        digest=hashlib.file_digest(f,"sha256").hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest)


def evaluated_nodes(name):
    out=RUN/("f_vr" if name.startswith("VR") else "c_dn")/name
    result={}
    for split in ("val","test"):
        nodes=[]
        for file in sorted((out/"evaluation"/split).glob("step-*.json")):
            row=read(file)
            d=row["depth_pixel_pooled"]
            den=d["reference_pixels"]
            nodes.append(dict(step=row["step"],images=row["frame_count"],reference_pixels=den,
                              invalid=d["invalid_prediction_pixels"],psnr=row["rgb_frame_means"]["psnr"],
                              mae_m=d["MAE_m"],p95_m=d.get("absolute_error_P95_m"),
                              correct_10cm_fraction=d["correct_10cm"]/den if den else None,source=ref(file)))
        if nodes:
            result[split]=nodes
    for crop in ("official418","strict376"):
        mesh=out/"mesh"/crop/"mesh_metrics.json"
        if mesh.exists():
            result[crop]=dict(metrics=read(mesh),source=ref(mesh))
    return result


def snapshot():
    q=read(RUN/"state/queue.json")
    if not q.get("training"):
        raise ValueError("Coordinator has not recorded a queue snapshot")
    src=read(ROOT/"configs/interactive_world_protocol.json")
    artifacts={}
    wanted=["restore/*.verified.json","a_depth/freeze.json",
            "a_depth/supervision/*/manifest.json","a_depth/boundary/manifest.json","a_depth/provenance/manifest.json",
            "a_depth/interface_check.json","a_depth/status/*.json","a_depth/quality/*/metrics.json","a_depth/quality/*/calibration.json",
            "a_depth/masked_manifest.json","a_depth/natural_manifest.json","a_depth/vr_transfer/hall_fold0_zero_shot.json",
            "a_depth/vr_transfer/vr_train_only_adaptation.json",
            "b_map/ordered_stream/verified.json","b_map/scale/audit.json","b_map/pose/matches_complete.json",
            "b_map/pose/accepted.json","b_map/equivalence/*.json","c_dn/*/training_complete.json","c_dn/*/evaluation/complete.json",
            "f_vr/*/training_complete.json","f_vr/*/evaluation/complete.json","f_vr/coverage.json",
            "c_dn/numeric_checks.json","c_dn/packet_checks.json","f_vr/blackwell_verification/blackwell.json",
            "d_world/proxy/functional.json","d_world/hall_sensor/attempt-2/support_audit.json",
            "d_world/hall_sensor/attempt-2/functional.json","d_world/memory/results.json","d_world/static_hall/results.json",
            "e_interaction/assets/pusht/source.json","e_interaction/assets/tblock/source.json",
            "e_interaction/assets/rope/source-*.json","e_interaction/prepare/split.json",
            "e_interaction/track_v2/tracking_all.json","e_interaction/calibration/*.json",
            "e_interaction/generated/manifest.json","e_interaction/probe/manifest.json","e_interaction/probe/baselines.json",
            "e_interaction/media/manifest.json","e_interaction/pusht_result_summary.json","e_interaction/phystwin/results_summary.json","e_interaction/phystwin/*/*_metrics.json",
            "e_interaction/phystwin/*/results.json","e_interaction/phystwin/*/complete.json",
            "b_map/pose/targets/frame_01304.json","b_map/pose/targets/frame_01305.json","b_map/pose/targets/frame_01514.json"]
    for pattern in wanted:
        for path in sorted(RUN.glob(pattern)):
            if path.is_file():
                artifacts[str(path.relative_to(RUN))]=ref(path)
    def registered(prefix):
        return [k for k in artifacts if k.startswith(prefix)]
    r1=read(RUN/"a_depth/supervision/R1/manifest.json")
    r2=read(RUN/"a_depth/supervision/R2/manifest.json")
    bound=read(RUN/"a_depth/boundary/manifest.json")
    m=read(RUN/"b_map/ordered_stream/verified.json")
    scale=read(RUN/"b_map/scale/audit.json")
    poss=read(RUN/"b_map/pose/match_progress.json")
    phys=read(RUN/"d_world/hall_sensor/attempt-2/functional.json")
    track=read(RUN/"e_interaction/track_v2/tracking_all.json")
    gen=read(RUN/"e_interaction/generated/manifest.json")
    probe=read(RUN/"e_interaction/probe/manifest.json")
    states={}
    for k,r in q["training"].items():
        reason=r.get("queue_reason","waiting for next sample")
        status="completed" if r["evaluation_complete"] else "running" if r["active"] else "pending"
        if reason.startswith("not_triggered"):
            status="not_triggered"
        if reason.startswith("failed repeatedly"):
            status="failed"
        states[k]={x:r[x] for x in ("durable_updates","verified_updates_lower_bound","training_complete","evaluation_complete","phase","pid","log","exit_code")}
        states[k].update(status=status,reason=reason,resources=r["measured"])
        measures=evaluated_nodes(k)
        if measures:
            states[k]["evaluated_nodes"]=measures
    counts={k:s["verified_updates_lower_bound"] for k,s in states.items()}
    lanes={
      "A":dict(status="running",r1_train_frames=len(r1.get("frames",[])),accepted_pixels=r1.get("accepted_pixels"),
               natural_holes=sum(r["natural_holes"] for r in r1.get("frames",[])),
               boundary_accepted=bound.get("accepted_pixels"),boundary_candidates=sum(r["candidates"] for r in bound.get("frames",[])),
               r2_accepted_pixels=r2.get("accepted_pixels"),r2_train_frames=len(r2.get("frames",[])),
               stage_receipts={Path(k).stem:read(RUN/k).get("status") for k in registered("a_depth/status/")}),
      "B":dict(status="completed" if (RUN/"b_map/pose/accepted.json").is_file() and m and scale else "running",
               stream_frames=m.get("frames"),persistent_frames=m.get("persistent_frames"),
               stream_fps=m.get("timing",{}).get("stream_fps"),end_to_end_fps=m.get("timing",{}).get("end_to_end_fps"),
               scale_audit_frames=scale.get("roles",{}).get("audit"),raw_depth_audit=scale.get("direct",{}),
               matched_pairs=poss.get("completed_pairs",0),planned_pairs=poss.get("pairs"),
               per_target_pose_audits=read(RUN/"b_map/pose/cpu_pipeline.json")),
      "C":dict(status="running",mandatory_updates_lower_bound=sum(counts[k] for k in ("R1","R2","R3","R4","R5")),
               mandatory_updates_target=1200000,variants={k:v for k,v in states.items() if k.startswith("R")}),
      "D":dict(status="running",rigid_sanity=phys,memory_evidence=registered("d_world/memory/")),
      "E":dict(status="running",episodes=track.get("episodes"),track_valid=track.get("valid"),
               track_reference_frames=track.get("frames"),generated_manifest=bool(gen),
               generated_episodes=gen.get("total_episodes",gen.get("episodes") if isinstance(gen.get("episodes"),int) else None),
               generated_control_steps=gen.get("control_steps"),paired_sets=gen.get("inputs",{}).get("episodes"),
               actual_probe_updates=probe.get("training_updates",0),probe_conditions_seeds=len(probe.get("results",[])),
               calibration_artifacts=registered("e_interaction/calibration/"),
               phystwin_conditions=registered("e_interaction/phystwin/")),
      "F":dict(status="running",external_evaluation="Faro",
               variants={k:v for k,v in states.items() if k.startswith("VR")})}
    # Episode summaries can carry all individual tracking records. Keep the fixed denominator.
    if isinstance(lanes["E"]["episodes"],(dict,list)):
        lanes["E"]["episodes"]=len(lanes["E"]["episodes"])
    stage_status=lanes["A"]["stage_receipts"]
    if any(v=="failed" for v in stage_status.values()):
        lanes["A"]["status"]="failed"
    elif stage_status.get("A_R3")=="completed" and stage_status.get("A_VR_adapt")=="completed":
        lanes["A"]["status"]="completed"
    if phys and lanes["D"]["memory_evidence"] and (RUN/"d_world/static_hall/results.json").is_file():
        lanes["D"].update(status="completed",scope="local measured support, synthetic rigid object and independent-teacher memory tests")
    if gen and probe and len(list((RUN/"e_interaction/phystwin").glob("*/G-*_metrics.json")))==6:
        lanes["E"]["status"]="completed"
    if all(s["status"] in ("completed","not_triggered") for s in lanes["C"]["variants"].values()):
        lanes["C"]["status"]="completed"
    if all(s["status"]=="completed" for s in lanes["F"]["variants"].values()) and lanes["C"]["status"]=="completed":
        lanes["F"]["status"]="completed"
    status="completed" if all(l["status"]=="completed" for l in lanes.values()) else "running"
    return dict(schema_version="2.0",stage="interactive-world",status=status,run_id=RUN.name,
        observed_at=q["host"]["time"],timezone="Asia/Shanghai",run_directory=str(RUN),source_baseline="f10e91805877b197d49107347844b675df856d3f",
        repository_head_at_observation=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        expected=dict(mandatory_dn_runs=5,iterations_each=240000,mandatory_vr_runs=2,generated_episodes=1536),
        actual_new_training_updates_lower_bound=sum(counts.values()),durable_training_updates=sum(s["durable_updates"] for s in states.values()),
        host=q["host"],coordinator_pid=q["coordinator_pid"],protocol=src["protocol_id"],lanes=lanes,artifacts=artifacts,
        resume="bash scripts/launch_interactive_world.sh status",
        links=dict(plan="docs/17_interactive_world_plan.md",daily_log="DAILY_RESEARCH_LOG.md",report="reports/interactive-world/20261128-v1/README.md"))


def journal(s):
    l=s["lanes"]; host=s["host"]; variants={**l["C"]["variants"],**l["F"]["variants"]}
    lines=[f'### {s["observed_at"]}｜交互世界服务器实测', "",
        f'运行 `{s["run_id"]}` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `{s.get("repository_head_at_observation",s["source_baseline"])}`。当前正式训练累计下界 {s["actual_new_training_updates_lower_bound"]:,} 步，其中可从已落盘 checkpoint 恢复 {s["durable_training_updates"]:,} 步。',
        "", "| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |",
        "|---|---|---:|---:|---:|---|"]
    for k,v in variants.items():
        lines.append(f'| {k} | {v["phase"]} | {v["verified_updates_lower_bound"]:,} | {v["durable_updates"]:,} | {v["pid"] or "—"} | {v["reason"]} |')
    lines.extend(["",f'A 线 sensor-only 在 {l["A"]["r1_train_frames"]:,} 个训练帧的 {l["A"]["natural_holes"]:,} 个自然孔洞中接受 {l["A"]["accepted_pixels"]} 个。B 线有序流已记录 {l["B"]["stream_frames"]} 帧，独立训练时组的尺度审计使用 {l["B"]["scale_audit_frames"]} 帧。固定大厅相机与传感器读数都来自同一采集。',
        "",f'D 线实测支撑面的代理物体接触为 {l["D"]["rigid_sanity"].get("contact_steps")}/{l["D"]["rigid_sanity"].get("compared_steps")} 控制步。E 线 RGB 跟踪的共同有效标签为 {l["E"]["track_valid"]}/{l["E"]["track_reference_frames"]} 帧。生成轨迹数为 {l["E"]["generated_episodes"]}，对应 {l["E"]["generated_control_steps"]} 个控制时刻。小预测器的 {l["E"]["probe_conditions_seeds"]} 项条件与种子组合累计训练 {l["E"]["actual_probe_updates"]} 步。',
        "",f'资源快照含 {host["cpu_quota"]} 核 CPU 配额，GPU 利用率 {host["gpu_util_percent"]}%，整卡已用显存 {(host["gpu_total_bytes"]-host["gpu_free_bytes"])/1024**3:.2f} GiB，cgroup 总用量 {host["cgroup_total_bytes"]/1024**3:.2f} GiB，清洁非活跃文件页 {host["clean_inactive_file_bytes"]/1024**3:.2f} GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。', "",
        '本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。'])
    if s.get("noon_progress"):
        speed=s["noon_progress"]
        rate=speed["last_30_minutes"].get("hall_updates_per_hour")
        required=speed["required_hall_updates_per_hour_with_tail"]["2h"]
        if rate is not None:
            text=f'六组大厅剩余 {speed["remaining_hall_updates"]:,} 步，最近 30 分钟合计吞吐 {rate:,.0f} 步/小时。'
            if required is not None:
                text+=f'按北京时间 11 月 29 日中午的目标、为最终评估和视频预留两小时，所需平均吞吐为 {required:,.0f} 步/小时。'
            lines.extend(["",text])
    if s.get("acceleration_deployment"):
        lines.extend(["","北京时间 22:42，R2 加入正式训练，原有两组继续运行。原调度器将 VR 终评期间整容器的文件缓存增长分别计入 R0C 和 R1，使 R2 的主存准入达到 87.64 GiB。现按进程树 PSS/USS、瞬态峰值及同一启动窗口内的 cgroup 增量调度，保留原 RAM 和 VRAM 余量。后续组按数值验证结果采用 GPU 图像预取、孔洞项缓存和诊断计算调度。大厅选中模型的渲染也在终评阶段复用。实测数字、分母、适用范围和哈希见[加速记录](reports/interactive-world/20261128-v1/acceleration/README.md)。"])
    if s.get("final_delivery",{}).get("status")=="completed":
        lines.extend(["","完整对照、质量头归因、真实交互、全部评价分母和视频见[最终报告](reports/interactive-world/20261128-v1/final/report.md)。"])
    return "\n".join(lines)


def record(first=False):
    PUBLIC.mkdir(parents=True,exist_ok=True)
    snap=snapshot()
    speed_path=RUN/"f_evaluation/acceleration_20261128/speed_report.json"
    if speed_path.exists():
        from world_speed_report import main as speed_report
        speed=speed_report(datetime.fromisoformat(read(speed_path)["planning_target"]),verbose=False)
        snap["noon_progress"]={k:speed[k] for k in ("planning_target","remaining_hall_updates","required_hall_updates_per_hour_with_tail")}
        snap["noon_progress"]["last_30_minutes"]=speed["rolling"]["last_30_minutes"] or {}
    if snap["status"]=="completed":
        folder=RUN/"f_evaluation/final_delivery"
        index=folder/"index.json"
        final=read(index)
        if final.get("run_id")!=RUN.name or final.get("status")!="completed":
            finished=subprocess.run([sys.executable,str(ROOT/"scripts/world_final_report.py"),"build"],cwd=ROOT)
            final=read(index) if finished.returncode==0 else {}
        if final.get("status")=="completed" and final.get("run_id")==RUN.name:
            snap["final_delivery"]=dict(status="completed",index=ref(index))
            snap["links"]["final_report"]="reports/interactive-world/20261128-v1/final/report.md"
        else:
            snap["status"]="running"
            snap["final_delivery"]={"status":"pending","reason":"awaiting the verified final file manifest"}
    last=read(PUBLIC/"last-entry.json")
    if "noon_progress" in snap and "planning_target" not in last.get("signature",{}):
        snap["acceleration_deployment"]=True
    write(PUBLIC/"snapshot.json",snap)
    date=snap["observed_at"][:10]
    v={**snap["lanes"]["C"]["variants"],**snap["lanes"]["F"]["variants"]}
    signature=dict(date=date,checkpoints={k:[r["durable_updates"],r["training_complete"],r["evaluation_complete"]] for k,r in v.items()},
                   artifacts=sorted(snap["artifacts"]))
    if "noon_progress" in snap:
        signature["planning_target"]=snap["noon_progress"]["planning_target"]
    if "final_delivery" in snap:
        signature["final_delivery"]=snap["final_delivery"]
    if last.get("signature")==signature and not first:
        return
    marker=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    entry=journal(snap)
    note=f"<!-- interactive-world-evidence {marker} -->"
    path=ROOT/"DAILY_RESEARCH_LOG.md"
    if note not in path.read_text(encoding="utf8"):
        with path.open("a",encoding="utf8",newline="\n") as f:
            f.write("\n\n"+entry+"\n\n"+note+"\n")
    write(REPORT/"latest.json",snap)
    # Each entry remains available for a Git publisher after a disconnected interval.
    key=snap["observed_at"].replace(":","").replace("+","_")+"-"+marker
    write(PUBLIC/"outbox"/(key+".json"),dict(key=key,marker=note,markdown=entry,snapshot=snap))
    write(PUBLIC/"last-entry.json",dict(signature=signature,marker=note))
    print(snap["observed_at"],"journaled",marker,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["snapshot","record","watch"])
    p.add_argument("--wait-for-owner",action="store_true",help="Wait for the previous journal writer to release its lock.")
    args=p.parse_args();mode=args.mode
    PUBLIC.mkdir(parents=True,exist_ok=True)
    if mode=="snapshot":
        write(PUBLIC/"snapshot.json",snapshot())
    else:
        import fcntl
        with (PUBLIC/"journal.lock").open("a+b") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX if args.wait_for_owner else fcntl.LOCK_EX|fcntl.LOCK_NB)
            while True:
                record()
                if mode=="record" or read(PUBLIC/"snapshot.json").get("status")=="completed": break
                time.sleep(60)
