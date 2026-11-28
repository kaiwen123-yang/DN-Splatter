"""Assemble the final or explicitly partial interactive-world report from server receipts."""
import argparse
from collections import Counter
from datetime import datetime
import fcntl
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/".local/runs/interactive-world-20261128-v1"
HALL=("R0C","R1","R2","R3","R4","R5")
VR=("VR0C","VR1","VR2")
DEN={"val":29222080,"test":29100252}
METHODS=("model_tsdf","supported_poisson")
MASKS=("native_footprint_bundle","contiguous_block","geometric_boundary_band")
TZ=ZoneInfo("Asia/Shanghai")


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):
            h.update(b)
    return h.hexdigest()


def write(p,doc):
    p.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


def require(test,why):
    if not test:
        raise ValueError(why)


class Evidence:
    def __init__(self):
        self.files={}

    def record(self,p,role,expected=None,expected_size=None):
        p=Path(p)
        if not p.is_absolute():
            p=ROOT/p
        p=p.resolve(strict=True)
        key=p.relative_to(RUN.resolve()).as_posix()
        size=p.stat().st_size
        require(expected_size is None or size==expected_size,f"Changed byte length: {key}")
        if key in self.files:
            row=self.files[key]
            require(not expected or row["sha256"]==expected,f"Conflicting digests: {key}")
            if role not in row["roles"]:
                row["roles"].append(role)
            return row
        if expected:
            require(bool(re.fullmatch("[0-9a-f]{64}",expected)),f"Invalid digest: {key}")
        # Reuse large producer-verified files. Newly inventoried checkpoints,
        # small receipts and unindexed files receive a digest at collection.
        if expected and size>2*1024**2:
            digest=expected
            origin="producer digest with a verified byte length"
        else:
            digest=sha(p)
            require(expected is None or digest==expected,f"Content hash mismatch: {key}")
            origin="SHA-256 readback"
        row=dict(path=str(p),bytes=size,sha256=digest,verification=origin,roles=[role])
        self.files[key]=row
        return row

    def json(self,p,role="result receipt"):
        self.record(p,role)
        return read(p)

    def pointer(self,row,role):
        return self.record(row["path"],role,row["sha256"],row.get("bytes"))


def required():
    names=list(HALL)
    prerequisites=[
        "a_depth/quality/summary.json","a_depth/supervision/R3/manifest.json","a_depth/status/A_R3.json",
        "a_depth/status/A_VR_adapt.json","a_depth/vr_transfer/hall_fold0_zero_shot.json","a_depth/vr_transfer/vr_train_only_adaptation.json",
        "b_map/pose/accepted.json","b_map/pose/matches_complete.json","b_map/scale/audit.json","b_map/ordered_stream/verified.json",
        "d_world/hall_sensor/attempt-2/functional.json","d_world/memory/results.json","d_world/static_hall/results.json",
        "e_interaction/pusht_result_summary.json","e_interaction/phystwin/results_summary.json","e_interaction/generated/manifest.json",
        "e_interaction/probe/manifest.json","e_interaction/media/manifest.json",
    ]
    prerequisites.extend(f"a_depth/quality/fold_{i}/{s}" for i in range(5) for s in ("outer_scores.json","R3_frames.json"))
    prerequisites.extend(f"e_interaction/phystwin/{case}/{method}_metrics.json" for case in
                         ("single_push_rope_1","single_push_rope_4") for method in ("G-NN","G-LingBot","G-Q1"))
    pose=RUN/"b_map/pose/accepted.json"
    decision={"R6":"待独立位姿验收","R7":"待独立位姿验收"}
    if pose.is_file():
        d=read(pose)
        require(d["audit_independent"] and d["total_targets"]==1478,"Incomplete or dependent pose audit")
        if d["accepted_frame_ids"]:
            names.append("R6")
            decision["R6"]="独立审核通过，已触发"
            prerequisites.append("c_dn/integration_decision.json")
            combined=RUN/"c_dn/integration_decision.json"
            if combined.exists():
                j=read(combined)
                if j["independent_pose_benefit"] and j["independent_boundary_benefit"]:
                    names.append("R7");decision["R7"]="两个单项均通过，已触发"
                else:
                    decision["R7"]="单项增益条件未满足"
            else:
                decision["R7"]="待边界与位姿单项判定"
        else:
            decision={k:"未触发，独立审核接受 0 个位姿" for k in decision}
    for k in names:
        prerequisites.extend(f"c_dn/{k}/{f}" for f in ("training_complete.json","evaluation/complete.json","media/receipt.json",
                              "meshes/export_freeze.json","evaluation/test_complete.json","media/local_geometry.json"))
    for k in VR:
        prerequisites.extend(f"f_vr/{k}/{f}" for f in ("training_complete.json","evaluation/complete.json"))
    missing=[s for s in prerequisites if not (RUN/s).is_file()]
    return names,decision,missing


@lru_cache(maxsize=2)
def frozen_ids(split):
    scene=RUN/"restore/tree/.local/runs/capture-full-quality-v3-20261122-v1/inputs/transforms.json"
    return [Path(n).stem for n in read(scene)[split+"_filenames"]]


def node(book,path,split,step):
    v=book.json(path)
    z=v["depth_pixel_pooled"]
    require(v["frame_count"]==184 and len(v["frames"])==184 and v["step"]==step and v["split"]==split and
            z["reference_pixels"]==DEN[split] and [r["id"] for r in v["frames"]]==frozen_ids(split),f"Incorrect denominator, frame identity or node: {path.name}")
    book.record(v["checkpoint"],"validation/development-test checkpoint",v["checkpoint_sha256"])
    return dict(step=step,updates=step+1,images=184,pixels=z["reference_pixels"],invalid=z["invalid_prediction_pixels"],
          psnr=v["rgb_frame_means"]["psnr"],ssim=v["rgb_frame_means"]["ssim"],lpips=v["rgb_frame_means"]["lpips"],
          mae_m=z["MAE_m"],p95_m=z.get("absolute_error_P95_m"),correct10=z["correct_10cm"],coverage10=z["correct_10cm"]/DEN[split])


def hall(book,k,queue,final):
    p=RUN/"c_dn"/k
    current=queue.get(k,{})
    row=dict(observed_updates=current.get("verified_updates_lower_bound",0),durable_updates=current.get("durable_updates",0),
             training_complete=False,evaluation_complete=False,val={},test={},meshes={},media={})
    for n in range(30000,240001,30000):
        f=p/"evaluation/val"/f"step-{n-1:09d}.json"
        if f.is_file():
            row["val"][n]=node(book,f,"val",n-1)
    checkpoint=p/"state/checkpoint.json"
    if checkpoint.is_file():
        d=book.json(checkpoint)
        book.pointer(d,"latest durable checkpoint")
        row["durable_updates"]=d["completed_updates"]
    trained=p/"training_complete.json"
    if not trained.is_file():
        return row
    t=book.json(trained)
    selected=book.json(p/"state/selection.json")
    require(t["variant"]==k and t["completed_updates"]==240000 and t["saved_step"]==239999 and
            t["selection"]==selected and selected["final_step"]==239999 and len(row["val"])==8,f"Inconsistent training freeze: {k}")
    chosen=selected["selected_step"]
    require(chosen+1 in row["val"],f"Invalid selected checkpoint: {k}")
    ck=book.pointer(t,"final 240k checkpoint")
    row.update(observed_updates=240000,durable_updates=240000,selected=chosen+1,training_complete=True)
    if final:
        # The 30k node digests already came from their evaluation receipts.
        directory=Path(ck["path"]).parent
        for n in range(10000,240001,10000):
            book.record(directory/f"step-{n-1:09d}.ckpt","10k recovery checkpoint")
    for step in sorted({chosen,239999}):
        f=p/"evaluation/test"/f"step-{step:09d}.json"
        if f.exists():
            row["test"][step+1]=node(book,f,"test",step)
    finished=p/"evaluation/complete.json"
    if not finished.exists():
        return row
    e=book.json(finished)
    require(e["variant"]==k and e["selected_step"]==chosen and set(row["test"])=={chosen+1,240000},f"Invalid final evaluation: {k}")
    receipt=book.json(p/"evaluation/test_complete.json")
    require(set(receipt["steps"])=={chosen,239999} and receipt["frames_per_step"]==184,f"Test identity differs: {k}")
    for product in e["artifacts"]:
        book.pointer(product,"hall final pipeline product")
    freeze=book.json(p/"meshes/export_freeze.json")
    require(freeze["checkpoint_selection"]==chosen and set(freeze["models"])==set(METHODS),f"Mesh selection: {k}")
    for method in METHODS:
        book.pointer(freeze["models"][method],k+" "+method+" mesh")
        values={}
        for split,den in DEN.items():
            d=book.json(p/"evaluation"/f"{split}_{method}.json")
            z=d["pooled"]
            require(d["method"]==method and d["split"]==split and d["reference_pixels"]==den==z["reference"]
                    and [r["id"] for r in d["frames"]]==frozen_ids(split) and z["hits"]+z["misses"]==den,f"Mesh denominator or frame identity differs: {k} {split} {method}")
            edges=d["edge_counts"]
            values[split]=dict(reference=den,hits=z["hits"],misses=z["misses"],coverage10=z["correct_10"]/den,
                front_conflict=z["foreground"]/den,hit_mae_m=z["MAE_hit_m"],hit_p95_m=z["P95_hit_m"],edge_counts=edges,
                edge_recall=edges["matched_reference_edges_1px"]/edges["reference_edges"] if edges["reference_edges"] else None)
        row["meshes"][method]=values
    row["fixed_views"]=book.json(p/"media/local_geometry.json")
    m=book.json(p/"media/receipt.json")
    require(m["views"]==1846 and m["public_withheld"]==30 and set(m["movies"])=={"full","summary"},f"Media timeline differs: {k}")
    for label,d in m["movies"].items():
        check=d["validation"]
        require(check["frames"]==1846==check["verified_pts"] and check["redacted_frames"]==30,f"Media frame verification: {k}")
        row["media"][label]=book.pointer(d,k+" "+label+" video")
    row["evaluation_complete"]=True
    return row


def vr(book,k,final):
    p=RUN/"f_vr"/k
    if not (p/"evaluation/complete.json").exists() or not (p/"training_complete.json").exists():
        return {"complete":False}
    t=book.json(p/"training_complete.json")
    require(t["variant"]==k and t["completed_updates"]==30000 and t["saved_step"]==29999,"VR update budget: "+k)
    model=book.pointer(t,"VR 30k checkpoint")
    if final:
        for n in (10000,20000,30000):
            book.record(Path(model["path"]).parent/f"step-{n-1:09d}.ckpt","VR recovery checkpoint")
    complete=book.json(p/"evaluation/complete.json")
    require(complete["updates"]==30000 and complete["faro_training_views"]==complete["faro_tuning_views"]==0
            and complete["train_crop_views"]==376 and complete["official_crop_views"]==418,"Faro roles or views differ: "+k)
    for obj in complete["artifacts"]:
        book.pointer(obj,"VR evaluation product")
    row={"complete":True}
    for name in ("official418","strict376"):
        row[name]=book.json(p/"mesh"/name/"mesh_metrics.json")
        mesh=p/"mesh"/name/"mesh_cull.ply"
        if mesh.is_file() and final:
            book.record(mesh,"VR culled mesh")
    row["depth"]={}
    d=book.json(p/"evaluation/faro_common_mask.json")
    for region,n in (("within",42),("with",186)):
        group=d["trajectories"][region]
        require(group["frame_count"]==n and group["pooled"]["reference"]>0,"Incomplete Faro split: "+region)
        z=group["pooled"]
        row["depth"][region]=dict(frames=n,reference=z["reference"],invalid=z["invalid_prediction"],**z["fixed_reference"])
    return row


def other_lanes(book,final):
    p=RUN
    def get(s):
        f=p/s
        return book.json(f) if f.is_file() else {}
    a={}
    for name in ("R1","R2","R3"):
        d=get(f"a_depth/supervision/{name}/manifest.json")
        if d:
            rows=d["frames"]
            require(len(rows)==1478 and d["scope"]=="train","Incomplete depth supervision "+name)
            a[name]=dict(frames=len(rows),accepted=d["accepted_pixels"],holes=sum(r["natural_holes"] for r in rows),
                accepted_by_source=dict(zip(d["source_codes"][1:],[sum(values) for values in zip(*(r["accepted_by_source"] for r in rows))],strict=True)))
    q=get("a_depth/quality/summary.json")
    if q:
        require(len(q["folds"])==5 and len({f["fold"] for f in q["folds"]})==5,"Outer folds")
        ops=[x for x in q["operations"] if x["rep"]=="footprint_aware" and x["point"]=="risk_5pct" and x["method"] in ("Q0","Q1")]
        require(len(ops)==6,"Missing registered operating points")
        a["oof"]={mask:{x["method"]:{k:x[k] for k in ("n","accepted","wrong","coverage","risk","coverage_ci95","risk_ci95")}
                         for x in ops if x["mask"]==mask} for mask in MASKS}
    for key in ("a_depth/status/A_R3.json","a_depth/status/A_VR_adapt.json"):
        d=get(key)
        if final:
            require(d["status"]=="completed" and d["exit_code"]==0,"Depth stage incomplete: "+key)
    b=get("b_map/pose/accepted.json");m=get("b_map/ordered_stream/verified.json");s=get("b_map/scale/audit.json")
    b={k:v for k,v in b.items() if k in ("total_targets","accepted_frame_ids","fit_solutions","audit_independent")}
    b.update(stream_frames=m.get("frames"),persistent=m.get("persistent_frames"),timing=m.get("timing"),scale_audit=s.get("direct"),scale_roles=s.get("roles"))
    d=get("d_world/static_hall/results.json")
    hall=get("d_world/hall_sensor/attempt-2/functional.json")
    if d:
        require(d["positions"]==1846 and set(d["methods"])=={"M0","M1","M2"},"Static hall population")
        book.pointer(d["frame_audit"],"full static memory event trace")
    d=dict(positions=d.get("positions"),source_groups=d.get("source_groups"),anchors=d.get("anchors"),counts=d.get("counts"),
           methods=d.get("methods"),synthetic_rigid_body=hall)
    e=get("e_interaction/pusht_result_summary.json")
    if e:
        gen=e["generation"]; probe=e["probe"]
        require(gen["paired_inputs"]==512 and sum(gen["episodes_per_version"].values())==1536 and gen["control_steps"]==230400
                and probe["total_updates"]==300000 and probe["updates_each"]==20000,"Incomplete paired data or probe")
        for k in ("manifest","bank"):
            book.pointer(gen[k],"generated data "+k)
        for k in ("manifest","preview"):
            book.pointer(e["media"][k],"real replay media "+k)
        book.pointer(probe["manifest"],"state probe index")
        e=dict(independent_tracks=e["independent_tracks"],replay={k:v["results"] for k,v in e["replay"].items()},
               generation={k:v for k,v in gen.items() if k not in ("manifest","bank")},probe_updates=probe["total_updates"],
               probe_2s={k:{seed:v["heldout"]["2s"] for seed,v in values.items()} for k,values in probe["conditions"].items()})
    ropes=get("e_interaction/phystwin/results_summary.json")
    done=sum(v.get("completed",False) for group in ropes.values() for v in group.values() if isinstance(v,dict))
    if final:
        require(done==6,"PhysTwin future comparisons are incomplete")
    if ropes:
        for target,conditions in ropes.items():
            for method in ("G-NN","G-LingBot","G-Q1"):
                if conditions.get(method,{}).get("completed"):
                    book.pointer(conditions[method]["source"],target+" "+method)
    e["rope"]=ropes
    return {"A":a,"B":b,"D":d,"E":e}


def resource_results():
    p=RUN/"state/throughput.jsonl"
    if not p.exists():
        return {}
    rows=[json.loads(line) for line in p.read_bytes().splitlines() if line]
    times=[datetime.fromisoformat(r["host"]["time"]) for r in rows]
    durations=[(b-a).total_seconds() for a,b in zip(times,times[1:])]
    normal=[i for i,seconds in enumerate(durations) if 0<seconds<=120]
    total=sum(durations[i] for i in normal)
    combos=Counter()
    for i in normal:
        combos[",".join(rows[i].get("active",[])) or "no active heavy worker"]+=durations[i]
    return dict(start=times[0].isoformat(),end=times[-1].isoformat(),samples=len(rows),measured_seconds=total,
        mean_gpu_percent=sum(rows[i]["host"]["gpu_util_percent"]*durations[i] for i in normal)/total if total else None,
        max_concurrent_workers=max(len(r["active"]) for r in rows),combination_seconds=dict(combos),
        measured_peak_device_GiB=max((r["host"]["gpu_total_bytes"]-r["host"]["gpu_free_bytes"])/1024**3 for r in rows),
        measurement="Time-weighted observed intervals up to 120 s; larger gaps excluded.")


def collect(names,decision,missing,final):
    q=read(RUN/"state/queue.json")
    book=Evidence()
    metrics=other_lanes(book,final)
    metrics.update(run_id=RUN.name,status="completed" if final else "partial",observed_at=q["host"]["time"],pending=missing,
          protocol_sha256=sha(ROOT/"configs/interactive_world_protocol.json"),
          conditional=decision,hall={k:hall(book,k,q["training"],final) for k in names},
          vr={k:vr(book,k,final) for k in VR},resources=resource_results())
    if final:
        require(all(d["evaluation_complete"] for d in metrics["hall"].values()) and all(v["complete"] for v in metrics["vr"].values()),"Incomplete terminal receipts")
    for f in ("a_depth/natural_manifest.json","a_depth/masked_manifest.json","a_depth/boundary/manifest.json",
              "a_depth/vr_transfer/hall_fold0_zero_shot.json","a_depth/vr_transfer/vr_train_only_adaptation.json",
              "b_map/pose/matches_complete.json","b_map/ordered_stream/verified.json","d_world/memory/results.json",
              "e_interaction/generated/manifest.json","e_interaction/probe/manifest.json","e_interaction/media/manifest.json"):
        p=RUN/f
        if p.is_file():
            book.record(p,"file-level artifact index or primary result")
    event_file=RUN/"state/coordinator-events.jsonl"
    failures=[]
    if event_file.exists():
        for line in event_file.read_bytes().splitlines():
            e=json.loads(line)
            if e["stage"]=="worker_exit":
                failures.append(dict(time=e["time"],variant=e["details"]["variant"],exit_code=e["details"]["code"]))
            elif e["stage"]=="coordinator_error":
                failures.append(dict(time=e["time"],stage="coordinator_error"))
    metrics["recorded_exits"]=failures
    return metrics,book.files


def fmt(v,digits=2,scale=1):
    return "—" if v is None else f"{scale*v:,.{digits}f}"


def table(lines,columns,rows):
    lines+=["","| "+" | ".join(columns)+" |","|"+"|".join("---" for _ in columns)+"|"]
    lines.extend("| "+" | ".join(map(str,row))+" |" for row in rows)
    lines.append("")


def markdown(m):
    complete=m["status"]=="completed"
    h=m["hall"]; a=m["A"]; b=m["B"]; d=m["D"]; e=m["E"]
    lines=["# 灵波支持的可校准交互世界"+("｜最终实验结果" if complete else "｜阶段结果"),
        "",f'北京时间 {m["observed_at"]}。大厅完整训练 {sum(r["training_complete"] for r in h.values())}/{len(h)} 组，完整评价 {sum(r["evaluation_complete"] for r in h.values())}/{len(h)} 组。VR/Faro 完整评价 {sum(r["complete"] for r in m["vr"].values())}/3 组。空栏对应尚无冻结结果的项目。',
        "",("[12 秒同相机重建对比](preview.mp4) · [固定视角预览](preview.jpg)" if any(r["media"] for r in h.values())
            else "[真实留出 PushT 阶段预览](preview.mp4) · [预览画面](preview.jpg)"),
        "", "## 大厅渲染与模型深度","",
        "验证集和 development-test 均按原顺序保留 184 帧，度量深度参考来自同次采集的传感器。验证的有效分母为 29,222,080 像素，开发测试为 29,100,252 像素。模型选点读取八个全验证节点，表中的步数使用累计更新数。"]
    rows=[]
    for k,r in h.items():
        if not r["training_complete"]:
            n=max(r["val"],default=None); v=r["val"].get(n,{})
            rows.append((k,"最近全验证",f'{n:,}' if n else "—",fmt(v.get("psnr")),fmt(v.get("mae_m"),2,100),"—","—","—","—"))
            continue
        for n,role in ((r["selected"],"验证选择"),(240000,"训练终点")):
            v=r["val"][n];t=r["test"].get(n,{})
            rows.append((k,role,f"{n:,}",fmt(v["psnr"]),fmt(v["mae_m"],2,100),fmt(t.get("psnr")),
                fmt(t.get("mae_m"),2,100),fmt(t.get("p95_m"),2,100),fmt(t.get("coverage10"),2,100)))
    table(lines,["组","节点","累计更新","val PSNR/dB","val MAE/cm","test PSNR/dB","test MAE/cm","test P95/cm","test 正确覆盖@10 cm/%"],rows)
    lines.extend(["## 同出口网格与方法对照","",
        "每组导出验证选中的模型。正确覆盖与前方冲突使用完整有效参考分母，命中 MAE 只在有有效预测的参考射线上汇总。边界召回衡量一像素邻域内匹配到的传感器边界比例。"])
    rows=[]
    for k,r in h.items():
        for method,t in r["meshes"].items():
            val,test=t["val"],t["test"]
            rows.append((k,method,fmt(val["coverage10"],2,100),fmt(test["coverage10"],2,100),fmt(test["front_conflict"],2,100),
                         f'{test["hits"]:,}',fmt(test["hit_mae_m"],2,100),fmt(test["edge_recall"],2,100)))
    if rows:
        table(lines,["组","出口","val 覆盖/%","test 覆盖/%","test 前方冲突/%","test 有效命中数","命中 MAE/cm","test 边界召回/%"],rows)
    else:
        lines.extend(["","大厅模型的最终网格评价仍在排队。"])
    comparisons=(("R1","R0C","传感器补洞"),("R2","R1","加入冻结灵波候选"),("R3","R2","Q1 选择与拒绝"),
                 ("R4","R3","几何支持下重新分配容量"),("R5","R3","关闭 normal TV"))
    pairs=[]
    for k0,k1,label in comparisons:
        one,two=h[k0],h[k1]
        if 240000 not in one["test"] or 240000 not in two["test"]:
            continue
        x,y=one["test"][240000],two["test"][240000]
        mesh=None
        if one["meshes"] and two["meshes"]:
            mesh=one["meshes"]["supported_poisson"]["test"]["coverage10"]-two["meshes"]["supported_poisson"]["test"]["coverage10"]
        err=(x["mae_m"]-y["mae_m"])*100 if x["mae_m"] is not None and y["mae_m"] is not None else None
        pairs.append((k0+" − "+k1,label,fmt(x["psnr"]-y["psnr"],3),fmt(err,3),fmt(x["coverage10"]-y["coverage10"],3,100),fmt(mesh,3,100)))
    if pairs:
        lines+=["","同预算差值按表中前组减后组计算。模型指标取 240k 终点，网格指标取两组分别由验证集冻结的节点。"]
        table(lines,["比较","实验变化","Δ PSNR/dB","Δ MAE/cm","Δ 模型正确覆盖/百分点","Δ Poisson 正确覆盖/百分点"],pairs)
    thin=[]
    for k in ("R3","R4","R5"):
        for obj,label in (("chair_legs","椅腿"),("foam_teeth","泡沫齿边")):
            geom=h[k].get("fixed_views",{}).get(obj)
            if not geom:
                continue
            result=geom["results"]["model"];edges=result["edge_counts"];two=result["two_mode_data"]
            ref_width=two["reference"]["horizontal"]["median_width_px"] if two else None
            new_width=two["prediction"]["horizontal"]["median_width_px"] if two else None
            thin.append((k,label+" "+geom["id"],result["measured_pixels"],edges["reference_edges"],edges["collapsed_reference_edges"],
                         fmt(ref_width,1),fmt(new_width,1)))
    if thin:
        lines+=["","固定 ROI 的边界坍缩计数来自模型 Z。宽度是在图像中有两端背景支持的水平前景段中位数。双深度模态分不开时保留空栏。"]
        table(lines,["组","固定位置","量测像素","参考边界数","坍缩边界数","观测宽度/px","预测宽度/px"],thin)
    lines+=["","## 几何来源与风险",""]
    if "oof" in a:
        lines+=["五折在十秒训练组间交叉拟合，各折用独立校准组确定错误接纳率目标为 5% 的阈值，再评价外折。下表报告 footprint-aware 输入的结果，误差超过 10 cm 计错。"]
        rows=[]
        for key,title in zip(MASKS,("原生量测足迹","连续块","几何边界"),strict=True):
            x,y=a["oof"][key]["Q0"],a["oof"][key]["Q1"]
            rows.append((title,f'{y["n"]:,}',fmt(x["coverage"],2,100),fmt(x["risk"],2,100),fmt(y["coverage"],2,100),
                         fmt(y["risk"],2,100),f'{y["wrong"]:,} / {y["accepted"]:,}'))
        table(lines,["隐藏区域","外折参考像素","Q0 覆盖/%","Q0 错误接纳/%","Q1 覆盖/%","Q1 错误接纳/%","Q1 错误/已接纳"],rows)
        over=[label for key,label in zip(MASKS,("原生量测足迹","连续块","几何边界"),strict=True)
              if (a["oof"][key]["Q1"]["risk"] or 0)>.05]
        if over:
            lines.append("Q1 在"+"、".join(over)+"的外折已接纳样本上超过 5% 错误接纳目标。")
    if all(k in a for k in ("R1","R2","R3")):
        lines.append(f'自然孔洞共 {a["R1"]["holes"]:,} 个。R1、R2、R3 分别接纳 {a["R1"]["accepted"]:,}、{a["R2"]["accepted"]:,}、{a["R3"]["accepted"]:,} 个。其中 R3 直接使用 MDM 深度 {a["R3"]["accepted_by_source"].get("MDM",0):,} 个。自然孔洞的真实误差需结合新视角或独立几何判断。')
    if b.get("total_targets"):
        lines+=["",f'Map 有序预取流覆盖 {b["stream_frames"]:,} 帧，保留 {b["persistent"]} 个持久帧。新匹配和两种初值覆盖 {b["total_targets"]:,} 个目标，独立审核接受 {len(b["accepted_frame_ids"])} 个局部位姿。R6 {m["conditional"]["R6"]}，R7 {m["conditional"]["R7"]}。']
    lines+=["","## 外部 Faro 评价","",
        "VR 保留 376 个训练视角、42 个同采集留出、186 个异采集视角。全 275,823,072 个训练 sensor 像素均有有效深度。VR1 与 VR2 在自然孔洞上的增量监督同为零。零自然孔洞条件下，组间差异也包含重复训练波动。"]
    rows=[]
    for k,v in m["vr"].items():
        if v["complete"]:
            same,diff=v["depth"]["within"],v["depth"]["with"];s=v["strict376"]
            rows.append((k,fmt(same["MAE_m"],3,100),fmt(diff["MAE_m"],3,100),f'{same["reference"]:,} / {diff["reference"]:,}',
                         fmt(v["official418"]["F-score"],5),fmt(s["F-score"],5),fmt(s["Acc"],3,100),fmt(s["Comp"],3,100)))
    if rows:
        table(lines,["组","同采集 MAE/cm","异采集 MAE/cm","同/异采集有效像素","418 F-score","376 F-score","376 accuracy/cm","376 completeness/cm"],rows)
    lines+=["## 交互、数据有效性与失败记录",""]
    if d.get("positions"):
        mem=d["methods"]["M2"];phys=d["synthetic_rigid_body"]
        lines.append(f'大厅静态诊断覆盖 {d["positions"]:,} 个位置，跨 {d["source_groups"]} 个时间组形成 {d["anchors"]} 个锚点。M2 的 {mem["foreground_occlusions"]:,} 次前方遮挡中有 {mem["new_retraction_on_occlusion"]} 次新撤回。{mem["clear_ray_comparisons"]:,} 次看穿旧位置的有效比较中有 {mem["new_static_retraction_events"]} 次新撤回。大厅缺少独立对象运动标签。实测支撑面上的合成刚体在 {phys["compared_steps"]} 个控制步中发生 {phys["contact_steps"]} 步接触，最大穿透 {fmt(phys["peak_penetration_m"],3,1000)} mm。')
    if "generation" in e:
        c=e["generation"];openloop=e["replay"]["open_loop"];online=e["replay"]["causal_online"];test=openloop["C0"]
        lines+=["",f'真实 PushT 测试覆盖 {test["episodes"]} 个 episode、{test["frames"]:,} 帧，冻结 RGB 跟踪器提供 {test["valid"]:,} 帧有效对象标签。']
        table(lines,["条件","开环位置误差/mm","在线位置误差/mm"],[(k,fmt(openloop[k]["mean_episode_position_m"],2,1000),
              fmt(online[k]["mean_episode_position_m"],2,1000)) for k in ("C0","C1","C2","C3")])
        lines.append(f'{c["paired_inputs"]} 组初态和动作生成 {sum(c["episodes_per_version"].values()):,} 条轨迹、{c["control_steps"]:,} 个控制时刻。各版本穿透超过 5 mm 的 episode 数分别为 '+ "、".join(f'{k} {v["penetration_over_5mm"]}/{c["episodes_per_version"][k]}' for k,v in c["failure_counts"].items())+
                     f'。C2/C3 有 {c["C2_C3_bit_identical_object_trajectories"]} 条对象轨迹逐位相同。')
        table(lines,["2 秒真实位置预测","有效窗口","seed 42/mm","seed 43/mm","seed 44/mm"],
              [(k,f'{s["42"]["windows"]:,}',*[fmt(s[str(seed)]["mean_episode_position_m"],2,1000) for seed in (42,43,44)])
               for k in ("RealOnly","Real+resampledReal","Real+C0","Real+C2","Real+C3") for s in [e["probe_2s"][k]]])
        lines.append("小预测器只评价状态与动作条件下的对象响应。现有实机数据未提供视觉策略的机器人成功率。")
    if e.get("rope"):
        lines+=["","真实 PhysTwin 使用仅由训练前缀建立的共同 spring–mass 先验，修订一次 camera 0 观测后，给定相同控制点开环预测 0.5 秒。"]
        table(lines,["对象","有效跟踪点","G-NN/cm","G-LingBot/cm","G-Q1/cm"],[(k,vs.get("G-NN",{}).get("endpoint",{}).get("valid_tracks","—"),
            *[fmt(vs.get(x,{}).get("endpoint",{}).get("track_mean_m"),2,100) for x in ("G-NN","G-LingBot","G-Q1")]) for k,vs in e["rope"].items()])
        matched=[vs for vs in e["rope"].values() if all(vs.get(k,{}).get("endpoint",{}).get("track_mean_m") is not None for k in ("G-NN","G-LingBot","G-Q1"))]
        if matched:
            wins=[sum(vs[k]["endpoint"]["track_mean_m"]<vs["G-NN"]["endpoint"]["track_mean_m"] for vs in matched) for k in ("G-LingBot","G-Q1")]
            lines.append(f'直接灵波补全和 Q1 筛选的末帧误差低于 NN 的样例数分别为 {wins[0]}/{len(matched)}、{wins[1]}/{len(matched)}。两例对象的形状与物性不同，结果适用于给定记录作用的局部状态预测。')
    r=m["resources"]
    if r:
        lines+=["","## 运行与恢复","",f'有效采样覆盖 {fmt(r["measured_seconds"]/3600)} 小时，时间加权 GPU 利用率 {fmt(r["mean_gpu_percent"])}%，观测整卡占用峰值 {fmt(r["measured_peak_device_GiB"])} GiB，最多同时存在 {r["max_concurrent_workers"]} 个重任务。分组吞吐由服务器 `state/throughput.jsonl` 记录。']
    if m["recorded_exits"]:
        lines+=["",f'调度器记录 {len(m["recorded_exits"])} 条异常退出或调度异常，各条时间和组别见机器结果。']
    lines+=["","[模型、网格、轨迹与视频路径和 SHA-256](artifacts.json) · [机器结果及分母](metrics.json) · [每日记录](../../../../DAILY_RESEARCH_LOG.md) · [采集与实现记录](../README.md)",
            "",f'服务器运行目录为 `{RUN}`。项目根目录执行 `bash scripts/launch_interactive_world.sh status` 可查看每个分支的最终回执或下一步。']
    return "\n".join(lines)+"\n"


def exact_frames(path,segments,stamps):
    """Seek once per contiguous excerpt and check each source PTS exactly."""
    import av
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        stream.thread_count=1
        for segment in segments:
            first=Fraction(stamps[segment[0]],1000000)
            container.seek(int(first/stream.time_base),stream=stream,backward=True,any_frame=False)
            idx=0
            for frame in container.decode(stream):
                actual=frame.pts*frame.time_base
                expected=Fraction(stamps[segment[idx]],1000000)
                if actual<expected:
                    continue
                require(actual==expected,f"Preview input PTS differs at frame {segment[idx]}")
                yield segment[idx],frame.to_ndarray(format="bgr24")
                idx+=1
                if idx==len(segment):
                    break
            require(idx==len(segment),"Preview excerpt truncated")


def preview_frames(metrics):
    import av
    import cv2
    import numpy as np
    complete={k:v for k,v in metrics["hall"].items() if v.get("media")}
    if not complete:
        # A draft can show actual finished interaction footage while hall training runs.
        source=RUN/"e_interaction/media/heldout_60s.mp4"
        if not source.exists():
            return
        with av.open(str(source)) as media:
            s=media.streams.video[0];s.thread_count=1
            for i,frame in enumerate(media.decode(s)):
                if i%6:
                    continue
                if i>=360:
                    break
                b=frame.to_ndarray(format="bgr24")
                height=max(2,round(b.shape[0]*960/b.shape[1]/2)*2)
                image=cv2.resize(b,(960,height),interpolation=cv2.INTER_AREA)
                banner=np.full((28,960,3),25,np.uint8)
                cv2.putText(banner,"HELD-OUT PUSHT  |  hall final renders pending",(10,19),0,.57,(240,240,240),1,cv2.LINE_AA)
                yield np.concatenate([banner,image],axis=0)
        return
    seq=read(ROOT/"reports/capture-full-dn-30k/20261119-v1/selection.json")["rows"]
    require(len(seq)==1846,"Preview timeline differs")
    t0=int(seq[0]["actual_rgb_timestamp_ns"])
    stamps=[(int(r["actual_rgb_timestamp_ns"])-t0+500)//1000 for r in seq]
    segments=[list(range(center-4,center+11)) for center in (120,240,1304,1420)]
    sources=[exact_frames(v["media"]["full"]["path"],segments,stamps) for v in complete.values()]
    try:
        for views in zip(*sources,strict=True):
            i=views[0][0]
            require(all(idx==i and img.shape==(768,1920,3) for idx,img in views),"Cross-model preview alignment")
            hs=122;rh=164;rows=(len(views)+1)//2
            canvas=np.full((hs+rh*rows,1024,3),20,np.uint8)
            src=views[0][1]
            canvas[8:116,4:196]=cv2.resize(src[48:408,:640],(192,108),interpolation=cv2.INTER_AREA)
            canvas[8:116,200:392]=cv2.resize(src[48:408,1280:],(192,108),interpolation=cv2.INTER_AREA)
            for j,line in enumerate((f'{seq[i]["id"]}   |   acquisition {stamps[i]/1e6:.2f} s',
                    "Reference RGB and sensor Z at left","Below: selected-checkpoint RGB and supported Poisson Z")):
                cv2.putText(canvas,line,(405,28+j*30),0,.52,(235,235,235),1,cv2.LINE_AA)
            for n,((k,v),(_,image)) in enumerate(zip(complete.items(),views,strict=True)):
                x=(n%2)*512;y=hs+(n//2)*rh
                cv2.putText(canvas,f'{k}   {v["selected"]:,} updates',(x+5,y+16),0,.50,(230,230,230),1,cv2.LINE_AA)
                canvas[y+20:y+164,x:x+256]=cv2.resize(image[48:408,640:1280],(256,144),interpolation=cv2.INTER_AREA)
                canvas[y+20:y+164,x+256:x+512]=cv2.resize(image[408:768,640:1280],(256,144),interpolation=cv2.INTER_AREA)
            yield canvas
    finally:
        for source in sources:
            source.close()


def render_preview(folder):
    import av
    import cv2
    cv2.setNumThreads(1)
    m=read(folder/"metrics.json");stream=None;count=0
    with av.open(str(folder/"preview.mp4"),"w",options={"movflags":"+faststart"}) as mux:
        for i,canvas in enumerate(preview_frames(m)):
            if stream is None:
                stream=mux.add_stream("libx264",rate=5)
                stream.width=canvas.shape[1];stream.height=canvas.shape[0]
                stream.pix_fmt="yuv420p";stream.codec_context.thread_count=2
                stream.options={"preset":"fast","crf":"26","maxrate":"1400k","bufsize":"1600k"}
                stream.codec_context.max_b_frames=0
            frame=av.VideoFrame.from_ndarray(canvas,format="bgr24")
            frame.pts=i;frame.time_base=Fraction(1,5)
            for pkt in stream.encode(frame):
                mux.mux(pkt)
            if i==20:
                require(cv2.imwrite(str(folder/"preview.jpg"),canvas,[cv2.IMWRITE_JPEG_QUALITY,85]),"Poster write")
            count+=1
        require(count==60 and stream is not None,"Preview requires all 60 source images")
        for pkt in stream.encode():
            mux.mux(pkt)
    with av.open(str(folder/"preview.mp4")) as mux:
        frames=0
        for i,frame in enumerate(mux.decode(video=0)):
            require(frame.pts*frame.time_base==Fraction(i,5),"Published preview timing differs")
            frames+=1
        require(frames==count and cv2.imread(str(folder/"preview.jpg")) is not None,"Published media could not be decoded")


def build(mode):
    names,decisions,missing=required()
    final=mode=="build"
    if final and missing:
        print(json.dumps(dict(status="pending",missing=missing),ensure_ascii=False))
        return 2
    folder=RUN/"f_evaluation"/("final_delivery" if final else "final_draft")
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/"writer.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=folder/"index.json"
        if final and receipt.exists():
            d=read(receipt)
            require(d["status"]=="completed" and d["run_id"]==RUN.name,"Final report identity")
            for p in d["files"]:
                item=folder/p["name"]
                require(item.stat().st_size==p["bytes"] and sha(item)==p["sha256"],"Published file changed")
            print(receipt)
            return 0
        metrics,artifacts=collect(names,decisions,missing,final)
        # The prior index continues to point to a complete set until the new
        # files are ready. The publisher verifies each digest before sending.
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            work=Path(tmp)
            (work/"report.md").write_text(markdown(metrics),encoding="utf8")
            write(work/"metrics.json",metrics)
            write(work/"artifacts.json",dict(run_id=RUN.name,status=metrics["status"],files=len(artifacts),
                nested_indices="Dataset shards and per-frame outputs are indexed by their own verified manifests.",
                artifacts=[artifacts[k] for k in sorted(artifacts)]))
            has_media=any(v["media"] for v in metrics["hall"].values()) or (RUN/"e_interaction/media/heldout_60s.mp4").exists()
            if has_media:
                env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONNOUSERSITE":"1"}
                subprocess.run([str(ROOT/".local/conda/dn-splatter-bw/bin/python"),str(Path(__file__).resolve()),
                                "render-preview","--folder",str(work)],env=env,check=True)
            require(not final or has_media,"Final aligned preview absent")
            rows=[]
            for f in sorted(work.iterdir()):
                require(f.stat().st_size<4*1024**2,"Oversized public result: "+f.name)
                rows.append(dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)))
            index=dict(run_id=RUN.name,status=metrics["status"],observed_at=metrics["observed_at"],files=rows)
            write(work/"index.json",index)
            for r in rows:
                os.replace(work/r["name"],folder/r["name"])
            os.replace(work/"index.json",receipt)
        print(json.dumps(index,ensure_ascii=False))
    return 0


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["draft","build","check","render-preview"])
    p.add_argument("--folder",type=Path)
    a=p.parse_args()
    if a.mode=="render-preview":
        if a.folder is None or not a.folder.resolve().is_relative_to((RUN/"f_evaluation").resolve()):
            p.error("A project evaluation output folder is required")
        render_preview(a.folder)
    elif a.mode=="check":
        _,_,gaps=required();print(json.dumps(dict(ready=not gaps,missing=gaps),ensure_ascii=False))
        sys.exit(2 if gaps else 0)
    else:
        sys.exit(build(a.mode))
