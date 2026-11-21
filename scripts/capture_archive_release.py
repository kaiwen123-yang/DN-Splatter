"""Prepare and publish the public derivatives of the frozen capture, separately from the private archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

import numpy as np

from capture_archive_transport import (ROOT, ID, RESULT, STAGE, REPORT, BASE_REPORT, B0, B1, VIDEO, atomic,
                                      check, checksum_record, digest, document, range_hashes, utc)

REPO="kaiwen123-yang/DN-Splatter"
TAG="capture-full-metric3d-30k-20261120-v1"
PUB=ROOT/STAGE/"public"
ASSETS=PUB/"assets"
RAW="DepthAndNormalMapsPoisson_"
TYPES={"double":"<f8","float":"<f4","float32":"<f4","float64":"<f8","uchar":"u1","uint8":"u1",
       "uint":"<u4","int":"<i4","int32":"<i4","uint32":"<u4"}


def frozen(rel):
    return json.loads(subprocess.check_output(["git","show",RESULT+":"+rel],cwd=ROOT))


def run(args,**kw):
    return subprocess.run([str(x) for x in args],check=True,**kw)


def zip_entry(name):
    info=zipfile.ZipInfo(name,date_time=(2026,11,20,0,0,0))
    info.compress_type=zipfile.ZIP_DEFLATED
    info.external_attr=0o100644<<16
    return info


def geometry(label,kind,rel,expected):
    src=ROOT/rel; check(checksum_record(src)==expected,rel)
    leaf=label+"_"+kind+".ply"
    target=ASSETS/(leaf+".zip");tmp=target.with_suffix(".pending")
    with src.open("rb") as f:
        header=[]
        while len(header)<60:
            line=f.readline(512).decode("ascii").strip()
            check(line,"PLY header "+rel)
            if line=="end_header":break
            header.append(line)
        check(header[:2]==["ply","format binary_little_endian 1.0"] and line=="end_header","PLY format")
        props=[];faceprop=None;vertices=faces=0;elem=None
        for line in header[2:]:
            fields=line.split()
            if fields[0]=="comment":continue
            if fields[0]=="element":
                _,elem,n=fields;check(elem in ("vertex","face"),line)
                if elem=="vertex": vertices=int(n)
                else: faces=int(n)
            elif fields[0]=="property" and elem=="vertex":
                _,t,k=fields;check(t in TYPES and k in ("x","y","z","nx","ny","nz","red","green","blue","alpha"),line)
                props.append((k,t))
            elif fields[0]=="property" and elem=="face":
                check(len(fields)==5 and fields[1]=="list" and fields[2]=="uchar" and fields[3] in ("uint","int","int32","uint32") and fields[4]=="vertex_indices",line)
                faceprop=line
            else:raise ValueError("Unsupported PLY content")
        keep=[(k,t) for k,t in props if k in ("x","y","z","nx","ny","nz")]
        check([k for k,t in keep]==["x","y","z","nx","ny","nz"],"geometry attributes")
        check(not faces or faceprop,"face schema")
        dtype=np.dtype([(k,TYPES[t]) for k,t in props])
        reduced=np.dtype([(k,TYPES[t]) for k,t in keep])
        lines=["ply","format binary_little_endian 1.0","comment Public geometry of "+label+" "+kind,
               f"element vertex {vertices}",*[f"property {t} {k}" for k,t in keep]]
        if faceprop:lines.extend([f"element face {faces}",faceprop])
        lines.append("end_header")
        h=hashlib.sha256();count=0
        with zipfile.ZipFile(tmp,"w",allowZip64=True) as archive, archive.open(zip_entry(leaf),"w",force_zip64=True) as out:
            def emit(b):
                nonlocal count
                out.write(b);h.update(b);count+=len(b)
            emit(("\n".join(lines)+"\n").encode("ascii"))
            for start in range(0,vertices,160000):
                n=min(160000,vertices-start)
                raw=f.read(n*dtype.itemsize);check(len(raw)==n*dtype.itemsize,rel)
                a=np.frombuffer(raw,dtype=dtype);b=np.empty(n,dtype=reduced)
                for k,t in keep:b[k]=a[k]
                emit(b.tobytes())
            for start in range(0,faces,160000):
                n=min(160000,faces-start);raw=f.read(n*13)
                check(len(raw)==n*13 and np.all(np.frombuffer(raw,np.uint8)[::13]==3),"triangle block")
                emit(raw)
            check(f.read(1)==b"","Trailing PLY payload")
    with zipfile.ZipFile(tmp) as archive,archive.open(leaf) as restored:
        actual=hashlib.sha256()
        for b in iter(lambda:restored.read(4*1024**2),b""):actual.update(b)
    check(actual.digest()==h.digest(),"PLY round trip")
    tmp.replace(target)
    return dict(name=target.name,kind="geometry-only PLY",model=label,geometry=kind,vertices=vertices,faces=faces,
                source_sha256=expected["sha256"],uncompressed_sha256=h.hexdigest(),uncompressed_bytes=count,**checksum_record(target))


def geometry_parameters(label,run_dir,source):
    import torch
    from vr_room_checkpoint import numpy_scalar_scope
    rec=document(ROOT/run_dir/"training_complete.json")
    ck=Path(rec["checkpoint"])
    check(ck.resolve().is_relative_to((ROOT/run_dir/"training").resolve()) and digest(ck)==source==rec["checkpoint_sha256"] and rec["step"]==29999,"final checkpoint")
    with numpy_scalar_scope():m=torch.load(ck,map_location="cpu",mmap=True,weights_only=True)
    arrays={}
    for key in ("means","scales","quats","opacities"):
        values=[v for k,v in m["pipeline"].items() if k.endswith("gauss_params."+key)]
        check(len(values)==1,key)
        arrays[key]=values[0].detach().cpu().numpy()
    count=arrays["means"].shape[0]
    check(count==rec["final_gaussians"] and all(len(a)==count and np.isfinite(a).all() for a in arrays.values()),label)
    f=ASSETS/(label+"_gaussian_geometry.npz")
    with f.open("wb") as out:np.savez_compressed(out,**arrays)
    with np.load(f,allow_pickle=False) as d:
        check(set(d.files)==set(arrays) and all(np.array_equal(d[k],a) for k,a in arrays.items()),"Gaussian geometry round trip")
    return dict(name=f.name,kind="Final Gaussian geometry parameters in native checkpoint coordinates",
                model=label,step=29999,count=count,fields=list(arrays),source_sha256=source,**checksum_record(f))


def movie_probe(p):
    args=["ffprobe","-v","error","-select_streams","v:0","-show_streams","-show_format","-show_packets",
          "-show_entries","stream=codec_name,width,height,time_base,nb_frames:format=duration:packet=pts","-of","json",p]
    data=json.loads(subprocess.check_output(args))
    s=data["streams"][0];tb=Fraction(s["time_base"])
    times=sorted(int(q["pts"])*tb for q in data["packets"])
    return s,Fraction(data["format"]["duration"]),times


def public_video():
    src=ROOT/VIDEO
    manifest=frozen(REPORT+"/video_manifest.json")
    check(checksum_record(src)=={k:manifest[k] for k in ("bytes","sha256")},"original timeline")
    seq=frozen(BASE_REPORT+"/selection.json")["rows"]
    # The close-up crosses the old preview box, so cover all three picture panels during the union.
    masked=[i for i,r in enumerate(seq) if 349<=r["relative_s"]<=353.7 or r.get("annotation")=="late dynamic interval"]
    lo,hi=min(masked),max(masked)
    check(masked==list(range(lo,hi+1)) and len(seq)==1846,"fixed published privacy interval")
    before,duration,pts=movie_probe(src)
    check((before["width"],before["height"])==(3840,762) and len(pts)==len(seq),"source video geometry")
    target=ASSETS/"reference_B0_B1_public.mp4";tmp=PUB/"public_video.pending.mp4"
    vf=f"drawbox=x=0:y=42:w=iw:h=ih-42:color=0x2d2d2d:t=fill:enable='between(n,{lo},{hi})'"
    run(["ffmpeg","-nostdin","-v","error","-y","-i",src,"-map","0:v:0","-an","-map_metadata","-1","-vf",vf,"-vsync","vfr",
         "-enc_time_base","1:25","-video_track_timescale","12800","-c:v","libx264","-threads","5","-preset","fast",
         "-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",tmp])
    after,new_duration,out_pts=movie_probe(tmp)
    check(pts==out_pts and duration==new_duration and
         [before[k] for k in ("width","height","nb_frames")]==[after[k] for k in ("width","height","nb_frames")],"presentation times changed")
    qa=f"select='between(n,{lo},{hi})',crop=3840:696:0:58,scale=96:24"
    pixels=subprocess.check_output(["ffmpeg","-nostdin","-v","error","-i",str(tmp),"-vf",qa,"-vsync","0",
                                    "-threads","2","-f","rawvideo","-pix_fmt","rgb24","-"])
    a=np.frombuffer(pixels,np.uint8).reshape(-1,24,96,3)
    check(len(a)==len(masked) and np.abs(a.astype(np.int16)-45).max()<=8,"redaction readback over all masked frames")
    tmp.replace(target)
    return dict(name=target.name,kind="Public reference/B0/B1 timeline",frames=len(pts),duration_seconds=float(duration),
                source_sha256=manifest["sha256"],width=after["width"],height=after["height"],
                original_frame_pts_preserved=True,
                privacy=dict(first_frame=lo,last_frame=hi,count=len(masked),first_acquisition_s=seq[lo]["relative_s"],
                             last_acquisition_s=seq[hi]["relative_s"],all_three_panels=True,
                             mask_top_y=42,verified_frames=len(a),interior_rgb_range=[int(a.min()),int(a.max())]),
                **checksum_record(target))


def data_assets():
    source=frozen(REPORT+"/artifact_manifest.json")
    artifacts={r["path"]:{k:r[k] for k in ("bytes","sha256")} for r in source["files"]}
    geometries=[];more=[]
    for label,run_dir in (("B0",B0),("B1",B1)):
        sup=B1+"/supported/"+label
        for kind,path in (("raw_points",run_dir+"/mesh/"+RAW+"pcd.ply"),("raw_mesh",run_dir+"/mesh/"+RAW+"poisson_mesh.ply"),
                          ("fused_points",sup+"/fused.ply"),("before_crop_mesh",sup+"/before_crop.ply"),("supported_mesh",sup+"/supported.ply")):
            geometries.append((label,kind,path,artifacts[path]))
        p=sup+"/face_votes.npz"
        check(checksum_record(ROOT/p)==artifacts[p],p)
        with np.load(ROOT/p,allow_pickle=False) as votes:
            check(set(votes.files)=={"surface","conflict"} and votes["surface"].shape==votes["conflict"].shape,"vote fields")
            for key in votes.files:
                x=votes[key];check(x.ndim==1 and x.dtype.kind in "iu" and x.min()>=0 and x.max()<=1478,key)
        target=ASSETS/(label+"_face_votes.npz")
        shutil.copyfile(ROOT/p,target)
        more.append(dict(name=target.name,kind="Per-face support and free-space conflict votes",model=label,**checksum_record(target)))
        key=document(ROOT/run_dir/"training_complete.json")["checkpoint_sha256"]
        check(key in {r["sha256"] for r in source["files"]},"frozen model")
        more.append(geometry_parameters(label,run_dir,key))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(lambda args:geometry(*args),geometries):
            more.append(result)
            print("Prepared",result["name"],flush=True)
    return more


def prepare():
    ASSETS.mkdir(exist_ok=True,parents=True)
    state=PUB/"prepare.json"
    if (PUB/"transfer.json").is_file():
        index=document(PUB/"transfer.json")
        for x in index["assets"]:
            check(checksum_record(ASSETS/x["name"])=={k:x[k] for k in ("bytes","sha256")},x["name"])
        return
    atomic(state,dict(stage="public_video",utc=utc()))
    media=public_video()
    atomic(PUB/"video_qa.json",media)
    atomic(state,dict(stage="geometry_and_reports",utc=utc()))
    rows=[media,*data_assets()]
    name="capture_full_comparison_reports.zip"
    run(["git","archive","--format=zip","-o",ASSETS/name,RESULT,"--",REPORT,
         "configs/capture_metric3d_protocol.json","docs/11_capture_metric3d_plan.md","docs/12_metric3d_normal_adapter.md",
         "docs/13_capture_supported_mesh.md"],cwd=ROOT)
    rows.append(dict(name=name,kind="Frozen reports, per-frame results, previews, protocol and configurations",**checksum_record(ASSETS/name)))
    atomic(ASSETS/"public-artifacts.json",dict(schema=1,run_id=ID,result_commit=RESULT,assets=rows,
          public_scope="Redacted RGB timeline, frozen scientific report, original geometry coordinates, normals, topology and per-face votes; geometry-only Gaussian parameters.",
          private_archive="Original RGB-D, unredacted timeline, appearance coefficients, training/optimizer checkpoints, priors and intermediate data."))
    rows.append(dict(name="public-artifacts.json",**checksum_record(ASSETS/"public-artifacts.json")))
    sums=ASSETS/"SHA256SUMS.txt"
    sums.write_text("".join(r["sha256"]+"  "+r["name"]+"\n" for r in sorted(rows,key=lambda a:a["name"])),"ascii")
    rows.append(dict(name=sums.name,**checksum_record(sums)))
    safe_names={r["name"] for r in rows}
    check(len(safe_names)==len(rows) and safe_names=={p.name for p in ASSETS.iterdir()},"public asset allowlist")
    desc=[dict(name=r["name"],bytes=r["bytes"],sha256=r["sha256"],chunks=range_hashes(ASSETS/r["name"])) for r in rows]
    atomic(PUB/"transfer.json",dict(run_id=ID,tag=TAG,result_commit=RESULT,assets=desc))
    atomic(state,dict(stage="ready",assets=len(rows),bytes=sum(r["bytes"] for r in rows),utc=utc()))


def gh(args,route):
    # The proxy endpoint itself never appears in command arguments or stored diagnostic output.
    pre='source /etc/network_turbo >/dev/null 2>&1 && ' if route=="acceleration" else ''
    proc=subprocess.run(["bash","-c",pre+'exec gh "$@"',"gh-project",*map(str,args)],cwd=ROOT,
                        env=dict(os.environ,GH_PROMPT_DISABLED="1",GH_NO_UPDATE_NOTIFIER="1",GIT_TERMINAL_PROMPT="0"),
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return proc


def request(args):
    for attempt in range(3):
        for route in ("acceleration","direct"):
            r=gh(args,route)
            if r.returncode==0:return r.stdout
        time.sleep(3*(attempt+1))
    raise RuntimeError("GitHub operation failed: "+str(args[0])+" (official acceleration and direct HTTPS)")


def api(path):
    return json.loads(request(["api",f"repos/{REPO}/"+path]))


def release_for_tag():
    return next((x for x in api("releases?per_page=100") if x["tag_name"]==TAG),None)


def asset_table(release_id):
    return {a["name"]:a for a in api(f"releases/{release_id}/assets?per_page=100")}


def matches(asset,row):
    return asset and asset.get("state")=="uploaded" and asset["size"]==row["bytes"] and asset.get("digest")=="sha256:"+row["sha256"]


def publish():
    plan=document(PUB/"transfer.json"); rows=plan["assets"]
    check(plan["result_commit"]==RESULT,"release source")
    repo=json.loads(request(["repo","view",REPO,"--json","nameWithOwner,visibility"]))
    check(repo["nameWithOwner"]==REPO and repo["visibility"]=="PUBLIC","release destination")
    notes=PUB/"notes.md"
    prefix=f"https://github.com/{REPO}/blob/{RESULT}/{REPORT}"
    notes.write_text(
        "同一完整采集上，B0 使用 Omnidata 法线，B1 使用 Metric3D v2 ViT-Large 法线。两组均由相同点云从头训练 30,000 次，保留原传感器深度监督。"
        "B1 在 184 张测试图上的 PSNR 为 16.885 dB，B0 为 15.826 dB。与采集传感器 Z 的合并像素 MAE 分别为 0.336 m 和 0.492 m。\n\n"
        "- 播放 \x60reference_B0_B1_public.mp4\x60 可对照 1,846 个位置、368.92 秒的实拍与两组重建。三列依次为实拍、B0、B1。"
        "公开版保留全部帧与时间戳，动态人物段的三列画面均已遮挡。\n"
        "- 点云和网格按模型、原始导出、1 cm 体素融合以及支持裁剪前后命名。每个 ZIP 内为标准 PLY，保留原始坐标、法线和三角拓扑，已移除顶点颜色。\n"
        "- 两份 Gaussian geometry NPZ 包含最终 checkpoint 的 means、scales、quats 和 opacities 原始参数。两份 face votes NPZ 保存裁剪前各三角面的表面支持与前方冲突票数。\n"
        "- 报告 ZIP 包含全部留出指标、逐帧与时间段对照及运行配置。测试集支持网格的正确覆盖率为 B0 53.55%、B1 53.85%，分母为全部有效参考像素。\n\n"
        f"[中文报告]({prefix}/overview.md) · [逐帧和表面指标]({prefix}/paired_summary.json) · [产物清单]({prefix}/artifact_manifest.json)\n",
        encoding="utf-8")
    rel=release_for_tag()
    if rel is None:
        args=["release","create",TAG,"-R",REPO,"--target",RESULT,"--draft","--title","Capture-full • Metric3D v2 30k 对照","--notes-file",str(notes)]
        for route in ("acceleration","direct"):
            gh(args,route)
            rel=release_for_tag()
            if rel:break
    check(rel and rel["target_commitish"]==RESULT and rel["tag_name"]==TAG,"same release tag and frozen commit")
    rid=rel["id"]
    table=asset_table(rid);planned={r["name"] for r in rows}
    check(set(table)<=planned,"unrelated release assets")
    def upload(row):
        name=row["name"];path=ASSETS/name
        check(checksum_record(path)=={k:row[k] for k in ("bytes","sha256")},name)
        if matches(table.get(name),row):return
        check(rel["draft"],"existing public release asset mismatch")
        for attempt in range(3):
            for route in ("acceleration","direct"):
                asset=asset_table(rid).get(name)
                if matches(asset,row):return
                if asset:
                    # Only a zero-length, failed placeholder in this exact unpublished release is disposable.
                    check(asset["state"]=="starter" and asset["size"]==0 and asset.get("digest") is None,"asset collision "+name)
                    request(["api","--method","DELETE",f"repos/{REPO}/releases/assets/{asset['id']}"])
                result=gh(["release","upload",TAG,str(path),"-R",REPO],route)
                if matches(asset_table(rid).get(name),row):return
                print("Retrying upload",name,route,"exit",result.returncode,flush=True)
            time.sleep(4*(attempt+1))
        raise RuntimeError("Could not verify GitHub asset "+name)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for n,_ in enumerate(pool.map(upload,rows),1):
            atomic(PUB/"publish.json",dict(stage="uploading",verified=n,total=len(rows),utc=utc()))
    table=asset_table(rid)
    check(set(table)==planned and all(matches(table[x["name"]],x) for x in rows),"draft release integrity")
    if rel["draft"]:
        request(["release","edit",TAG,"-R",REPO,"--draft=false","--latest=false"])
    published=release_for_tag();tag=api("git/ref/tags/"+TAG)["object"]
    commit=api("git/tags/"+tag["sha"])["object"]["sha"] if tag["type"]=="tag" else tag["sha"]
    table=asset_table(rid)
    check(published and not published["draft"] and commit==RESULT and all(matches(table[x["name"]],x) for x in rows),"published release verification")
    atomic(PUB/"published.json",dict(status="published_verified",tag=TAG,release_id=rid,result_commit=commit,
          url=published["html_url"],verified_utc=utc(),assets=[dict(name=r["name"],bytes=r["bytes"],sha256=r["sha256"],
          url=table[r["name"]]["browser_download_url"]) for r in rows]))
    atomic(PUB/"publish.json",dict(stage="published_verified",assets=len(rows),utc=utc()))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("prepare","publish","all"))
    mode=ap.parse_args().mode;PUB.mkdir(parents=True,exist_ok=True)
    with (PUB/"release.lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if mode in ("prepare","all"):prepare()
            if mode in ("publish","all"):publish()
        except Exception as e:
            atomic(PUB/"last_error.json",dict(mode=mode,error_type=type(e).__name__,reason=str(e)[:400],utc=utc()))
            raise
