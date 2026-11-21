"""Independently verify GitHub Release assets on the private disk and seal its archival receipt."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from capture_archive_transport import (B0,B1,ID,RESULT,REPORT,ROOT,STAGE,VIDEO,atomic,check,
                                      checksum_record,digest,document,download,remote_bytes,safe_rel,ssh_args,utc)
from capture_archive_windows import exclusive

REPO="kaiwen123-yang/DN-Splatter"
TAG="capture-full-metric3d-30k-20261120-v1"


def github(path):
    for attempt in range(4):
        try:
            p=subprocess.run(["gh","api","repos/"+REPO+"/"+path],stdin=subprocess.DEVNULL,
                    capture_output=True,timeout=55,env=dict(os.environ,GH_PROMPT_DISABLED="1",GH_NO_UPDATE_NOTIFIER="1"))
            if p.returncode==0:return json.loads(p.stdout)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2*(attempt+1))
    raise OSError("Cannot verify GitHub metadata for "+path)


def get_official_asset(row,tmp):
    # Prior partial bytes go straight to the range-resumable SSH reader.
    if tmp.exists() and tmp.stat().st_size:
        return False
    with tmp.open("wb") as out,subprocess.Popen(
            ["gh","release","download",TAG,"-R",REPO,"--pattern",row["name"],"--output","-"],stdin=subprocess.DEVNULL,
            stdout=out,stderr=subprocess.DEVNULL,env=dict(os.environ,GH_PROMPT_DISABLED="1",GH_NO_UPDATE_NOTIFIER="1")) as proc:
        previous=0;last=time.monotonic()
        while proc.poll() is None:
            time.sleep(2)
            n=os.fstat(out.fileno()).st_size
            if n!=previous:
                previous=n;last=time.monotonic()
            if time.monotonic()-last>55:
                proc.kill()
                break
        rc=proc.wait()
    return rc==0 and tmp.stat().st_size==row["bytes"] and digest(tmp)==row["sha256"]


def collect_release(dest,ctrl):
    progress=ctrl/"release_progress.json"
    while True:
        payload=remote_bytes(STAGE+"/public/published.json",optional=True)
        if payload:break
        atomic(progress,dict(stage="waiting_for_verified_publication",pid=os.getpid(),utc=utc()))
        time.sleep(15)
    published=json.loads(payload)
    check(published["status"]=="published_verified" and published["result_commit"]==RESULT and published["tag"]==TAG,
          "server publication receipt")
    remote=github("releases/tags/"+TAG)
    check(remote["id"]==published["release_id"] and not remote["draft"] and remote["html_url"]==published["url"],"public release identity")
    tag=github("git/ref/tags/"+TAG)["object"]
    commit=github("git/tags/"+tag["sha"])["object"]["sha"] if tag["type"]=="tag" else tag["sha"]
    check(commit==RESULT,"GitHub tag target")
    listed=github(f"releases/{remote['id']}/assets?per_page=100")
    receipt={r["name"]:r for r in published["assets"]}
    check(len(listed)==len(receipt),"GitHub release cardinality")
    for item in listed:
        r=receipt[item["name"]]
        check(item["size"]==r["bytes"] and item["digest"]=="sha256:"+r["sha256"] and item["state"]=="uploaded",item["name"])
    descriptors=json.loads(remote_bytes(STAGE+"/public/transfer.json"))["assets"]
    check({x["name"] for x in descriptors}==set(receipt),"transport and GitHub allowlists")
    folder=dest/"public-release";folder.mkdir(exist_ok=True)
    lock=threading.Lock();finished=[]
    def receive(d):
        name=safe_rel(d["name"]);check("/" not in name,"release member")
        info={k:receipt[name][k] for k in ("bytes","sha256")}
        check(info=={k:d[k] for k in info},"server and GitHub identity: "+name)
        p=folder/name;tmp=p.with_suffix(p.suffix+".ranges")
        if not p.exists():
            if get_official_asset(d,tmp):
                tmp.replace(p)
            else:
                download(ssh_args(),STAGE+"/public/assets/"+name,p,d,6,tmp)
        check(checksum_record(p)==info,"H public release readback: "+name)
        with lock:
            finished.append(name)
            atomic(progress,dict(stage="verifying_release_on_disk",pid=os.getpid(),assets_done=len(finished),
                                 assets_total=len(receipt),utc=utc()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(receive,descriptors))
    check(set(finished)==set(receipt),"H public asset completeness")
    public_meta=document(folder/"public-artifacts.json")
    video=next(x for x in public_meta["assets"] if x["name"]=="reference_B0_B1_public.mp4")
    check(video["frames"]==1846 and video["original_frame_pts_preserved"] and video["privacy"]["verified_frames"]==video["privacy"]["count"],
          "public video QA")
    result=dict(**published,verified_on_disk_utc=utc(),asset_count=len(receipt),
                total_bytes=sum(x["bytes"] for x in receipt.values()),video=video)
    atomic(ctrl/"public_complete.json",result)
    atomic(progress,dict(stage="public_release_verified_on_disk",asset_count=len(receipt),utc=utc()))
    return result


def finish(dest,ctrl,public):
    priv=document(ctrl/"private_complete.json");catalog=document(ctrl/"catalog.json")
    check(priv["status"]=="verified" and priv["result_commit"]==public["result_commit"]==RESULT,"private/public binding")
    video_rel="project/"+VIDEO
    rawmeta=document(dest/"project"/REPORT/"video_manifest.json")
    check(checksum_record(dest/video_rel)=={k:rawmeta[k] for k in ("bytes","sha256")},"private playback file")
    ledger=sorted((ctrl/"volumes").glob("*.json"))
    check(len(ledger)==priv["volumes"]==catalog["volume_count"],"volume count at handoff")
    source=priv["source_bundle"]
    check(checksum_record(dest/"DN-Splatter.bundle")=={k:source[k] for k in ("bytes","sha256")},"source bundle readback")
    stamp=utc()
    combined=dict(status="complete_verified",verified_utc=stamp,run_id=ID,result_commit=RESULT,
        private_receipt="verification/private_complete.json",public_receipt="verification/public_complete.json",
        private_receipt_sha256=digest(ctrl/"private_complete.json"),public_receipt_sha256=digest(ctrl/"public_complete.json"),
        logical_paths=priv["logical_files"],logical_bytes=priv["logical_bytes"],unique_contents=priv["unique_files"],
        unique_bytes=priv["unique_bytes"],disk_readback_bytes=priv["storage_readback_bytes"],
        B1_manifest_file_count=priv["b1_files"],B0_manifest_file_count=priv["b0_files"],volumes=priv["volumes"],
        frozen_tracked_files=priv["tracked_files"],source_bundle=source,original_bag=document(ctrl/"original.json"),
        original_video=dict(path=video_rel,**checksum_record(dest/video_rel)),
        public_release_url=public["url"],public_asset_count=public["asset_count"],public_asset_bytes=public["total_bytes"])
    original=combined["original_bag"]
    movie_pub="public-release/reference_B0_B1_public.mp4"
    md=f"""# DN-Splatter 全程 Metric3D v2 对照归档

归档对应科学结果提交 \x60{RESULT}\x60，已在 H 盘读回核验。B0 使用 Omnidata 法线，B1 使用 Metric3D v2 ViT-Large 法线。原始 RGB、传感器 Z、相机、划分和训练设置相同。两组均有完整的 30,000 次更新。

## 打开结果

- [完整原始三列对照视频]({video_rel})，共 1,846 帧、368.92 秒，从左到右依次为实拍、B0、B1。
- [可分享的公开视频]({movie_pub})，动态人物段已遮挡。
- [中文实验报告](project/{REPORT}/overview.md) · [模型测试集结果](project/{REPORT}/metrics_test.json) · [网格覆盖评价](project/{REPORT}/surface_test_metrics.json)。
- [本轮公开 Release]({public['url']}) · [公开文件及 SHA256](public-release/public-artifacts.json)。

## 完整性和目录

`project/` 保留冻结的代码、配置、报告与实验相对路径。`project/{B1}/` 为 B1 的全部逐帧先验、训练输入、各保存步模型、渲染、网格与表面评价。`project/{B0}/` 保存 B0 对照材料。`project/{B1}/supported/` 内的 B0 和 B1 子目录分别保留采样来源、融合体素、裁剪前后网格与支持票数。本次涉及的全程几何工作区、尺度、相机、紧凑包和 s01 同源对照几何也在原位置。

源实验清单含 {priv['logical_files']:,} 个逻辑路径和 {priv['logical_bytes']:,} 字节，按 SHA256 去重后为 {priv['unique_files']:,} 个内容、{priv['unique_bytes']:,} 字节。相同内容通过归档内部的 NTFS 硬链接还原路径。如需编辑，先复制到工作目录。[逐路径 SHA256 清单](verification/inventory.jsonl.gz) · [源清单与规模](verification/catalog.json) · [H 盘回读凭据](verification/private_complete.json)。

[原始采集 bag]({original['member'].replace(chr(92),'/')}) 为归档中的独立副本，大小 {original['bytes']:,} 字节。SHA256 为 \x60{original['sha256']}\x60。[软件源码快照、补丁和三个独立环境的版本清单](project/archive-recovery/)与 [Git 历史 bundle](DN-Splatter.bundle) 可供恢复。运行缓存、登录凭据及其他场景的训练环境不属于本轮实验文件。

## 恢复和核验

`project/` 中的小型跟踪文件逐个对应冻结提交，合计 {priv['tracked_files']:,} 个。需要 Git 工作区时，可将 \x60DN-Splatter.bundle\x60 克隆到新的位置并检出 \x60{RESULT}\x60，再按项目内原相对路径访问归档数据。环境重建依据各 prefix 的包版本清单、历史源码和项目补丁。

公开 Release 共 {public['asset_count']} 个文件、{public['total_bytes']:,} 字节，均在 `public-release/` 保存并与 GitHub 的 SHA256 独立比对。原始彩色视频、带外观参数的完整 checkpoint、第三方预训练权重和原始逐帧数据位于私有归档。公开网格保留几何、法线和拓扑，公开高斯参数保留均值、尺度、四元数和不透明度。[逐文件发布核验](verification/public_complete.json) · [整体验收凭据](archive_complete.json)。
"""
    (dest/"归档说明.md").write_text(md,encoding="utf-8")
    atomic(dest/"archive_complete.json",combined)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--destination",type=Path,required=True)
    ap.add_argument("--public-only",action="store_true")
    args=ap.parse_args()
    dest=args.destination.resolve();ctrl=dest/"verification"
    identity=document(ctrl/"identity.json")
    check(identity["result_commit"]==RESULT and identity["run_id"]==ID,"existing private archive destination")
    with exclusive(ctrl/"publication_receiver.lock"):
        try:
            public=collect_release(dest,ctrl)
            if not args.public_only:
                while not (ctrl/"private_complete.json").is_file():
                    time.sleep(20)
                finish(dest,ctrl,public)
        except Exception as exc:
            atomic(ctrl/"finalize_error.json",dict(utc=utc(),error_type=type(exc).__name__,reason=str(exc)[:600]))
            raise
