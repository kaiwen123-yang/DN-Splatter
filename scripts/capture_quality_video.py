"""Render all timeline cameras and validate the public reference/B1/B2 video against B1 PTS."""
from fractions import Fraction
import gzip
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType

from capture_metric3d_common import REPORT as B1REPORT, selection
from capture_quality_paths import B1, ROOT, RUN, SCENE, STATE, check, read, record, sha, write
from capture_quality_tsdf import load_selected


def probe(path):
    data=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_streams",
         "-show_format","-show_packets","-show_entries","stream=width,height,time_base,nb_frames:format=duration:packet=pts",
         "-of","json",path]))
    stream=data["streams"][0]; tick=Fraction(stream["time_base"])
    return (stream["width"],stream["height"],int(stream["nb_frames"]),Fraction(data["format"]["duration"]),
            sorted(int(p["pts"])*tick for p in data["packets"]))


def main():
    _,pipe,ck,_=load_selected()
    seq=selection(); meta=read(SCENE/"transforms.json")
    frames={Path(f["file_path"]).stem:f for f in meta["frames"]}
    check(len(seq)==len(frames)==1846,"full trajectory")
    index=read(B1REPORT/"artifact_manifest.json")["index"]
    check(sha(ROOT/index["path"])==index["sha256"],"B1 timeline source manifest")
    with gzip.open(ROOT/index["path"],"rt") as f:
        source={r["path"]:r["sha256"] for line in f for r in [json.loads(line)] if "_comparison.jpg" in r["path"]}
    old=B1/"video/reference_B0_B1.mp4"
    root=RUN/"video"; root.mkdir(exist_ok=True)
    previews=RUN/"previews"; previews.mkdir(exist_ok=True)
    sel=read(STATE/"selection.json")
    worst=set()
    for split in ("val","test"):
        rows=read(RUN/"evaluation"/split/f"step-{sel['selected_step']:09d}.json")["frames"]
        worst.add(min(rows,key=lambda r:r["rgb"]["psnr"])["id"])
        valid=[r for r in rows if r["depth"]["MAE_m"] is not None]
        if valid: worst.add(max(valid,key=lambda r:r["depth"]["MAE_m"])["id"])
    masked=[i for i,r in enumerate(seq) if 349<=r["relative_s"]<=353.7 or r.get("annotation")=="late dynamic interval"]
    check(len(masked)==30 and masked==list(range(min(masked),max(masked)+1)),"published privacy interval")
    tape=["ffconcat version 1.0"]
    with torch.no_grad():
        for i,row in enumerate(seq):
            stem=row["id"]; frame=frames[stem]
            cam=Cameras(camera_to_worlds=torch.tensor(frame["transform_matrix"],dtype=torch.float32)[None,:3,:],
                fx=meta["fl_x"],fy=meta["fl_y"],cx=meta["cx"],cy=meta["cy"],width=meta["w"],height=meta["h"],
                camera_type=CameraType.PERSPECTIVE).to(pipe.device)
            out=pipe.model.get_outputs_for_camera(camera=cam)
            rgb=out["rgb"].detach().cpu().numpy().reshape(720,1280,3)
            new=np.uint8(np.rint(255*np.clip(np.nan_to_num(rgb),0,1)))[:,:,::-1]
            path=B1/"video"/(stem+"_comparison.jpg")
            check(sha(path)==source[path.relative_to(ROOT).as_posix()],"B1 comparison frame "+stem)
            archived=cv2.imread(str(path)); check(archived.shape==(762,3840,3),"B1 panel geometry")
            ref=cv2.imread(str(SCENE/frame["file_path"]))
            canvas=cv2.copyMakeBorder(np.hstack((ref,archived[42:,2560:],new)),42,0,0,0,cv2.BORDER_CONSTANT)
            if i in masked: canvas[42:]=45
            for j,label in enumerate((f"Reference  {stem}  {row['requested_split']}  {row['relative_s']:.3f}s",
                                      "B1  Metric3D 30k",f"B2  {sel['selected_step']+1} updates")):
                cv2.putText(canvas,label,(j*1280+12,29),0,.83,(250,250,250),1,cv2.LINE_AA)
            target=root/(stem+".jpg"); check(cv2.imwrite(str(target),canvas),stem)
            if stem in worst or i in {0,120,240,1293,1294,1306,1307,1514}:
                cv2.imwrite(str(previews/(stem+"_timeline.jpg")),cv2.resize(canvas,(1440,286),interpolation=cv2.INTER_AREA))
            tape.append(f"file '{target.name}'")
            if i+1<len(seq):
                dt=(int(seq[i+1]["actual_rgb_timestamp_ns"])-int(row["actual_rgb_timestamp_ns"]))/1e9
                tape.append(f"duration {dt:.9f}")
            if (i+1)%100==0: print("timeline",i+1,flush=True)
    manifest=root/"timeline.ffconcat"; manifest.write_text("\n".join(tape)+"\n")
    movie=root/"reference_B1_B2_public.mp4"
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-f","concat","-safe","0","-i",manifest,
        "-vsync","vfr","-c:v","libx264","-threads",str(min(12,int(os.environ["DN_CPU_BUDGET"]))),"-crf","18",
        "-pix_fmt","yuv420p","-movflags","+faststart",movie],check=True)
    actual=probe(movie)
    check(actual==probe(old) and actual[2]==1846,"exact full B1 PTS, duration and dimensions")
    vf=f"select='between(n,{min(masked)},{max(masked)})',crop=3840:696:0:58,scale=96:24"
    pixels=subprocess.check_output(["ffmpeg","-nostdin","-v","error","-i",movie,"-vf",vf,"-vsync","0",
                                  "-threads","2","-f","rawvideo","-pix_fmt","rgb24","-"])
    a=np.frombuffer(pixels,np.uint8).reshape(-1,24,96,3)
    check(len(a)==30 and np.max(np.abs(a.astype(np.int16)-45))<=8,"all masked pictures read back")
    write(root/"video_manifest.json",dict(video=record(movie),frames=1846,duration_s=float(actual[3]),
          redacted_frames=masked,all_pts_equal_B1=True,selected_checkpoint=record(ck),worst_frames=sorted(worst)))


if __name__=="__main__":
    main()
