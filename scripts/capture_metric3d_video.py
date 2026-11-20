"""Join the full timestamped trajectory using matched B0 renders and the B1 final checkpoint."""
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from capture_metric3d_common import BASE, B0_REPORT, REPORT, RUN, SCENE, P, demand, read, selection, sha, write
from capture_s01_eval import final_setup
from capture_rectify import imwrite
from vr_room_run import CPU


def main():
    seq = selection()
    old = read(B0_REPORT/"video_manifest.json")
    demand(old["frames"]==len(seq)==1846 and not old["gaps"] and old["checkpoint_sha256"]==P["baseline"]["checkpoint_sha256"], "B0 video")
    inventory = {r["path"]:r["sha256"] for r in read(B0_REPORT/"artifact_manifest.json")["files"]}
    _,pipe,ck,step = final_setup("test")
    ds = pipe.datamanager.train_dataset
    cams = {p.stem:i for i,p in enumerate(ds.image_filenames)}
    where = set(old["previews"]) | {"frame_01409"}
    for split in ("val","test"):
        m = read(REPORT/f"metrics_{split}.json")["frames"]
        where.add(min(m,key=lambda r:r["rgb"]["psnr"])["id"])
        valid = [r for r in m if r["depth_conditional"]]
        if valid:
            where.add(max(valid,key=lambda r:r["depth_conditional"]["MAE_m"])["id"])
    root = RUN/"video"
    root.mkdir(exist_ok=True)
    tape = ["ffconcat version 1.0"]
    for i, row in enumerate(seq):
        stem, split = row["id"], row["requested_split"]
        ref = cv2.imread(str(SCENE/"images"/(stem+".png")))
        source = BASE/("video/"+stem+"_pair.jpg" if split=="train" else f"renders/{split}/rgb/{stem}.png")
        demand(sha(source)==inventory[source.relative_to(BASE).as_posix()], "B0 render "+stem)
        b0 = cv2.imread(str(source))
        if split=="train":
            demand(b0.shape[:2]==(762,2560), "archived B0 video panel")
            b0 = b0[42:,1280:]
            j = cams[stem]
            with torch.no_grad():
                out = pipe.model.get_outputs_for_camera(camera=ds.cameras[j:j+1].to(pipe.device))
                def bgr(key):
                    return np.uint8(np.rint(255*np.clip(out[key].detach().cpu().numpy().reshape(720,1280,3),0,1)))[...,::-1]
                b1 = bgr("rgb")
                if stem in {"frame_00000","frame_00120","frame_00240"}:
                    imwrite(root/(stem+"_normal.png"),bgr("normal"))
                    np.save(root/(stem+"_depth.npy"),out["depth"].detach().cpu().numpy().reshape(720,1280))
        else:
            b1 = cv2.imread(str(RUN/"renders"/split/"rgb"/(stem+".png")))
        demand(ref.shape==b0.shape==b1.shape==(720,1280,3), stem)
        canvas = cv2.copyMakeBorder(np.hstack((ref,b0,b1)),42,0,0,0,cv2.BORDER_CONSTANT)
        for col,label in enumerate((f"Reference  {stem}  {split}  {row['relative_s']:.3f} s","B0  Omnidata","B1  Metric3D v2 ViT-Large")):
            cv2.putText(canvas,label,(col*1280+12,29),0,.83,(250,250,250),1,cv2.LINE_AA)
        frame = root/(stem+"_comparison.jpg")
        imwrite(frame,canvas)
        if stem in where:
            public = canvas.copy()
            if 349<=row["relative_s"]<=353.7:
                for x in (0,1280,2560):
                    public[42:542,x:x+700]=45
            imwrite(REPORT/"previews"/(stem+"_timeline.jpg"),cv2.resize(public,(1440,286),interpolation=cv2.INTER_AREA))
        tape.append(f"file '{frame.name}'")
        if i+1<len(seq):
            dt = (int(seq[i+1]["actual_rgb_timestamp_ns"])-int(row["actual_rgb_timestamp_ns"]))/1e9
            tape.append(f"duration {dt:.9f}")
    index = root/"trajectory.ffconcat"
    index.write_text("\n".join(tape)+"\n")
    movie = root/"reference_B0_B1.mp4"
    subprocess.run(["ffmpeg","-v","error","-y","-f","concat","-safe","0","-i",index,"-vsync","vfr",
                    "-c:v","libx264","-threads",str(min(12,CPU)),"-crf","18","-pix_fmt","yuv420p",
                    "-movflags","+faststart",movie],check=True)
    probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_streams",
                              "-show_format","-of","json",movie]))
    duration = float(probe["format"]["duration"])
    demand(int(probe["streams"][0]["nb_frames"])==len(seq) and abs(duration-(seq[-1]["relative_s"]-seq[0]["relative_s"]))<.1, "timeline duration")
    write(REPORT/"video_manifest.json",dict(path=str(movie),bytes=movie.stat().st_size,sha256=sha(movie),frames=len(seq),
           duration_s=duration,B0_checkpoint_sha256=old["checkpoint_sha256"],B1_checkpoint_sha256=sha(ck),step=step,
           B0_train_source="SHA-matched B0 timeline JPEG render panels; B0 heldout uses lossless PNG",
           source_duration_s=seq[-1]["relative_s"]-seq[0]["relative_s"],previews=sorted(where)))


if __name__=="__main__":
    main()
