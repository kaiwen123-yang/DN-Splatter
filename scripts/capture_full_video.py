"""Render the capture timeline at recorded times, retaining explicit coverage gaps."""
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from capture_rectify import read, save, sha, imwrite
from capture_s01_eval import final_setup
from capture_s01_inputs import GEO_RUN, REPORT, ensure
from vr_room_run import CPU, RUN, SCENE


def main():
    rows=read(GEO_RUN/"transfer/selection.json")
    meta=read(SCENE/"transforms.json")
    _,pipe,checkpoint,step=final_setup("test")
    train=pipe.datamanager.train_dataset
    camera={p.stem:i for i,p in enumerate(train.image_filenames)}
    posed={Path(f["file_path"]).stem for f in meta["frames"]}
    metrics=[r for s in ("val","test") for r in read(REPORT/f"metrics_{s}.json")["frames"]]
    choices={0,len(rows)//2,len(rows)-1,238,239}
    for t in [*range(15,370,30),353.25]:
        choices.add(min(range(len(rows)),key=lambda i:abs(rows[i]["relative_s"]-t)))
    if metrics:
        choices.add(int(min(metrics,key=lambda r:r["rgb"]["psnr"])["id"][-5:]))
        depth=[r for r in metrics if r["depth_full_reference"]]
        if depth:
            choices.add(int(max(depth,key=lambda r:r["depth_full_reference"]["MAE_m"])["id"][-5:]))
    root=RUN/"video"
    root.mkdir(exist_ok=True)
    tape=["ffconcat version 1.0"]
    gaps=[]
    for i,r in enumerate(rows):
        stem=r["id"]
        split=r["requested_split"]
        ref_path=GEO_RUN/"rect/1280/images"/(stem+".png")
        ref=cv2.imread(str(ref_path)) if ref_path.is_file() else None
        if ref is None:
            native=GEO_RUN/"transfer/rgb"/(stem+".jpg")
            ref=cv2.imread(str(native)) if native.is_file() else None
        if ref is None:
            ref=np.zeros((720,1280,3),np.uint8)
        if stem not in posed:
            render=np.zeros_like(ref)
            cv2.putText(render,"Missing camera pose",(100,360),0,2,(220,220,255),3)
            gaps.append(stem)
        elif split=="train":
            j=camera[stem]
            with torch.no_grad():
                out=pipe.model.get_outputs_for_camera(camera=train.cameras[j:j+1].to(pipe.device))
                def bgr(key):
                    arr=out[key].detach().cpu().numpy().reshape(720,1280,3)
                    return np.uint8(np.rint(np.clip(arr,0,1)*255))[...,::-1].copy()
                render=bgr("rgb")
                if i in (0,120,240):
                    imwrite(root/(stem+"_normal.png"),bgr("normal"))
                    np.save(root/(stem+"_depth.npy"),out["depth"].detach().cpu().numpy().reshape(720,1280))
        else:
            render=cv2.imread(str(RUN/"renders"/split/"rgb"/(stem+".png")))
            ensure(render is not None,"missing heldout render "+stem)
        label=f"{stem}   {split}   {r['relative_s']:.3f} s   device {r['actual_rgb_timestamp_ns']} ns"
        pair=np.hstack((ref,render))
        def title(im):
            tile=cv2.copyMakeBorder(im,42,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(tile,"Reference   |   "+label,(12,29),0,.73,(245,245,245),1,cv2.LINE_AA)
            cv2.putText(tile,"DN-Splatter",(im.shape[1]//2+12,29),0,.73,(245,245,245),1,cv2.LINE_AA)
            return tile
        image=root/(stem+"_pair.jpg")
        imwrite(image,title(pair))
        if i in choices:
            public=pair.copy()
            if 349<=r["relative_s"]<=353.7:
                for x in (0,1280):
                    public[:500,x:x+700]=45
            imwrite(REPORT/"previews"/(stem+"_timeline.jpg"),cv2.resize(title(public),(1152,344),interpolation=cv2.INTER_AREA))
        tape.extend(["file '"+image.name+"'","option framerate 1000"])
        if i+1<len(rows):
            dt=(int(rows[i+1]["actual_rgb_timestamp_ns"])-int(r["actual_rgb_timestamp_ns"]))/1e9
            tape.append(f"duration {dt:.9f}")
    index=root/"trajectory.ffconcat"
    index.write_text("\n".join(tape)+"\n")
    movie=root/"reference_render.mp4"
    subprocess.run(["ffmpeg","-v","error","-y","-f","concat","-safe","0","-i",str(index),"-vsync","vfr",
                    "-c:v","libx264","-threads",str(min(CPU,12)),"-crf","18","-pix_fmt","yuv420p",
                    "-movflags","+faststart",str(movie)],check=True)
    import json
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_streams",
                                               "-show_format","-of","json",movie],text=True))
    s=probe["streams"][0]
    ensure(int(s["nb_frames"])==len(rows) and abs(float(probe["format"]["duration"])-(rows[-1]["relative_s"]-rows[0]["relative_s"]))<.1,"video timing")
    save(REPORT/"video_manifest.json",dict(path=str(movie),bytes=movie.stat().st_size,sha256=sha(movie),
         frames=len(rows),gaps=gaps,source_time_s=[rows[0]["relative_s"],rows[-1]["relative_s"]],
         duration_s=float(probe["format"]["duration"]),checkpoint_sha256=sha(checkpoint),step=step,
         previews=[rows[i]["id"] for i in sorted(choices)],timing="Variable frame durations from consecutive device timestamps."))


if __name__=="__main__":
    main()
