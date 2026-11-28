"""Produce the hall RGB-D, paired point clouds, fixed-view diagnostics and timestamped replays."""
import argparse
import csv
import gzip
from functools import lru_cache
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lingbot_common import HARD, P0, REPORT, ROOT, RUN, SEQ, by_id, depth, masks, member, read, rgb, save
from lingbot_depth_evaluate import prediction
from lingbot_map_evaluate import load_prediction, sample
from lingbot_metrics import measures, parts


MEDIA=RUN/"media"
VISIBLE=REPORT/"previews"
BLUE="#0072B2"
ORANGE="#D55E00"
GRAY=(52,60,68)
LIGHT=(245,247,249)


def put(im,text,xy,scale=.52,color=GRAY,weight=1):
    cv2.putText(im,text,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,weight,cv2.LINE_AA)


def colour(z):
    valid=np.isfinite(z)&(z>0)
    u=np.zeros(z.shape,np.uint8)
    u[valid]=np.rint(255*np.clip((z[valid]-.3)/5.7,0,1)).astype(np.uint8)
    c=cv2.applyColorMap(u,cv2.COLORMAP_VIRIDIS)
    c[~valid]=(28,24,22)
    c[valid&((z<.3)|(z>6))]=(172,110,178)
    return c


def delta_colour(raw,pred):
    valid=np.isfinite(pred)&(pred>0)&(raw>=.3)&(raw<=6)
    a=np.zeros(raw.shape,np.float32)
    a[valid]=np.clip((pred[valid]-raw[valid])/.2,-1,1)
    white=np.array([245,245,245])
    blue=np.array([156,91,24]); orange=np.array([11,113,205])
    c=white+(np.where((a>0)[...,None],orange,blue)-white)*np.abs(a[...,None])
    c[~valid]=(33,30,29)
    return c.astype(np.uint8)


def legend(image,y,offset=12):
    vals=np.linspace(.3,6,180,dtype=np.float32)[None,:]
    image[y:y+12,offset:offset+180]=np.repeat(colour(vals),12,axis=0)
    put(image,"0.3 m", (offset,y+31),.43)
    put(image,"6 m", (offset+155,y+31),.43)
    put(image,"Black: missing     Mauve: outside 0.3-6 m",(offset+197,y+13),.44)


def tile(images,titles,width=340,height=190):
    out=np.full((height+42,len(images)*width,3),LIGHT,np.uint8)
    for i,(im,title) in enumerate(zip(images,titles)):
        scale=min(width/im.shape[1],height/im.shape[0])
        w,h=round(im.shape[1]*scale),round(im.shape[0]*scale)
        scaled=cv2.resize(im,(w,h),interpolation=cv2.INTER_NEAREST)
        x=i*width+(width-w)//2; y=38+(height-h)//2
        out[y:y+h,x:x+w]=scaled
        put(out,title,(i*width+9,25),.55)
    return out


def imwrite(path,image):
    path.parent.mkdir(exist_ok=True,parents=True)
    if not cv2.imwrite(str(path),image): raise IOError(path)


def cloud(z,image,k,valid,step):
    y,x=np.nonzero(valid[::step,::step])
    y*=step; x*=step
    d=z[y,x]
    xyz=np.column_stack(((x-k[0,2])*d/k[0,0],(y-k[1,2])*d/k[1,1],d)).astype(np.float32)
    return xyz,image[y,x,::-1].copy(),(y,x)


def ply(path,points,colours):
    path.parent.mkdir(parents=True,exist_ok=True)
    dtype=np.dtype([("xyz","<f4",3),("rgb","u1",3)])
    with path.open("wb") as f:
        n=sum(len(p) for p in points)
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\n"
                 "property float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        for p,c in zip(points,colours):
            block=np.empty(len(p),dtype=dtype); block["xyz"]=p;block["rgb"]=c;block.tofile(f)


@lru_cache(maxsize=1)
def encoder():
    """Probe the allocated GPU's encoder once. An FFmpeg build can list an unusable backend."""
    binary=imageio_ffmpeg.get_ffmpeg_exe()
    options=["-c:v","h264_nvenc","-preset","p4","-rc","vbr","-cq","21","-b:v","0"]
    probe=subprocess.run([binary,"-hide_banner","-loglevel","error","-f","lavfi","-i","color=size=320x180:rate=5",
                          "-frames:v","2",*options,"-f","null","-"],capture_output=True)
    return options if probe.returncode==0 else ["-c:v","libx264","-threads","4","-preset","veryfast","-crf","21"]


class Movie:
    """A 5 Hz viewing clock holds past images across removed split positions."""
    def __init__(self,path,w,h):
        self.path=path
        self.process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-y",
            "-f","rawvideo","-pix_fmt","bgr24","-s",f"{w}x{h}","-r","5","-i","-","-an",
            *encoder(),"-pix_fmt","yuv420p","-movflags","+faststart",str(path)],stdin=subprocess.PIPE)
        self.frames=0
    def write(self,frame,n=1):
        data=np.ascontiguousarray(frame).tobytes()
        for _ in range(n): self.process.stdin.write(data)
        self.frames+=n
    def close(self):
        self.process.stdin.close()
        if self.process.wait(): raise RuntimeError("FFmpeg failed on "+str(self.path))


def preview(path,start,length,target):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-y","-ss",str(start),"-i",str(path),
        "-t",str(length),"-vf","scale=800:-2","-an","-c:v","libx264","-threads","3","-crf","24","-movflags","+faststart",str(target)],check=True)


def recovery_plot():
    with gzip.open(RUN/"depth/evaluation/masked_per_frame.csv.gz","rt") as f:
        values=[r for r in csv.DictReader(f) if r["stratum"]=="all"]
    fig,axs=plt.subplots(1,2,figsize=(8.7,4.25),layout="constrained")
    for ax,mode,label in zip(axs,("random_points","blocks"),("10% scattered measurements","64 x 64 image blocks")):
        z={m:{r["id"]:r for r in values if r["method"]==m and r["mode"]==mode} for m in ("nearest","lingbot")}
        hi=max(max(float(r["mae_m"])*100 for r in rows.values() if r["mae_m"]) for rows in z.values())*1.07
        ax.plot([0,hi],[0,hi],ls="--",color=".45",lw=1)
        for group,symbol,color,label2 in (("ordinary_complete","o",BLUE,"Ordinary"),("natural_holes","s",ORANGE,"Holes"),
                                         ("edges_and_thin_structures","^","#009E73","Edges")):
            pairs=[(float(a["mae_m"])*100,float(z["lingbot"][fid]["mae_m"])*100) for fid,a in z["nearest"].items()
                   if a["group"]==group and a["mae_m"] and z["lingbot"][fid]["mae_m"]]
            if pairs: ax.scatter(*zip(*pairs),s=24,marker=symbol,color=color,alpha=.82,label=label2)
        ax.set(xlabel="2D nearest-neighbour MAE (cm)",ylabel="LingBot-Depth MAE (cm)",title=label,xlim=(0,hi),ylim=(0,hi),aspect="equal")
        ax.grid(alpha=.16)
    axs[0].legend(fontsize=8,loc="lower right"); fig.savefig(VISIBLE/"depth_recovery.png",dpi=160); plt.close(fig)


def depth_media(m,frames,sel):
    k=np.array(m["K_cv"]); canvas=np.load(P0/"canvas.npy")
    directory=MEDIA/"depth"; directory.mkdir(parents=True,exist_ok=True)
    diagnostic=[]
    panels={}
    for s in sel["samples"]:
        fid=s["id"]; raw=depth(frames[fid]); pred,v=prediction(fid); image=rgb(frames[fid])
        d=directory/fid; d.mkdir(exist_ok=True)
        observed,primary,band,_=masks(raw,canvas)
        np.savez_compressed(d/"sensor_validity.npz",observed=observed,primary=primary,canvas=canvas)
        for name,z,valid in (("raw",raw,observed),("refined",pred,canvas&v)):
            p,c,_=cloud(z,image,k,valid,3)
            ply(d/(name+"_camera.ply"),[p],[c])
        ims=[image,colour(raw),colour(pred),delta_colour(raw,pred)]
        panel=tile(ims,["RGB","Sensor Z","LingBot Z","Change on observed Z"])
        foot=np.full((54,panel.shape[1],3),LIGHT,np.uint8); legend(foot,5)
        put(foot,"Change: blue = closer, orange = farther; saturates at +/-20 cm",(600,22),.45)
        imwrite(d/"comparison.png",np.vstack((panel,foot)))
        x0,y0,x1,y1=s["roi_xyxy"]
        roi=np.zeros_like(canvas);roi[y0:y1,x0:x1]=True
        row=dict(id=fid,group=s["group"],roi=s["roi_xyxy"])
        for label,fixed in (("observed",roi&primary),("near_edge",roi&primary&band)):
            row[label]=measures(*parts(raw,pred,fixed,v))
        evidence=RUN/"depth/hole_evidence"/(fid+".npz")
        if evidence.is_file():
            with np.load(evidence) as e:
                row["natural_holes"]=int((roi&canvas&~observed).sum())
                row["two_view_support"]=int((roi&e["observed_in_two"]).sum())
                row["two_view_agree"]=int((roi&e["agrees"]).sum())
        diagnostic.append(row)
        if fid in ("frame_00120","frame_00240","frame_01420"):
            name={"frame_00120":"Table edge","frame_00240":"Chair legs","frame_01420":"Foam teeth"}[fid]
            panels[fid]=tile([a[y0:y1,x0:x1] for a in ims],
                [f"{name} | RGB","Sensor Z","LingBot Z","Change on observed Z"],width=275,height=190)
    board=np.vstack([panels[i] for i in ("frame_00120","frame_00240","frame_01420")])
    foot=np.full((66,board.shape[1],3),LIGHT,np.uint8);legend(foot,6)
    put(foot,"Change: blue = closer, orange = farther; +/-20 cm",(550,23),.48)
    imwrite(VISIBLE/"depth_structures.png",np.vstack((board,foot)))
    save(directory/"fixed_roi_metrics.json",diagnostic)
    ids=sel["video_train_ids"]
    video=directory/"depth_hall.mp4"; writer=Movie(video,1152,268)
    for i,fid in enumerate(ids):
        f=frames[fid]; raw=depth(f); pred,_=prediction(fid)
        t=(f["rgb_timestamp_ns"]-m["frames"][0]["rgb_timestamp_ns"])*1e-9
        frame=tile([rgb(f),colour(raw),colour(pred)],["RGB","Sensor Z","LingBot Z"],width=384,height=204)
        footer=np.full((22,1152,3),LIGHT,np.uint8)
        put(footer,f"{fid}   Capture {t:.2f} s     Z 0.3-6 m  |  black: missing   mauve: outside range",(10,16),.45)
        dt=(frames[ids[i+1]]["rgb_timestamp_ns"]-f["rgb_timestamp_ns"])*1e-9 if i+1<len(ids) else .2
        writer.write(np.vstack((frame,footer)),max(1,round(dt*5)) if dt<3 else 1)
    writer.close()
    preview(video,20,12,VISIBLE/"depth_preview.mp4")
    save(directory/"video.json",dict(file=member(video),ids=ids,display_frames=writer.frames,fps=5,encoder=encoder(),
        timeline="original capture seconds printed; jumps between three fixed excerpts; shorter train gaps hold previous frame"))
    recovery_plot()


def trajectory_plot(seq,trajectory,rows,out):
    ref=np.array([np.array(a["reference_c2w"])[:3,3] for a in trajectory])
    est=np.array([np.array(a["aligned_c2w"])[:3,3] for a in trajectory])
    fig,axs=plt.subplots(1,2,figsize=(10,4.15),layout="constrained",width_ratios=[1,1.4])
    axs[0].plot(ref[:,0],ref[:,2],color=BLUE,lw=1.3,ls="--",label="B2 reference")
    axs[0].plot(est[:,0],est[:,2],color=ORANGE,lw=1.1,label="Map 2, aligned")
    axs[0].scatter(*ref[0,[0,2]],color="black",s=15,marker="o",label="Start")
    axs[0].set(aspect="equal",xlabel="Reference X (m)",ylabel="Reference Z (m)",title="Camera centres")
    axs[0].legend(fontsize=8)
    axs[1].plot([float(r["seconds"]) for r in rows],[float(r["ref_centre_difference_m"]) for r in rows],color=BLUE,lw=1)
    axs[1].set(xlabel="Capture time (s)",ylabel="Centre difference from B2 (m)",title="One prefix fit applied to the whole stream",ylim=(0,None))
    for ax in axs: ax.grid(alpha=.18)
    fig.suptitle(seq.replace("_"," "),fontsize=13)
    fig.savefig(out,dpi=160);plt.close(fig)


def map_media(seq,m,frames):
    directory=MEDIA/"map"/seq
    directory.mkdir(exist_ok=True,parents=True)
    seqdir=RUN/"map"/seq
    audit=read(seqdir/"complete.json")["audit"]
    db=Path(audit["directory"])
    align=read(seqdir/"evaluation/alignment.json")
    trajectory=read(seqdir/"evaluation/trajectory.json")
    with gzip.open(seqdir/"evaluation/per_frame.csv.gz","rt") as f: rows=list(csv.DictReader(f))
    trajectory_plot(seq,trajectory,rows,VISIBLE/(seq+"_trajectory.png"))
    s=align["scale_m_per_model_unit"]
    centers=np.array([np.array(t["aligned_c2w"])[:3,3] for t in trajectory])
    refs=np.array([np.array(t["reference_c2w"])[:3,3] for t in trajectory])
    coords=np.vstack((centers,refs))[:,[0,2]]
    lower=coords.min(0)-4;upper=coords.max(0)+4
    px=385/max(upper-lower)
    def projectxz(points):
        u=np.rint((points[:,0]-lower[0])*px+12).astype(int)
        v=np.rint(400-(points[:,2]-lower[1])*px).astype(int)
        return u,v
    bg=np.full((414,414,3),[35,31,28],np.uint8)
    video=directory/"trajectory_cloud.mp4"
    writer=Movie(video,1054,450)
    points=[];colors=[];keep=[]
    outside=0
    fixed={*HARD,"frame_00120","frame_00240","frame_01420"}
    redacted=set(read(ROOT/"reports/capture-full-quality-v3/20261122-v1/video.json")["movies"]["B2_RGB_reference"]["redacted_frames"])
    for i,(info,t) in enumerate(zip(audit["frames"],trajectory)):
        fid=info["id"]; f=frames[fid]
        z,conf,k,_=load_prediction(db,info)
        image=cv2.resize(rgb(f),(512,288),interpolation=cv2.INTER_AREA)
        pose=np.array(t["aligned_c2w"])
        good=np.isfinite(z)&(z>0)
        p,c,idx=cloud(z*s,image,k,good,12)
        p=(p@pose[:3,:3].T+pose[:3,3]).astype(np.float32)
        filt=conf[idx]>1.5
        points.append(p);colors.append(c);keep.append(filt)
        u,v=projectxz(p[filt])
        inside=(u>=0)&(u<414)&(v>=0)&(v<414)
        outside+=int((~inside).sum())
        redact=int(fid.split("_")[1]) in redacted
        if not redact: bg[v[inside],u[inside]]=c[filt][inside,::-1]
        view=bg.copy()
        for path,color in ((refs,(215,170,90)),(centers,(44,151,245))):
            x,y=projectxz(path[:i+1])
            poly=np.column_stack((x,y)).astype(np.int32)
            cv2.polylines(view,[poly],False,color,1,cv2.LINE_AA)
            cv2.circle(view,tuple(poly[-1]),3,color,-1)
        canvas=np.full((450,1054,3),LIGHT,np.uint8)
        canvas[0:414,640:1054]=view
        canvas[0:180,0:320]=cv2.resize(image,(320,180))
        raw=sample(depth(f))
        canvas[0:180,320:640]=cv2.resize(colour(raw),(320,180),interpolation=cv2.INTER_NEAREST)
        canvas[218:398,0:320]=cv2.resize(colour(z*s),(320,180),interpolation=cv2.INTER_NEAREST)
        cscore=np.clip((conf-1)/4,0,1)
        canvas[218:398,320:640]=cv2.resize(cv2.applyColorMap((cscore*255).astype(np.uint8),cv2.COLORMAP_MAGMA),(320,180))
        put(canvas,"RGB",(10,204));put(canvas,"Sensor Z",(330,204))
        put(canvas,"Scaled Map depth (0.3-6 m)",(10,420),.46);put(canvas,"Confidence score (1-5 display)",(330,420),.46)
        put(canvas,"Accumulated X-Z view  |  orange: Map, blue: B2",(642,436),.43)
        put(canvas,f"{fid}   {float(rows[i]['seconds']):.2f} s    {seq}",(10,444),.46)
        if redact:
            canvas[:414,:640]=32
            put(canvas,"Source frame withheld in published replay",(50,200),.6,(230,230,230))
        if fid in fixed:
            imwrite(directory/(fid+".png"),canvas)
        dt=float(rows[i+1]["seconds"])-float(rows[i]["seconds"]) if i+1<len(rows) else .2
        writer.write(canvas,max(1,round(dt*5)))
    writer.close()
    ply(directory/"visible_all_stride12.ply",points,colors)
    ply(directory/"confidence_gt1p5_stride12.ply",[p[v] for p,v in zip(points,keep)],[c[v] for c,v in zip(colors,keep)])
    if seq=="full_train_1478":
        preview(video,252,16,VISIBLE/"map_preview.mp4")
        imwrite(VISIBLE/"map_difficult.png",cv2.imread(str(directory/"frame_01304.png")))
    save(directory/"display.json",dict(fps=5,encoded_frames=writer.frames,source_frames=len(points),encoder=encoder(),
         source_timestamps="original timestamps shown, previous frame held over train split gaps, nearest 0.2 s",
         point_sample="every twelfth source row and column, each frame",all_sample_points=sum(map(len,points)),
         confidence_kept=sum(int(v.sum()) for v in keep),outside_top_down_display=outside,
         view_bounds_xz_m=[lower.tolist(),upper.tolist()],source_redaction="historical 30-frame schedule in public video",
         video=member(video)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch",choices=("depth","map",*SEQ))
    a=parser.parse_args()
    cv2.setNumThreads(1)
    VISIBLE.mkdir(parents=True,exist_ok=True)
    m,frames=by_id()
    if a.branch=="depth": depth_media(m,frames,read(P0/"selection.json"))
    else:
        for sequence in (SEQ if a.branch=="map" else (a.branch,)): map_media(sequence,m,frames)
