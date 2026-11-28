"""Record all original hall timestamps and an orbit of the saved RGB-only Map reconstruction.

Make the small Chinese caption assets on a machine with a Chinese font, then use
the isolated Map environment with PyAV 15.1 and a CUDA GPU for render and verify.
No model is loaded. Only the output movie needs to be copied to a playback device.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lingbot_common import ROOT, RUN, by_id, depth, member, read, require, rgb, save, sha


SEQUENCE = "full_all_1846_exploratory"
WIDTH, HEIGHT = 1920, 1080
PW, PH = 880, 495
POSITIONS = ((72, 54), (968, 54), (72, 557), (968, 557))
ORBIT_RECT = (35, 57, 1850, 960)
STRIDE = 6
ORBIT_FRAMES = 250
TIME_BASE = Fraction(1, 1_000_000)
BG = (17, 25, 33)
FG = (235, 241, 245)
MUTED = (167, 185, 197)
TRAIL = (76, 183, 255)  # BGR
SELECTED = (0, 120, 240, 1304, 1743, 1845)


def labels(folder, font_path, bold_path):
    folder.mkdir(parents=True, exist_ok=True)
    def font(size, bold=False):
        return ImageFont.truetype(str(bold_path if bold else font_path), size)
    for kind, title, subtitle in [
            ("capture", "大厅全程记录", "Gemini 335   /   LingBot-Map 2   ·   完整采集探索"),
            ("orbit", "整厅三维回看", "累积点云   /   360° 旋转")]:
        im = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(im)
        draw.text((72, 5), title, font=font(30, True), fill=FG)
        draw.text((286, 13), subtitle, font=font(21), fill=MUTED)
        if kind == "capture":
            draw.text((72, 1053), "两张深度图共用光轴 Z 色标", font=font(16), fill=FG)
            draw.text((327, 1053), "0.3 m", font=font(16), fill=FG)
            ramp = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None], cv2.COLORMAP_VIRIDIS)
            im.paste(Image.fromarray(cv2.cvtColor(ramp, cv2.COLOR_BGR2RGB)).resize((170, 10)), (379, 1061))
            draw.text((559, 1053), "6 m", font=font(16), fill=FG)
            draw.rectangle((650, 1059, 663, 1072), fill=(22, 24, 28), outline=MUTED)
            draw.text((674, 1053), "缺测", font=font(16), fill=MUTED)
            draw.rectangle((750, 1059, 763, 1072), fill=(178, 110, 172))
            draw.text((775, 1053), "区间外", font=font(16), fill=MUTED)
            draw.text((968, 1053), "橙线为相机轨迹   ·   三维展示采用置信分数 > 1.5 的点", font=font(16), fill=MUTED)
        else:
            draw.text((72, 1034), "同一批已保存的 Map 预测   ·   橙线为完整相机轨迹", font=font(22), fill=MUTED)
        im.save(folder / (kind + ".png"))
    for name, title in [("rgb", "实拍 RGB"), ("cloud", "累积三维点云"),
                        ("sensor", "传感器深度"), ("map", "Map 预测深度")]:
        ft = font(21, True)
        width = int(ft.getlength(title)) + 32
        im = Image.new("RGB", (width, 36), (26, 38, 48))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 3, 35), fill=(92, 199, 188))
        d.text((14, 1), title, font=ft, fill=FG)
        im.save(folder / (name + ".png"))
    im = Image.new("RGB", (PW, PH), (40, 45, 50))
    d = ImageDraw.Draw(im)
    d.text((PW//2, PH//2), "此采集时刻按原发布规则遮挡", anchor="mm", font=font(24), fill=MUTED)
    im.save(folder / "withheld.png")


def get_inputs():
    manifest, frames = by_id()
    root = RUN / "map" / SEQUENCE
    receipt = read(root / "complete.json")["audit"]
    track = read(root / "evaluation/trajectory.json")
    alignment = read(root / "evaluation/alignment.json")
    ids = [r["id"] for r in receipt["frames"]]
    require(len(ids) == 1846 and ids == [r["id"] for r in manifest["frames"]]
            and ids == [r["id"] for r in track], "Full primary sequence / trajectory identity")
    require([r["raw_index"] for r in receipt["frames"]] == list(range(len(ids))), "Original Map raw indices")
    start = frames[ids[0]]["rgb_timestamp_ns"]
    times = [(frames[i]["rgb_timestamp_ns"] - start + 500)//1000 for i in ids]
    require(all(b > a for a, b in zip(times, times[1:])), "Source timestamps")
    orbit_start = times[-1] + int(np.median(np.diff(times)))
    pts = times + [orbit_start + 40_000*i for i in range(ORBIT_FRAMES)]
    hidden = set(read(ROOT / "reports/capture-full-quality-v3/20261122-v1/video.json")
                 ["movies"]["B2_RGB_reference"]["redacted_frames"])
    return frames, receipt, np.array([r["aligned_c2w"] for r in track]), alignment["scale_m_per_model_unit"], pts, hidden


def view_basis(yaw, elevation=34):
    az, el = np.deg2rad([yaw, elevation])
    forward = -np.array([np.sin(az)*np.cos(el), -np.sin(el), np.cos(az)*np.cos(el)])
    right = np.cross(forward, [0.,-1.,0.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.array([right, down, forward])


def framing(ply, centres):
    with ply.open("rb") as f:
        while True:
            line = f.readline()
            require(bool(line), "Point-cloud header")
            if line.strip() == b"end_header": break
        points = np.fromfile(f, dtype=np.dtype([("xyz","<f4",3),("rgb","u1",3)]))["xyz"][::24].astype(float)
    require(len(points) > 100, "Empty reference display cloud")
    low, high = np.percentile(points, [2,98], axis=0)
    centre = (low+high)/2
    scene = points-centre
    distance = 0.
    # Use the cloud only to frame the virtual camera. All selected points are still rendered.
    for yaw in np.arange(0,360,15):
        x,y,z = (scene @ view_basis(yaw).T).T
        required = np.maximum(np.abs(x)/(0.84*1.77*np.tan(np.deg2rad(20))),
                              np.abs(y)/(0.84*np.tan(np.deg2rad(20)))) - z
        distance = max(distance, float(np.percentile(required, 98.7)))
    return centre, distance, float(np.percentile(points[:,1],92)), np.vstack((centres.min(0), centres.max(0)))


class View:
    def __init__(self, size, yaw, geometry):
        import torch
        self.w, self.h = size
        centre, distance, ground, bounds = geometry
        self.rotation = view_basis(yaw)
        self.eye = centre - distance*self.rotation[2]
        self.f = .5*self.h/np.tan(np.deg2rad(20))
        self.offset = -self.rotation @ self.eye
        self.gpu_rotation = torch.as_tensor(self.rotation.T.copy(), dtype=torch.float32, device="cuda")
        self.gpu_offset = torch.as_tensor(self.offset, dtype=torch.float32, device="cuda")
        self.empty = np.iinfo(np.int64).max
        self.buffer = torch.full((self.h,self.w),self.empty,dtype=torch.int64,device="cuda")
        grid = np.empty((self.h,self.w,3),np.uint8)
        grid[:] = (42,35,28)
        x0,z0 = 2*np.floor(bounds[0,[0,2]]/2)-2
        x1,z1 = 2*np.ceil(bounds[1,[0,2]]/2)+2
        for x in np.arange(x0,x1+1,2):
            line = self.project_cpu([[x,ground,z0],[x,ground,z1]])
            cv2.line(grid,tuple(line[0]),tuple(line[1]),(56,49,42),1,cv2.LINE_AA)
        for z in np.arange(z0,z1+1,2):
            line = self.project_cpu([[x0,ground,z],[x1,ground,z]])
            cv2.line(grid,tuple(line[0]),tuple(line[1]),(56,49,42),1,cv2.LINE_AA)
        self.grid = torch.from_numpy(grid).to("cuda")

    def project_cpu(self, xyz):
        cam=np.asarray(xyz)@self.rotation.T+self.offset
        return np.rint(cam[:,:2]*(self.f/np.maximum(cam[:,2:],.01))+[self.w/2,self.h/2]).astype(np.int32)

    def add(self, points, colors):
        import torch
        if not len(points): return
        cam=points @ self.gpu_rotation+self.gpu_offset
        z=cam[:,2].contiguous()
        factor=self.f/z.clamp_min(.01)
        u=(cam[:,0]*factor+self.w/2).round().long()
        v=(cam[:,1]*factor+self.h/2).round().long()
        valid=(z>.01)&(u>=0)&(u<self.w)&(v>=0)&(v<self.h)
        # Positive IEEE float bits sort monotonically. The nearest point supplies its own colour.
        key=(z.view(torch.int32).long()<<24)|colors.long()
        self.buffer.view(-1).scatter_reduce_(0,(v*self.w+u)[valid],key[valid],reduce="amin",include_self=True)

    def image(self, trail, pose, splat_radius=1):
        import torch
        img=self.buffer
        for _ in range(splat_radius):
            wide=img.clone()
            wide[1:]=torch.minimum(wide[1:],img[:-1])
            wide[:-1]=torch.minimum(wide[:-1],img[1:])
            wide[:,1:]=torch.minimum(wide[:,1:],img[:,:-1])
            wide[:,:-1]=torch.minimum(wide[:,:-1],img[:,1:])
            img=wide
        bgr=torch.stack(tuple((img>>shift)&255 for shift in (0,8,16)),dim=-1).to(torch.uint8)
        out=torch.where((img==self.empty)[...,None],self.grid,bgr).cpu().numpy()
        path=self.project_cpu(trail)
        cv2.polylines(out,[path],False,(39,37,29),3,cv2.LINE_AA)
        cv2.polylines(out,[path],False,TRAIL,1 if self.w<1000 else 2,cv2.LINE_AA)
        cv2.circle(out,tuple(path[-1]),4 if self.w<1000 else 6,TRAIL,-1,cv2.LINE_AA)
        if pose is not None:
            corners=np.array([[-.27,-.15,.40],[.27,-.15,.40],[.27,.15,.40],[-.27,.15,.40]])
            corners=self.project_cpu(corners@pose[:3,:3].T+pose[:3,3])
            cv2.polylines(out,[corners],True,TRAIL,1,cv2.LINE_AA)
            for xy in corners: cv2.line(out,tuple(path[-1]),tuple(xy),TRAIL,1,cv2.LINE_AA)
        return out


def prepare_frame(row, record, pose, scale, db, withheld):
    from lingbot_map_evaluate import load_prediction
    from lingbot_media import colour
    if withheld: return None
    image=rgb(row)
    raw=depth(row)
    z,confidence,k,_=load_prediction(db,record)
    measured=cv2.resize(raw,(PW,PH),interpolation=cv2.INTER_NEAREST_EXACT)
    scaled=z*scale
    predicted=cv2.resize(scaled,(PW,PH),interpolation=cv2.INTER_NEAREST_EXACT)
    small=cv2.resize(image,(512,288),interpolation=cv2.INTER_AREA)
    v,u=np.nonzero((np.isfinite(scaled)&(scaled>0)&(confidence>1.5))[::STRIDE,::STRIDE])
    v*=STRIDE; u*=STRIDE
    d=scaled[v,u]
    local=np.column_stack(((u-k[0,2])*d/k[0,0],(v-k[1,2])*d/k[1,1],d))
    xyz=(local@pose[:3,:3].T+pose[:3,3]).astype(np.float32)
    c=small[v,u].astype(np.int32)
    packed=c[:,0]|(c[:,1]<<8)|(c[:,2]<<16)
    return cv2.resize(image,(PW,PH),interpolation=cv2.INTER_AREA),colour(measured),colour(predicted),xyz,packed


def ordered(pool, function, count, ahead=12):
    pending=deque()
    for i in range(count):
        pending.append(pool.submit(function,i))
        if len(pending)>=ahead: yield pending.popleft().result()
    while pending: yield pending.popleft().result()


def stamp(tick):
    ms=(int(tick)+500)//1000
    return f"{ms//60000:02d}:{ms//1000%60:02d}.{ms%1000:03d}"


def write_frame(mux,stream,canvas,ts):
    import av
    f=av.VideoFrame.from_ndarray(canvas,format="bgr24")
    f.time_base=TIME_BASE
    f.pts=int(ts)
    for packet in stream.encode(f): mux.mux(packet)


def verify(movie, asset_dir, pts, hidden, capture_n, qa):
    import av
    qa.mkdir(parents=True,exist_ok=True)
    withheld=cv2.imread(str(asset_dir/"withheld.png"))
    frames,_,_,_,_,_=get_inputs()
    ids=list(frames)
    source_errors={}
    hidden_errors=[]
    rotations=[]
    selected=set(SELECTED)|{capture_n,capture_n+62,capture_n+125,capture_n+187}
    with av.open(str(movie)) as c:
        stream=c.streams.video[0]
        require(stream.codec_context.name=="h264" and (stream.width,stream.height)==(WIDTH,HEIGHT),"Output codec or size")
        count=0
        for i,f in enumerate(c.decode(video=0)):
            require(i<len(pts) and f.pts*f.time_base==pts[i]*TIME_BASE,f"Output timestamp {i}")
            count+=1
            if i not in selected and i not in hidden: continue
            image=f.to_ndarray(format="bgr24")
            x,y=POSITIONS[0]
            if i in hidden:
                for px,py in (POSITIONS[0],POSITIONS[2],POSITIONS[3]):
                    field=image[py+70:py+PH-20,px+15:px+PW-15]
                    hidden_errors.append(float(np.mean(cv2.absdiff(field,withheld[70:PH-20,15:PW-15]))))
            elif i<capture_n:
                reference=cv2.resize(rgb(frames[ids[i]]),(PW,PH),interpolation=cv2.INTER_AREA)[56:-10,12:-12]
                recon=image[y+56:y+PH-10,x+12:x+PW-12]
                source_errors[ids[i]]=float(np.mean(cv2.absdiff(cv2.resize(reference,(160,90),interpolation=cv2.INTER_AREA),
                                                                cv2.resize(recon,(160,90),interpolation=cv2.INTER_AREA))))
            else:
                rotations.append(cv2.resize(image[100:960,70:1850],(160,90),interpolation=cv2.INTER_AREA))
            if i in selected:
                cv2.imwrite(str(qa/f"decoded_{i:04d}.jpg"),cv2.resize(image,(960,540),interpolation=cv2.INTER_AREA))
        require(count==len(pts),"Missing encoded frames")
        duration=float(stream.duration*stream.time_base)
    require(len(hidden_errors)==3*len(hidden) and max(hidden_errors)<5,"Published redaction")
    require(source_errors and max(source_errors.values())<12,"Source / output RGB frame matching")
    motion=[float(np.mean(cv2.absdiff(a,b))) for a,b in zip(rotations,rotations[1:])]
    require(len(motion)==3 and min(motion)>1.0,"Orbit playback")
    return dict(decoded_frames=count,exact_pts_checked=count,duration_s=duration,redacted_positions=len(hidden),
                matching_rgb_mae_8bit=source_errors,redacted_tile_max_mae_8bit=max(hidden_errors),orbit_view_differences=motion)


def render(asset_dir, movie):
    import av
    import torch
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    require(torch.cuda.is_available(),"CUDA is required for the point renderer")
    require(not movie.exists(),"Choose a new output movie path")
    movie.parent.mkdir(parents=True,exist_ok=True)
    pending=movie.with_suffix(".pending.mp4")
    require(not pending.exists(),"Previous unfinished recording exists")
    frames,receipt,poses,scale,pts,hidden=get_inputs()
    records=receipt["frames"]
    centres=poses[:,:3,3]
    geometry=framing(RUN/"media"/"map"/SEQUENCE/"confidence_gt1p5_stride12.ply",centres)
    azimuth=145.
    view=View((PW,PH),azimuth,geometry)
    cap=len(records)*((512+STRIDE-1)//STRIDE)*((288+STRIDE-1)//STRIDE)
    xyz=torch.empty((cap,3),device="cuda",dtype=torch.float32)
    color=torch.empty(cap,device="cuda",dtype=torch.int32)
    used=0
    art={n:cv2.imread(str(asset_dir/(n+".png"))) for n in ("capture","orbit","withheld","rgb","cloud","sensor","map")}
    require(all(a is not None for a in art.values()),"Generate caption assets first")
    db=Path(receipt["directory"])
    def load(i):
        r=records[i]
        return prepare_frame(frames[r["id"]],r,poses[i],scale,db,int(r["id"].split("_")[1]) in hidden)
    with av.open(str(pending),"w",format="mp4",options={"movflags":"+faststart","video_track_timescale":"1000000"}) as mux:
        st=mux.add_stream("h264_nvenc",rate=25)
        st.width,st.height=WIDTH,HEIGHT
        st.pix_fmt="yuv420p"
        st.time_base=st.codec_context.time_base=TIME_BASE
        st.codec_context.max_b_frames=0
        st.codec_context.gop_size=25
        st.options={"preset":"p5","rc":"vbr","cq":"21","b":"0"}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i,chunk in enumerate(ordered(pool,load,len(records))):
                if chunk is None:
                    camera=measured=predicted=art["withheld"]
                else:
                    camera,measured,predicted,p,c=chunk
                    end=used+len(p)
                    xyz[used:end].copy_(torch.from_numpy(p))
                    color[used:end].copy_(torch.from_numpy(c))
                    view.add(xyz[used:end],color[used:end])
                    used=end
                panel=view.image(centres[:i+1],poses[i])
                canvas=art["capture"].copy()
                for (x,y),im,name in zip(POSITIONS,(camera,panel,measured,predicted),("rgb","cloud","sensor","map")):
                    canvas[y:y+PH,x:x+PW]=im
                    tag=art[name]
                    canvas[y+9:y+9+tag.shape[0],x+10:x+10+tag.shape[1]]=tag
                text=f"{stamp(pts[i])}    {records[i]['id']}    {i+1:04d} / {len(records)}"
                cv2.putText(canvas,text,(1260,36),cv2.FONT_HERSHEY_SIMPLEX,.75,(231,238,242),1,cv2.LINE_AA)
                write_frame(mux,st,canvas,pts[i])
                if (i+1)%200==0: print(f"Capture {i+1}/{len(records)}, {used:,} points",flush=True)
        x,y,w,h=ORBIT_RECT
        for i in range(ORBIT_FRAMES):
            orbit_view=View((w,h),azimuth+360*i/ORBIT_FRAMES,geometry)
            for begin in range(0,used,500_000):
                end=min(used,begin+500_000)
                orbit_view.add(xyz[begin:end],color[begin:end])
            canvas=art["orbit"].copy()
            canvas[y:y+h,x:x+w]=orbit_view.image(centres,None,splat_radius=2)
            cv2.putText(canvas,f"{360*i/ORBIT_FRAMES:05.1f} deg   |   {used:,} points",
                        (1350,36),cv2.FONT_HERSHEY_SIMPLEX,.73,(231,238,242),1,cv2.LINE_AA)
            write_frame(mux,st,canvas,pts[len(records)+i])
            if (i+1)%50==0: print("Orbit",i+1,"/",ORBIT_FRAMES,flush=True)
        for packet in st.encode(): mux.mux(packet)
    check=verify(pending,asset_dir,pts,hidden,len(records),movie.parent/"qa")
    pending.rename(movie)
    output=dict(video=member(movie),capture_frames=len(records),capture_last_pts_s=pts[len(records)-1]/1e6,
                orbit_start_s=pts[len(records)]/1e6,orbit_frames=ORBIT_FRAMES,orbit_fps=25,
                full_sequence=SEQUENCE,map_config_scale_m_per_unit=scale,source_frames_displayed=len(records)-len(hidden),
                display_point_count=used,point_stride=STRIDE,confidence_threshold=1.5,
                depth_display_m=[.3,6],size=[WIDTH,HEIGHT],codec="H.264 / NVENC, yuv420p",pyav=av.__version__,
                original_data_use="train + val + development-test RGB, exploratory",
                pts="RGB device timestamps relative to first frame in microseconds; orbit follows final capture",
                source_inputs=member(RUN/"prepare/freeze.json"),map_audit=member(RUN/"map"/SEQUENCE/"complete.json"),
                registration_sha256=sha(RUN/"map"/SEQUENCE/"evaluation/alignment.json"),verification=check)
    save(movie.with_suffix(".json"),output)
    print("Verified",movie,check["decoded_frames"],"frames",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="action",required=True)
    make=subs.add_parser("labels")
    make.add_argument("folder",type=Path)
    make.add_argument("--font",type=Path,required=True)
    make.add_argument("--bold",type=Path,required=True)
    record=subs.add_parser("render")
    record.add_argument("folder",type=Path)
    record.add_argument("movie",type=Path)
    args=parser.parse_args()
    if args.action=="labels": labels(args.folder,args.font,args.bold)
    else: render(args.folder,args.movie)
