"""Independent frozen two-box silhouette tracker on the recorded front RGB camera."""

import argparse
import concurrent.futures
import json
import math
from pathlib import Path

import av
import cv2
import numpy as np

from interactive_push_assets import file_hash, write_json
from interactive_push_prepare import read_episode


# Frozen side-camera calibration from kywind/real2sim-eval, cfg/env/xarm_pusher.yaml.
K_RAW=np.array([[427.292022705,0.,429.999359131],[0.,426.792694092,242.811523438],[0.,0.,1.]])
T_CV=np.array([[.005258014,.612551232,-.790413399,.883026390],
               [.999986009,-.003677991,.003801786,.053908462],
               [-.000578345,-.790422330,-.612562001,.397603386],[0.,0.,0.,1.]])
RES=.0035
LOW=np.array([.19,-.37])
HIGH=np.array([.79,.37])


def project(points,k,t=T_CV):
    camera=(np.asarray(points)-t[:3,3])@t[:3,:3]
    pixels=camera@k.T
    return pixels[...,:2]/pixels[...,2:]


def cropped_k(mode):
    a,s = (.5,184.) if mode=="center_square" else (240/848,0.)
    b=.5
    # Pixel center resize convention used by OpenCV.
    affine=np.array([[a,0.,a*(-s+.5)-.5],[0.,b,b*.5-.5],[0.,0.,1.]])
    return affine@K_RAW


def front_file(data,e):
    return data/f"videos/chunk-000/observation.images.front/episode_{e:06d}.mp4"


def images(path):
    with av.open(str(path)) as video:
        stream=video.streams.video[0]
        stream.codec_context.thread_count=1
        for frame in video.decode(stream):
            if (frame.width,frame.height)!=(240,240):
                raise ValueError("Unexpected RGB canvas")
            yield frame.to_ndarray(format="bgr24")


def shaft_scores(gray,xy,mode,offsets):
    k=cropped_k(mode)
    z=np.linspace(.045,.13,9)
    points=np.column_stack((np.broadcast_to(xy,(len(z),2)),z))
    p=project(points,k)
    u=p[-1]-p[0]; orth=np.array([-u[1],u[0]])/np.linalg.norm(u)
    p=p[None,:,:]+offsets[:,None,:]
    def intens(d):
        q=np.rint(p+orth*d).astype(int)
        return gray[q[:,:,1].clip(0,239),q[:,:,0].clip(0,239)].astype(float)
    # Robust center brightness relative to shaft flanks on felt; object overlap can remove the contrast.
    score=np.clip(intens(0)-np.maximum(intens(-6),intens(6)),-70,70)
    return np.median(score,axis=1)


def tip_pixel(gray, xy):
    """Find the lower cap of the long cylindrical pusher using its two image edges."""
    expected=project(np.r_[xy,.02],cropped_k("center_square"))
    lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if lines is None:
        return None
    segments=[]
    for raw in lines[:,0].reshape(-1,2,2):
        top,bottom=sorted(raw,key=lambda p:p[1])
        l=bottom-top
        if l[1]<24 or abs(l[0])>.48*l[1] or abs(bottom[0]-expected[0])>52 or abs(bottom[1]-expected[1])>80:
            continue
        segments.append((top,bottom,l[0]/l[1]))
    best=(-1.,None)
    for i,(a,b,s) in enumerate(segments):
        for c,d,v in segments[:i]:
            y0,y1=max(a[1],c[1]),min(b[1],d[1])
            if y1-y0<22 or abs(b[1]-d[1])>13 or abs(s-v)>.10:
                continue
            y=np.linspace(y0+5,y1-4,12)
            u=b[0]+(y-b[1])*s
            w=d[0]+(y-d[1])*v
            separation=np.median(abs(u-w))
            if not 3<=separation<=11:
                continue
            mid=(u+w)/2
            def at(x,y):
                return gray[y.round().astype(int).clip(0,239),x.round().astype(int).clip(0,239)].astype(float)
            contrast=np.median(at(mid,y)-np.maximum(at(np.minimum(u,w)-3,y),at(np.maximum(u,w)+3,y)))
            end_y=np.arange(int(y1)-5,int(y1)+13)
            x=(b[0]+(end_y-b[1])*s+d[0]+(end_y-d[1])*v)/2
            cap=at(x,end_y-2)-at(x,end_y+3)
            k=int(np.argmax(cap))
            score=float(contrast+min(y1-y0,80)*.25+np.clip(cap[k],-70,70)*.24)
            if score>best[0] and contrast>6 and cap[k]>4:
                best=score,np.array([x[k],end_y[k]],float)
    return best[1]


def calibrate(data,prep,out):
    roles=json.loads((prep/"split.json").read_text())["roles"]
    scene=json.loads((prep/"scene_C0.json").read_text())
    offsets=np.array([(x,y) for x in range(-8,9,2) for y in range(-8,9,2)],float)
    scores={}; anchors=[]
    for mode in ("center_square","stretch"):
        scores[mode]=[]
    for e in roles["fit"]:
        _,_,robot=read_episode(data,e)
        sampled=set(np.linspace(0,len(robot)-1,9).round().astype(int))
        for i,im in enumerate(images(front_file(data,e))):
            if i in sampled:
                gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                tip=tip_pixel(gray,robot[i])
                if tip is not None:
                    anchors.append((e,i,robot[i].tolist(),tip.tolist()))
                for mode in scores:
                    scores[mode].append(shaft_scores(gray,robot[i],mode,offsets))
    result={}
    for mode,s in scores.items():
        response=np.median(np.asarray(s),axis=0)
        idx=int(np.argmax(response))
        result[mode]={"median_shaft_contrast":float(response[idx]),"offset_uv":offsets[idx].tolist(),"observations":len(s)}
    mode=max(result,key=lambda x:result[x]["median_shaft_contrast"])
    k=cropped_k(mode)
    xyz=np.array([[*row[2],.02] for row in anchors],np.float64)
    uv=np.array([row[3] for row in anchors],np.float64)
    if len(anchors)<20:
        raise RuntimeError("Too few independently detected pusher caps to calibrate the camera")
    cv2.setRNGSeed(42)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,uv,k,None,iterationsCount=2000,reprojectionError=3.,confidence=.999)
    if not ok or len(inliers)<20:
        raise RuntimeError("Static front calibration could not resolve pusher landmarks")
    ix=inliers[:,0]
    rvec,tvec=cv2.solvePnPRefineLM(xyz[ix],uv[ix],k,None,rvec,tvec)
    w2c=np.eye(4);w2c[:3,:3]=cv2.Rodrigues(rvec)[0];w2c[:3,3]=tvec[:,0]
    c2w=np.linalg.inv(w2c)
    errors=np.linalg.norm(project(xyz,k,c2w)-uv,axis=1)
    freeze={"tracker":"fit-calibrated camera with frozen bright-material silhouette templates, sampled at 3 degree yaw",
            "code_sha256":file_hash(Path(__file__))[0],"role":"fit only","used_episodes":roles["fit"],"camera_candidates":result,
            "selected_crop":mode,"K":k.tolist(),"T_world_camera_cv":c2w.tolist(),"raw_K":K_RAW.tolist(),
            "pusher_tip_plane_z_m_assumed":.02,"fit_cap_landmarks":anchors,"fit_PnP_inlier_indices":ix.tolist(),
            "fit_cap_inlier_median_px":float(np.median(errors[ix])),"fit_all_cap_median_px":float(np.median(errors)),
            "raw_wh":[848,480],"output_wh":[240,240],"pixel_grid_m":RES,"shape":scene["objects"][0],
            "color_rule":{"min_R":174,"min_G":115,"min_R_minus_G":16,"min_R_minus_B":34,"max_hue_opencv":23},
            "validity":{"min_dice":.65,"min_visible_fraction":.45},"label_source":"independent front RGB plus static known shape",
            "backend":"CPU/OpenCV","goal_overlay":"yellow translucent fixed goal excluded by color; foreground can remain visible through overlay"}
    write_json(out/"tracker_freeze.json",freeze)
    print(json.dumps(result),flush=True)


class Tracker:
    def __init__(self,freeze):
        self.f=freeze
        self.k=np.array(freeze["K"])
        self.t=np.array(freeze["T_world_camera_cv"])
        self.xy=np.stack(np.meshgrid(np.arange(LOW[0],HIGH[0]+RES/2,RES),
                                    np.arange(LOW[1],HIGH[1]+RES/2,RES)),axis=-1)
        z=float(freeze["shape"]["z_extent_m"])
        uv=project(np.concatenate((self.xy,np.full((*self.xy.shape[:2],1),z)),axis=-1),self.k,self.t)
        self.u,self.v=(np.ascontiguousarray(uv[...,i],dtype=np.float32) for i in (0,1))
        self.workspace=(self.v>=62)&(self.v<239)&(self.u>=1)&(self.u<239)
        parts=freeze["shape"]["parts"]
        bound=max(np.linalg.norm(np.abs(p["center"][:2])+np.array(p["half"][:2])) for p in parts)
        self.r=math.ceil(bound/RES)+2
        extent=np.arange(-self.r,self.r+1)*RES
        mesh=np.stack(np.meshgrid(extent,extent),axis=-1)
        self.templates={}
        for degree in range(0,360,3):
            angle=np.deg2rad(degree)
            mat=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            local=mesh@mat
            inside=np.zeros(mesh.shape[:2],bool)
            for part in parts:
                inside|=np.all(np.abs(local-part["center"][:2])<=part["half"][:2],axis=-1)
            self.templates[degree]=inside.astype(np.float32)
        self.last=None

    def observation(self,im,pusher):
        seen=np.full(im.shape[:2],255,np.uint8)
        ends=project(np.column_stack((np.broadcast_to(pusher,(3,2)),[.015,.16,.31])),self.k,self.t).round().astype(int)
        cv2.line(seen,tuple(ends[0]),tuple(ends[1]),0,10)
        cv2.line(seen,tuple(ends[1]),tuple(ends[2]),0,36)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        b,g,r=cv2.split(im.astype(np.int16))
        rule=self.f["color_rule"]
        color=((r>=rule["min_R"])&(g>=rule["min_G"])&(r-g>=rule["min_R_minus_G"])&
               (r-b>=rule["min_R_minus_B"])&(hsv[...,0]<=rule["max_hue_opencv"])).astype(np.uint8)
        mask=cv2.remap(color,self.u,self.v,cv2.INTER_NEAREST)
        vis=(cv2.remap(seen,self.u,self.v,cv2.INTER_NEAREST)>0)&self.workspace
        return mask*vis,vis

    def estimate(self,im,pusher,frame):
        measured,visible=self.observation(im,pusher)
        r=self.r
        masks=[cv2.copyMakeBorder(x.astype(np.float32),r,r,r,r,cv2.BORDER_CONSTANT) for x in (measured,visible)]
        # Translation window for nearby and global RGB-only reacquisition. The prior pose is just a proposal.
        neighborhood=cv2.boxFilter(measured.astype(np.float32),-1,(2*r+1,2*r+1),normalize=False,borderType=cv2.BORDER_CONSTANT)
        ny,nx=measured.shape
        def search(angles,bounds):
            x0,y0,x1,y1=bounds
            obs,vis=(x[y0:y1+2*r,x0:x1+2*r] for x in masks)
            total=neighborhood[y0:y1,x0:x1]
            best=(-1.,0,0,0,0.)
            for degree in angles:
                theta=int(degree)%360
                temp=self.templates[theta]
                hit=cv2.matchTemplate(obs,temp,cv2.TM_CCORR)
                area=cv2.matchTemplate(vis,temp,cv2.TM_CCORR)
                frac=area/temp.sum()
                dice=2*hit/np.maximum(total+area,1)
                dice[frac<self.f["validity"]["min_visible_fraction"]]=-1.
                _,peak,_,(xi,yi)=cv2.minMaxLoc(dice)
                if peak>best[0]:
                    best=peak,theta,xi+x0,yi+y0,float(frac[yi,xi])
            return best
        if self.last is not None and frame%30:
            _,deg,x,y,_=self.last
            bounds=(max(0,x-10),max(0,y-10),min(nx,x+11),min(ny,y+11))
            best=search(range(deg-18,deg+19,6),bounds)
        else:
            best=(-1.,0,0,0,0.)
        if best[0]<.65:
            best=search(range(0,360,15),(0,0,nx,ny))
        _,deg,x,y,_=best
        best=search(range(deg-9,deg+10,3),(max(0,x-3),max(0,y-3),min(nx,x+4),min(ny,y+4)))
        score,degree,x,y,frac=best
        valid=score>=self.f["validity"]["min_dice"] and frac>=self.f["validity"]["min_visible_fraction"]
        self.last=best if valid else None
        pose=np.array([LOW[0]+x*RES,LOW[1]+y*RES,np.deg2rad(degree)])
        return pose, bool(valid), float(score),float(frac)

    def outline(self,im,pose,valid):
        out=im.copy()
        angle=pose[2]
        mat=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        for part in self.f["shape"]["parts"]:
            xy=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*part["half"][:2]+part["center"][:2]
            xyz=np.column_stack((xy@mat.T+pose[:2],np.full(4,self.f["shape"]["z_extent_m"])))
            cv2.polylines(out,[project(xyz,self.k,self.t).round().astype(np.int32)],True,(40,230,40) if valid else (40,40,240),1)
        return out


def track_episode(work):
    data,out,e=work
    cv2.setNumThreads(1)
    base=out/"labels"/f"episode_{e:06d}"
    receipt=base.with_suffix(".json")
    freeze=out/"tracker_freeze.json"
    identity={"tracker":file_hash(freeze)[0],"clip":file_hash(front_file(data,e))[0]}
    if receipt.is_file():
        old=json.loads(receipt.read_text()); path=base.with_suffix(".npz")
        if old["identity"]==identity and file_hash(path)[0]==old["labels_sha256"]:
            return old
        raise ValueError(f"Frozen labels changed for episode {e}; use a new attempt")
    t,cmd,robot=read_episode(data,e)
    tr=Tracker(json.loads(freeze.read_text()))
    keys=set(np.rint(np.linspace(0,len(t)-1,6)).astype(int))
    poses=[];valid=[];score=[];visibility=[];panels=[]
    for i,im in enumerate(images(front_file(data,e))):
        pos,ok,s,v=tr.estimate(im,robot[i],i)
        poses.append(pos);valid.append(ok);score.append(s);visibility.append(v)
        if i in keys:
            overlay=tr.outline(im,pos,ok)
            cv2.putText(overlay,f"e{e} f{i} F1 {s:.2f}",(3,18),0,.43,(255,255,255),1)
            panels.append(overlay)
    if len(poses)!=len(t):
        raise ValueError(f"Video/time count mismatch for episode {e}")
    base.parent.mkdir(parents=True,exist_ok=True)
    f=base.with_suffix(".npz")
    np.savez_compressed(f,time_s=t,action_command=cmd,executed_pusher_xy=robot,
                        object_xytheta=np.array(poses),object_valid=valid,mask_dice=score,visible_fraction=visibility)
    cv2.imwrite(str(base.with_suffix(".jpg")),np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])]))
    result={"episode":e,"frames":len(t),"valid":int(np.sum(valid)),"first_frame_valid":bool(valid[0]),
            "mean_dice":float(np.mean(score)), "identity":identity,"labels_sha256":file_hash(f)[0],
            "audit_frames":sorted(map(int,keys)),"audit_jpeg":str(base.with_suffix(".jpg"))}
    write_json(receipt,result)
    print(f"episode {e}: {result['valid']}/{len(t)} independent RGB labels",flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["calibrate","track"]);p.add_argument("data",type=Path)
    p.add_argument("prepare",type=Path);p.add_argument("out",type=Path);p.add_argument("--workers",type=int,default=1)
    p.add_argument("--role",choices=["fit","calibration","test","all"],default="all")
    a=p.parse_args();cv2.setNumThreads(1)
    a.out.mkdir(parents=True,exist_ok=True)
    if a.mode=="calibrate":
        calibrate(a.data,a.prepare,a.out);return
    roles=json.loads((a.prepare/"split.json").read_text())["roles"]
    ids=roles[a.role] if a.role!="all" else sum(roles.values(),[])
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(track_episode,[(a.data,a.out,i) for i in ids]))
    write_json(a.out/("tracking_"+a.role+".json"),{"role":a.role,"episodes":results,"frames":sum(x["frames"] for x in results),
               "valid":sum(x["valid"] for x in results),"source":"frozen RGB tracker; visual reference, not motion capture"})


if __name__=="__main__":
    main()
