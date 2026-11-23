"""Full frozen trajectory comparisons, fixed native-resolution crops and privacy verification."""
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import cv2
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras,CameraType

import capture_quality_eval as evaluator
from capture_quality_video import probe
from capture_surface_evaluate import make_scene, rays, FLIP
from capture_surface_submaps import B2,ROOT,RUN,check,read,sha,write
from capture_metric3d_common import selection,member,B0_REPORT
from vr_room_run import CPU

VIDEO=RUN/'video'
PRE=RUN/'previews'


def render_model(pipe,pose,meta):
    cam=Cameras(camera_to_worlds=torch.tensor(pose,dtype=torch.float32)[None,:3,:],
                fx=meta['fl_x'],fy=meta['fl_y'],cx=meta['cx'],cy=meta['cy'],
                width=1280,height=720,camera_type=CameraType.PERSPECTIVE).to(pipe.device)
    with torch.inference_mode():
        rgb=pipe.model.get_outputs_for_camera(camera=cam)['rgb'].reshape(720,1280,3).cpu().numpy()
    check(np.isfinite(rgb).all(),'nonfinite timeline RGB')
    return np.uint8(np.rint(255*np.clip(rgb,0.,1.)))[:,:,::-1]


def panels(images,labels):
    canvas=cv2.copyMakeBorder(np.hstack(images),42,0,0,0,cv2.BORDER_CONSTANT)
    x=0
    for image,label in zip(images,labels):
        cv2.putText(canvas,label,(x+8,28),0,.72,(250,250,250),1,cv2.LINE_AA)
        x+=image.shape[1]
    return canvas


def finish_video(folder,seq,masked,old,identity):
    tape=['ffconcat version 1.0']
    for i,r in enumerate(seq):
        tape.append("file '"+r['id']+".jpg'")
        if i+1<len(seq):
            dt=(int(seq[i+1]['actual_rgb_timestamp_ns'])-int(r['actual_rgb_timestamp_ns']))/1.e9
            tape.append(f'duration {dt:.9f}')
    manifest=folder/'timeline.ffconcat'; manifest.write_text('\n'.join(tape)+'\n')
    temp=folder/'public.pending.mp4'; movie=folder/'public.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','concat','-safe','0','-i',manifest,
                    '-vsync','vfr','-c:v','libx264','-threads',str(min(CPU,12)),'-crf','18',
                    '-pix_fmt','yuv420p','-movflags','+faststart',temp],check=True)
    actual=probe(temp)
    check(actual==probe(old) and actual[2]==1846,'public movie PTS, resolution or frame count')
    vf=f"select='between(n,{min(masked)},{max(masked)})',crop=3840:696:0:58,scale=96:24"
    raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',temp,'-vf',vf,'-vsync','0',
        '-threads','2','-f','rawvideo','-pix_fmt','rgb24','-'])
    a=np.frombuffer(raw,np.uint8).reshape(-1,24,96,3)
    check(len(a)==30 and np.max(np.abs(a.astype('i2')-45))<=8,'encoded privacy readback')
    temp.replace(movie)
    receipt=dict(video=member(movie),frames=1846,duration_s=float(actual[3]),timeline=member(manifest),
                 redacted_frames=masked,all_pts_equal_B2=True,privacy_verified=True,identity=identity)
    write(folder/'receipt.json',receipt)
    return receipt


def shade(scene,pose,intr,crop):
    x0,y0,x1,y1=crop
    image=np.full((y1-y0,x1-x0,3),245,'u1')
    for y in range(y0,y1,120):
        yy,xx=np.mgrid[y:min(y+120,y1),x0:x1]
        hit=scene.cast_rays(rays(xx,yy,pose,intr),nthreads=CPU)
        z=hit['t_hit'].numpy(); valid=np.isfinite(z)&(z>0)
        normals=hit['primitive_normals'].numpy()[valid]
        light=.24+.76*np.abs(normals@np.array([-.4,-.6,-.69])).clip(0,1)
        image[y-y0:y-y0+len(yy)][valid]=(light[:,None]*[204,212,219]).astype('u1')
    return image


def cached_movie(folder,identity):
    p=folder/'receipt.json'
    if not p.exists():
        return None
    m=read(p)
    check(m['identity']==identity and m['frames']==1846 and m['privacy_verified'] and
          member(ROOT/m['video']['path'])==m['video'],'video receipt mismatch')
    return m


def main():
    cv2.setNumThreads(1)
    freeze=RUN/'state/selection_freeze.json'; f=read(freeze)
    test=read(RUN/'evaluation/development_test.json')
    check(test['freeze_sha256']==sha(freeze),'timeline freeze')
    seq=selection()
    old=read(B2/'video/video_manifest.json')
    movie=ROOT/old['video']['path']
    check(member(movie)==old['video'] and old['frames']==len(seq)==1846,'B2 reference movie')
    masked=[i for i,r in enumerate(seq) if 349<=r['relative_s']<=353.7 or r.get('annotation')=='late dynamic interval']
    check(masked==old['redacted_frames'] and len(masked)==30,'privacy schedule')
    VIDEO.mkdir(exist_ok=True);PRE.mkdir(exist_ok=True)
    times=VIDEO/'timestamps.json'
    write(times,[dict(frame=r['id'],actual_rgb_timestamp_ns=r['actual_rgb_timestamp_ns'],relative_s=r['relative_s'],
                      split=r['requested_split'],redacted=i in masked) for i,r in enumerate(seq)])
    views=[dict(name=r['object'],stem=f"frame_{r['train_id']:05d}",xy=r['xyxy'])
           for r in read(B0_REPORT/'mesh_validation.json')['local_views']]
    views.append(dict(name='foam_teeth',stem='frame_01420',xy=[720,224,1104,608]))
    movies={}; model_crops={}
    for name in ('B2','B3') if f['trained'] else ('B2',):
        mr=B2 if name=='B2' else RUN; scene=mr/'inputs'
        meta=read(scene/'transforms.json'); frames={Path(x['file_path']).stem:x for x in meta['frames']}
        step=f['checkpoint_steps'][name][0]
        with patch.object(evaluator,'SCENE',scene),patch.object(evaluator,'RUN',mr):
            _,pipe,ck,_=evaluator.load_model(Path(f['configs'][name]),step,'val')
        check(sha(Path(ck))==f['checkpoint_sha256'][name][str(step)],'timeline checkpoint')
        for obj in views:
            stem=obj['stem']; x,y,x1,y1=obj['xy']
            model_crops[name,obj['name']]=render_model(pipe,frames[stem]['transform_matrix'],meta)[y:y1,x:x1].copy()
        if name=='B3':
            folder=VIDEO/'rgb_B2_B3';folder.mkdir(exist_ok=True)
            identity=dict(freeze=sha(freeze),checkpoint=f['checkpoint_sha256'][name][str(step)],B2=old['video'])
            receipt=cached_movie(folder,identity)
            if not receipt:
                for i,r in enumerate(seq):
                    stem=r['id']
                    if i in masked:
                        images=[np.full((720,1280,3),45,'u1')]*3
                    else:
                        archive=cv2.imread(str(B2/'video'/(stem+'.jpg')))
                        check(archive is not None and archive.shape==(762,3840,3),'B2 RGB archive '+stem)
                        images=[cv2.imread(str(scene/frames[stem]['file_path'])),archive[42:,2560:],
                                render_model(pipe,frames[stem]['transform_matrix'],meta)]
                    canvas=panels(images,[f'Reference {stem} {r["relative_s"]:.3f}s','B2 210k (archived JPEG)',f'B3 {step+1} updates'])
                    check(cv2.imwrite(str(folder/(stem+'.jpg')),canvas),'RGB timeline write')
                receipt=finish_video(folder,seq,masked,movie,identity)
            movies['RGB']=receipt
        else:
            movies['B2_RGB_reference']=old
        del pipe
        torch.cuda.empty_cache()
    local={}
    for label,base,new in [('B2','supported_poisson','new_model_surface')]+(
            [('B3','B3_supported_poisson','B3_new_model_surface')] if f['trained'] else []):
        folder=VIDEO/('meshes_'+label);folder.mkdir(exist_ok=True)
        identity=dict(freeze=sha(freeze),old=f['mesh_sha256'][base],new=f['mesh_sha256'][new])
        dest=RUN/'inputs' if label=='B3' else B2/'inputs'
        meta=read(dest/'transforms.json'); frames={Path(row['file_path']).stem:row for row in meta['frames']}
        intr=[meta[k] for k in ('fl_x','fl_y','cx','cy')]
        old_scene,new_scene=[make_scene(Path(f['mesh_paths'][k])) for k in (base,new)]
        for obj in views:
            x,y,x1,y1=obj['xy']; frame=frames[obj['stem']]; pose=np.array(frame['transform_matrix'])@FLIP
            images=[cv2.imread(str(dest/frame['file_path']))[y:y1,x:x1],model_crops[label,obj['name']],
                    shade(old_scene,pose,intr,obj['xy']),shade(new_scene,pose,intr,obj['xy'])]
            path=PRE/(label+'_'+obj['name']+'.jpg')
            check(cv2.imwrite(str(path),panels(images,['Reference',label+' RGB','Old mesh','New mesh'])),'crop')
            local[label+'/'+obj['name']]=dict(frame=obj['stem'],crop=obj['xy'],image=member(path))
        receipt=cached_movie(folder,identity)
        if not receipt:
            for i,r in enumerate(seq):
                stem=r['id']; frame=frames[stem]; pose=np.array(frame['transform_matrix'])@FLIP
                if i in masked:
                    images=[np.full((720,1280,3),45,'u1')]*3
                else:
                    images=[cv2.imread(str(dest/frame['file_path'])),shade(old_scene,pose,intr,(0,0,1280,720)),
                            shade(new_scene,pose,intr,(0,0,1280,720))]
                check(cv2.imwrite(str(folder/(stem+'.jpg')),panels(images,[f'Reference {stem} {r["relative_s"]:.3f}s',
                      label+' supported Poisson',label+' multi-view surface'])),'mesh movie frame')
                if (i+1)%100==0: print(label,'mesh timeline',i+1,flush=True)
            receipt=finish_video(folder,seq,masked,movie,identity)
        movies[label+'_meshes']=receipt
        del old_scene,new_scene
    write(VIDEO/'receipt.json',dict(movies=movies,fixed_views=local,timestamps=member(times),frames=1846,
                                    redacted_frames=masked,privacy_verified=True,freeze_sha256=sha(freeze)))


if __name__=='__main__':
    main()
