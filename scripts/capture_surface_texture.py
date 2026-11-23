"""Freeze repeated training observations and measure native-supported texture evidence."""
import argparse
import inspect
from pathlib import Path

import cv2
import numpy as np

from capture_surface_submaps import ROOT, RUN, FLIP, check, read, sha, write


def image(path):
    result = cv2.imread(str(path), cv2.IMREAD_COLOR)
    check(result is not None and result.shape == (720, 1280, 3), 'missing or invalid image: ' + str(path))
    return result.astype(np.float32) / 255.


def highpass(a):
    return a - cv2.GaussianBlur(a, (0, 0), 1.)


def gradient(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))


def warp(source_depth, target_depth, source_pose, target_pose, meta, crop):
    x0,y0,x1,y1 = crop
    y,x = np.mgrid[y0:y1,x0:x1]
    z = source_depth[y,x]
    kx,ky,cx,cy = meta['fl_x'],meta['fl_y'],meta['cx']-.5,meta['cy']-.5
    p = np.stack(((x-cx)*z/kx,(y-cy)*z/ky,z),axis=-1).reshape(-1,3)
    rel = np.linalg.solve(target_pose,source_pose)
    q = p @ rel[:3,:3].T + rel[:3,3]
    valid = np.isfinite(p).all(1) & (p[:,2]>=.3) & (p[:,2]<=6) & np.isfinite(q).all(1) & (q[:,2]>=.3) & (q[:,2]<=6)
    u = np.zeros(len(q)); v = np.zeros(len(q))
    u[valid] = kx*q[valid,0]/q[valid,2]+cx; v[valid] = ky*q[valid,1]/q[valid,2]+cy
    valid &= (u>=0)&(u<1279)&(v>=0)&(v<719)
    ui=np.clip(np.rint(u).astype(int),0,1279); vi=np.clip(np.rint(v).astype(int),0,719)
    observed=target_depth[vi,ui]
    valid &= np.isfinite(observed)&(observed>=.3)&(observed<=6)&(np.abs(observed-q[:,2])<=np.maximum(.05,.03*observed))
    # Nearest source surface at target pixel, avoiding hidden source points.
    ids=np.flatnonzero(valid); key=vi[ids]*1280+ui[ids]
    order=np.lexsort((q[ids,2],key)); sorted_ids=ids[order]; sorted_key=key[order]
    keep=sorted_ids[np.r_[True,sorted_key[1:]!=sorted_key[:-1]]] if len(ids) else ids
    valid[:]=False; valid[keep]=True
    world = p @ source_pose[:3,:3].T + source_pose[:3,3]
    va = source_pose[:3,3] - world; vb = target_pose[:3,3] - world
    denom = np.linalg.norm(va,axis=1)*np.linalg.norm(vb,axis=1)
    cosine = np.divide((va*vb).sum(1),denom,out=np.ones_like(denom),where=denom>0)
    angle = np.degrees(np.arccos(np.clip(cosine,-1,1))).reshape(z.shape)
    return u.reshape(z.shape).astype(np.float32),v.reshape(z.shape).astype(np.float32),valid.reshape(z.shape),angle


def choose_views(obj, meta, frames, groups, scene):
    name = obj['source']
    train = meta['train_filenames']
    source = frames[name]
    pose = np.asarray(source['transform_matrix']) @ FLIP
    depth = np.load(scene / source['depth_file_path'], mmap_mode='r')
    pos = train.index(name)
    candidates = sorted(range(len(train)), key=lambda i: (abs(i-pos), i))
    scored = []
    for i in [i for i in candidates if groups[train[i]] != groups[name]][:100]:
        other = train[i]
        target = frames[other]
        target_pose = np.asarray(target['transform_matrix']) @ FLIP
        baseline = float(np.linalg.norm(pose[:3,3]-target_pose[:3,3]))
        if baseline < .1:
            continue
        z = np.load(scene / target['depth_file_path'], mmap_mode='r')
        _, _, valid, angle = warp(depth,z,pose,target_pose,meta,obj['crop'])
        if not valid.any():
            continue
        median_angle = float(np.median(angle[valid]))
        if median_angle >= 3.:
            scored.append(dict(name=other, group=groups[other], baseline_m=baseline,
                               median_angle_deg=median_angle, observed_overlap=int(valid.sum())))
    chosen = []
    used = set()
    for row in sorted(scored, key=lambda r: (-r['observed_overlap'], r['name'])):
        if row['group'] not in used:
            chosen.append(row)
            used.add(row['group'])
            if len(chosen) == 4:
                break
    return chosen


def supported_depth(rendered, alpha, rel, meta, native, k):
    """Verify each model surface pixel in the native depth camera before warping it."""
    y,x = np.indices(rendered.shape, dtype=np.float64)
    z = np.asarray(rendered, dtype=np.float64)
    points = np.stack(((x-meta['cx']+.5)*z/meta['fl_x'],
                       (y-meta['cy']+.5)*z/meta['fl_y'], z), axis=-1)
    q = (points-rel[:3,3]) @ rel[:3,:3]
    ok = np.isfinite(q).all(axis=-1) & (q[:,:,2]>=.3) & (q[:,:,2]<=6) & (alpha>=.5)
    div = np.where(ok, q[:,:,2], 1.)
    u = np.rint(k['fx']*q[:,:,0]/div+k['cx']).astype(int)
    v = np.rint(k['fy']*q[:,:,1]/div+k['cy']).astype(int)
    ok &= (u>=0)&(v>=0)&(u<k['width'])&(v<k['height'])
    measured = native[np.clip(v,0,k['height']-1),np.clip(u,0,k['width']-1)]
    ok &= (measured>=.3)&(measured<=6.)&(np.abs(q[:,:,2]-measured)<=np.maximum(.05,.03*measured))
    return np.where(ok, z, 0.).astype('f4')


def patch_features(real, pred, depth):
    grad = gradient(real)
    err = np.abs(highpass(real)-highpass(pred)).mean(axis=2)
    mask = depth[:704] > 0
    def blocks(a):
        return a[:704].reshape(22,32,40,32).sum(axis=(1,3))
    count = blocks(mask)
    valid = count >= 512
    q = float(np.quantile(blocks(err[:704]*mask)[valid]/count[valid],.9)) if valid.any() else None
    gray = np.uint8(np.rint(cv2.cvtColor(real,cv2.COLOR_BGR2GRAY)*255))
    return dict(gray=gray, grad=grad, err=err, z=depth, q=q, q_blocks=int(valid.sum()))


def aligned_edges(source, target, u, v, valid, crop):
    """Track identifiable corners on source edges, initialized by the geometric warp."""
    x0,y0,x1,y1=crop
    mask=np.zeros(source.shape,'u1')
    mask[y0:y1,x0:x1]=valid.astype('u1')*255
    key=cv2.goodFeaturesToTrack(source,4096,.01,3,mask=mask)
    offsets=np.full(valid.shape,np.nan,dtype='f4')
    if key is None:
        return offsets
    x,y=np.rint(key[:,0]).astype(int).T
    predicted=np.stack((u[y-y0,x-x0],v[y-y0,x-x0]),axis=-1)[:,None,:]
    got,ok,_=cv2.calcOpticalFlowPyrLK(source,target,key,predicted.copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,winSize=(15,15),maxLevel=2)
    back,ok2,_=cv2.calcOpticalFlowPyrLK(target,source,got,key.copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,winSize=(15,15),maxLevel=2)
    keep=ok.ravel().astype(bool)&ok2.ravel().astype(bool)&(np.linalg.norm((back-key)[:,0],axis=1)<=1.)
    offsets[y[keep]-y0,x[keep]-x0]=np.linalg.norm((got-predicted)[keep,0],axis=1)
    return offsets


def diagnose(freeze_only=False):
    import torch
    from unittest.mock import patch
    import capture_quality_eval as evaluator
    from capture_surface_submaps import B2, PACKET, camera_parameters, immutable
    from dn_splatter.dn_model import DNSplatterModel
    out = RUN/'texture'/'B2'
    ip = RUN/'export/B2/identity.json'
    identity = read(ip)
    for key in ('config','cameras','checkpoint','selection','calibration'):
        check(sha(ROOT/identity[key]['path'])==identity[key]['sha256'], 'identity mismatch '+key)
    meta=read(ROOT/identity['cameras']['path'])
    scene=(ROOT/identity['cameras']['path']).parent
    train=meta['train_filenames']
    frames={f['file_path']:f for f in meta['frames']}
    groups={n:g['id'] for g in identity['groups'] for n in g['names']}
    spec=ROOT/'configs/capture_surface_texture_regions.json'
    rows=read(spec)['objects']
    check(len(rows)==3 and all(r['source'] in train for r in rows),'train-only fixed texture crops')
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        manifest=read(manifest_path)
        check(manifest['regions_sha256']==sha(spec) and manifest['identity_sha256']==sha(ip),'texture freeze mismatch')
    else:
        for obj in rows:
            obj['targets']=choose_views(obj,meta,frames,groups,scene)
        manifest=dict(regions_sha256=sha(spec),identity_sha256=sha(ip),objects=rows)
        immutable(manifest_path,manifest)
    rows=manifest['objects']
    if freeze_only:
        return
    names=sorted({n for o in rows for n in [o['source']]+[r['name'] for r in o['targets']]})
    selection={r['id']:r for r in read(ROOT/identity['selection']['path'])}
    cal=read(ROOT/identity['calibration']['path'])
    _,_,rel=camera_parameters(meta,cal)
    with patch.object(evaluator,'RUN',B2), patch.object(evaluator,'SCENE',scene):
        _,pipe,loaded,step=evaluator.load_model(ROOT/identity['config']['path'],identity['step'],'val')
    check(Path(loaded).resolve()==(ROOT/identity['checkpoint']['path']).resolve(),'texture checkpoint')
    ds=pipe.datamanager.train_dataset
    check([p.relative_to(scene).as_posix() for p in ds.image_filenames]==train,'training camera order')
    views={}; provenance={}; outputs_keys=set()
    for name in names:
        stem=Path(name).stem
        with torch.inference_mode():
            i=train.index(name)
            prediction=pipe.model.get_outputs_for_camera(camera=ds.cameras[i:i+1].to(pipe.device))
            outputs_keys.update(prediction.keys())
            rgb=prediction['rgb'].reshape(720,1280,3).cpu().numpy()[:,:,::-1]
            z=prediction['depth'].reshape(720,1280).cpu().numpy()
            alpha=prediction['accumulation'].reshape(720,1280).cpu().numpy()
        with np.load(PACKET/selection[stem]['depth']['file']) as data:
            native=data['depth'].astype('f4')/1000.
        pose=np.asarray(frames[name]['transform_matrix'])@FLIP
        depth=supported_depth(z,alpha,rel,meta,native,cal['cameras']['depth'])
        path=out/(stem+'.png')
        check(cv2.imwrite(str(path),np.uint8(np.rint(255*np.clip(rgb,0.,1.)))), 'lossless render')
        views[name]=patch_features(image(scene/name),rgb,depth)
        views[name]['pose']=pose
        provenance[name]=dict(rgb_sha256=sha(scene/name),render_sha256=sha(path),supported_pixels=int((depth>0).sum()))
    results=[]
    for obj in rows:
        name=obj['source']; crop=obj['crop']; x0,y0,x1,y1=crop
        src=views[name]; region=np.s_[y0:y1,x0:x1]
        pairs=[]
        for target in obj['targets']:
            target_name=target['name']
            dst=views[target_name]
            u,v,visible,angle=warp(src['z'],dst['z'],src['pose'],dst['pose'],meta,crop)
            errors=aligned_edges(src['gray'],dst['gray'],u,v,visible,crop)
            pairs.append(dict(name=target_name,mask=visible,angle=angle,edges=errors,
                              grad=cv2.remap(dst['grad'],u,v,cv2.INTER_LINEAR),
                              err=cv2.remap(dst['err'],u,v,cv2.INTER_LINEAR)))
        blocks=[]
        for y in range(0,y1-y0-31,32):
            for x in range(0,x1-x0-31,32):
                s=np.s_[y:y+32,x:x+32]; sm=src['z'][region][s]>0
                source_grad=float(src['grad'][region][s][sm].mean()) if sm.any() else None
                measured=[]
                for pair in pairs:
                    m=pair['mask'][s]; n=int(m.sum()); edges=pair['edges'][s]
                    edge=edges[np.isfinite(edges)&m]
                    measured.append(dict(name=pair['name'],visible_pixels=n,overlap=n/1024,
                        gradient=float(pair['grad'][s][m].mean()) if n else None,
                        edge_matches=len(edge),edge_median_px=float(np.median(edge)) if len(edge) else None,
                        angle_deg=float(np.median(pair['angle'][s][m])) if n else None,
                        target_error=float(pair['err'][s][m].mean()) if n else None,
                        source_error=float(src['err'][region][s][m].mean()) if n else None,
                        target_q90=views[pair['name']]['q']))
                energies=[m['gradient'] for m in measured if m['overlap']>=.5]
                median=float(np.median([source_grad]+energies)) if source_grad is not None and energies else None
                for m in measured:
                    m['passes']=bool(median is not None and m['overlap']>=.5 and m['angle_deg']>=3
                        and source_grad>=median and m['gradient']>=median
                        and m['edge_median_px'] is not None and m['edge_median_px']<=1.
                        and src['q'] is not None and m['target_q90'] is not None
                        and m['source_error']>src['q'] and m['target_error']>m['target_q90'])
                blocks.append(dict(x=x0+x,y=y0+y,source_pixels=int(sm.sum()),source_gradient=source_grad,
                                   clarity_median=median,source_q90=src['q'],pairs=measured,
                                   image_evidence_passed=any(m['passes'] for m in measured)))
        results.append(dict(name=obj['name'],source=name,crop=crop,targets=obj['targets'],blocks=blocks))
    # Inspect both the live return and the exact forward code that built it.
    code=inspect.getsource(DNSplatterModel.get_outputs)
    keys=sorted(outputs_keys)
    if any(any(word in key.lower() for word in ('contrib','gaussian_id','weights_per_pixel')) for key in keys):
        raise RuntimeError('Renderer exposes attribution candidates; verify pixel contributions before a split decision')
    path=Path(inspect.getsourcefile(DNSplatterModel))
    from hashlib import sha256
    passed=sum(b['image_evidence_passed'] for o in results for b in o['blocks'])
    write(out/'receipt.json',dict(status='diagnosed',run_id=RUN.name,texture_split_enabled=False,
        reason='The installed model returned aggregate image buffers with no per-pixel contributor IDs and weights.',
        attribution=dict(model_source_sha256=sha(path),get_outputs_sha256=sha256(code.encode()).hexdigest(),
            runtime_output_keys=keys,verified_contributor_ids=False),objects=results,eligible_image_blocks=passed,
        extra_parent_ids=[],manifest_sha256=sha(manifest_path),identity_sha256=sha(ip),
        checkpoint_sha256=identity['checkpoint']['sha256'],cameras_sha256=identity['cameras']['sha256'],
        frames=provenance,reference_quantile_blocks={n:v['q_blocks'] for n,v in views.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    diagnose(parser.parse_args().freeze)
