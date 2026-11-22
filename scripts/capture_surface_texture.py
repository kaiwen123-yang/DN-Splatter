"""Measure fixed train-view texture evidence; never infer Gaussian IDs from depth.

Manifest JSON contains exactly three objects, each with name, source, target and
crop [x0,y0,x1,y1]. Frame names must belong to the frozen training split. RGB
renders are supplied as lossless PNGs in --renders/<frame stem>.png. Run after
freezing the manifest, before examining this run's validation or test outputs.
"""
import argparse
import json
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


def block_quantile(real, pred):
    delta = np.abs(highpass(real) - highpass(pred)).mean(axis=2)
    # 720 is not divisible by 32: retain all complete blocks; report their count.
    h, w = delta.shape
    values = delta[:h//32*32, :w//32*32].reshape(h//32,32,w//32,32).mean((1,3))
    return float(np.quantile(values, .9)), int(values.size), delta


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


def measure(manifest, label, renders):
    identity_path=RUN/'export'/label/'identity.json'; identity=read(identity_path)
    for key in ('cameras','checkpoint'):
        check(sha(ROOT/identity[key]['path'])==identity[key]['sha256'], 'identity mismatch '+key)
    meta=read(ROOT/identity['cameras']['path']); scene=(ROOT/identity['cameras']['path']).parent
    frames={f['file_path']:f for f in meta['frames']}; train=set(meta['train_filenames'])
    groups={name:g['id'] for g in identity['groups'] for name in g['names']}
    objects=manifest['objects']; check(len(objects)==3 and len({o['name'] for o in objects})==3,'three distinct fixed objects required')
    provenance=read(renders/'receipt.json')
    check(provenance['identity_sha256']==sha(identity_path), 'render identity mismatch')
    check(provenance['checkpoint_sha256']==identity['checkpoint']['sha256'], 'render checkpoint mismatch')
    check(provenance['cameras_sha256']==identity['cameras']['sha256'], 'render camera mismatch')
    results=[]
    for obj in objects:
        a,b=obj['source'],obj['target']; crop=obj['crop']
        check(a in train and b in train and groups[a]!=groups[b], 'views must be distinct training groups')
        x0,y0,x1,y1=map(int,crop); check(0<=x0<x1<=1280 and 0<=y0<y1<=720,'invalid crop')
        fa,fb=frames[a],frames[b]; pa=np.asarray(fa['transform_matrix'])@FLIP; pb=np.asarray(fb['transform_matrix'])@FLIP
        ra,rb=image(scene/a),image(scene/b)
        for name in (a,b):
            entry=provenance['frames'][name]; path=renders/(Path(name).stem+'.png')
            check(entry['sha256']==sha(path), 'render hash mismatch '+name)
        ma,mb=image(renders/(Path(a).stem+'.png')),image(renders/(Path(b).stem+'.png'))
        za=np.load(scene/fa['depth_file_path']); zb=np.load(scene/fb['depth_file_path'])
        u,v,valid,angle=warp(za,zb,pa,pb,meta,(x0,y0,x1,y1)); sl=np.s_[y0:y1,x0:x1]
        ga,gb=gradient(ra),gradient(rb); warped=cv2.remap(gb,u,v,cv2.INTER_LINEAR)
        src_edges=(ga[sl] > np.quantile(ga[sl][valid],.75)) & valid if valid.any() else np.zeros_like(valid)
        dst_edges=(gb > np.quantile(gb,.75)).astype(np.uint8)
        distances=cv2.distanceTransform(1-dst_edges,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        edge=cv2.remap(distances,u,v,cv2.INTER_LINEAR)
        qa,na,ea=block_quantile(ra,ma); qb,nb,eb=block_quantile(rb,mb)
        ebt=cv2.remap(eb,u,v,cv2.INTER_LINEAR); center=pa[:3,3]-pb[:3,3]
        records=[]
        for yy in range(0,y1-y0-31,32):
            for xx in range(0,x1-x0-31,32):
                s=np.s_[yy:yy+32,xx:xx+32]; m=valid[s]; n=int(m.sum())
                if not n: continue
                se=src_edges[s]; ed=edge[s][se]
                g0=float(ga[sl][s][m].mean()); g1=float(warped[s][m].mean()); median=(g0+g1)/2
                e0=float(ea[sl][s][m].mean()); e1=float(ebt[s][m].mean())
                med=float(np.median(ed)) if len(ed) else None
                angle_median=float(np.median(angle[s][m]))
                independent=float(np.linalg.norm(center)) >= .1 and angle_median >= 3
                # Two views alone cannot establish that both are above the
                # median of a larger set of observations of the same block.
                clear0=g0>=median; clear1=g1>=median
                eligible=False
                records.append(dict(x=x0+xx,y=y0+yy,valid_pixels=n,overlap=n/1024,edge_samples=len(ed),edge_median_px=med,
                    source_gradient=g0,target_gradient=g1,clarity_median=median,source_clear=clear0,target_clear=clear1,clear_both=clear0 and clear1,
                    surface_ray_angle_median_deg=angle_median,independent_viewpoints=independent,
                    source_highpass_error=e0,target_highpass_error=e1,source_q90=qa,target_q90=qb,
                    eligible=eligible))
        rays_a=pa[:3,3]- (pa[:3,3]+pa[:3,2])
        rays_b=pb[:3,3]- (pb[:3,3]+pb[:3,2])
        view_angle=float(np.degrees(np.arccos(np.clip(np.dot(rays_a,rays_b)/(np.linalg.norm(rays_a)*np.linalg.norm(rays_b)),-1,1))))
        results.append(dict(name=obj['name'],source=a,target=b,crop=[x0,y0,x1,y1],source_group=groups[a],target_group=groups[b],
            baseline_m=float(np.linalg.norm(center)),camera_axis_angle_deg=view_angle,valid_pixels=int(valid.sum()),crop_pixels=int(valid.size),
            edge_median_px=float(np.median(edge[src_edges])) if src_edges.any() else None,
            residual_quantiles=dict(source=qa,target=qb,source_blocks=na,target_blocks=nb),blocks=records))
    # Record the inspected renderer implementation rather than inventing IDs from
    # a projected position or treating rasterizer tile intersections as visibility.
    from dn_splatter.dn_model import DNSplatterModel
    renderer_source=inspect.getsourcefile(DNSplatterModel)
    renderer_text=Path(renderer_source).read_text(encoding='utf-8')
    attribution=dict(renderer_source=str(renderer_source),renderer_sha256=sha(Path(renderer_source)),
        get_outputs_source=inspect.getsource(DNSplatterModel.get_outputs),
        mentions_gaussian_ids=('gaussian_ids' in renderer_text),
        contributing_ids_verified=False,
        reason='No per-pixel contribution IDs were supplied and verified against the rendered pixels; spatial or tile candidates are insufficient.')
    write(RUN/'texture'/label/'receipt.json',dict(schema_version=1,identity_sha256=sha(identity_path),manifest=manifest,
        manifest_sha256=sha(Path(manifest['_path'])),render_receipt_sha256=sha(renders/'receipt.json'),results=results,eligible_blocks=sum(r['eligible'] for o in results for r in o['blocks']),
        texture_split_enabled=False,extra_parent_ids=[],attribution=attribution,
        status='incomplete', note='Two-view clarity cannot establish the multi-view median; actual contributing IDs are also unverified. Do not use for training decisions.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('model',choices=('B2','B3'))
    parser.add_argument('--manifest',type=Path,required=True); parser.add_argument('--renders',type=Path,required=True)
    args=parser.parse_args(); manifest=read(args.manifest); manifest['_path']=str(args.manifest.resolve())
    measure(manifest,args.model,args.renders)

