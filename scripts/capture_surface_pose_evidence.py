"""Check the remaining blurred camera against measured planes and RGB correspondences."""
from pathlib import Path

import cv2
import numpy as np

from capture_native_neighbor_check import native_observation
from capture_surface_repeatability import edge_distance
from capture_surface_submaps import ROOT,RUN,B2,PACKET,FLIP,read,sha,write,check,camera_parameters


def main():
    original=RUN/'geometry/pose/diagnostics.json'
    receipt=read(original)
    inp=B2/'inputs'; meta=read(inp/'transforms.json')
    check(receipt['input_sha256']==sha(inp/'transforms.json'),'previous map identity')
    names={int(Path(f['file_path']).stem[-5:]):f for f in meta['frames'] if f['file_path'] in set(meta['train_filenames'])}
    ids=(1513,1514,1515)
    cal=read(PACKET/'calibration.json'); k=cal['cameras']['depth']
    _,_,rel=camera_parameters(meta,cal)
    packets={f['id']:f for f in read(PACKET/'selection.json')}
    obs={i:native_observation(PACKET,names[i],packets,rel) for i in ids}
    focus={}
    for i in ids:
        image=cv2.imread(str(inp/names[i]['file_path']),cv2.IMREAD_GRAYSCALE)
        check(image is not None and image.shape==(720,1280),'RGB input')
        lap=cv2.Laplacian(image,cv2.CV_64F)
        focus[str(i)]=dict(laplacian_variance=float(lap.var()),image_sha256=sha(inp/names[i]['file_path']))
    separate=receipt['separate_1514']
    z,pose=obs[1514]
    y,x=np.mgrid[4:k['height']:8,4:k['width']:8]
    zz=z[y,x]
    good=(zz>=.3)&(zz<=6)&(edge_distance(z)[y,x]>2)
    ps=np.column_stack(((x[good]-k['cx'])*zz[good]/k['fx'],(y[good]-k['cy'])*zz[good]/k['fy'],zz[good]))
    planes=[]
    for plane in separate['native_planes']:
        normal=np.array(plane['normal_depth_camera']); center=np.array(plane['center_depth_camera_m'])
        same=np.abs((ps-center)@normal)<=.02
        points=ps[same]@pose[:3,:3].T+pose[:3,3]
        row=dict(source_interior_samples=len(points),neighbours={})
        for target in (1513,1515):
            depth,other=obs[target]
            q=(points-other[:3,3])@other[:3,:3]
            q=q[(q[:,2]>=.3)&(q[:,2]<=6)]
            u=np.rint(k['fx']*q[:,0]/q[:,2]+k['cx']).astype(int)
            v=np.rint(k['fy']*q[:,1]/q[:,2]+k['cy']).astype(int)
            keep=(u>=0)&(u<k['width'])&(v>=0)&(v<k['height'])
            q,u,v=q[keep],u[keep],v[keep]
            d=depth[v,u]
            keep=(edge_distance(depth)[v,u]>2)&(d>=.3)&(d<=6)
            q,u,v,d=q[keep],u[keep],v[keep],d[keep]
            dz=q[:,2]-d
            reliable=np.abs(dz)<=np.maximum(.05,.03*d)
            surface=np.column_stack(((u-k['cx'])*d/k['fx'],(v-k['cy'])*d/k['fy'],d))
            src=((surface@other[:3,:3].T+other[:3,3])-pose[:3,3])@pose[:3,:3]
            signed=(src-center)@normal
            row['neighbours'][str(target)]=dict(projected_interior=len(q),same_surface=int(reliable.sum()),
                free_space_conflict=int((dz < -np.maximum(.05,.03*d)).sum()),
                occluded=int((dz>np.maximum(.05,.03*d)).sum()),
                median_point_to_plane_m=float(np.median(signed[reliable])) if reliable.any() else None)
        planes.append(row)
    matches=separate['rgb_neighbour_checks']
    prior_path=B2/'geometry/pose.json'; previous=read(prior_path)['results']['1514']
    spec=read(ROOT/'configs/capture_surface_quality_protocol.json')['pose']
    max_local=sum(m['matches'] for m in matches)
    # Reuse the real wider-map PnP attempt at the same geometry. If evidence supports
    # a fit, it needs a separate joint plane/RGB solve before deciding this camera.
    check(previous['inliers']<spec['fit_pnp_inliers_min'] and max_local<spec['fit_pnp_inliers_min'],
          '1514 correspondence population can support fitting; solve and validate a constrained correction')
    check(not separate['accepted'] and receipt['accepted_pose_changes']==0,'unexpected previous pose decision')
    details=dict(sharpness=focus,planes=planes,neighbour_RGB=matches,total_local_RGB_before_dedup=max_local,
        wider_train_correspondences=previous['correspondences'],wider_train_anchors=previous['anchors'],
        wider_PnP_inliers=previous['inliers'],wider_receipt_sha256=sha(prior_path),
        reason='The two local RGB match lists together contain fewer than 20 matches before deduplication. The wider fixed-map PnP fit returned zero inliers. Native planes alone do not establish the rigid correction.')
    write(RUN/'geometry/pose/decision.json',dict(receipt,separate_1514=dict(separate,verification=details),
        joint_receipt_sha256=sha(original),final_cameras_sha256=sha(RUN/'geometry/pose/transforms.json')))


if __name__=='__main__':
    main()
