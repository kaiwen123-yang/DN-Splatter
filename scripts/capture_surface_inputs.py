"""Freeze B3 input identities and decide once whether training changes the optimization."""
import os
from pathlib import Path
import shutil
import subprocess

from capture_surface_submaps import B2, ROOT, RUN, check, read, sha, write
from capture_surface_train import training_decision


def independent_copy(source, target, digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        check(not source.samefile(target) and sha(target)==digest, 'existing input differs: '+str(target))
        return
    tmp=target.with_suffix(target.suffix+'.part')
    subprocess.run(['cp','--reflink=auto','--',str(source),str(tmp)],check=True)
    check(not source.samefile(tmp) and sha(tmp)==digest, 'copy identity: '+str(source))
    os.replace(tmp,target)


def prepare():
    receipt_paths=dict(pose=RUN/'geometry/pose/diagnostics.json',
                       weights=RUN/'weights/B2/receipt.json',texture=RUN/'texture/B2/receipt.json')
    results={k:read(p) for k,p in receipt_paths.items()}
    binding={k:dict(path=str(p.relative_to(RUN)),sha256=sha(p)) for k,p in receipt_paths.items()}
    decision=training_decision(**results)
    meta=read(B2/'inputs/transforms.json')
    pose=RUN/'geometry/pose/transforms.json'
    check(sha(pose)==results['pose']['input_sha256'] and decision['accepted_pose_changes']==0,
          'existing pose receipt no longer represents unchanged B2 cameras')
    weights=results['weights']
    check(weights['valid_pixels']>0 and len(weights['diagnostics'])==4 and
          [f['frame'] for f in weights['frames']]==meta['train_filenames'], 'incomplete train weights')
    check(results['texture']['status']=='diagnosed' and len(results['texture']['objects'])==3 and
          results['texture']['attribution']['runtime_output_keys'], 'texture diagnosis missing')
    write(RUN/'state/training_decision.json', dict(decision,diagnostic_receipts=binding))
    if decision['decision']=='skipped_equivalent':
        return
    scene=RUN/'inputs'
    inventory=read(B2/'inputs/training_manifest.json')['files']
    files={row['path']:row['sha256'] for row in inventory}
    check(len(files)==len(inventory) and all(sum(Path(k).parts[0]==d for k in files)==1846
          for d in ('images','depth','normals_from_pretrain')), 'B2 signed population')
    check(files['transforms.json']==sha(pose) and files[meta['ply_file_path']]==
          read(B2/'geometry/initialization.json')['ply']['sha256'], 'initialization pose or PLY mismatch')
    for name,digest in files.items():
        independent_copy(B2/'inputs'/name,scene/name,digest)
    aux={}
    for name in ('tracks.npz','added_tracks.npz'):
        p=B2/'geometry'/name
        aux[name]=sha(p)
        independent_copy(p,RUN/'geometry'/name,aux[name])
    geometry=read(scene/'geometry.json')
    manifest=dict(run_id=RUN.name, files=files, diagnostic_receipts=binding, **results,
                  initialization=dict(source='B2 training-only sensor and qualified SfM seed cloud reused at identical camera SHA',
                      ply=meta['ply_file_path'],sha256=files[meta['ply_file_path']],points=geometry['init_points'],
                      sensor=geometry['sensor_points'],sfm=geometry['sfm_points'],track_sha256=aux))
    target=scene/'training_manifest.json'
    if target.exists():
        check(read(target)==manifest,'B3 manifest changed across recovery')
    else:
        write(target,manifest)


if __name__=='__main__':
    prepare()
